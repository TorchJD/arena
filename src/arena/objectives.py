import time
from abc import ABC, abstractmethod
from typing import Callable, Never

import torch
from torch import Tensor
from torch.nn import Linear, MSELoss, ReLU, Sequential, Module
from torch.optim import SGD
from torchjd.aggregation import Mean, Aggregator, UPGrad

from arena.matrix_samplers import MatrixSampler, NonWeakSampler, NormalSampler, StrictlyWeakSampler, StrongSampler


class Objective(ABC):
    @abstractmethod
    def __call__(self, func: Callable[..., Never]) -> float:
        """Returns the value of the objective obtained for the provided function."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AggregatorObjective(Objective, ABC):
    @abstractmethod
    def __call__(self, A: Callable[[Tensor], Tensor]) -> float:
        """Returns the value of the objective obtained for the provided aggregator."""


class AggregationTime(AggregatorObjective):
    def __init__(self, matrix_sampler: MatrixSampler, device: str, iterations: int):
        self.matrix_sampler = matrix_sampler
        self.device = device
        self.iterations = iterations

    def __call__(self, A: Callable[[Tensor], Tensor]) -> float:
        J = self.matrix_sampler().to(device=self.device)
        A(J)

        # Synchronize before timing if using CUDA
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        start = time.perf_counter()
        for _ in range(self.iterations):
            A(J)
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()
        end = time.perf_counter()

        average_runtime = (end - start) / self.iterations
        return average_runtime

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(matrix_sampler={self.matrix_sampler}, device={self.device}," f" iterations={self.iterations})"

    def __str__(self) -> str:
        return f"AT({self.matrix_sampler}, {self.device}, x{self.iterations})"


class ForwardBackwardTime(Objective):
    def __init__(self, ns: list[int], device: str, iterations: int):
        self.ns = ns
        self.device = device
        shapes = zip(ns[:-1], ns[1:])
        layers = [Linear(n, m) for n, m in shapes]
        self.model = Sequential(*layers)
        self.iterations = iterations

    def __call__(self, forward_backward: Callable):
        aggregator = UPGrad()
        total_time = 0.0
        for i in range(self.iterations + 1):
            x = torch.randn(self.ns[0], device=self.device)

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            start = time.perf_counter()

            forward_backward(self.model, x, aggregator)

            if self.device.startswith("cuda"):
                torch.cuda.synchronize()
            end = time.perf_counter()
            if i > 0:
                total_time += end - start
        average_runtime = total_time / self.iterations
        return average_runtime

        def __repr__(self) -> str:
            return f"{self.__class__.__name__}(ns={self.ns}, device={self.device}," f" iterations={self.iterations})"

        def __str__(self) -> str:
            return f"AT({self.matrix_sampler}, {self.device}, x{self.iterations})"


class DualProjectionPrimalFeasibilityObjective(Objective):
    def __init__(self, matrix_sampler: MatrixSampler, device: str, iterations: int):
        self.matrix_sampler = matrix_sampler
        self.device = device
        self.iterations = iterations

    def __call__(self, project_weights: Callable[[Tensor, Tensor], Tensor]) -> float:
        """Returns the primal feasibility gap."""
        _, _, slackness = compute_kkt_conditions(self.matrix_sampler, self.device, self.iterations, project_weights)
        return slackness

    def __str__(self) -> str:
        return f"KKTP({self.matrix_sampler}, {self.device}, x{self.iterations})"


class DualProjectionDualFeasibilityObjective(Objective):
    def __init__(self, matrix_sampler: MatrixSampler, device: str, iterations: int):
        self.matrix_sampler = matrix_sampler
        self.device = device
        self.iterations = iterations

    def __call__(self, project_weights: Callable[[Tensor, Tensor], Tensor]) -> float:
        """Returns the primal feasibility gap."""
        _, dual_gap, _ = compute_kkt_conditions(self.matrix_sampler, self.device, self.iterations, project_weights)
        return dual_gap

    def __str__(self) -> str:
        return f"KKTD({self.matrix_sampler}, {self.device}, x{self.iterations})"


class DualProjectionSlacknessFeasibilityObjective(Objective):
    def __init__(self, matrix_sampler: MatrixSampler, device: str, iterations: int):
        self.matrix_sampler = matrix_sampler
        self.device = device
        self.iterations = iterations

    def __call__(self, project_weights: Callable[[Tensor, Tensor], Tensor]) -> float:
        """Returns the primal feasibility gap."""
        primal_gap, _, _ = compute_kkt_conditions(self.matrix_sampler, self.device, self.iterations, project_weights)
        return primal_gap

    def __str__(self) -> str:
        return f"KKTS({self.matrix_sampler}, {self.device}, x{self.iterations})"


class MTLBackwardTime(Objective):
    def __init__(self, n_tasks: int, device: str, iterations: int):
        self.device = device
        self.iterations = iterations
        self.shared_module = Sequential(Linear(10, 5), ReLU(), Linear(5, 3), ReLU()).to(device=self.device)

        self.n_tasks = n_tasks
        self.task_modules = [Linear(3, 1).to(device=self.device) for _ in range(n_tasks)]
        params = list(self.shared_module.parameters())
        for head in self.task_modules:
            params += list(head.parameters())

        self.loss_fn = MSELoss()
        self.optimizer = SGD(params, lr=0.1)
        self.inputs = torch.randn(8, 16, 10, device=self.device)  # 8 batches of 16 random input vectors of length 10
        self.targets = [torch.randn(8, 16, 1, device=self.device) for _ in range(self.n_tasks)]
        self.aggregator = Mean()

    def __call__(self, mtl_backward) -> float:
        total_time = 0.0
        for _ in range(self.iterations):
            for i in range(len(self.inputs)):
                input = self.inputs[i]
                targets = [target[i] for target in self.targets]
                features = self.shared_module(input)
                outputs = [head(features) for head in self.task_modules]
                losses = [self.loss_fn(output, target) for output, target in zip(outputs, targets)]

                self.optimizer.zero_grad()
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                start = time.perf_counter()
                mtl_backward(losses=losses, features=features, aggregator=self.aggregator)
                if self.device.startswith("cuda"):
                    torch.cuda.synchronize()
                end = time.perf_counter()
                total_time += end - start
                self.optimizer.step()
        average_runtime = total_time / self.iterations
        return average_runtime

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(device={self.device}," f" iterations={self.iterations})"

    def __str__(self) -> str:
        return f"MTLBT({self.device}, x{self.iterations})"


def compute_kkt_conditions(
    matrix_sampler: MatrixSampler, device: str, iterations: int, project_weights: Callable[[Tensor, Tensor], Tensor]
) -> tuple[float, float, float]:
    J = matrix_sampler().to(device=device)
    G = J @ J.T
    u = torch.rand(G.shape[0], device=G.device, dtype=G.dtype)
    _ = project_weights(u, G)

    zero = torch.tensor([0.0], device=G.device, dtype=G.dtype)

    cumulative_primal_gap_differences = 0.0
    cumulative_dual_gap_differences = 0.0
    cumulative_slackness = 0.0
    for _ in range(iterations):
        J = matrix_sampler().to(device=device)
        G = J @ J.T
        u = torch.rand(G.shape[0], device=G.device, dtype=G.dtype)
        w = project_weights(u, G)
        dual_gap = w - u
        primal_gap = G @ w

        # Primal gap
        primal_gap_positive_part = primal_gap[primal_gap >= 0]
        primal_gap_norm = primal_gap.norm()
        if primal_gap_norm <= 1e-08:
            difference = zero
        else:
            difference = torch.abs(primal_gap_positive_part.norm() - primal_gap_norm) / primal_gap_norm
        cumulative_primal_gap_differences += difference.item()

        # Dual gap
        dual_gap_positive_part = dual_gap[dual_gap >= 0.0]
        dual_gap_norm = dual_gap.norm()
        if dual_gap_norm <= 1e-08:
            difference = zero
        else:
            difference = torch.abs(dual_gap_positive_part.norm() - dual_gap_norm) / dual_gap_norm
        cumulative_dual_gap_differences += difference.item()

        # Slackness
        norm_product = dual_gap.norm() * primal_gap.norm()
        if norm_product <= 1e-08:
            slackness = zero
        else:
            slackness = dual_gap @ primal_gap / norm_product
        cumulative_slackness += slackness.abs().item()

    average_primal_gap = cumulative_primal_gap_differences / iterations
    average_dual_gap = cumulative_dual_gap_differences / iterations
    average_slackness = cumulative_slackness / iterations
    return average_primal_gap, average_dual_gap, average_slackness


OBJECTIVE_LISTS = {
    "runtime": [
        AggregationTime(matrix_sampler=cls(m, m, m - 1, torch.float32), device=device, iterations=1)
        for cls in [NormalSampler, StrongSampler, StrictlyWeakSampler, NonWeakSampler]
        for device in ["cpu", "cuda"]
        for m in [2, 4, 32, 128]
    ],
    "mtl_backward_runtime": [MTLBackwardTime(n_tasks=50, device=device, iterations=100) for device in ["cpu", "cuda"]],
    "project_weights": [
        DualProjectionPrimalFeasibilityObjective(matrix_sampler=cls(m, m, m - 1, torch.float32), device=device, iterations=10)
        for cls in [NormalSampler, StrongSampler, StrictlyWeakSampler, NonWeakSampler]
        for device in ["cpu", "cuda"]
        for m in [2, 4, 32, 128]
    ]
    + [
        DualProjectionDualFeasibilityObjective(matrix_sampler=cls(m, m, m - 1, torch.float32), device=device, iterations=10)
        for cls in [NormalSampler, StrongSampler, StrictlyWeakSampler, NonWeakSampler]
        for device in ["cpu", "cuda"]
        for m in [2, 4, 32, 128]
    ]
    + [
        DualProjectionSlacknessFeasibilityObjective(matrix_sampler=cls(m, m, m - 1, torch.float32), device=device, iterations=10)
        for cls in [NormalSampler, StrongSampler, StrictlyWeakSampler, NonWeakSampler]
        for device in ["cpu", "cuda"]
        for m in [2, 4, 32, 128]
    ],
}
