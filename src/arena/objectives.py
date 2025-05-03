import time
from abc import ABC, abstractmethod
from typing import Callable

import torch
from torch import Tensor
from torchjd.aggregation import Aggregator

from arena.matrix_samplers import generate_gramian, MatrixSampler


class Objective(ABC):
    @abstractmethod
    def __call__(self, func: Callable) -> float:
        """Returns the value of the objective obtained for the provided function."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AggregatorObjective(Objective, ABC):
    @abstractmethod
    def __call__(self, A: Aggregator) -> float:
        """Returns the value of the objective obtained for the provided aggregator."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class AggregationTime(AggregatorObjective):
    def __init__(self, matrix_sampler: MatrixSampler, device: str, iterations: int):
        self.matrix_sampler = matrix_sampler
        self.device = device
        self.iterations = iterations

    def __call__(self, A: Aggregator) -> float:
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
        return (
            f"{self.__class__.__name__}(matrix_sampler={self.matrix_sampler}, device={self.device},"
            f" iterations={self.iterations})"
        )

    def __str__(self) -> str:
        return f"AT({self.matrix_sampler}, {self.device}, x{self.iterations})"


class DualProjectionPrimalFeasibilityObjective(Objective):
    def __init__(self, matrix_sampler: MatrixSampler, device: str, iterations: int):
        self.matrix_sampler = matrix_sampler
        self.device = device
        self.iterations = iterations

    def __call__(self, project_weights: Callable[[Tensor, Tensor], Tensor]) -> float:
        """Returns the primal feasibility gap."""

        J = self.matrix_sampler().to(device=self.device)
        G = J @ J.T
        u = torch.rand(G.shape[0], device=G.device, dtype=G.dtype)
        _ = project_weights(u, G)

        cumulative_primal_gap_differences = 0.0
        for _ in range(self.iterations):
            J = self.matrix_sampler().to(device=self.device)
            G = J @ J.T
            u = torch.rand(G.shape[0], device=G.device, dtype=G.dtype)
            w = project_weights(u, G)
            primal_gap = G @ w
            primal_gap_positive_part = primal_gap[primal_gap >= 0]
            primal_gap_norm = primal_gap.norm()
            if primal_gap_norm <= 1e-08:
                difference = 0.0
            else:
                difference = torch.abs(primal_gap_positive_part.norm() - primal_gap_norm) / primal_gap_norm
            cumulative_primal_gap_differences += difference.item()

        average_primal_gap = cumulative_primal_gap_differences / self.iterations
        return average_primal_gap


class DualProjectionDualFeasibilityObjective(Objective):
    def __init__(self, matrix_sampler: MatrixSampler, device: str, iterations: int):
        self.matrix_sampler = matrix_sampler
        self.device = device
        self.iterations = iterations

    def __call__(self, project_weights: Callable[[Tensor, Tensor], Tensor]) -> float:
        """Returns the primal feasibility gap."""
        J = self.matrix_sampler().to(device=self.device)
        G = J @ J.T
        u = torch.rand(G.shape[0], device=G.device, dtype=G.dtype)
        _ = project_weights(u, G)

        cumulative_dual_gap_differences = 0.0
        for _ in range(self.iterations):
            J = self.matrix_sampler().to(device=self.device)
            G = J @ J.T
            u = torch.rand(G.shape[0], device=G.device, dtype=G.dtype)
            w = project_weights(u, G)
            dual_gap = w - u
            dual_gap_positive_part = dual_gap[dual_gap >= 0.0]
            dual_gap_norm = dual_gap.norm()
            if dual_gap_norm <= 1e-08:
                difference = 0.0
            else:
                difference = torch.abs(dual_gap_positive_part.norm() - dual_gap_norm) / dual_gap_norm
            cumulative_dual_gap_differences += difference.item()

        average_primal_gap = cumulative_dual_gap_differences / self.iterations
        return average_primal_gap


class DualProjectionSlacknessFeasibilityObjective(Objective):
    def __init__(self, matrix_sampler: MatrixSampler, device: str, iterations: int):
        self.matrix_sampler = matrix_sampler
        self.device = device
        self.iterations = iterations

    def __call__(self, project_weights: Callable[[Tensor, Tensor], Tensor]) -> float:
        """Returns the primal feasibility gap."""
        J = self.matrix_sampler().to(device=self.device)
        G = J @ J.T
        u = torch.rand(G.shape[0], device=G.device, dtype=G.dtype)
        _ = project_weights(u, G)

        cumulative_slackness = 0.0
        for _ in range(self.iterations):
            J = self.matrix_sampler().to(device=self.device)
            G = J @ J.T
            u = torch.rand(G.shape[0], device=G.device, dtype=G.dtype)
            w = project_weights(u, G)
            dual_gap = w - u
            primal_gap = G @ w
            norm_product = dual_gap.norm() * primal_gap.norm()
            if norm_product <= 1e-08:
                slackness = 0.0
            else:
                slackness = dual_gap @ primal_gap / norm_product
            cumulative_slackness += slackness.abs().item()

        average_primal_gap = cumulative_slackness / self.iterations
        return average_primal_gap
