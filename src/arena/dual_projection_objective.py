from typing import Callable

import torch
from torch import Tensor

from arena.objective import Objective


class DualProjectionPrimalFeasibilityObjective(Objective):
    def __init__(self, m: int, device: str, dtype: torch.dtype, iterations: int):
        self.m = m
        self.device = device
        self.dtype = dtype
        self.iterations = iterations

    def __call__(self, project_weights: Callable[[Tensor, Tensor], Tensor]) -> float:
        """Returns the primal feasibility gap."""
        u = torch.rand(self.m, device=self.device, dtype=self.dtype)
        G = _generate_gramian(self.m, self.device, self.dtype)
        _ = project_weights(u, G)

        # Synchronize before timing if using CUDA
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        cumulative_primal_gap_differences = 0.0
        for _ in range(self.iterations):
            u = torch.rand(self.m, device=self.device, dtype=self.dtype)
            G = _generate_gramian(self.m, self.device, self.dtype)
            w = project_weights(u, G)
            primal_gap = G @ w
            primal_gap_positive_part = primal_gap[primal_gap >= 0]
            difference = torch.abs(primal_gap_positive_part.norm() - primal_gap.norm())
            cumulative_primal_gap_differences += difference.item()

        average_primal_gap = cumulative_primal_gap_differences / self.iterations
        return average_primal_gap


class DualProjectionDualFeasibilityObjective(Objective):
    def __init__(self, m: int, device: str, dtype: torch.dtype, iterations: int):
        self.m = m
        self.device = device
        self.dtype = dtype
        self.iterations = iterations

    def __call__(self, project_weights: Callable[[Tensor, Tensor], Tensor]) -> float:
        """Returns the primal feasibility gap."""
        u = torch.rand(self.m, device=self.device, dtype=self.dtype)
        G = _generate_gramian(self.m, self.device, self.dtype)
        _ = project_weights(u, G)

        # Synchronize before timing if using CUDA
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        cumulative_dual_gap_differences = 0.0
        for _ in range(self.iterations):
            u = torch.rand(self.m, device=self.device, dtype=self.dtype)
            G = _generate_gramian(self.m, self.device, self.dtype)
            w = project_weights(u, G)
            dual_gap = w - u
            dual_gap_positive_part = dual_gap[dual_gap >= 0.0]
            difference = torch.abs(dual_gap_positive_part.norm() - dual_gap.norm())
            cumulative_dual_gap_differences += difference.item()

        average_primal_gap = cumulative_dual_gap_differences / self.iterations
        return average_primal_gap


class DualProjectionSlacknessFeasibilityObjective(Objective):
    def __init__(self, m: int, device: str, dtype: torch.dtype, iterations: int):
        self.m = m
        self.device = device
        self.dtype = dtype
        self.iterations = iterations

    def __call__(self, project_weights: Callable[[Tensor, Tensor], Tensor]) -> float:
        """Returns the primal feasibility gap."""
        u = torch.rand(self.m, device=self.device, dtype=self.dtype)
        G = _generate_gramian(self.m, self.device, self.dtype)
        _ = project_weights(u, G)

        # Synchronize before timing if using CUDA
        if self.device.startswith("cuda"):
            torch.cuda.synchronize()

        cumulative_slackness = 0.0
        for _ in range(self.iterations):
            u = torch.rand(self.m, device=self.device, dtype=self.dtype)
            G = _generate_gramian(self.m, self.device, self.dtype)
            w = project_weights(u, G)
            dual_gap = w - u
            primal_gap = G @ w
            slackness = dual_gap @ primal_gap
            cumulative_slackness += slackness.item()

        average_primal_gap = cumulative_slackness / self.iterations
        return average_primal_gap

def _generate_gramian(m: int, device: str, dtype: torch.dtype) -> Tensor:
    matrix = torch.randn([m, m], device=device, dtype=dtype)
    return matrix @ matrix.T

