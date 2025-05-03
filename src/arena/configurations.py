import torch

from arena.matrix_samplers import NormalSampler, StrongSampler, \
    StrictlyWeakSampler, NonWeakSampler
from arena.objectives import (
    AggregationTime,
    DualProjectionDualFeasibilityObjective,
    DualProjectionSlacknessFeasibilityObjective,
    DualProjectionPrimalFeasibilityObjective,
)

OBJECTIVE_LISTS = {
    "runtime": [
        AggregationTime(matrix_sampler=cls(m, m, m-1, dtype), device=device, iterations=1)
        for dtype in [torch.float32, torch.float64]
        for device in ["cpu", "cuda"]
        for m in [2, 4, 32, 128]
        for cls in [NormalSampler, StrongSampler, StrictlyWeakSampler, NonWeakSampler]
    ],
    "project_weights": [
        DualProjectionPrimalFeasibilityObjective(m=10, device="cpu", dtype=torch.float32, iterations=10),
        DualProjectionDualFeasibilityObjective(m=10, device="cpu", dtype=torch.float32, iterations=10),
        DualProjectionSlacknessFeasibilityObjective(m=10, device="cpu", dtype=torch.float32, iterations=10),
    ],
}
