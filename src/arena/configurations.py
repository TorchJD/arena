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
        AggregationTime(matrix_sampler=cls(m, m, m-1, torch.float32), device=device, iterations=1)
        for cls in [NormalSampler, StrongSampler, StrictlyWeakSampler, NonWeakSampler]
        for device in ["cpu", "cuda"]
        for m in [2, 4, 32, 128]
    ],
    "project_weights": [
        DualProjectionPrimalFeasibilityObjective(m=10, device="cpu", dtype=torch.float32, iterations=10),
        DualProjectionDualFeasibilityObjective(m=10, device="cpu", dtype=torch.float32, iterations=10),
        DualProjectionSlacknessFeasibilityObjective(m=10, device="cpu", dtype=torch.float32, iterations=10),
    ],
}
