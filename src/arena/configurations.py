import torch

from arena.matrix_samplers import NonWeakSampler, NormalSampler, StrictlyWeakSampler, StrongSampler
from arena.objectives import (
    AggregationTime,
    DualProjectionDualFeasibilityObjective,
    DualProjectionPrimalFeasibilityObjective,
    DualProjectionSlacknessFeasibilityObjective,
)

OBJECTIVE_LISTS = {
    "runtime": [
        AggregationTime(matrix_sampler=cls(m, m, m - 1, torch.float32), device=device, iterations=1)
        for cls in [NormalSampler, StrongSampler, StrictlyWeakSampler, NonWeakSampler]
        for device in ["cpu", "cuda"]
        for m in [2, 4, 32, 128]
    ],
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
