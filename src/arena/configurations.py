import torch

from arena.objectives import (
    AggregationTime,
    DualProjectionDualFeasibilityObjective,
    DualProjectionSlacknessFeasibilityObjective,
    DualProjectionPrimalFeasibilityObjective,
)

OBJECTIVE_LISTS = {
    "runtime": [
        AggregationTime(m=m, n=m, device=device, dtype=dtype, iterations=10)
        for dtype in [torch.float32]
        for device in ["cpu", "cuda"]
        for m in [2, 4, 32, 128]
    ],
    "project_weights": [
        DualProjectionPrimalFeasibilityObjective(m=10, device="cuda", dtype=torch.float32, iterations=10),
        DualProjectionDualFeasibilityObjective(m=10, device="cuda", dtype=torch.float32, iterations=10),
        DualProjectionSlacknessFeasibilityObjective(m=10, device="cuda", dtype=torch.float32, iterations=10),
    ],
}
