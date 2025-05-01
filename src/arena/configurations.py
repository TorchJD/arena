import torch

from arena.aggregation_time import AggregationTime
from arena.dual_projection_objective import DualProjectionPrimalFeasibilityObjective, \
    DualProjectionDualFeasibilityObjective, DualProjectionSlacknessFeasibilityObjective

OBJECTIVE_LISTS = {
    "runtime": [
        AggregationTime(m=m, n=m, device=device, dtype=dtype, iterations=10)
        for dtype in [torch.float32]
        for device in ["cpu", "cuda"]
        for m in [2, 5, 100]
    ],
    "project_weights": [
        DualProjectionPrimalFeasibilityObjective(m=10, device="cuda", dtype=torch.float32, iterations=10),
        DualProjectionDualFeasibilityObjective(m=10, device="cuda", dtype=torch.float32, iterations=10),
        DualProjectionSlacknessFeasibilityObjective(m=10, device="cuda", dtype=torch.float32, iterations=10),
    ],
}
