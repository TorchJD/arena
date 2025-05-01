import torch

from arena.aggregation_time import AggregationTime

OBJECTIVE_LISTS = {
    "runtime_0": [
        AggregationTime(m=1, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=10, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=20, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=30, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=40, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=50, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=60, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=70, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=80, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=90, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=100, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        AggregationTime(m=120, n=1000, device="cuda", dtype=torch.float32, iterations=10),
    ],
}
