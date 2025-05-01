import torch

from arena.runtime import Runtime

OBJECTIVE_LISTS = {
    "runtime_0": [
        Runtime(m=1, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=10, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=20, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=30, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=40, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=50, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=60, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=70, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=80, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=90, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=100, n=1000, device="cuda", dtype=torch.float32, iterations=10),
        Runtime(m=120, n=1000, device="cuda", dtype=torch.float32, iterations=10),
    ],
}
