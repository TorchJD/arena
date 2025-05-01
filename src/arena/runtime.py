import time

import torch
from torchjd.aggregation import Aggregator

from .objective import Objective


class Runtime(Objective):
    def __init__(self, m: int, n: int, device: str, dtype: torch.dtype, iterations: int):
        self.m = m
        self.n = n
        self.device = device
        self.dtype = dtype
        self.iterations = iterations

    def __call__(self, A: Aggregator) -> float:
        J = torch.ones(self.m, self.n, device=self.device, dtype=self.dtype)
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
            f"{self.__class__.__name__}(m={self.m}, n={self.n}, device={self.device}, dtype="
            f"{self.dtype}, iterations={self.iterations})"
        )
