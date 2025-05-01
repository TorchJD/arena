from typing import Callable

import torch
from torch import Tensor
from torch.nn.functional import normalize

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

        cumulative_slackness = 0.0
        for _ in range(self.iterations):
            u = torch.rand(self.m, device=self.device, dtype=self.dtype)
            G = _generate_gramian(self.m, self.device, self.dtype)
            w = project_weights(u, G)
            dual_gap = w - u
            primal_gap = G @ w
            slackness = dual_gap @ primal_gap
            cumulative_slackness += slackness.abs().item()

        average_primal_gap = cumulative_slackness / self.iterations
        return average_primal_gap


def _generate_gramian(m: int, device: str, dtype: torch.dtype) -> Tensor:
    matrix = _sample_strictly_weak_matrix(m, m, m-2).to(device=device, dtype=dtype)
    # matrix = torch.randn([m, m], device=device, dtype=dtype)
    return matrix @ matrix.T


def _sample_matrix(m: int, n: int, rank: int) -> Tensor:
    """Samples a random matrix A of shape [m, n] with provided rank."""

    U = _sample_orthonormal_matrix(m)
    Vt = _sample_orthonormal_matrix(n)
    S = torch.diag(torch.abs(torch.randn([rank])))
    A = U[:, :rank] @ S @ Vt[:rank, :]
    return A


def _sample_strong_matrix(m: int, n: int, rank: int) -> Tensor:
    """
    Samples a random strongly stationary matrix A of shape [m, n] with provided rank.

    Definition: A matrix A is said to be strongly stationary if there exists a vector 0 < v such
    that v^T A = 0.

    This is done by sampling a positive v, and by then sampling a matrix orthogonal to v.
    """

    assert 1 < m
    assert 0 < rank <= min(m - 1, n)

    v = torch.abs(torch.randn([m]))
    U1 = normalize(v, dim=0).unsqueeze(1)
    U2 = _sample_semi_orthonormal_complement(U1)
    Vt = _sample_orthonormal_matrix(n)
    S = torch.diag(torch.abs(torch.randn([rank])))
    A = U2[:, :rank] @ S @ Vt[:rank, :]
    return A


def _sample_strictly_weak_matrix(m: int, n: int, rank: int) -> Tensor:
    """
    Samples a random strictly weakly stationary matrix A of shape [m, n] with provided rank.

    Definition: A matrix A is said to be weakly stationary if there exists a vector 0 <= v, v != 0,
    such that v^T A = 0.

    Definition: A matrix A is said to be strictly weakly stationary if it is weakly stationary and
    not strongly stationary, i.e. if there exists a vector 0 <= v, v != 0, such that v^T A = 0 and
    there exists no vector 0 < w with w^T A = 0.

    This is done by sampling two unit-norm vectors v, v', whose sum u is a positive vector. These
    two vectors are also non-negative and non-zero, and are furthermore orthogonal. Then, a matrix
    A, orthogonal to v, is sampled. By its orthogonality to v, A is weakly stationary. Moreover,
    since v' is a non-negative left-singular vector of A with positive singular value s, any 0 < w
    satisfies w^T A != 0. Otherwise, we would have 0 = w^T A A^T v' = s w^T v' > 0, which is a
    contradiction. A is thus also not strongly stationary.
    """

    assert 1 < m
    assert 0 < rank <= min(m - 1, n)

    u = torch.abs(torch.randn([m]))
    split_index = torch.randint(1, m, []).item()
    shuffled_range = torch.randperm(m)
    v = torch.zeros(m)
    v[shuffled_range[:split_index]] = normalize(u[shuffled_range[:split_index]], dim=0)
    v_prime = torch.zeros(m)
    v_prime[shuffled_range[split_index:]] = normalize(u[shuffled_range[split_index:]], dim=0)
    U1 = torch.stack([v, v_prime]).T
    U2 = _sample_semi_orthonormal_complement(U1)
    U = torch.hstack([U1, U2])
    Vt = _sample_orthonormal_matrix(n)
    S = torch.diag(torch.abs(torch.randn([rank])))
    A = U[:, 1 : rank + 1] @ S @ Vt[:rank, :]
    return A


def _sample_non_weak_matrix(m: int, n: int, rank: int) -> Tensor:
    """
    Samples a random non weakly-stationary matrix A of shape [m, n] with provided rank.

    This is done by sampling a positive u, and by then sampling a matrix A that has u as one of its
    left-singular vectors, with positive singular value s. Any 0 <= v, v != 0, satisfies v^T A != 0.
    Otherwise, we would have 0 = v^T A A^T u = s v^T u > 0, which is a contradiction. A is thus not
    weakly stationary.
    """

    assert 0 < rank <= min(m, n)

    u = torch.abs(torch.randn([m]))
    U1 = normalize(u, dim=0).unsqueeze(1)
    U2 = _sample_semi_orthonormal_complement(U1)
    U = torch.hstack([U1, U2])
    Vt = _sample_orthonormal_matrix(n)
    S = torch.diag(torch.abs(torch.randn([rank])))
    A = U[:, :rank] @ S @ Vt[:rank, :]
    return A


def _sample_orthonormal_matrix(dim: int) -> Tensor:
    """Uniformly samples a random orthonormal matrix of shape [dim, dim]."""

    return _sample_semi_orthonormal_complement(torch.zeros([dim, 0]))


def _sample_semi_orthonormal_complement(Q: Tensor) -> Tensor:
    """
    Uniformly samples a random semi-orthonormal matrix Q' (i.e. Q'^T Q' = I) of shape [m, m-k]
    orthogonal to Q, i.e. such that the concatenation [Q, Q'] is an orthonormal matrix.

    :param Q: A semi-orthonormal matrix (i.e. Q^T Q = I) of shape [m, k], with k <= m.
    """

    m, k = Q.shape
    A = torch.randn([m, m - k])

    # project A onto the orthogonal complement of Q
    A_proj = A - Q @ (Q.T @ A)

    Q_prime, _ = torch.linalg.qr(A_proj)
    return Q_prime
