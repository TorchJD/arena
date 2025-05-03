from abc import ABC, abstractmethod

import torch
from torch import Tensor
from torch.nn.functional import normalize


def generate_gramian(m: int, device: str, dtype: torch.dtype) -> Tensor:
    matrix = _sample_strictly_weak_matrix(m, m, m - 2, dtype=dtype).to(device=device)
    # matrix = torch.randn([m, m], device=device, dtype=dtype)
    return matrix @ matrix.T


class MatrixSampler(ABC):
    def __init__(self, m: int, n: int, rank: int, dtype: torch.dtype):
        self.m = m
        self.n = n
        self.rank = rank
        self.dtype = dtype

    @abstractmethod
    def __call__(self) -> Tensor:
        """Samples a random matrix."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(m={self.m}, n={self.n}, rank={self.rank})"

    def __str__(self) -> str:
        return f"{self.__class__.__name__.replace("MatrixSampler", "")}({self.m}x{self.n}r{self.rank}:{str(self.dtype)[6:]})"


class NormalSampler(MatrixSampler):
    def __call__(self) -> Tensor:
        return _sample_matrix(self.m, self.n, self.rank, self.dtype)


class StrongSampler(MatrixSampler):
    def __call__(self) -> Tensor:
        return _sample_strong_matrix(self.m, self.n, self.rank, self.dtype)


class StrictlyWeakSampler(MatrixSampler):
    def __call__(self) -> Tensor:
        return _sample_strictly_weak_matrix(self.m, self.n, self.rank, self.dtype)


class NonWeakSampler(MatrixSampler):
    def __call__(self) -> Tensor:
        return _sample_non_weak_matrix(self.m, self.n, self.rank, self.dtype)


def _sample_matrix(m: int, n: int, rank: int, dtype: torch.dtype) -> Tensor:
    """Samples a random matrix A of shape [m, n] with provided rank."""

    U = _sample_orthonormal_matrix(m, dtype=dtype)
    Vt = _sample_orthonormal_matrix(n, dtype=dtype)
    S = torch.diag(torch.abs(torch.randn([rank], dtype=dtype)))
    A = U[:, :rank] @ S @ Vt[:rank, :]
    return A


def _sample_strong_matrix(m: int, n: int, rank: int, dtype: torch.dtype) -> Tensor:
    """
    Samples a random strongly stationary matrix A of shape [m, n] with provided rank.

    Definition: A matrix A is said to be strongly stationary if there exists a vector 0 < v such
    that v^T A = 0.

    This is done by sampling a positive v, and by then sampling a matrix orthogonal to v.
    """

    assert 1 < m
    assert 0 < rank <= min(m - 1, n)

    v = torch.abs(torch.randn([m], dtype=dtype))
    U1 = normalize(v, dim=0).unsqueeze(1)
    U2 = _sample_semi_orthonormal_complement(U1)
    Vt = _sample_orthonormal_matrix(n, dtype=dtype)
    S = torch.diag(torch.abs(torch.randn([rank], dtype=dtype)))
    A = U2[:, :rank] @ S @ Vt[:rank, :]
    return A


def _sample_strictly_weak_matrix(m: int, n: int, rank: int, dtype: torch.dtype) -> Tensor:
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

    u = torch.abs(torch.randn([m], dtype=dtype))
    split_index = torch.randint(1, m, []).item()
    shuffled_range = torch.randperm(m)
    v = torch.zeros(m, dtype=dtype)
    v[shuffled_range[:split_index]] = normalize(u[shuffled_range[:split_index]], dim=0)
    v_prime = torch.zeros(m, dtype=dtype)
    v_prime[shuffled_range[split_index:]] = normalize(u[shuffled_range[split_index:]], dim=0)
    U1 = torch.stack([v, v_prime]).T
    U2 = _sample_semi_orthonormal_complement(U1)
    U = torch.hstack([U1, U2])
    Vt = _sample_orthonormal_matrix(n, dtype=dtype)
    S = torch.diag(torch.abs(torch.randn([rank], dtype=dtype)))
    A = U[:, 1 : rank + 1] @ S @ Vt[:rank, :]
    return A


def _sample_non_weak_matrix(m: int, n: int, rank: int, dtype: torch.dtype) -> Tensor:
    """
    Samples a random non weakly-stationary matrix A of shape [m, n] with provided rank.

    This is done by sampling a positive u, and by then sampling a matrix A that has u as one of its
    left-singular vectors, with positive singular value s. Any 0 <= v, v != 0, satisfies v^T A != 0.
    Otherwise, we would have 0 = v^T A A^T u = s v^T u > 0, which is a contradiction. A is thus not
    weakly stationary.
    """

    assert 0 < rank <= min(m, n)

    u = torch.abs(torch.randn([m], dtype=dtype))
    U1 = normalize(u, dim=0).unsqueeze(1)
    U2 = _sample_semi_orthonormal_complement(U1)
    U = torch.hstack([U1, U2])
    Vt = _sample_orthonormal_matrix(n, dtype=dtype)
    S = torch.diag(torch.abs(torch.randn([rank], dtype=dtype)))
    A = U[:, :rank] @ S @ Vt[:rank, :]
    return A


def _sample_orthonormal_matrix(dim: int, dtype: torch.dtype) -> Tensor:
    """Uniformly samples a random orthonormal matrix of shape [dim, dim]."""

    return _sample_semi_orthonormal_complement(torch.zeros([dim, 0], dtype=dtype))


def _sample_semi_orthonormal_complement(Q: Tensor) -> Tensor:
    """
    Uniformly samples a random semi-orthonormal matrix Q' (i.e. Q'^T Q' = I) of shape [m, m-k]
    orthogonal to Q, i.e. such that the concatenation [Q, Q'] is an orthonormal matrix.

    :param Q: A semi-orthonormal matrix (i.e. Q^T Q = I) of shape [m, k], with k <= m.
    """

    dtype = Q.dtype
    m, k = Q.shape
    A = torch.randn([m, m - k], dtype=dtype)

    # project A onto the orthogonal complement of Q
    A_proj = A - Q @ (Q.T @ A)

    Q_prime, _ = torch.linalg.qr(A_proj)
    return Q_prime
