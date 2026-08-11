"""Invariants for src/indbw/algebra.py that must hold over random input,
per CLAUDE.md's property-test list. Closed-form point checks live in
tests/unit/test_algebra.py; this file checks they hold generically.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from indbw.algebra import phi, principal_angles, sym_antisym_split, truncate_svd

_seeds = st.integers(min_value=0, max_value=2**31 - 1)
_sizes = st.integers(min_value=2, max_value=8)


def _random_square(seed: int, n: int) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((n, n))


def _random_orthogonal(seed: int, n: int) -> np.ndarray:
    A = np.random.default_rng(seed).standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    return Q


@given(seed=_seeds, n=_sizes)
def test_frobenius_norm_decomposes(seed: int, n: int) -> None:
    M = _random_square(seed, n)
    S, Lam = sym_antisym_split(M)
    lhs = float(np.sum(M * M))
    rhs = float(np.sum(S * S)) + float(np.sum(Lam * Lam))
    assert lhs == pytest.approx(rhs, rel=1e-10)


@given(seed=_seeds, n=_sizes)
def test_sym_antisym_frobenius_orthogonal(seed: int, n: int) -> None:
    M = _random_square(seed, n)
    S, Lam = sym_antisym_split(M)
    inner = float(np.sum(S * Lam))
    assert inner == pytest.approx(0.0, abs=1e-10)


@given(seed=_seeds, n=_sizes)
def test_phi_in_unit_interval(seed: int, n: int) -> None:
    M = _random_square(seed, n)
    if np.allclose(M, 0):
        return
    value = phi(M)
    assert 0.0 <= value <= 1.0


@given(seed=_seeds, n=_sizes)
def test_phi_invariant_under_orthogonal_conjugation(seed: int, n: int) -> None:
    M = _random_square(seed, n)
    Q = _random_orthogonal(seed + 1, n)
    conjugated = Q @ M @ Q.T
    # Chained QR + SVD accumulates more float error than a single closed-form
    # identity; 1e-8 reflects that chain, not a loosened definition.
    assert phi(conjugated) == pytest.approx(phi(M), rel=1e-8, abs=1e-8)


@given(seed=_seeds, n=_sizes)
def test_principal_angles_in_valid_range(seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    k1, k2 = min(2, n), min(2, n)
    U = rng.standard_normal((n + 2, k1))
    V = rng.standard_normal((n + 2, k2))
    angles = principal_angles(U, V)
    assert np.all(angles >= -1e-9)
    assert np.all(angles <= np.pi / 2 + 1e-9)


@given(seed=_seeds, n=_sizes)
def test_truncate_svd_rank_bounded(seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    M = rng.standard_normal((n, n))
    r = max(1, n - 1)
    approx = truncate_svd(M, r)
    assert np.linalg.matrix_rank(approx) <= r
