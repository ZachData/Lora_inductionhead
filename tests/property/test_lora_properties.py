"""Invariants for src/indbw/lora.py that must hold over random input, per
CLAUDE.md's property-test list. Closed-form point checks live in
tests/unit/test_lora.py; this file checks they hold generically, not just
on the hand-picked cases -- in particular "rank(Delta W) <= r for every
parameterization", the invariant CLAUDE.md names explicitly.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from indbw.lora import antisymmetric_delta, symmetric_delta, unconstrained_delta

_seeds = st.integers(min_value=0, max_value=2**31 - 1)
_ranks = st.integers(min_value=1, max_value=4)
_pair_ranks = st.integers(min_value=1, max_value=3).map(lambda k: 2 * k)  # even, >= 2
_dims = st.integers(min_value=6, max_value=10)


@given(seed=_seeds, r=_ranks, d_out=_dims, d_in=_dims)
def test_unconstrained_rank_bounded_by_r(seed: int, r: int, d_out: int, d_in: int) -> None:
    rng = np.random.default_rng(seed)
    B = rng.standard_normal((d_out, r))
    A = rng.standard_normal((r, d_in))
    delta = unconstrained_delta(B, A, alpha=float(r))
    assert np.linalg.matrix_rank(delta) <= r


@given(seed=_seeds, rank=_pair_ranks, d=_dims)
def test_symmetric_rank_bounded_by_rank(seed: int, rank: int, d: int) -> None:
    rng = np.random.default_rng(seed)
    k = rank // 2
    U = rng.standard_normal((d, k))
    V = rng.standard_normal((d, k))
    delta = symmetric_delta(U, V, alpha=float(rank), rank=rank)
    assert np.linalg.matrix_rank(delta) <= rank


@given(seed=_seeds, rank=_pair_ranks, d=_dims)
def test_antisymmetric_rank_bounded_by_rank(seed: int, rank: int, d: int) -> None:
    rng = np.random.default_rng(seed)
    k = rank // 2
    U = rng.standard_normal((d, k))
    V = rng.standard_normal((d, k))
    delta = antisymmetric_delta(U, V, alpha=float(rank), rank=rank)
    assert np.linalg.matrix_rank(delta) <= rank


@given(seed=_seeds, rank=_pair_ranks, d=_dims)
def test_antisymmetric_delta_always_even_rank(seed: int, rank: int, d: int) -> None:
    # docs/mathematics.md §6: a nonzero real antisymmetric matrix has even
    # rank, generically for any random pair of factors.
    rng = np.random.default_rng(seed)
    k = rank // 2
    U = rng.standard_normal((d, k))
    V = rng.standard_normal((d, k))
    delta = antisymmetric_delta(U, V, alpha=float(rank), rank=rank)
    assert np.linalg.matrix_rank(delta) % 2 == 0


@given(seed=_seeds, rank=_pair_ranks, d=_dims)
def test_symmetric_delta_always_symmetric(seed: int, rank: int, d: int) -> None:
    rng = np.random.default_rng(seed)
    k = rank // 2
    U = rng.standard_normal((d, k))
    V = rng.standard_normal((d, k))
    delta = symmetric_delta(U, V, alpha=float(rank), rank=rank)
    np.testing.assert_allclose(delta, delta.T, rtol=1e-8, atol=1e-8)


@given(seed=_seeds, rank=_pair_ranks, d=_dims)
def test_antisymmetric_delta_always_antisymmetric(seed: int, rank: int, d: int) -> None:
    rng = np.random.default_rng(seed)
    k = rank // 2
    U = rng.standard_normal((d, k))
    V = rng.standard_normal((d, k))
    delta = antisymmetric_delta(U, V, alpha=float(rank), rank=rank)
    np.testing.assert_allclose(delta, -delta.T, rtol=1e-8, atol=1e-8)
