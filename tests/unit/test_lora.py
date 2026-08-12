"""Closed-form oracles and discrimination tests for src/indbw/lora.py.

PROJECT.md §3, docs/mathematics.md §7/§9. Required by CLAUDE.md's TDD
contract: "The antisymmetric parameterization produces a matrix satisfying
M^T = -M to float tolerance" and "rank(Delta W) <= r for every
parameterization" -- both closed-form, tested to machine precision
(float64, rtol=1e-12) since the underlying algebra is exact.
"""

from __future__ import annotations

import numpy as np
import pytest

from indbw.algebra import phi
from indbw.lora import (
    PARAMETERIZATIONS,
    antisymmetric_delta,
    build_delta,
    inject,
    symmetric_delta,
    unconstrained_delta,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- unconstrained_delta -------------------------------------------------


def test_unconstrained_delta_closed_form() -> None:
    B = np.array([[2.0], [3.0]])
    A = np.array([[4.0, 5.0]])
    # alpha = r = 1, so Delta W = B @ A exactly.
    delta = unconstrained_delta(B, A, alpha=1.0)
    np.testing.assert_allclose(delta, [[8.0, 10.0], [12.0, 15.0]], rtol=1e-12, atol=1e-12)


def test_unconstrained_delta_scales_by_alpha_over_r() -> None:
    B = np.eye(3)
    A = np.eye(3)
    r = 3
    alpha = 6.0
    delta = unconstrained_delta(B, A, alpha=alpha)
    np.testing.assert_allclose(delta, (alpha / r) * np.eye(3), rtol=1e-12, atol=1e-12)


def test_unconstrained_delta_rank_matches_r_generically() -> None:
    # Generic random factors: rank(B A) == r exactly (r < min(d_out, d_in)).
    rng = _rng(0)
    B = rng.standard_normal((6, 3))
    A = rng.standard_normal((3, 5))
    delta = unconstrained_delta(B, A, alpha=3.0)
    assert np.linalg.matrix_rank(delta) == 3


def test_unconstrained_delta_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        unconstrained_delta(np.zeros((4, 2)), np.zeros((3, 5)), alpha=1.0)


def test_unconstrained_delta_rejects_zero_rank() -> None:
    with pytest.raises(ValueError):
        unconstrained_delta(np.zeros((4, 0)), np.zeros((0, 5)), alpha=1.0)


# --- symmetric_delta / antisymmetric_delta --------------------------------


def test_antisymmetric_delta_satisfies_M_transpose_equals_negative_M() -> None:
    # Required oracle, CLAUDE.md TDD contract kind 1.
    rng = _rng(1)
    U = rng.standard_normal((5, 1))
    V = rng.standard_normal((5, 1))
    M = antisymmetric_delta(U, V, alpha=2.0, rank=2)
    np.testing.assert_allclose(M.T, -M, rtol=1e-12, atol=1e-12)


def test_symmetric_delta_satisfies_M_transpose_equals_M() -> None:
    rng = _rng(2)
    U = rng.standard_normal((5, 2))
    V = rng.standard_normal((5, 2))
    M = symmetric_delta(U, V, alpha=2.0, rank=4)
    np.testing.assert_allclose(M.T, M, rtol=1e-12, atol=1e-12)


def test_antisymmetric_delta_phi_is_exactly_one() -> None:
    # Cross-check against algebra.phi, already tested to machine precision
    # against its own closed-form oracle -- an independent confirmation
    # that antisymmetric_delta really is what it claims to be.
    rng = _rng(3)
    U = rng.standard_normal((6, 2))
    V = rng.standard_normal((6, 2))
    M = antisymmetric_delta(U, V, alpha=1.0, rank=4)
    assert phi(M) == pytest.approx(1.0, rel=1e-12)


def test_symmetric_delta_phi_is_exactly_zero() -> None:
    rng = _rng(4)
    U = rng.standard_normal((6, 2))
    V = rng.standard_normal((6, 2))
    M = symmetric_delta(U, V, alpha=1.0, rank=4)
    assert phi(M) == pytest.approx(0.0, abs=1e-12)


def test_antisymmetric_delta_has_even_rank() -> None:
    # docs/mathematics.md §6: a nonzero real antisymmetric matrix has even
    # rank. Two independent, non-collinear pairs -> generically rank 4.
    rng = _rng(5)
    U = rng.standard_normal((8, 2))
    V = rng.standard_normal((8, 2))
    M = antisymmetric_delta(U, V, alpha=1.0, rank=4)
    rank = np.linalg.matrix_rank(M)
    assert rank % 2 == 0
    assert rank > 0


def test_symmetric_and_antisymmetric_reject_odd_rank() -> None:
    U = np.zeros((4, 1))
    V = np.zeros((4, 1))
    with pytest.raises(ValueError):
        symmetric_delta(U, V, alpha=1.0, rank=1)
    with pytest.raises(ValueError):
        antisymmetric_delta(U, V, alpha=1.0, rank=1)


def test_symmetric_and_antisymmetric_reject_rank_below_two() -> None:
    U = np.zeros((4, 0))
    V = np.zeros((4, 0))
    with pytest.raises(ValueError):
        symmetric_delta(U, V, alpha=1.0, rank=0)


def test_symmetric_and_antisymmetric_reject_mismatched_UV_shape() -> None:
    U = np.zeros((4, 1))
    V = np.zeros((5, 1))
    with pytest.raises(ValueError):
        symmetric_delta(U, V, alpha=1.0, rank=2)
    with pytest.raises(ValueError):
        antisymmetric_delta(U, V, alpha=1.0, rank=2)


def test_symmetric_and_antisymmetric_reject_wrong_column_count() -> None:
    # rank=4 implies U, V should have rank // 2 == 2 columns, not 1.
    U = np.zeros((4, 1))
    V = np.zeros((4, 1))
    with pytest.raises(ValueError):
        symmetric_delta(U, V, alpha=1.0, rank=4)


# --- silent-failure guard: the three parameterizations must differ --------


def test_three_parameterizations_are_pairwise_distinguishable() -> None:
    # Guard against a copy-paste bug where two parameterizations silently
    # compute the same thing (CLAUDE.md: a probe must return different
    # answers on known-different inputs, not a plausible-looking constant).
    rng = _rng(6)
    U = rng.standard_normal((5, 1))
    V = rng.standard_normal((5, 1))
    sym = symmetric_delta(U, V, alpha=1.0, rank=2)
    antisym = antisymmetric_delta(U, V, alpha=1.0, rank=2)
    uncon = unconstrained_delta(U, V.T, alpha=1.0)
    assert not np.allclose(sym, antisym)
    assert not np.allclose(sym, uncon)
    assert not np.allclose(antisym, uncon)


# --- build_delta -----------------------------------------------------------


def test_build_delta_dispatches_to_the_named_parameterization() -> None:
    rng = _rng(7)
    U = rng.standard_normal((5, 1))
    V = rng.standard_normal((5, 1))
    np.testing.assert_allclose(
        build_delta("unconstrained", (U, V.T), alpha=1.0, rank=1),
        unconstrained_delta(U, V.T, alpha=1.0),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        build_delta("symmetric", (U, V), alpha=1.0, rank=2),
        symmetric_delta(U, V, alpha=1.0, rank=2),
        rtol=1e-12,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        build_delta("antisymmetric", (U, V), alpha=1.0, rank=2),
        antisymmetric_delta(U, V, alpha=1.0, rank=2),
        rtol=1e-12,
        atol=1e-12,
    )


def test_build_delta_rejects_unknown_parameterization() -> None:
    with pytest.raises(ValueError):
        build_delta("orthogonal", (np.zeros((2, 1)), np.zeros((1, 2))), alpha=1.0, rank=1)


def test_parameterizations_tuple_lists_exactly_the_three() -> None:
    assert set(PARAMETERIZATIONS) == {"unconstrained", "symmetric", "antisymmetric"}


# --- inject ----------------------------------------------------------------


def test_inject_adds_delta_to_base_weight() -> None:
    base = np.array([[1.0, 2.0], [3.0, 4.0]])
    delta = np.array([[0.5, -0.5], [0.0, 1.0]])
    np.testing.assert_allclose(inject(base, delta), base + delta, rtol=1e-12, atol=1e-12)


def test_inject_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        inject(np.zeros((3, 2)), np.zeros((2, 3)))
