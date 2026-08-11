"""Closed-form oracles for src/indbw/algebra.py, per PROJECT.md §3.

Every identity here has an exact answer, so assertions are rtol=1e-12
(float64 machine precision), never a loosened tolerance (CLAUDE.md,
"Numerical tolerance policy").
"""

from __future__ import annotations

import numpy as np
import pytest

from indbw.algebra import phi, principal_angles, sym_antisym_split, truncate_svd


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- sym_antisym_split -------------------------------------------------


def test_split_reconstructs_original() -> None:
    M = _rng(0).standard_normal((5, 5))
    S, Lam = sym_antisym_split(M)
    np.testing.assert_allclose(S + Lam, M, rtol=1e-12, atol=1e-12)


def test_split_parts_are_symmetric_and_antisymmetric() -> None:
    M = _rng(1).standard_normal((6, 6))
    S, Lam = sym_antisym_split(M)
    np.testing.assert_allclose(S, S.T, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(Lam, -Lam.T, rtol=1e-12, atol=1e-12)


def test_split_requires_square_matrix() -> None:
    with pytest.raises(ValueError):
        sym_antisym_split(np.zeros((3, 4)))


def test_antisymmetric_matrix_has_even_rank() -> None:
    # A nonzero real antisymmetric matrix has even rank (PROJECT.md §3).
    # Odd ambient dimension forces at least one zero eigenvalue, so this
    # is the sharpest case to check.
    for n in (5, 7, 9):
        M = _rng(n).standard_normal((n, n))
        _, Lam = sym_antisym_split(M)
        rank = np.linalg.matrix_rank(Lam)
        assert rank % 2 == 0, f"antisymmetric {n}x{n} matrix has odd rank {rank}"


# --- phi -----------------------------------------------------------------


def test_phi_rank_one_closed_form() -> None:
    # phi(b a^T) = 1/2 (1 - c^2), c = cosine(a, b).
    rng = _rng(2)
    a = rng.standard_normal(8)
    b = rng.standard_normal(8)
    a_hat, b_hat = a / np.linalg.norm(a), b / np.linalg.norm(b)
    c = float(a_hat @ b_hat)
    M = np.outer(b, a)
    expected = 0.5 * (1 - c**2)
    assert phi(M) == pytest.approx(expected, rel=1e-12, abs=1e-12)


def test_phi_rank_one_orthogonal_hits_max() -> None:
    a = np.array([1.0, 0.0, 0.0])
    b = np.array([0.0, 1.0, 0.0])
    assert phi(np.outer(b, a)) == pytest.approx(0.5, rel=1e-12)


def test_phi_zero_for_symmetric_matrix() -> None:
    A = _rng(3).standard_normal((5, 5))
    S = A + A.T
    assert phi(S) == pytest.approx(0.0, abs=1e-12)


def test_phi_one_for_antisymmetric_matrix() -> None:
    A = _rng(4).standard_normal((5, 5))
    Lam = A - A.T
    assert phi(Lam) == pytest.approx(1.0, rel=1e-12)


def test_phi_half_for_off_diagonal_block() -> None:
    # M = P_U M P_V with U perp V (disjoint coordinate blocks) forces
    # tr(M^2) = 0, hence phi = 1/2 exactly, at any rank (PROJECT.md §3).
    rng = _rng(5)
    A = rng.standard_normal((3, 3))
    M = np.zeros((6, 6))
    M[:3, 3:] = A  # rows 0-2 = U, cols 3-5 = V, block off-diagonal
    assert phi(M) == pytest.approx(0.5, rel=1e-12)


def test_phi_zero_matrix_raises() -> None:
    with pytest.raises(ValueError):
        phi(np.zeros((4, 4)))


def test_phi_constant_matrix_is_not_silently_wrong() -> None:
    # A constant matrix is symmetric, so phi = 0 is the correct closed-form
    # answer here (not a silent-failure case) -- guard that it isn't nan.
    M = np.full((4, 4), 3.0)
    assert phi(M) == pytest.approx(0.0, abs=1e-12)


# --- principal_angles ------------------------------------------------------


def test_principal_angles_zero_for_identical_subspace() -> None:
    U = _rng(6).standard_normal((10, 3))
    angles = principal_angles(U, U)
    # arccos near cos=1 is ill-conditioned; 1e-7 reflects that, not laziness.
    np.testing.assert_allclose(angles, 0.0, atol=1e-7)


def test_principal_angles_half_pi_for_orthogonal_subspaces() -> None:
    U = np.eye(6)[:, :3]
    V = np.eye(6)[:, 3:]
    angles = principal_angles(U, V)
    np.testing.assert_allclose(angles, np.pi / 2, rtol=1e-12, atol=1e-12)


def test_principal_angles_requires_matching_row_dimension() -> None:
    with pytest.raises(ValueError):
        principal_angles(np.zeros((5, 2)), np.zeros((6, 2)))


# --- truncate_svd ------------------------------------------------------


def test_truncate_svd_matches_eckart_young() -> None:
    M = _rng(7).standard_normal((6, 4))
    r = 2
    approx = truncate_svd(M, r)
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    expected = (U[:, :r] * s[:r]) @ Vt[:r, :]
    np.testing.assert_allclose(approx, expected, rtol=1e-12, atol=1e-12)
    assert np.linalg.matrix_rank(approx) <= r


def test_truncate_svd_full_rank_reconstructs_exactly() -> None:
    M = _rng(8).standard_normal((5, 5))
    approx = truncate_svd(M, 5)
    np.testing.assert_allclose(approx, M, rtol=1e-12, atol=1e-12)


def test_truncate_svd_rejects_out_of_range_rank() -> None:
    M = np.zeros((4, 3))
    with pytest.raises(ValueError):
        truncate_svd(M, 0)
    with pytest.raises(ValueError):
        truncate_svd(M, 4)
