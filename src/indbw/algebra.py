"""Symmetric/antisymmetric decomposition, phi, principal angles, SVD truncation.

PROJECT.md §3. Test file: tests/unit/test_algebra.py — closed-form oracles,
exact to rtol=1e-12. Write the test before implementing anything here.
"""

from __future__ import annotations

from typing import cast

import numpy as np


def sym_antisym_split(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split M = S + Lambda, S = (M+M^T)/2 symmetric, Lambda = (M-M^T)/2 antisymmetric."""
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"sym_antisym_split requires a square matrix, got shape {M.shape}")
    S = 0.5 * (M + M.T)
    Lam = 0.5 * (M - M.T)
    return S, Lam


def phi(M: np.ndarray) -> float:
    """Antisymmetric fraction ||Lambda||_F^2 / ||M||_F^2 (PROJECT.md §3)."""
    _, Lam = sym_antisym_split(M)
    norm_sq = float(np.sum(M * M))
    if norm_sq == 0.0:
        raise ValueError("phi is undefined for the zero matrix")
    value = float(np.sum(Lam * Lam)) / norm_sq
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"phi out of range: {value}")
    return value


def principal_angles(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Principal angles (radians, in [0, pi/2]) between the column spaces of U and V."""
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("principal_angles requires 2D matrices (columns = basis vectors)")
    if U.shape[0] != V.shape[0]:
        raise ValueError(
            f"principal_angles requires matrices with the same row dimension, "
            f"got {U.shape[0]} and {V.shape[0]}"
        )
    Qu, _ = np.linalg.qr(U)
    Qv, _ = np.linalg.qr(V)
    sigma = np.linalg.svd(Qu.T @ Qv, compute_uv=False)
    sigma = np.clip(sigma, -1.0, 1.0)
    return cast(np.ndarray, np.arccos(sigma))


def truncate_svd(M: np.ndarray, r: int) -> np.ndarray:
    """Best rank-r Frobenius approximation of M (Eckart-Young)."""
    if M.ndim != 2:
        raise ValueError("truncate_svd requires a 2D matrix")
    if r < 1 or r > min(M.shape):
        raise ValueError(f"rank {r} out of bounds for shape {M.shape}")
    U, s, Vt = np.linalg.svd(M, full_matrices=False)
    return cast(np.ndarray, (U[:, :r] * s[:r]) @ Vt[:r, :])
