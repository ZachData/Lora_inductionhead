"""LoRA injection: unconstrained / symmetric / antisymmetric parameterizations.

PROJECT.md §3. Adapt exactly one matrix per bilinear form -- W_Q for QK,
W_O for OV (docs/mathematics.md §9). Same pattern as algebra.py/probes.py:
pure functions over plain arrays, no model loading (CLAUDE.md, fixtures
rule) -- train.py is responsible for wiring these into a real
HookedTransformer's weights.

Three parameterizations of the trainable rank-r delta:

  - unconstrained: standard LoRA, Delta W = (alpha/r) B A, any rectangular
    shape (d_out, d_in). This is what the QK/OV arms inject into W_Q / W_O
    respectively (one matrix per form -- PROJECT.md §3).

  - symmetric / antisymmetric: only meaningful on a *square* delta
    (symmetry is undefined on a rectangular matrix -- PROJECT.md §3).
    Built from r // 2 vector pairs U, V in R^{d x r//2}:

        symmetric:     Delta M = (alpha/r) (U V^T + V U^T)
        antisymmetric: Delta M = (alpha/r) (U V^T - V U^T)

    Each pair contributes rank <= 2 (docs/mathematics.md §6: a nonzero
    antisymmetric matrix has even rank), so r must be even here -- not a
    convenience choice, it is forced by the parity theorem -- and the
    resulting matrix has rank <= r, the same budget as the unconstrained
    arm at the same r (CLAUDE.md invariant: rank(Delta W) <= r for every
    parameterization).
"""

from __future__ import annotations

from typing import Literal, cast

import numpy as np

Parameterization = Literal["unconstrained", "symmetric", "antisymmetric"]
PARAMETERIZATIONS: tuple[Parameterization, ...] = ("unconstrained", "symmetric", "antisymmetric")


def unconstrained_delta(B: np.ndarray, A: np.ndarray, alpha: float) -> np.ndarray:
    """Delta W = (alpha/r) B A. B: [d_out, r], A: [r, d_in]. rank(Delta W) <= r."""
    if B.ndim != 2 or A.ndim != 2:
        raise ValueError("B and A must be 2D")
    r = B.shape[1]
    if r < 1:
        raise ValueError(f"rank must be >= 1, got {r}")
    if A.shape[0] != r:
        raise ValueError(f"rank mismatch: B is {B.shape}, A is {A.shape}")
    return cast(np.ndarray, (alpha / r) * (B @ A))


def symmetric_delta(U: np.ndarray, V: np.ndarray, alpha: float, rank: int) -> np.ndarray:
    """Delta M = (alpha/rank) (U V^T + V U^T). U, V: [d, rank // 2], rank even >= 2."""
    _check_pair_shapes(U, V, rank)
    return cast(np.ndarray, (alpha / rank) * (U @ V.T + V @ U.T))


def antisymmetric_delta(U: np.ndarray, V: np.ndarray, alpha: float, rank: int) -> np.ndarray:
    """Delta M = (alpha/rank) (U V^T - V U^T). U, V: [d, rank // 2], rank even >= 2."""
    _check_pair_shapes(U, V, rank)
    return cast(np.ndarray, (alpha / rank) * (U @ V.T - V @ U.T))


def build_delta(
    parameterization: Parameterization,
    factors: tuple[np.ndarray, np.ndarray],
    alpha: float,
    rank: int,
) -> np.ndarray:
    """Dispatch to the named parameterization.

    factors = (B, A) for "unconstrained"; (U, V) for "symmetric" /
    "antisymmetric".
    """
    P, Q = factors
    if parameterization == "unconstrained":
        return unconstrained_delta(P, Q, alpha)
    if parameterization == "symmetric":
        return symmetric_delta(P, Q, alpha, rank)
    if parameterization == "antisymmetric":
        return antisymmetric_delta(P, Q, alpha, rank)
    raise ValueError(
        f"unknown parameterization {parameterization!r}, expected one of {PARAMETERIZATIONS}"
    )


def bandwidth(rank: int, d_in: int, d_out: int) -> int:
    """beta = r(d_in + d_out): the update's free-parameter count (PROJECT.md §5).

    Lives here rather than in probes.py because it is a property of the
    parameterization, not a measurement of a model -- it is arithmetic on
    a config, computable before a single forward pass, and keeping it out
    of probes.py keeps the METRIC_VERSION surface to things that actually
    read a model (see metric_hash.py). It is what the h0_3 control is
    matched *on* (indbw.nulls.matched_beta_support).
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if d_in < 1 or d_out < 1:
        raise ValueError(f"d_in and d_out must be >= 1, got {d_in}, {d_out}")
    return rank * (d_in + d_out)


def bandwidth_fraction(rank: int, d_in: int, d_out: int) -> float:
    """beta as a fraction of the adapted matrix's own parameter count (PROJECT.md §5).

    Exceeds 1.0 once r(d_in + d_out) > d_in * d_out -- that is a real
    property of a generous rank, not an error, and it is exactly the
    regime where `matched_beta_support` refuses to build a control.
    """
    return bandwidth(rank, d_in, d_out) / float(d_in * d_out)


def inject(base_weight: np.ndarray, delta: np.ndarray) -> np.ndarray:
    """base_weight + delta, for adding a trained update to a frozen weight (PROJECT.md §2)."""
    if base_weight.shape != delta.shape:
        raise ValueError(f"base_weight shape {base_weight.shape} != delta shape {delta.shape}")
    return cast(np.ndarray, base_weight + delta)


def _check_pair_shapes(U: np.ndarray, V: np.ndarray, rank: int) -> None:
    if U.ndim != 2 or V.ndim != 2:
        raise ValueError("U and V must be 2D")
    if U.shape != V.shape:
        raise ValueError(f"U and V must share shape, got {U.shape} and {V.shape}")
    if rank < 2 or rank % 2 != 0:
        raise ValueError(
            f"symmetric/antisymmetric parameterizations require an even rank >= 2, got {rank} "
            "(a nonzero antisymmetric matrix has even rank -- docs/mathematics.md §6)"
        )
    if U.shape[1] != rank // 2:
        raise ValueError(
            f"expected U, V with {rank // 2} columns (rank {rank} // 2), got {U.shape[1]}"
        )
