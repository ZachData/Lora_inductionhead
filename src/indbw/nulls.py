"""Matched-norm random updates, random subspaces.

Controls for h0_3, h0_4 (PROJECT.md §6/§9).

PROJECT.md §8's "Mandatory controls" are not optional decoration: no
rank-level failure may be reported without them (CLAUDE.md, "No failure
claim without its controls attached"). This module builds the *objects*
those controls need; running a model on them belongs to the row scripts,
so -- same pattern as algebra.py / probes.py / lora.py -- everything here
is a pure function over plain arrays and an explicit `numpy.Generator`.

Three controls, each tied to a pre-registered null:

  - **Matched-norm random rank-r update** (§8; h0_3's "matched-beta
    unstructured update performs identically" and h0_4's "matched-norm
    random-null band"). `matched_norm_random_update` draws a rank-r
    matrix of the same shape and *exactly* the same Frobenius norm as a
    trained Delta W. Behavioural use: inject it instead of the trained
    update and measure R. Structural use: `phi_null_band`.

  - **phi null band** (h0_4). Random matrices concentrate near
    phi = 1/2 (PROJECT.md §6's stated limitation), so a phi value is
    only interpretable against the band its own matched-norm null
    produces. `phi_null_band` takes the *composition* as a callable
    rather than hardcoding it: the QK arm's object is
    Delta M_QK = Delta W_Q W_K^T and the OV arm's is
    Delta M_OV = Delta W_O^T W_V^T (PROJECT.md §3), and getting that
    transpose convention wrong in a null generator would produce a
    plausible band for the wrong operator. The caller, which already
    holds the frozen weights, supplies it.

  - **Matched-beta unstructured update** (h0_3). See
    `matched_beta_support` -- its construction resolves a PROJECT.md §11
    open question and is flagged in REVIEW.md rather than assumed.

Random-subspace draws (`random_subspace_basis`) are the shared primitive
behind h0_6 and behind G1's already-closed overlap null. `gates.py`
inlines its own equivalent draw; that code is deliberately untouched
here, since re-routing it through this module could perturb the
committed G1 numbers for no scientific gain.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from indbw.algebra import phi


def random_subspace_basis(d: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Orthonormal basis [d, k] spanning a uniformly-random k-dim subspace of R^d.

    QR of a Gaussian matrix. The Q factor's *signs* are not Haar-corrected,
    which would matter if the basis vectors themselves were the object of
    study; only their span is, and the span of a Gaussian matrix is exactly
    uniformly distributed on the Grassmannian.
    """
    if d < 1:
        raise ValueError(f"d must be >= 1, got {d}")
    if not (1 <= k <= d):
        raise ValueError(f"k must satisfy 1 <= k <= d, got k={k}, d={d}")
    Q, _ = np.linalg.qr(rng.standard_normal((d, k)))
    return np.asarray(Q[:, :k])


def matched_norm_random_update(
    shape: tuple[int, int], rank: int, target_norm: float, rng: np.random.Generator
) -> np.ndarray:
    """Random rank-`rank` matrix of shape `shape` with ||.||_F == `target_norm` exactly.

    Built as B @ A with Gaussian factors of the same shape convention as
    `indbw.lora.unconstrained_delta` (B: [d_out, rank], A: [rank, d_in]),
    then rescaled -- so the null occupies the same matrix manifold as the
    trained update it is matched against, differing only in *which*
    rank-r matrix it is.

    `target_norm` must be strictly positive: a zero-norm null is a
    degenerate control (phi is undefined on the zero matrix, and an
    all-zero update trivially reproduces the base model), so it raises
    rather than returning a plausible-looking zero matrix.
    """
    d_out, d_in = _check_shape(shape)
    if rank < 1 or rank > min(d_out, d_in):
        raise ValueError(f"rank {rank} out of bounds for shape {shape}")
    if not np.isfinite(target_norm):
        raise ValueError(f"target_norm must be finite, got {target_norm}")
    if target_norm <= 0.0:
        raise ValueError(
            f"target_norm must be > 0, got {target_norm} -- a zero-norm null is degenerate"
        )
    B = rng.standard_normal((d_out, rank))
    A = rng.standard_normal((rank, d_in))
    M = B @ A
    norm = float(np.linalg.norm(M))
    if norm == 0.0:  # probability zero, but a silent divide-by-zero here would be a NaN update
        raise FloatingPointError("matched_norm_random_update drew an all-zero factor product")
    return np.asarray(M * (target_norm / norm))


@dataclass(frozen=True)
class NullBand:
    """Empirical band of a statistic over matched null draws.

    `values` is kept alongside the summary so a record can report the
    band without the draw that produced it becoming unreproducible.
    """

    values: np.ndarray
    percentile: float
    percentile_value: float
    mean: float
    n_draws: int

    def exceeds(self, observed: float) -> bool:
        """True iff `observed` lies outside the band's upper edge.

        The same convention `gates.py` used for G1 and that PROJECT.md §6
        pre-registers for h0_6: observed vs. the `percentile`-th
        percentile of the null draws.
        """
        return observed > self.percentile_value


def phi_null_band(
    shape: tuple[int, int],
    rank: int,
    target_norm: float,
    compose: Callable[[np.ndarray], np.ndarray],
    *,
    n_draws: int = 100,
    percentile: float = 95.0,
    seed: int = 0,
) -> NullBand:
    """Band of phi over `n_draws` matched-norm random rank-r updates (h0_4).

    `compose` maps a candidate Delta W to the square composed form phi is
    defined on (PROJECT.md §3: symmetry is undefined on the rectangular
    Delta W itself). For the QK arm that is
    `lambda dWq: dWq @ W_K.T`; for the OV arm, `lambda dWo: dWo.T @ W_V.T`.
    Passing the identity is meaningful only when `shape` is already square.

    Note what this band is and is not. It is the distribution of phi for
    a *structurally meaningless* update of the same shape, rank and norm
    as the trained one. PROJECT.md §6: random matrices concentrate near
    phi = 1/2, so phi_QK ~ 0.5 sitting inside this band is uninformative
    on its own and must be reported only as a contrast against phi_OV.
    """
    if n_draws < 2:
        raise ValueError(f"n_draws must be >= 2 for a band, got {n_draws}")
    if not (0.0 < percentile < 100.0):
        raise ValueError(f"percentile must be in (0, 100), got {percentile}")
    rng = np.random.default_rng(seed)
    values = np.empty(n_draws)
    for i in range(n_draws):
        delta = matched_norm_random_update(shape, rank, target_norm, rng)
        values[i] = phi(compose(delta))  # phi range-guards itself (algebra.py)
    return NullBand(
        values=values,
        percentile=float(percentile),
        percentile_value=float(np.percentile(values, percentile)),
        mean=float(np.mean(values)),
        n_draws=n_draws,
    )


def phi_separation_null_band(
    shape_qk: tuple[int, int],
    shape_ov: tuple[int, int],
    rank: int,
    target_norm_qk: float,
    target_norm_ov: float,
    compose_qk: Callable[[np.ndarray], np.ndarray],
    compose_ov: Callable[[np.ndarray], np.ndarray],
    *,
    n_draws: int = 100,
    percentile: float = 95.0,
    seed: int = 0,
) -> NullBand:
    """Band of |phi_QK - phi_OV| over *paired* matched-norm null draws.

    h0_4 is falsified by "separation exceeding the matched-norm random-null
    band" (PROJECT.md §6) -- a statement about the *difference* of the two
    arms' phi, not about either one alone, so the null it needs is the
    difference between two independent matched nulls, one per arm, drawn
    with each arm's own shape and norm.

    The only free choice here is `percentile`, defaulted to the 95.0 this
    repo already uses for G1's overlap null and for h0_6. This is an
    operationalization of a human-fixed null, not a new test -- but it is
    the first place the *separation* band is made concrete, so it is
    logged in REVIEW.md for sign-off before M4/M6 report against it.
    """
    if n_draws < 2:
        raise ValueError(f"n_draws must be >= 2 for a band, got {n_draws}")
    rng = np.random.default_rng(seed)
    values = np.empty(n_draws)
    for i in range(n_draws):
        d_qk = matched_norm_random_update(shape_qk, rank, target_norm_qk, rng)
        d_ov = matched_norm_random_update(shape_ov, rank, target_norm_ov, rng)
        values[i] = abs(phi(compose_qk(d_qk)) - phi(compose_ov(d_ov)))
    return NullBand(
        values=values,
        percentile=float(percentile),
        percentile_value=float(np.percentile(values, percentile)),
        mean=float(np.mean(values)),
        n_draws=n_draws,
    )


def matched_beta_support(shape: tuple[int, int], beta: int, rng: np.random.Generator) -> np.ndarray:
    """Fixed random support of exactly `beta` entries: a bool mask [d_out, d_in].

    The h0_3 control is an update with the same parameter budget
    beta = r(d_in + d_out) (PROJECT.md §5) as the rank-r LoRA arm but
    *without* low-rank structure. This operationalizes "unstructured" as
    a random **coordinate** subspace of matrix space -- beta free entries,
    every other entry pinned to zero -- fixed before training and never
    resampled.

    Two choices are being made here, both flagged in REVIEW.md rather
    than buried:

    1. *Fixed, not resampled.* PROJECT.md §11 leaves this open but states
       the leaning ("Fixed is the right analogue to LoRA"); a LoRA
       adapter's subspace is likewise fixed once initialized, and only
       its coordinates within that subspace train.

    2. *Coordinate subspace, not a dense Gaussian one.* The Li et al.
       2018 construction §12 cites uses a dense random projection
       P: R^beta -> R^{d_out x d_in}. At this project's shapes that P is
       not representable: the QK arm's Delta W_Q is 512 x 64 = 32768
       entries and beta = 576r, so at the generous rank r=64 a dense P
       would be 32768 x 36864 floats (~9.7 GB) -- and it would also
       exceed the ambient dimension, making the "control" a full
       unconstrained matrix. A coordinate subspace is a random subspace
       of the same dimension, costs a bool mask, and keeps the
       parameter budget exactly matched.

    Raises if beta exceeds the ambient dimension: at that point the
    control is not a restricted update at all, and silently clamping it
    would produce a control that cannot fail to match the LoRA arm's
    expressiveness.
    """
    d_out, d_in = _check_shape(shape)
    ambient = d_out * d_in
    if beta < 1:
        raise ValueError(f"beta must be >= 1, got {beta}")
    if beta > ambient:
        raise ValueError(
            f"beta={beta} exceeds the ambient dimension {d_out}x{d_in}={ambient}: at this "
            "rank the matched-beta control is an unconstrained full-rank update, not a "
            "restricted one, and comparing against it would be vacuous"
        )
    flat_idx = rng.choice(ambient, size=beta, replace=False)
    mask = np.zeros(ambient, dtype=bool)
    mask[flat_idx] = True
    return mask.reshape(d_out, d_in)


def _check_shape(shape: tuple[int, int]) -> tuple[int, int]:
    if len(shape) != 2:
        raise ValueError(f"expected a 2D shape (d_out, d_in), got {shape}")
    d_out, d_in = int(shape[0]), int(shape[1])
    if d_out < 1 or d_in < 1:
        raise ValueError(f"shape entries must be >= 1, got {shape}")
    return d_out, d_in
