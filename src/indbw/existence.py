"""Closed-form quantities from the existence derivation, docs/mathematics.md §12-§18.

These answer question (E)/(A) of §12 -- does a rank-limited update installing
the behaviour *exist*, and how does the achievable margin scale with rank --
as opposed to (L), whether training finds one. Everything here is either a
closed form or a function of frozen matrices: nothing in this module trains,
loads a model, or reads a checkpoint.

Same pattern as algebra.py / probes.py / lora.py: pure functions over plain
arrays, so the derivation's load-bearing identities are testable against the
oracles in docs/mathematics.md rather than against whatever the code
currently returns. Test file: tests/unit/test_existence.py.

**Deliberately not implemented here:** §16's linear program. Deciding
feasibility of $\\mathcal{F}_\\Delta$ needs an LP solver (no scipy dependency
in this project) *and* a recorded set of layer-input activations from
checkpoint A, which is orchestration, not algebra. §16 stands as derivation;
see REVIEW.md.

**Not metrics.** Nothing here lands in a results record, so this module is
outside metric_hash.py's METRIC_MODULES by design (same reasoning as
gates.py / lora.py). If any of these ever becomes a reported readout it moves
under the METRIC_VERSION lock first.
"""

from __future__ import annotations

from typing import cast

import numpy as np

from indbw.algebra import subspace_projector


def required_logit_margin(tau: float, n_positions: int) -> float:
    """Score margin that *guarantees* attention >= tau on the target position.

    docs/mathematics.md Corollary 14.2: Delta_tau(m) = log((m-1) tau / (1-tau)).
    Sufficient, not necessary -- PMS averages attention over query positions,
    so the behaviour can clear tau with less margin than this on some of them.
    """
    if not (0.0 < tau < 1.0):
        raise ValueError(f"tau must lie strictly in (0, 1), got {tau}")
    if n_positions < 2:
        raise ValueError(f"need at least 2 competing positions, got {n_positions}")
    return float(np.log((n_positions - 1) * tau / (1.0 - tau)))


def attention_lower_bound(margin: float, n_positions: int) -> float:
    """Attention floor on the target given a uniform score margin over the rest.

    docs/mathematics.md Proposition 14.1: alpha >= 1 / (1 + (m-1) e^{-Delta}).
    Exact inverse of `required_logit_margin` at the boundary.
    """
    if n_positions < 2:
        raise ValueError(f"need at least 2 competing positions, got {n_positions}")
    value = 1.0 / (1.0 + (n_positions - 1) * float(np.exp(-margin)))
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"attention lower bound out of range: {value}")
    return value


def rank_ceiling(d_out: int, d_in: int) -> int:
    """min(d_out, d_in) -- the largest rank a delta of this shape can have.

    docs/mathematics.md Lemma 13.2. For the QK arm the adapted weight is
    W_Q [d_model, d_head], so this is d_head.
    """
    if d_out < 1 or d_in < 1:
        raise ValueError(f"d_out and d_in must be >= 1, got {d_out}, {d_in}")
    return min(d_out, d_in)


def rank_constraint_is_vacuous(rank: int, d_out: int, d_in: int) -> bool:
    """True when rank-`rank` LoRA on a [d_out, d_in] weight constrains nothing.

    docs/mathematics.md Lemma 13.2: at rank >= min(d_out, d_in) the set of
    representable deltas is all of R^{d_out x d_in}, so a cell run at that
    rank measures the *unconstrained* arm and carries no rank information.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    return rank >= rank_ceiling(d_out, d_in)


def welch_coherence_bound(n_vectors: int, dim: int) -> float:
    """Welch bound: min achievable max-|inner product| for n unit vectors in R^dim.

    docs/mathematics.md A4. sqrt((N - n) / (n (N - 1))), and 0 when the
    vectors fit orthogonally (N <= n). At N = |V| = 50304 in n = d_head = 64
    this is 0.125: key-side interference of order 1/8 is forced by counting,
    at any rank.
    """
    if n_vectors < 1 or dim < 1:
        raise ValueError(f"n_vectors and dim must be >= 1, got {n_vectors}, {dim}")
    if n_vectors <= dim:
        return 0.0
    return float(np.sqrt((n_vectors - dim) / (dim * (n_vectors - 1))))


def sketch_rank_threshold(n_competitors: int, eps: float = 1.0) -> float:
    """Rank at which a random rank-r sketch still separates 1 target from m-1 others.

    docs/mathematics.md Theorem 15.2: r >~ 2 log(m) / eps^2, where eps is the
    fraction of the unsketched margin being given up. Logarithmic in the
    number of competitors, and independent of the ambient dimension.

    Returned as a float: it is a scaling threshold, not an integer rank, and
    rounding it here would misrepresent an order-of-magnitude bound as exact.
    """
    if n_competitors < 2:
        raise ValueError(f"need at least 2 competitors, got {n_competitors}")
    if not (0.0 < eps <= 1.0):
        raise ValueError(f"eps must lie in (0, 1], got {eps}")
    return float(2.0 * np.log(n_competitors) / (eps * eps))


def random_projector(dim: int, rank: int, rng: np.random.Generator) -> np.ndarray:
    """Uniformly random rank-`rank` orthogonal projector on R^dim (docs §15.2's Pi_r)."""
    if dim < 1:
        raise ValueError(f"dim must be >= 1, got {dim}")
    if not (1 <= rank <= dim):
        raise ValueError(f"rank {rank} out of bounds for dim {dim}")
    return subspace_projector(rng.standard_normal((dim, rank)))


def _unit_columns(M: np.ndarray, name: str) -> np.ndarray:
    if M.ndim != 2:
        raise ValueError(f"{name} must be 2D [dim, n_codes], got shape {M.shape}")
    norms = np.linalg.norm(M, axis=0)
    if np.any(norms == 0.0):
        raise ValueError(f"{name} has an all-zero column; its direction is undefined")
    return cast(np.ndarray, M / norms)


def matched_filter_update(
    query_codes: np.ndarray,
    key_codes: np.ndarray,
    projector: np.ndarray,
    scale: float = 1.0,
) -> np.ndarray:
    """The §15.2 construction: Delta W_Q = scale * Qhat (Pi_r Khat)^T.

    query_codes: [d_model, n_codes], one column per token's query-side
    direction q_t. key_codes: [d_head, n_codes], one column per matched key
    code kappa_t = W_K^T p_t (docs §15.1, A3). Columns are normalized here --
    the hats in the formula are part of the construction, not a precondition
    on the caller.

    Returns [d_model, d_head] with rank <= rank(projector), so it is a legal
    LoRA delta for the QK arm at that rank. This is an *explicit witness*:
    §11's "sufficiency is constructive" without a training run.
    """
    Q = _unit_columns(query_codes, "query_codes")
    K = _unit_columns(key_codes, "key_codes")
    if Q.shape[1] != K.shape[1]:
        raise ValueError(
            f"query_codes and key_codes must have the same number of columns, "
            f"got {Q.shape[1]} and {K.shape[1]}"
        )
    if projector.ndim != 2 or projector.shape[0] != projector.shape[1]:
        raise ValueError(f"projector must be square, got shape {projector.shape}")
    if projector.shape[0] != K.shape[0]:
        raise ValueError(
            f"projector dimension {projector.shape[0]} != key code dimension {K.shape[0]}"
        )
    return cast(np.ndarray, scale * (Q @ (projector @ K).T))


def sketch_signal_and_interference(
    key_codes: np.ndarray, projector: np.ndarray, target: int = 0
) -> tuple[float, float]:
    """(signal, max interference) for the sketched matched filter at one target.

    docs/mathematics.md §15.3: signal = <Pi_r kappa_hat_t, kappa_hat_t>,
    interference = max over the other codes of <Pi_r kappa_hat_t, kappa_hat_s>.
    Separation holds iff signal > interference; Theorem 15.2 says that happens
    around rank ~ 2 log(n_codes).
    """
    K = _unit_columns(key_codes, "key_codes")
    n = K.shape[1]
    if n < 2:
        raise ValueError(f"need at least 2 codes to have any interference, got {n}")
    if not (0 <= target < n):
        raise ValueError(f"target index {target} out of bounds for {n} codes")
    if projector.shape != (K.shape[0], K.shape[0]):
        raise ValueError(f"projector must be [{K.shape[0]}, {K.shape[0]}], got {projector.shape}")
    sketched = projector @ K[:, target]
    overlaps = sketched @ K
    signal = float(overlaps[target])
    interference = float(np.max(np.delete(overlaps, target)))
    return signal, interference


def _check_attention(attn: np.ndarray) -> None:
    if attn.ndim != 1:
        raise ValueError(f"attn must be a 1D row of attention weights, got shape {attn.shape}")
    if attn.size < 1:
        raise ValueError("attn is empty")
    if np.any(attn < 0.0):
        raise ValueError("attn has negative entries; it is not a softmax output")
    total = float(np.sum(attn))
    # 1e-6: attn comes from a float32 softmax over <= 256 positions, whose
    # row sum is exact to ~1e-7; anything looser than that is a real bug.
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"attn must sum to 1, got {total}")


def attention_score_gradient(
    attn: np.ndarray, values: np.ndarray, grad_out: np.ndarray
) -> np.ndarray:
    """dL/ds_j = alpha_j <g, z_j - zbar>, the exact QK score gradient (Theorem 17.1).

    attn: [m] attention row. values: [m, d] the head's per-position write
    z_j = M_OV x_j. grad_out: [d] the upstream dL/do_i.

    **An all-constant `values` returns exactly zeros, and that is the
    theorem, not a silent failure** (Corollary 17.2): a head that writes the
    same vector whichever position it attends to has an identically zero QK
    gradient, at every rank. Do not "fix" this by adding an epsilon --
    `value_spread` is the readout that tells you the gate is closed.
    """
    _check_attention(attn)
    if values.ndim != 2 or values.shape[0] != attn.shape[0]:
        raise ValueError(f"values must be [{attn.shape[0]}, d], got shape {values.shape}")
    if grad_out.ndim != 1 or grad_out.shape[0] != values.shape[1]:
        raise ValueError(f"grad_out must be [{values.shape[1]}], got shape {grad_out.shape}")
    projected = values @ grad_out
    return cast(np.ndarray, attn * (projected - float(attn @ projected)))


def value_spread(attn: np.ndarray, values: np.ndarray) -> float:
    """sigma_OV = max_j ||z_j - zbar||, the bound on the QK gradient (Corollary 17.2).

    Zero exactly when the head's write is position-independent. That is a
    meaningful measurement (the QK arm has no gradient to descend), not a
    degenerate input -- see `attention_score_gradient`.
    """
    _check_attention(attn)
    if values.ndim != 2 or values.shape[0] != attn.shape[0]:
        raise ValueError(f"values must be [{attn.shape[0]}, d], got shape {values.shape}")
    centered = values - (attn @ values)
    return float(np.max(np.linalg.norm(centered, axis=1)))
