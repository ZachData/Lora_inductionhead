"""PMS, prev-token score, copying score, ICL score, recovery R.

PROJECT.md §4/§5. Every probe here is a pure function over tensors already
extracted from a model (attention patterns, embedding/unembedding/OV
matrices, per-token NLL) — it never loads or runs a model itself, so it can
be tested against hand-constructed toy inputs (CLAUDE.md TDD contract, kind
3) without depending on models.py.

Every probe ships with a negative control test: a known-positive and a
known-negative input it must tell apart, plus a guard against degenerate
input (all-zero / all-constant) that would otherwise produce a
plausible-looking but meaningless number.
"""

from __future__ import annotations

import numpy as np


def _require_square_attention(attn: np.ndarray) -> int:
    if attn.ndim != 2 or attn.shape[0] != attn.shape[1]:
        raise ValueError(f"expected a square attention pattern, got shape {attn.shape}")
    if np.any(attn < -1e-8):
        raise ValueError("attention pattern must be nonnegative (post-softmax)")
    row_sums = attn.sum(axis=1)
    if not np.allclose(row_sums, 1.0, atol=1e-5):
        raise ValueError(
            "attention pattern rows must sum to 1 (a valid post-softmax pattern); "
            "got an input that does not (e.g. an all-zero matrix)"
        )
    return int(attn.shape[0])


def prefix_matching_score(attn: np.ndarray, T: int) -> float:
    """Mean attention from position t to t-T+1, over t in the second copy.

    attn: [2T, 2T] post-softmax attention pattern for one head, on a
    repeated-random sequence of period T (PROJECT.md §5).
    """
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    seq_len = _require_square_attention(attn)
    if seq_len != 2 * T:
        raise ValueError(f"expected seq_len == 2*T ({2 * T}), got {seq_len}")
    t_idx = np.arange(T, 2 * T)
    src_idx = t_idx - T + 1
    value = float(np.mean(attn[t_idx, src_idx]))
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"prefix_matching_score out of range: {value}")
    return value


def prev_token_score(attn: np.ndarray) -> float:
    """Mean attention from position t to t-1, over all valid t.

    attn: [seq_len, seq_len] post-softmax attention pattern for one head.
    """
    seq_len = _require_square_attention(attn)
    if seq_len < 2:
        raise ValueError(f"prev_token_score requires seq_len >= 2, got {seq_len}")
    t_idx = np.arange(1, seq_len)
    src_idx = t_idx - 1
    value = float(np.mean(attn[t_idx, src_idx]))
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"prev_token_score out of range: {value}")
    return value


def copying_score(W_U: np.ndarray, M_OV: np.ndarray, W_E: np.ndarray) -> float:
    """Fraction of vocab tokens t for which argmax_s (W_U @ M_OV @ e_t)_s == t.

    W_U: [vocab, d] unembedding. M_OV: [d, d] composed OV matrix.
    W_E: [vocab, d] embedding, e_t = W_E[t] (PROJECT.md §4).
    """
    if W_U.ndim != 2 or M_OV.ndim != 2 or W_E.ndim != 2:
        raise ValueError("copying_score requires 2D matrices for W_U, M_OV, W_E")
    vocab_u, d_u = W_U.shape
    d_ov_out, d_ov_in = M_OV.shape
    vocab_e, d_e = W_E.shape
    if d_ov_out != d_ov_in:
        raise ValueError(f"M_OV must be square, got shape {M_OV.shape}")
    if d_u != d_ov_out or d_ov_in != d_e:
        raise ValueError(
            f"dimension mismatch: W_U is [.,{d_u}], M_OV is {M_OV.shape}, W_E is [.,{d_e}]"
        )
    if vocab_u != vocab_e:
        raise ValueError(f"W_U and W_E must share vocab size, got {vocab_u} and {vocab_e}")
    if vocab_u < 2:
        raise ValueError("copying_score requires vocab size >= 2 to be meaningful")
    if not np.any(M_OV):
        raise ValueError("copying_score: M_OV is all-zero — the OV circuit is undefined")
    logits = (W_E @ M_OV.T) @ W_U.T  # logits[t] = W_U @ M_OV @ e_t
    predicted = np.argmax(logits, axis=1)
    value = float(np.mean(predicted == np.arange(vocab_u)))
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"copying_score out of range: {value}")
    return value


def icl_score(nll_first: np.ndarray, nll_second: np.ndarray) -> float:
    """NLL on the first copy minus NLL on the second copy. Higher is stronger ICL.

    nll_first, nll_second: per-example negative log-likelihood on a
    repeated-random eval set, same shape (PROJECT.md §5).
    """
    nll_first = np.asarray(nll_first, dtype=float)
    nll_second = np.asarray(nll_second, dtype=float)
    if nll_first.shape != nll_second.shape:
        raise ValueError(
            f"nll_first and nll_second must have matching shape, "
            f"got {nll_first.shape} and {nll_second.shape}"
        )
    if nll_first.size == 0:
        raise ValueError("icl_score requires a nonempty eval set")
    if not (np.all(np.isfinite(nll_first)) and np.all(np.isfinite(nll_second))):
        raise ValueError("icl_score received a non-finite NLL value")
    return float(np.mean(nll_first) - np.mean(nll_second))


def recovery(icl_x: float, icl_a: float, icl_b: float) -> float:
    """R(X) = (ICL(X) - ICL(A)) / (ICL(B) - ICL(A)). R(A)=0, R(B)=1 exactly.

    Unclamped: PROJECT.md §5 logs the raw value and clamps only for
    reporting — use clamp_recovery for that.
    """
    denom = icl_b - icl_a
    if denom == 0.0:
        raise ValueError("recovery is undefined when ICL(B) == ICL(A)")
    return float((icl_x - icl_a) / denom)


def clamp_recovery(r: float) -> float:
    """Clamp an unclamped recovery value to [0, 1] for reporting (PROJECT.md §5)."""
    return float(np.clip(r, 0.0, 1.0))
