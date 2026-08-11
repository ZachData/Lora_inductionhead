"""Invariants for src/indbw/probes.py that must hold over random input,
per CLAUDE.md's property-test list. Closed-form/discrimination point
checks live in tests/unit/test_probes.py; this file checks the range
guards hold generically, not just on the hand-picked cases.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from indbw.probes import clamp_recovery, copying_score, prefix_matching_score, prev_token_score

_seeds = st.integers(min_value=0, max_value=2**31 - 1)
_periods = st.integers(min_value=1, max_value=6)
_dims = st.integers(min_value=2, max_value=6)
_vocabs = st.integers(min_value=2, max_value=10)


def _random_attention(seed: int, seq_len: int) -> np.ndarray:
    # Random softmax rows: a valid, generically-structureless attention
    # pattern (row sums to 1, all entries nonnegative).
    logits = np.random.default_rng(seed).standard_normal((seq_len, seq_len))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    return exp / exp.sum(axis=1, keepdims=True)


@given(seed=_seeds, T=_periods)
def test_pms_in_unit_interval(seed: int, T: int) -> None:
    attn = _random_attention(seed, 2 * T)
    value = prefix_matching_score(attn, T)
    assert 0.0 <= value <= 1.0


@given(seed=_seeds, T=_periods)
def test_prev_token_score_in_unit_interval(seed: int, T: int) -> None:
    attn = _random_attention(seed, 2 * T)
    value = prev_token_score(attn)
    assert 0.0 <= value <= 1.0


@given(seed=_seeds, d=_dims, vocab=_vocabs)
def test_copying_score_in_unit_interval(seed: int, d: int, vocab: int) -> None:
    rng = np.random.default_rng(seed)
    W_E = rng.standard_normal((vocab, d))
    W_U = rng.standard_normal((vocab, d))
    M_OV = rng.standard_normal((d, d))
    if not np.any(M_OV):
        return
    value = copying_score(W_U, M_OV, W_E)
    assert 0.0 <= value <= 1.0


@given(st.floats(min_value=-1e6, max_value=1e6, allow_nan=False))
def test_clamp_recovery_always_in_unit_interval(r: float) -> None:
    value = clamp_recovery(r)
    assert 0.0 <= value <= 1.0
