"""Invariants for src/indbw/gates.py that must hold over random input.
Closed-form point checks live in tests/unit/test_gates.py.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given
from hypothesis import strategies as st

from indbw.gates import k_composition_overlap, locate_prev_token_head, locate_transition

_seeds = st.integers(min_value=0, max_value=2**31 - 1)
_sizes = st.integers(min_value=4, max_value=30)


@given(seed=_seeds, n=_sizes)
def test_a_before_b_whenever_found(seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    steps = list(range(n))
    pms = np.clip(rng.standard_normal(n).cumsum() * 0.05 + 0.05, 0.0, 1.0)
    icl = rng.standard_normal(n).cumsum() * 0.1
    result = locate_transition(steps, pms, icl)
    if result.found:
        assert result.a_step is not None and result.b_step is not None
        assert result.a_step < result.b_step
        assert result.bracket_width == result.b_step - result.a_step
        assert result.bracket_width >= 1


@given(seed=_seeds, n=_sizes)
def test_passed_implies_found_and_narrow_bracket(seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    steps = list(range(n))
    pms = np.clip(rng.standard_normal(n).cumsum() * 0.05 + 0.05, 0.0, 1.0)
    icl = rng.standard_normal(n).cumsum() * 0.1
    result = locate_transition(steps, pms, icl)
    if result.passed:
        assert result.found
        assert result.bracket_width is not None and result.bracket_width <= 3


@given(seed=_seeds, n=_sizes)
def test_not_found_implies_no_steps_reported(seed: int, n: int) -> None:
    rng = np.random.default_rng(seed)
    steps = list(range(n))
    pms = np.clip(rng.standard_normal(n).cumsum() * 0.05 + 0.05, 0.0, 1.0)
    icl = rng.standard_normal(n).cumsum() * 0.1
    result = locate_transition(steps, pms, icl)
    if not result.found:
        assert result.a_step is None
        assert result.b_step is None
        assert result.bracket_width is None
        assert not result.passed


@given(
    seed=_seeds,
    n_layers=st.integers(min_value=1, max_value=6),
    n_heads=st.integers(min_value=1, max_value=8),
)
def test_prev_token_head_found_implies_score_above_threshold(
    seed: int, n_layers: int, n_heads: int
) -> None:
    rng = np.random.default_rng(seed)
    scores = rng.uniform(0.0, 1.0, size=(n_layers, n_heads))
    result = locate_prev_token_head(scores, threshold=0.3)
    assert 0 <= result.layer < n_layers
    assert 0 <= result.head < n_heads
    assert result.score == pytest.approx(scores.max())
    assert result.found == (result.score >= 0.3)


@given(seed=_seeds, d_model=st.integers(min_value=3, max_value=12))
def test_k_composition_overlap_ratio_in_unit_interval(seed: int, d_model: int) -> None:
    rng = np.random.default_rng(seed)
    d_head = max(1, d_model // 2)
    W_O_prev = rng.standard_normal((d_head, d_model))
    W_K = rng.standard_normal((d_model, d_head))
    result = k_composition_overlap(W_O_prev, W_K, n_null_draws=10, seed=seed)
    assert 0.0 <= result.ratio <= 1.0
    assert result.significant == (result.ratio > result.null_percentile_value)
