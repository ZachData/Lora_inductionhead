"""Invariants for src/indbw/gates.py that must hold over random input.
Closed-form point checks live in tests/unit/test_gates.py.
"""

from __future__ import annotations

import numpy as np
from hypothesis import given
from hypothesis import strategies as st

from indbw.gates import locate_transition

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
