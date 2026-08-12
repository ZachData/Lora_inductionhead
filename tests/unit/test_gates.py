"""Closed-form and discrimination tests for src/indbw/gates.py.

PROJECT.md §2 (A/B definitions) and §8 (G0 pass/fail criterion). Every
case is hand-constructed so the expected A/B/bracket/pass verdict is
known by construction, not eyeballed (CLAUDE.md TDD contract).
"""

from __future__ import annotations

import numpy as np
import pytest

from indbw.gates import locate_transition


def _series(pms: list[float], icl: list[float]) -> tuple[list[int], np.ndarray, np.ndarray]:
    steps = list(range(len(pms)))
    return steps, np.array(pms), np.array(icl)


# --- clean, narrow-bracket transition -------------------------------------


def test_locates_clean_sharp_transition_within_bracket() -> None:
    # PMS crosses 0.1 at step 3; ICL jumps at step 3 and is flat (stabilized)
    # by step 3 itself -- narrowest possible bracket.
    steps, pms, icl = _series(
        pms=[0.0, 0.02, 0.05, 0.5, 0.6, 0.62, 0.63],
        icl=[0.0, 0.0, 0.01, 1.0, 1.0, 1.0, 1.0],
    )
    result = locate_transition(steps, pms, icl)
    assert result.found
    assert result.a_step == 2
    assert result.b_step == 3
    assert result.bracket_width == 1
    assert result.passed


def test_wide_bracket_fails_even_when_transition_is_found() -> None:
    # PMS crosses at step 2 (A=1), but ICL keeps drifting until step 8
    # (lookahead=2 first satisfied at j=8) -- bracket_width = 8 - 1 = 7 > 3.
    steps, pms, icl = _series(
        pms=[0.0, 0.05, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5],
        icl=[0.0, 0.0, 0.2, 0.4, 0.55, 0.68, 0.79, 0.88, 0.95, 0.99, 1.0],
    )
    result = locate_transition(steps, pms, icl)
    assert result.found
    assert not result.passed
    assert result.bracket_width is not None and result.bracket_width > 3


def test_no_head_ever_crosses_threshold_is_not_found() -> None:
    steps, pms, icl = _series(
        pms=[0.0, 0.02, 0.03, 0.04, 0.05],
        icl=[0.0, 0.0, 0.01, 0.0, 0.01],
    )
    result = locate_transition(steps, pms, icl)
    assert not result.found
    assert not result.passed
    assert result.a_step is None
    assert result.b_step is None
    assert result.bracket_width is None


def test_transition_at_step_zero_has_no_valid_A() -> None:
    # Step 0 is already above threshold: there is no pre-transition
    # checkpoint on this grid, so A is undefined.
    steps, pms, icl = _series(
        pms=[0.5, 0.5, 0.5, 0.5],
        icl=[1.0, 1.0, 1.0, 1.0],
    )
    result = locate_transition(steps, pms, icl)
    assert not result.found


def test_pms_crosses_but_icl_has_already_moved_at_candidate_A_is_not_found() -> None:
    # max_pms says step 1 is pre-transition (crosses at step 2), but ICL at
    # step 1 has already moved well past baseline -- A's two conditions
    # (PROJECT.md §2) are inconsistent, so no valid A exists.
    steps, pms, icl = _series(
        pms=[0.0, 0.05, 0.5, 0.5],
        icl=[0.0, 0.9, 1.0, 1.0],
    )
    result = locate_transition(steps, pms, icl, icl_baseline_tol=0.05)
    assert not result.found


def test_transition_never_stabilizes_is_not_found() -> None:
    # PMS crosses early but ICL keeps climbing monotonically to the end of
    # the grid -- never satisfies the stabilization check.
    steps, pms, icl = _series(
        pms=[0.0, 0.5, 0.5, 0.5, 0.5, 0.5],
        icl=[0.0, 0.1, 0.3, 0.6, 1.0, 1.5],
    )
    result = locate_transition(steps, pms, icl, icl_stabilize_rtol=0.01)
    assert not result.found


# --- input validation (silent-failure guards) -----------------------------


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        locate_transition([0, 1, 2], np.zeros(3), np.zeros(2))


def test_rejects_grid_shorter_than_lookahead_plus_two() -> None:
    with pytest.raises(ValueError):
        locate_transition([0, 1], np.zeros(2), np.zeros(2), lookahead=2)


def test_rejects_non_ascending_steps() -> None:
    steps = [0, 2, 1, 4, 5]
    with pytest.raises(ValueError):
        locate_transition(steps, np.zeros(5), np.zeros(5))


def test_rejects_non_finite_input() -> None:
    steps, pms, icl = _series(pms=[0.0, np.nan, 0.5, 0.5, 0.5], icl=[0.0, 0.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        locate_transition(steps, pms, icl)


def test_rejects_pms_out_of_range() -> None:
    steps, pms, icl = _series(pms=[0.0, 1.5, 0.5, 0.5, 0.5], icl=[0.0, 0.0, 1.0, 1.0, 1.0])
    with pytest.raises(ValueError):
        locate_transition(steps, pms, icl)
