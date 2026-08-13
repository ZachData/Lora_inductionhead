"""Closed-form and discrimination tests for src/indbw/gates.py.

PROJECT.md §2 (A/B definitions) and §8 (G0 pass/fail criterion). Every
case is hand-constructed so the expected A/B/bracket/pass verdict is
known by construction, not eyeballed (CLAUDE.md TDD contract).
"""

from __future__ import annotations

import numpy as np
import pytest

from indbw.algebra import subspace_projector
from indbw.gates import k_composition_overlap, locate_prev_token_head, locate_transition


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


# --- locate_prev_token_head (PROJECT.md §8 G1) ----------------------------


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def test_locate_prev_token_head_finds_clear_winner_above_threshold() -> None:
    scores = np.array([[0.02, 0.05, 0.01], [0.9, 0.03, 0.02]])
    result = locate_prev_token_head(scores)
    assert result.layer == 1
    assert result.head == 0
    assert result.score == pytest.approx(0.9)
    assert result.found


def test_locate_prev_token_head_below_threshold_is_not_found() -> None:
    # Best head only reaches 0.2 -- below the 0.3 pre-registered threshold,
    # so no previous-token head exists on this grid at this checkpoint.
    scores = np.array([[0.02, 0.05, 0.2], [0.1, 0.03, 0.02]])
    result = locate_prev_token_head(scores)
    assert result.score == pytest.approx(0.2)
    assert not result.found


def test_locate_prev_token_head_rejects_wrong_ndim() -> None:
    with pytest.raises(ValueError):
        locate_prev_token_head(np.zeros(5))


def test_locate_prev_token_head_rejects_non_finite() -> None:
    scores = np.array([[0.1, np.nan], [0.2, 0.3]])
    with pytest.raises(ValueError):
        locate_prev_token_head(scores)


def test_locate_prev_token_head_rejects_out_of_range() -> None:
    scores = np.array([[0.1, 1.5], [0.2, 0.3]])
    with pytest.raises(ValueError):
        locate_prev_token_head(scores)


# --- k_composition_overlap (PROJECT.md §8/§9) -----------------------------


def test_k_composition_overlap_is_one_when_W_K_lies_in_prev_tok_span() -> None:
    # W_O_prev writes into a 3-dim subspace of an 8-dim residual stream;
    # build W_K entirely inside that same subspace -- the overlap ratio
    # must be exactly 1.0, and comfortably clears any random-subspace null.
    rng = _rng(0)
    d_model, d_head = 8, 3
    basis = rng.standard_normal((d_model, d_head))
    W_O_prev = basis.T  # [d_head, d_model]
    U = subspace_projector(basis)
    W_K = U @ rng.standard_normal((d_model, d_head))
    result = k_composition_overlap(W_O_prev, W_K, n_null_draws=20, seed=1)
    assert result.ratio == pytest.approx(1.0, rel=1e-8)
    assert result.significant


def test_k_composition_overlap_is_zero_when_W_K_is_orthogonal_to_prev_tok_span() -> None:
    # Disjoint coordinate blocks: W_O_prev's write subspace is axes 0-2,
    # W_K lives entirely in axes 3-5 -- the projection is exactly zero and
    # cannot exceed any random-subspace null.
    d_model = 6
    W_O_prev = np.eye(d_model)[:3, :]  # [3, 6]: writes into axes 0-2
    W_K = np.eye(d_model)[:, 3:]  # [6, 3]: lives in axes 3-5
    result = k_composition_overlap(W_O_prev, W_K, n_null_draws=20, seed=2)
    assert result.ratio == pytest.approx(0.0, abs=1e-12)
    assert not result.significant


def test_k_composition_overlap_rejects_d_model_mismatch() -> None:
    W_O_prev = np.zeros((3, 8))
    W_K = np.zeros((6, 3))
    with pytest.raises(ValueError):
        k_composition_overlap(W_O_prev, W_K)
