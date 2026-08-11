"""Closed-form oracles and discrimination tests for src/indbw/probes.py.

PROJECT.md §4/§5. Every probe here needs a discrimination test (CLAUDE.md
TDD contract, kind 3): a known-positive and a known-negative case it must
tell apart. Silent failure — a constant or plausible-looking wrong number —
is the risk these guard against, not crashes.
"""

from __future__ import annotations

import numpy as np
import pytest

from indbw.probes import (
    clamp_recovery,
    copying_score,
    icl_score,
    prefix_matching_score,
    prev_token_score,
    recovery,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _uniform_attn(seq_len: int) -> np.ndarray:
    return np.full((seq_len, seq_len), 1.0 / seq_len)


# --- prefix_matching_score --------------------------------------------


def test_pms_hand_constructed_induction_head_is_one() -> None:
    # Perfect prefix matcher: position t attends entirely to t - T + 1.
    T = 8
    attn = np.zeros((2 * T, 2 * T))
    for t in range(T, 2 * T):
        attn[t, t - T + 1] = 1.0
    for t in range(T):
        attn[t, t] = 1.0  # first copy: irrelevant to PMS, just needs to be valid
    assert prefix_matching_score(attn, T) == pytest.approx(1.0, rel=1e-12)


def test_pms_uniform_attention_is_near_chance() -> None:
    # No structure: every row is uniform, so PMS collapses to 1/seq_len,
    # far below the PROJECT.md §5 induction-bearing threshold of 0.3.
    T = 16
    attn = _uniform_attn(2 * T)
    value = prefix_matching_score(attn, T)
    assert value == pytest.approx(1.0 / (2 * T), rel=1e-12)
    assert value < 0.3


def test_pms_requires_square_matrix() -> None:
    with pytest.raises(ValueError):
        prefix_matching_score(np.zeros((4, 6)), 2)


def test_pms_requires_seq_len_matches_2T() -> None:
    with pytest.raises(ValueError):
        prefix_matching_score(np.zeros((5, 5)), 2)


def test_pms_rejects_invalid_attention_pattern() -> None:
    # Rows must sum to 1 (a real post-softmax pattern); an all-zero
    # matrix is the canonical silent-failure input (CLAUDE.md).
    with pytest.raises(ValueError):
        prefix_matching_score(np.zeros((8, 8)), 4)


def test_pms_rejects_negative_entries() -> None:
    T = 2
    attn = _uniform_attn(2 * T)
    attn[0, 0] += 0.5
    attn[0, 1] -= 0.5  # still sums to 1, but now has a negative entry
    with pytest.raises(ValueError):
        prefix_matching_score(attn, T)


# --- prev_token_score ----------------------------------------------------


def test_prev_token_score_hand_constructed_shifted_diagonal_is_one() -> None:
    seq_len = 10
    attn = np.zeros((seq_len, seq_len))
    attn[0, 0] = 1.0
    for t in range(1, seq_len):
        attn[t, t - 1] = 1.0
    assert prev_token_score(attn) == pytest.approx(1.0, rel=1e-12)


def test_prev_token_score_uniform_attention_is_near_chance() -> None:
    seq_len = 20
    attn = _uniform_attn(seq_len)
    value = prev_token_score(attn)
    assert value == pytest.approx(1.0 / seq_len, rel=1e-12)
    assert value < 0.3


def test_prev_token_score_requires_at_least_two_positions() -> None:
    with pytest.raises(ValueError):
        prev_token_score(np.ones((1, 1)))


def test_prev_token_score_rejects_invalid_attention_pattern() -> None:
    with pytest.raises(ValueError):
        prev_token_score(np.zeros((5, 5)))


# --- copying_score ---------------------------------------------------------


def test_copying_score_identity_ov_circuit_is_one() -> None:
    d = 6
    W_E = np.eye(d)
    M_OV = np.eye(d)
    W_U = np.eye(d)
    assert copying_score(W_U, M_OV, W_E) == pytest.approx(1.0, rel=1e-12)


def test_copying_score_random_ov_is_near_chance() -> None:
    d, vocab = 32, 200
    rng = _rng(0)
    W_E = rng.standard_normal((vocab, d))
    W_U = rng.standard_normal((vocab, d))
    M_OV = rng.standard_normal((d, d))
    value = copying_score(W_U, M_OV, W_E)
    # Chance level is 1/vocab; a random OV circuit should land far below
    # the "near 1" identity case, not merely below it.
    assert value < 0.05


def test_copying_score_matches_naive_per_token_definition() -> None:
    # The identity-circuit oracle above is transpose-invariant (I.T == I),
    # so it can't catch an argument-order bug in the vectorized formula.
    # Recompute the literal per-token definition independently and compare.
    d, vocab = 5, 7
    rng = _rng(11)
    W_E = rng.standard_normal((vocab, d))
    W_U = rng.standard_normal((vocab, d))
    M_OV = rng.standard_normal((d, d))
    naive_matches = 0
    for t in range(vocab):
        logits_t = W_U @ (M_OV @ W_E[t])
        if int(np.argmax(logits_t)) == t:
            naive_matches += 1
    expected = naive_matches / vocab
    assert copying_score(W_U, M_OV, W_E) == pytest.approx(expected, rel=1e-12)


def test_copying_score_rejects_dimension_mismatch() -> None:
    with pytest.raises(ValueError):
        copying_score(np.zeros((10, 4)), np.zeros((5, 5)), np.zeros((10, 5)))


def test_copying_score_rejects_zero_ov() -> None:
    d = 4
    W_E = np.eye(d)
    W_U = np.eye(d)
    with pytest.raises(ValueError):
        copying_score(W_U, np.zeros((d, d)), W_E)


# --- icl_score ---------------------------------------------------------


def test_icl_score_closed_form_difference_of_means() -> None:
    nll_first = np.array([2.0, 3.0, 4.0])
    nll_second = np.array([1.0, 1.0, 1.0])
    # mean(first) - mean(second) = 3.0 - 1.0 = 2.0, exactly.
    assert icl_score(nll_first, nll_second) == pytest.approx(2.0, rel=1e-12)


def test_icl_score_zero_when_no_improvement() -> None:
    nll = np.array([1.5, 2.5, 3.5])
    assert icl_score(nll, nll) == pytest.approx(0.0, abs=1e-12)


def test_icl_score_rejects_shape_mismatch() -> None:
    with pytest.raises(ValueError):
        icl_score(np.zeros(3), np.zeros(4))


def test_icl_score_rejects_empty_input() -> None:
    with pytest.raises(ValueError):
        icl_score(np.zeros(0), np.zeros(0))


def test_icl_score_rejects_non_finite_input() -> None:
    with pytest.raises(ValueError):
        icl_score(np.array([1.0, np.nan]), np.array([1.0, 1.0]))


# --- recovery ------------------------------------------------------------


def test_recovery_zero_at_A_by_construction() -> None:
    icl_a, icl_b = 0.4, 2.1
    assert recovery(icl_a, icl_a, icl_b) == pytest.approx(0.0, abs=1e-12)


def test_recovery_one_at_B_by_construction() -> None:
    icl_a, icl_b = 0.4, 2.1
    assert recovery(icl_b, icl_a, icl_b) == pytest.approx(1.0, rel=1e-12)


def test_recovery_linear_between_A_and_B() -> None:
    icl_a, icl_b = 0.0, 4.0
    assert recovery(1.0, icl_a, icl_b) == pytest.approx(0.25, rel=1e-12)


def test_recovery_undefined_when_A_equals_B() -> None:
    with pytest.raises(ValueError):
        recovery(1.0, 0.5, 0.5)


def test_recovery_unclamped_can_exceed_unit_interval() -> None:
    # Unclamped is logged as-is (PROJECT.md §5); only reporting clamps.
    icl_a, icl_b = 0.0, 1.0
    assert recovery(1.6, icl_a, icl_b) == pytest.approx(1.6, rel=1e-12)
    assert recovery(-0.3, icl_a, icl_b) == pytest.approx(-0.3, rel=1e-12)


def test_clamp_recovery_bounds_to_unit_interval() -> None:
    assert clamp_recovery(-0.3) == 0.0
    assert clamp_recovery(1.6) == 1.0
    assert clamp_recovery(0.42) == pytest.approx(0.42, rel=1e-12)
