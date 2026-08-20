"""Closed-form oracles for src/indbw/existence.py, per docs/mathematics.md §12-§18.

Every identity checked here has an exact answer in the derivation, so the
assertions are machine-precision (rtol=1e-12) wherever the mathematics is
exact. The two places they are not are labelled with where the tolerance
comes from, per CLAUDE.md's numerical tolerance policy.

Three of these are discrimination guards rather than oracles -- Theorem
17.1's vanishing condition and Theorem 15.2's rank threshold are claims that
a quantity is *different* in two regimes, and a test that only ever saw one
regime could not catch a function that returned the same thing always.
"""

from __future__ import annotations

import numpy as np
import pytest

from indbw.existence import (
    attention_lower_bound,
    attention_score_gradient,
    matched_filter_update,
    random_projector,
    rank_ceiling,
    rank_constraint_is_vacuous,
    required_logit_margin,
    sketch_rank_threshold,
    sketch_signal_and_interference,
    value_spread,
    welch_coherence_bound,
)


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def _softmax(s: np.ndarray) -> np.ndarray:
    e = np.exp(s - s.max())
    return e / e.sum()


# --- §14: margin <-> attention -----------------------------------------


def test_margin_and_attention_bound_are_exact_inverses() -> None:
    """Corollary 14.2 is Proposition 14.1 solved for Delta; round-tripping is exact."""
    for tau, m in [(0.3, 256), (0.5, 2), (0.9, 50304)]:
        assert attention_lower_bound(required_logit_margin(tau, m), m) == pytest.approx(
            tau, rel=1e-12
        )


def test_required_margin_at_the_preregistered_pms_threshold() -> None:
    """tau = 0.3 (PROJECT.md §5) over m = 2T = 256 positions needs 4.69 score units."""
    assert required_logit_margin(0.3, 256) == pytest.approx(4.693965684771222, rel=1e-12)


def test_margin_bound_is_achieved_by_an_actual_softmax() -> None:
    """The bound is tight: it is attained when every distractor sits exactly at
    the margin, and is a strict under-estimate when they sit further below."""
    m, delta = 32, 3.0
    tight = np.full(m, -delta)
    tight[0] = 0.0
    assert _softmax(tight)[0] == pytest.approx(attention_lower_bound(delta, m), rel=1e-12)

    slack = np.full(m, -delta - 1.0)
    slack[0] = 0.0
    assert _softmax(slack)[0] > attention_lower_bound(delta, m)


def test_margin_rejects_degenerate_inputs() -> None:
    with pytest.raises(ValueError):
        required_logit_margin(0.0, 256)
    with pytest.raises(ValueError):
        required_logit_margin(1.0, 256)
    with pytest.raises(ValueError):
        required_logit_margin(0.3, 1)


# --- §13: the rank ceiling ---------------------------------------------


def test_rank_ceiling_is_d_head_for_the_qk_arm() -> None:
    """Lemma 13.2: W_Q is [d_model, d_head], so rank(Delta W_Q) <= d_head = 64."""
    assert rank_ceiling(512, 64) == 64
    assert rank_ceiling(64, 512) == 64  # OV arm: W_O is [d_head, d_model]


def test_generous_rank_64_constrains_nothing() -> None:
    """G3's r = d_head = 64 cell is the unconstrained arm, not a rank-limited one."""
    assert rank_constraint_is_vacuous(64, 512, 64)
    assert rank_constraint_is_vacuous(65, 512, 64)
    assert not rank_constraint_is_vacuous(63, 512, 64)


def test_any_delta_is_exactly_representable_at_the_ceiling_rank() -> None:
    """The claim behind Lemma 13.2: at r = d_head the LoRA image is all of
    R^{d x d_head}, so an arbitrary target is hit exactly, not approximated."""
    d, d_h = 40, 8
    target = _rng(0).standard_normal((d, d_h))
    U, s, Vt = np.linalg.svd(target, full_matrices=False)
    B, A = U * s, Vt  # a rank-d_h factorization
    np.testing.assert_allclose(B @ A, target, rtol=1e-12, atol=1e-12)
    assert B.shape[1] == rank_ceiling(d, d_h)


# --- §15: the construction and its rate --------------------------------


def test_welch_bound_at_the_real_vocab_and_head_dimension() -> None:
    """A4: 50304 token codes in 64 dimensions cannot be more separated than 0.125."""
    assert welch_coherence_bound(50304, 64) == pytest.approx(0.12492169982676785, rel=1e-12)
    assert welch_coherence_bound(64, 64) == 0.0  # they fit orthogonally


def test_matched_filter_has_the_rank_of_its_projector() -> None:
    """§15.2: rank(Delta W_Q^{(r)}) <= r, so it is a legal rank-r LoRA delta."""
    rng = _rng(1)
    d, d_h, n_codes, r = 30, 12, 50, 4
    P = random_projector(d_h, r, rng)
    dW = matched_filter_update(
        rng.standard_normal((d, n_codes)), rng.standard_normal((d_h, n_codes)), P
    )
    assert dW.shape == (d, d_h)
    assert np.linalg.matrix_rank(dW, tol=1e-10) <= r


def test_unsketched_matched_filter_is_exact_on_orthonormal_codes() -> None:
    """The closed-form limit: with orthonormal codes and Pi = I the filter gives
    signal exactly 1 and interference exactly 0, at any number of codes."""
    d_h = 8
    K = np.linalg.qr(_rng(2).standard_normal((d_h, d_h)))[0]  # d_h orthonormal codes
    signal, interference = sketch_signal_and_interference(K, np.eye(d_h))
    assert signal == pytest.approx(1.0, rel=1e-12)
    assert interference == pytest.approx(0.0, abs=1e-12)


def test_sketch_statistics_match_the_r_over_n_and_sqrt_r_over_n_laws() -> None:
    """Lemma 15.1: E[signal] = r/n and sd[interference] = sqrt(r)/n.

    Tolerance is 8% relative, not machine precision: these are expectations
    estimated from 400 draws, whose own standard error is ~5%. The point of
    the test is the *scaling law*, which a wrong formula (say r/n^2, or
    r/sqrt(n)) misses by orders of magnitude, not by 8%.
    """
    rng = _rng(3)
    n, reps = 64, 400
    for r in (4, 16, 32):
        sig, inter = [], []
        for _ in range(reps):
            K = rng.standard_normal((n, 2))
            P = random_projector(n, r, rng)
            s, i = sketch_signal_and_interference(K, P)
            sig.append(s)
            inter.append(i)
        assert np.mean(sig) == pytest.approx(r / n, rel=0.08)
        assert np.std(inter) == pytest.approx(np.sqrt(r) / n, rel=0.08)


def test_sketch_rank_threshold_is_logarithmic_in_competitors_only() -> None:
    """Theorem 15.2. The 2 log m form, and the H3 direction it implies (Cor 15.3)."""
    assert sketch_rank_threshold(256) == pytest.approx(2 * np.log(256), rel=1e-12)
    assert sketch_rank_threshold(256, eps=0.5) == pytest.approx(4 * 2 * np.log(256), rel=1e-12)
    # OV competes against the vocabulary, QK against the context: ratio ~1.95.
    ratio = sketch_rank_threshold(50304) / sketch_rank_threshold(256)
    assert ratio == pytest.approx(1.9522981877823258, rel=1e-12)


def test_separation_fails_below_the_threshold_and_holds_above_it() -> None:
    """Discrimination guard for Theorem 15.2: the sketch must actually *stop*
    separating at low rank, or the threshold is not measuring anything.

    Seeded and stated as a wide-band claim (r=1 almost never separates,
    r = 4 * 2 log m almost always does), so it tests the regime change rather
    than a hand-tuned crossing point.
    """
    rng = _rng(4)
    n, m, reps = 64, 256, 100

    def win_rate(r: int) -> float:
        wins = 0
        for _ in range(reps):
            K = rng.standard_normal((n, m))
            signal, interference = sketch_signal_and_interference(K, random_projector(n, r, rng))
            wins += signal > interference
        return wins / reps

    assert win_rate(1) < 0.10
    assert win_rate(int(4 * sketch_rank_threshold(m))) > 0.99


# --- §17: the gradient gate --------------------------------------------


def test_score_gradient_matches_finite_differences() -> None:
    """Theorem 17.1 against a numerical derivative of the actual attention output.

    atol=1e-8: a central difference at h=1e-6 on a float64 objective carries
    ~1e-10 truncation and ~1e-10 cancellation error; 1e-8 is two decades of
    headroom and still three decades tighter than any plausible sign or
    index error in the formula.
    """
    rng = _rng(5)
    m, d, h = 7, 5, 1e-6
    s = rng.standard_normal(m)
    Z = rng.standard_normal((m, d))
    g = rng.standard_normal(d)

    def loss(scores: np.ndarray) -> float:
        return float(g @ (_softmax(scores) @ Z))

    numerical = np.zeros(m)
    for j in range(m):
        bump = np.zeros(m)
        bump[j] = h
        numerical[j] = (loss(s + bump) - loss(s - bump)) / (2 * h)

    np.testing.assert_allclose(attention_score_gradient(_softmax(s), Z, g), numerical, atol=1e-8)


def test_constant_values_give_exactly_zero_gradient_but_varying_values_do_not() -> None:
    """Corollary 17.2, as a discrimination test: the gate is closed on a
    position-independent write and open otherwise. A function that always
    returned zeros would pass the first half alone."""
    rng = _rng(6)
    m, d = 9, 4
    attn = _softmax(rng.standard_normal(m))
    g = rng.standard_normal(d)

    constant = np.tile(rng.standard_normal(d), (m, 1))
    np.testing.assert_allclose(attention_score_gradient(attn, constant, g), 0.0, atol=1e-15)
    assert value_spread(attn, constant) == pytest.approx(0.0, abs=1e-15)

    varying = rng.standard_normal((m, d))
    assert np.max(np.abs(attention_score_gradient(attn, varying, g))) > 1e-3
    assert value_spread(attn, varying) > 1e-3


def test_score_gradient_sums_to_zero() -> None:
    """A softmax row is normalized, so no score gradient can move total attention
    mass. Exact identity, and the cheapest available check on the formula's shape."""
    rng = _rng(7)
    attn = _softmax(rng.standard_normal(11))
    grad = attention_score_gradient(attn, rng.standard_normal((11, 6)), rng.standard_normal(6))
    assert float(np.sum(grad)) == pytest.approx(0.0, abs=1e-14)


def test_gradient_is_invariant_to_shifting_all_values_by_a_constant() -> None:
    """z_j - zbar is shift-invariant, so adding the same vector to every position's
    write leaves the QK gradient untouched -- only the *spread* matters (Cor 17.2)."""
    rng = _rng(8)
    m, d = 8, 5
    attn = _softmax(rng.standard_normal(m))
    Z = rng.standard_normal((m, d))
    g = rng.standard_normal(d)
    shifted = Z + rng.standard_normal(d)
    np.testing.assert_allclose(
        attention_score_gradient(attn, Z, g),
        attention_score_gradient(attn, shifted, g),
        rtol=1e-12,
        atol=1e-12,
    )


def test_readouts_reject_a_non_normalized_attention_row() -> None:
    """Silent-failure guard: an un-normalized row is not a softmax output, and
    both readouts return plausible-looking numbers on one if they do not check."""
    Z = _rng(9).standard_normal((5, 3))
    bad = np.full(5, 0.5)
    with pytest.raises(ValueError):
        attention_score_gradient(bad, Z, np.ones(3))
    with pytest.raises(ValueError):
        value_spread(bad, Z)
    with pytest.raises(ValueError):
        value_spread(np.zeros(5), Z)
