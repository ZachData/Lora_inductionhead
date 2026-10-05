"""Spec for selora.stats: closed-form oracles, calibration, discrimination, guards."""

from __future__ import annotations

import numpy as np
import pytest

from indbw.selora import stats


def test_rank_pvalue_closed_form() -> None:
    ref = np.arange(10, dtype=float)
    assert stats.rank_pvalue(9.5, ref) == 1 / 11  # beats every reference draw
    assert stats.rank_pvalue(-1.0, ref) == 1.0  # beaten by every draw
    assert stats.rank_pvalue(4.5, ref) == 6 / 11  # 5,6,7,8,9 are >= 4.5, plus itself


def test_rank_pvalue_never_zero_and_floor_matches() -> None:
    ref = np.zeros(24)
    assert stats.rank_pvalue(1e9, ref) == stats.min_attainable_p(24) == 1 / 25


def test_rank_pvalue_type_i_error_is_calibrated() -> None:
    # Under exchangeability P(p <= 1/20) = 1/20 exactly with 19 reference draws.
    # 2000 trials: binomial sd = 0.0049, so 0.05 + 4 sd = 0.07 is the bound.
    rng = np.random.default_rng(0)
    p = np.array([stats.rank_pvalue(rng.normal(), rng.normal(size=19)) for _ in range(2000)])
    assert np.mean(p <= 0.05) <= 0.07
    assert np.mean(p <= 0.05) >= 0.03


def test_sign_flip_discriminates_signal_from_noise() -> None:
    rng = np.random.default_rng(1)
    noise = rng.normal(size=60)
    signal = noise + 1.0
    assert stats.sign_flip_pvalue(noise, 2000, rng) > 0.05
    assert stats.sign_flip_pvalue(signal, 2000, rng) <= 2 / 2001


def test_sign_flip_margin_shift() -> None:
    # True mean is exactly -0.02; non-inferiority at margin 0.05 holds, at 0.01 it does not.
    rng = np.random.default_rng(2)
    d = rng.normal(-0.02, 0.01, size=200)
    assert stats.sign_flip_pvalue(d, 2000, rng, shift=-0.05) <= 0.01
    assert stats.sign_flip_pvalue(d, 2000, rng, shift=-0.01) > 0.5


def test_sign_flip_type_i_error() -> None:
    # 300 null datasets; sd of the rejection rate is sqrt(.05*.95/300)=0.0126; bound .05+4sd.
    rng = np.random.default_rng(3)
    rej = [stats.sign_flip_pvalue(rng.normal(size=30), 400, rng) <= 0.05 for _ in range(300)]
    assert np.mean(rej) <= 0.10


def test_bootstrap_ci_degenerate_and_covering() -> None:
    rng = np.random.default_rng(4)
    m, lo, hi = stats.cluster_bootstrap_ci(np.full(20, 0.3), 500, rng)
    assert m == lo == hi == pytest.approx(0.3, rel=1e-12)
    x = rng.normal(5.0, 1.0, size=400)
    _, lo, hi = stats.cluster_bootstrap_ci(x, 2000, rng)
    assert lo < 5.0 < hi
    assert stats.one_sided_lower_bound(x, 2000, rng) > lo  # one-sided is tighter than two-sided
    assert stats.one_sided_upper_bound(x, 2000, rng) < hi


def test_holm_and_bh_hand_computed() -> None:
    p = np.array([0.01, 0.04, 0.03])
    np.testing.assert_allclose(stats.holm(p), [0.03, 0.06, 0.06], rtol=1e-12)
    np.testing.assert_allclose(stats.benjamini_hochberg(p), [0.03, 0.04, 0.04], rtol=1e-12)


def test_mcnemar_exact() -> None:
    assert stats.mcnemar_exact_pvalue(0, 10) == pytest.approx(2 / 1024, rel=1e-12)
    assert stats.mcnemar_exact_pvalue(5, 5) == 1.0
    assert stats.mcnemar_exact_pvalue(0, 0) == 1.0


@pytest.mark.parametrize(
    "bad", [np.array([]), np.array([1.0, np.nan]), np.array([[1.0, 2.0]]), np.array([np.inf])]
)
def test_guards_raise_on_bad_input(bad: np.ndarray) -> None:
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        stats.rank_pvalue(0.0, bad)
    with pytest.raises(ValueError):
        stats.sign_flip_pvalue(bad, 10, rng)
    with pytest.raises(ValueError):
        stats.cluster_bootstrap_ci(bad, 10, rng)


def test_p_values_outside_unit_interval_raise() -> None:
    with pytest.raises(ValueError):
        stats.holm(np.array([0.5, 1.2]))
    with pytest.raises(ValueError):
        stats.benjamini_hochberg(np.array([-0.1, 0.2]))
