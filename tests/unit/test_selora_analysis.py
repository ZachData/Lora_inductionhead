"""Discrimination tests: each claim's observed->verdict path says pass on a planted truth
and fail on a planted counter-truth. This is the silent-failure guard for the analysis layer."""

from __future__ import annotations

import numpy as np
import pytest

from indbw.selora import analysis, prereg
from indbw.selora.adjudicate import adjudicate

CFG = {"t": 1}


def _verdict(claim_id: str, observed: dict) -> str:  # type: ignore[type-arg]
    ctrl = list(prereg.claim(claim_id).controls)
    return adjudicate(claim_id, observed, ctrl, CFG, 0, "r", "h", 1.0).verdict


def _fvu(rng: np.random.Generator, shift: float, n: int = 200) -> tuple[np.ndarray, np.ndarray]:
    base = rng.normal(0.10, 0.01, n)
    return base, base + shift + rng.normal(0, 0.005, n)


def test_s0_exact_zero_passes_and_any_difference_fails() -> None:
    x = np.random.default_rng(0).normal(size=(3, 5, 8))
    assert _verdict("S0", analysis.s0_observed(x, x.copy())) == "pass"
    y = x.copy()
    y[0, 0, 0] += 1e-9
    assert _verdict("S0", analysis.s0_observed(x, y)) == "fail"


def test_s1a_no_shift_passes_big_shift_fails() -> None:
    rng = np.random.default_rng(1)
    ok = {f"c{i}": _fvu(rng, 0.005) for i in range(6)}
    bad = {**ok, "c_bad": _fvu(rng, 0.12)}
    assert _verdict("S1a", analysis.s1a_observed(ok)) == "pass"
    assert _verdict("S1a", analysis.s1a_observed(bad)) == "fail"


def test_s1a_absolute_bound_binds_even_when_shift_is_small() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(0.30, 0.01, 200)  # SAE already poor at 30% FVU
    obs = analysis.s1a_observed({"c": (base, base + 0.001)})
    assert obs["n_cells_holm_p_above_alpha"] == 0 and _verdict("S1a", obs) == "fail"


def test_s1b_trained_beats_random_only_when_it_does() -> None:
    rng = np.random.default_rng(3)
    trained = rng.normal(0.05, 0.01, 200)
    assert (
        _verdict("S1b", analysis.s1b_observed({"c": (trained, rng.normal(0.01, 0.01, 200))}))
        == "pass"
    )
    assert (
        _verdict("S1b", analysis.s1b_observed({"c": (trained, rng.normal(0.05, 0.01, 200))}))
        == "fail"
    )


def test_s2a_counts_layers() -> None:
    assert _verdict("S2a", analysis.s2a_observed([0.0004] * 3 + [0.9] * 23)) == "pass"
    assert _verdict("S2a", analysis.s2a_observed([0.5] * 26)) == "fail"


def test_s2b_peak() -> None:
    ex = np.zeros(26)
    ex[8] = 5.0
    assert _verdict("S2b", analysis.s2b_observed(ex, np.full(26, 0.1))) == "pass"
    ex2 = np.zeros(26)
    ex2[20] = 5.0
    assert _verdict("S2b", analysis.s2b_observed(ex2, np.full(26, 0.1))) == "fail"


def test_s3a_noninferiority_discriminates() -> None:
    rng = np.random.default_rng(4)
    n = 1319
    allc = (rng.random((3, n)) < 0.38).astype(float)
    same = np.where(rng.random((3, n)) < 0.97, allc, 1 - allc)  # ~equal accuracy, noisy pairing
    worse = allc * (rng.random((3, n)) < 0.80)  # drops ~8 points
    ok = analysis.s3a_observed(same, allc)
    bad = analysis.s3a_observed(worse, allc)
    assert ok["holm_p"] <= prereg.ALPHA and bad["holm_p"] > prereg.ALPHA
    assert _verdict("S3a", ok) == "pass" and _verdict("S3a", bad) == "fail"


def test_rank_claim_reaches_alpha_only_if_best_of_24() -> None:
    rng = np.random.default_rng(5)
    draws = rng.normal(0.3, 0.02, prereg.N_RANDOM_DRAWS)
    assert _verdict("S3b", analysis.rank_claim_observed(draws.max() + 0.1, draws)) == "pass"
    assert _verdict("S3b", analysis.rank_claim_observed(draws.mean(), draws)) == "fail"


def test_s4_formatting_vs_forgetting() -> None:
    rng = np.random.default_rng(6)
    fmt = analysis.s4_observed(rng.random(500) < 0.6, rng.random(500) < 0.45)
    forgot = analysis.s4_observed(rng.random(500) < 0.10, rng.random(500) < 0.10)
    assert _verdict("S4", fmt) == "pass" and _verdict("S4", forgot) == "fail"


def test_analysis_guards() -> None:
    with pytest.raises(ValueError):
        analysis.s1a_observed({})
    with pytest.raises(ValueError):
        analysis.s1a_observed({"c": (np.zeros(3), np.zeros(4))})
    with pytest.raises(ValueError):
        analysis.s3a_observed(np.zeros((2, 5)), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        analysis.s0_observed(np.zeros(3), np.zeros(4))
