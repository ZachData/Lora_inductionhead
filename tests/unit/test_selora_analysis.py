"""Discrimination tests: each claim's observed->verdict path says pass on a planted truth
and fail on a planted counter-truth. The silent-failure guard for the analysis layer."""

from __future__ import annotations

import numpy as np
import pytest

from indbw.selora import analysis, prereg
from indbw.selora.adjudicate import adjudicate

CFG = {"t": 1}
IT, BASE = "google/gemma-2-2b-it", "google/gemma-2-2b"


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


def test_s1a_is_intersection_union_not_holm() -> None:
    # 156 cells each with raw p just under alpha: IUT rejects all; Holm would reject none.
    rng = np.random.default_rng(7)
    cells = {f"c{i}": _fvu(rng, 0.0) for i in range(156)}
    obs = analysis.s1a_observed(cells)
    assert obs["n_cells_p_above_alpha"] == 0.0 and obs["n_cells"] == 156.0


def test_s1a_absolute_bound_binds_even_when_shift_is_small() -> None:
    rng = np.random.default_rng(2)
    base = rng.normal(0.30, 0.01, 200)
    obs = analysis.s1a_observed({"c": (base, base + 0.001 + rng.normal(0, 0.002, 200))})
    assert obs["n_cells_p_above_alpha"] == 0 and _verdict("S1a", obs) == "fail"


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


def test_seed_noise_floor_reports_whether_margin_is_resolvable() -> None:
    rng = np.random.default_rng(8)
    quiet = {"c": (rng.normal(0.1, 0.01, 200), rng.normal(0.1, 0.01, 200))}
    loud = {"c": (rng.normal(0.1, 0.01, 200), rng.normal(0.2, 0.01, 200))}
    q, ld = analysis.seed_noise_floor(quiet), analysis.seed_noise_floor(loud)
    assert q["margin_resolvable"] == 1.0 and ld["margin_resolvable"] == 0.0
    assert q["noise_floor_upper95"] < prereg.FVU_MARGIN < ld["noise_floor_upper95"]


def test_s2a_counts_layers() -> None:
    assert _verdict("S2a", analysis.s2a_observed([0.0004] * 3 + [0.9] * 23)) == "pass"
    assert _verdict("S2a", analysis.s2a_observed([0.5] * 26)) == "fail"


# ---- gatekeeping and the A3/A4 fixes -------------------------------------------------------


def _s3a_inputs(rng: np.random.Generator, drop: float) -> tuple[np.ndarray, np.ndarray]:
    allc = (rng.random((3, 1319)) < 0.38).astype(float)
    late = np.where(rng.random((3, 1319)) < 0.97, allc, 1 - allc)
    if drop:
        late = late * (rng.random((3, 1319)) > drop)
    return late, allc


def _steps(rng: np.random.Generator, s3a_drop: float, late_best: bool) -> dict:  # type: ignore[type-arg]
    draws = rng.normal(0.3, 0.02, prereg.N_RANDOM_DRAWS)
    named = {0: draws.max() + 0.1 if late_best else draws.mean(), 1: 0.0, 2: 0.0}
    out = {}
    for model in (IT, BASE):
        out[("S3a", model)] = analysis.s3a_step(*_s3a_inputs(rng, s3a_drop))
        out[("S3b", model)] = analysis.rank_step(named, draws, random_seed=0)
    return out


def test_raw_steps_cannot_be_adjudicated_directly() -> None:
    # A4: only gatekeep() produces the criterion field `p_adjusted`.
    rng = np.random.default_rng(4)
    step = analysis.s3a_step(*_s3a_inputs(rng, 0.0))
    assert "p_adjusted" not in step
    with pytest.raises(ValueError, match="p_adjusted"):
        _verdict("S3a", step)


def test_gatekeeping_passes_all_when_all_hold() -> None:
    obs = analysis.gatekeep(_steps(np.random.default_rng(5), 0.0, True))
    assert [_verdict(cid, obs[(cid, m)]) for cid, m in prereg.GATE_SEQUENCE] == ["pass"] * 4


def test_gatekeeping_failure_blocks_everything_after_it() -> None:
    # S3a holds, S3b (2nd in sequence) fails -> steps 3 and 4 fail even though S3a-base holds.
    obs = analysis.gatekeep(_steps(np.random.default_rng(6), 0.0, False))
    verdicts = [_verdict(cid, obs[(cid, m)]) for cid, m in prereg.GATE_SEQUENCE]
    assert verdicts == ["pass", "fail", "fail", "fail"]


def test_gatekeep_requires_exactly_the_registered_sequence() -> None:
    steps = _steps(np.random.default_rng(7), 0.0, True)
    del steps[("S3b", BASE)]
    with pytest.raises(ValueError, match="sequence"):
        analysis.gatekeep(steps)


def test_rank_step_uses_only_the_same_seed_as_the_random_draws() -> None:
    draws = np.linspace(0.2, 0.4, prereg.N_RANDOM_DRAWS)
    # Seed 0 is unremarkable; the mean over seeds would be best-of-25. Only seed 0 may count.
    named = {0: 0.30, 1: 0.9, 2: 0.9}
    assert analysis.rank_step(named, draws, random_seed=0)["p_raw"] > 0.3
    with pytest.raises(KeyError):
        analysis.rank_step({1: 0.9}, draws, random_seed=0)


def test_s5_single_test() -> None:
    draws = np.linspace(0.2, 0.4, prereg.N_RANDOM_DRAWS)
    assert _verdict("S5", analysis.s5_observed({0: 0.5}, draws, 0)) == "pass"
    assert _verdict("S5", analysis.s5_observed({0: 0.3}, draws, 0)) == "fail"


def test_s4_formatting_vs_forgetting() -> None:
    rng = np.random.default_rng(6)
    fmt = analysis.s4_observed(rng.random(500) < 0.6, rng.random(14000) < 0.45)
    forgot = analysis.s4_observed(rng.random(500) < 0.10, rng.random(14000) < 0.10)
    assert _verdict("S4", fmt) == "pass" and _verdict("S4", forgot) == "fail"


def test_require_complete_refuses_partial_sweeps() -> None:
    analysis.require_complete({"a", "b"}, ["a", "b"])
    with pytest.raises(RuntimeError, match="1 of 3"):
        analysis.require_complete({"a", "b"}, ["a", "b", "c"])


def test_analysis_guards() -> None:
    with pytest.raises(ValueError):
        analysis.s1a_observed({})
    with pytest.raises(ValueError):
        analysis.s1a_observed({"c": (np.zeros(3), np.zeros(4))})
    with pytest.raises(ValueError):
        analysis.s3a_step(np.zeros((2, 5)), np.zeros((3, 5)))
    with pytest.raises(ValueError):
        analysis.s0_observed(np.zeros(3), np.zeros(4))
