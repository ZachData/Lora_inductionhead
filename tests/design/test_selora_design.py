"""Design-level tests: the pre-registered procedure is feasible, calibrated and powered.

These simulate the *procedure*, not the code paths: planted-true and planted-false data
under `prereg.ASSUMPTIONS`. A threshold, sample-size or multiplicity edit must keep them green
(STATE.md practice 1). Simulation tolerances are binomial: stated per test.
"""

from __future__ import annotations

import numpy as np
import pytest

from indbw.selora import analysis, prereg, stats

A = prereg.ASSUMPTIONS


# ---- feasibility (STATE A1) -------------------------------------------------------------


def test_every_registered_test_can_clear_its_multiplicity_threshold() -> None:
    assert prereg.feasibility_problems() == []


def test_feasibility_check_catches_the_original_s3b_design() -> None:
    # 24 rank draws (floor 0.04) inside a Holm family of 4 (strictest threshold 0.0125).
    bad = {
        "S3b": prereg.TestSpec(floor=stats.min_attainable_p(24), multiplicity="holm", family_size=4)
    }
    probs = prereg.feasibility_problems(bad)
    assert len(probs) == 1 and "S3b" in probs[0]


def test_every_claim_declares_a_known_multiplicity_and_has_a_test_spec() -> None:
    for c in prereg.CLAIMS:
        assert c.multiplicity in {"iut", "holm", "gatekeep", "single"}
        if c.id != "S0":  # S0 is an exact equality, no p-value
            assert c.id in prereg.TEST_SPECS


def test_gate_sequence_names_only_gatekeep_claims() -> None:
    ids = {cid for cid, _ in prereg.GATE_SEQUENCE}
    assert ids == {c.id for c in prereg.CLAIMS if c.multiplicity == "gatekeep"}


# ---- fixed-sequence FWER (STATE A1) -------------------------------------------------------


def test_fixed_sequence_controls_fwer_under_global_null() -> None:
    # 4 uniform p-values, 4000 replications: FWER = P(p1 <= a) = a exactly; sd ~0.0034.
    rng = np.random.default_rng(0)
    p = rng.random((4000, 4))
    adj = np.array([stats.fixed_sequence_adjust(row) for row in p])
    assert np.mean((adj <= prereg.ALPHA).any(axis=1)) <= prereg.ALPHA + 0.015


def test_fixed_sequence_stops_at_first_failure() -> None:
    adj = stats.fixed_sequence_adjust(np.array([0.01, 0.20, 0.001, 0.001]))
    np.testing.assert_allclose(adj, [0.01, 0.20, 0.20, 0.20])


# ---- S3a pooled non-inferiority (STATE B1) ------------------------------------------------


def _paired_diffs(
    rng: np.random.Generator, true_diff: float, disc: float, n_seeds: int = 3
) -> np.ndarray:
    """[seeds, items] late-minus-all in {-1,0,1}; discordance partly shared across seeds."""
    n = A["gsm8k_n_items"]
    p_plus, p_minus = disc / 2 + true_diff / 2, disc / 2 - true_diff / 2
    shared = rng.random(n)
    out = np.zeros((n_seeds, n))
    for s in range(n_seeds):
        u = np.where(rng.random(n) < A["seed_item_correlation"], shared, rng.random(n))
        out[s] = np.where(u < p_plus, 1.0, np.where(u < p_plus + p_minus, -1.0, 0.0))
    return out


@pytest.mark.parametrize("disc", [A["gsm8k_discordance_low"], A["gsm8k_discordance_high"]])
def test_s3a_type_i_at_margin_boundary(disc: float) -> None:
    # 150 reps at the null boundary (true diff = -margin): sd of the rate ~0.018; bound a + 3 sd.
    rng = np.random.default_rng(1)
    rej = [
        analysis.s3a_from_diffs(_paired_diffs(rng, -prereg.ACC_MARGIN, disc), seed=i)["p_raw"]
        <= prereg.ALPHA
        for i in range(150)
    ]
    assert np.mean(rej) <= prereg.ALPHA + 0.055


@pytest.mark.parametrize("disc", [A["gsm8k_discordance_low"], A["gsm8k_discordance_high"]])
def test_s3a_power_when_arms_are_equal(disc: float) -> None:
    rng = np.random.default_rng(2)
    ok = [
        analysis.s3a_from_diffs(_paired_diffs(rng, 0.0, disc), seed=i)["p_raw"] <= prereg.ALPHA
        for i in range(100)
    ]
    assert np.mean(ok) >= 0.8


# ---- S1a bootstrap-t (STATE B4) -----------------------------------------------------------


def test_s1a_bootstrap_t_type_i_on_skewed_fvu_differences() -> None:
    # Lognormal-skewed per-prompt dFVU with mean exactly at the margin (null boundary).
    # 200 reps: sd ~0.015; bound a + 0.045.
    rng = np.random.default_rng(3)
    rej = 0
    for _ in range(200):
        x = rng.lognormal(0.0, 0.8, A["sae_prompts"])
        x = x / x.mean() * prereg.FVU_MARGIN
        rej += stats.bootstrap_t_pvalue(x, prereg.FVU_MARGIN, 1000, rng, "less") <= prereg.ALPHA
    assert rej / 200 <= prereg.ALPHA + 0.045


def test_s1a_power_with_assumed_spread() -> None:
    rng = np.random.default_rng(4)
    hits = 0
    for _ in range(100):
        x = rng.normal(0.0, A["fvu_delta_sd"], A["sae_prompts"])
        hits += stats.bootstrap_t_pvalue(x, prereg.FVU_MARGIN, 1000, rng, "less") <= prereg.ALPHA
    assert hits / 100 >= 0.8
