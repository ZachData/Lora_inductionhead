"""Spec for selora.prereg / adjudicate: the pre-registration is machine-checkable and power-feasible."""

from __future__ import annotations

import pytest

from indbw.schema import record_from_dict
from indbw.selora import prereg
from indbw.selora.adjudicate import adjudicate
from indbw.selora.stats import min_attainable_p

OBS_S0_PASS = {"max_abs_resid_diff_below_first_adapted": 0.0}
CFG = {"model": "tiny"}


def _adj(claim_id: str, observed: dict, controls: list[str]):  # type: ignore[no-untyped-def]
    return adjudicate(claim_id, observed, controls, CFG, 0, "tiny-rev", "evalhash", 1.0)


def test_claim_ids_unique_and_every_criterion_is_machine_evaluable() -> None:
    ids = [c.id for c in prereg.CLAIMS]
    assert len(ids) == len(set(ids))
    for c in prereg.CLAIMS:
        assert c.criteria and c.controls and c.null
        for crit in c.criteria:
            assert crit.op in {"<=", "<", ">=", ">", "=="}


def test_rank_tests_are_powered_to_reach_alpha() -> None:
    # A rank test against N random draws cannot return p below 1/(N+1). The design is
    # infeasible (a 'significant' result is impossible) unless that floor is below alpha.
    assert min_attainable_p(prereg.N_RANDOM_DRAWS) < prereg.ALPHA


def test_draft_status_blocks_real_runs() -> None:
    assert prereg.PREREG_STATUS == "draft"
    with pytest.raises(RuntimeError, match="draft"):
        prereg.require_registered()


def test_prereg_hash_changes_with_any_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    h = prereg.prereg_hash()
    assert h == prereg.prereg_hash()
    monkeypatch.setattr(prereg, "ACC_MARGIN", 0.04)
    assert prereg.prereg_hash() != h


def test_verdict_is_computed_and_discriminates() -> None:
    ok = _adj("S0", OBS_S0_PASS, ["late-layer arm"])
    bad = _adj("S0", {"max_abs_resid_diff_below_first_adapted": 1e-9}, ["late-layer arm"])
    assert ok.verdict == "pass" and bad.verdict == "fail"
    assert ok.is_self_consistent() and bad.is_self_consistent()


def test_compound_claim_fails_if_any_clause_fails() -> None:
    ctrl = list(prereg.claim("S1a").controls)
    good = {"n_cells_p_above_alpha": 0.0, "max_fvu_ft_upper95": 0.10}
    assert _adj("S1a", good, ctrl).verdict == "pass"
    assert _adj("S1a", {**good, "max_fvu_ft_upper95": 0.20}, ctrl).verdict == "fail"
    assert _adj("S1a", {**good, "n_cells_p_above_alpha": 3.0}, ctrl).verdict == "fail"


def test_missing_control_is_refused() -> None:
    with pytest.raises(ValueError, match="controls"):
        _adj(
            "S1a",
            {"n_cells_p_above_alpha": 0.0, "max_fvu_ft_upper95": 0.1},
            ["spectrum-matched random LoRA"],
        )


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), None, "0.1", True])
def test_missing_or_nonfinite_metric_is_refused(bad: object) -> None:
    with pytest.raises(ValueError):
        _adj("S0", {"max_abs_resid_diff_below_first_adapted": bad}, ["late-layer arm"])
    with pytest.raises(ValueError):
        _adj("S0", {}, ["late-layer arm"])


def test_record_roundtrips_through_schema() -> None:
    rec = _adj("S0", OBS_S0_PASS, ["late-layer arm"])
    again = record_from_dict(__import__("json").loads(rec.to_json_line()))
    assert again.is_self_consistent() and again.observed["prereg_hash"] == prereg.prereg_hash()


def test_assumptions_are_recorded_and_complete() -> None:
    a = prereg.ASSUMPTIONS
    for k in (
        "gsm8k_accuracy",
        "gsm8k_n_items",
        "gsm8k_discordance_low",
        "gsm8k_discordance_high",
        "seed_item_correlation",
        "fvu_delta_sd",
        "sae_prompts",
    ):
        assert k in a
    assert 0 < a["gsm8k_discordance_low"] < a["gsm8k_discordance_high"] < 1


def test_prereg_hash_covers_assumptions_and_multiplicity(monkeypatch: pytest.MonkeyPatch) -> None:
    h = prereg.prereg_hash()
    monkeypatch.setitem(prereg.ASSUMPTIONS, "fvu_delta_sd", 0.123)
    assert prereg.prereg_hash() != h
