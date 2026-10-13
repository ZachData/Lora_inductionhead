"""Spec for selora.report: complete sweep rows -> one schema-valid record per claim.

Synthetic rows stand in for a finished sweep. Controls are attached only when the rows
actually contain the data that computes them (STATE.md: a control is a number, not a label).
"""

from __future__ import annotations

from dataclasses import asdict

import numpy as np
import pytest

from indbw.selora import cells, prereg, report

N_PROMPTS, N_GSM, N_MMLU_LP, N_MMLU_GEN = 12, 60, 80, 20
PROV = report.Provenance({"test": 1}, 0, "synthetic", "synthetic", 0.0)
SCORES = {m: list(np.linspace(0, 1, cells.N_LAYERS)) for m in cells.MODELS}


def _row(c: cells.Cell, rng: np.random.Generator, good: bool) -> dict:  # type: ignore[type-arg]
    layers = [str(li) for li in range(cells.N_LAYERS)]
    base = {li: list(rng.normal(0.10, 0.01, N_PROMPTS)) for li in layers}
    shift = 0.002 if good else 0.2
    adapted = {str(li) for li in c.layers}
    ft = {
        li: [v + (shift if li in adapted else 0.0) + rng.normal(0, 0.003) for v in base[li]]
        for li in layers
    }
    rand = {
        li: [v + (0.0005 if li in adapted else 0.0) + rng.normal(0, 0.003) for v in base[li]]
        for li in layers
    }
    p_gsm = 0.40 if c.arm in ("late", "all") else 0.35
    acc_lp = 0.48 if c.arm == "late" else 0.40
    row = {
        "run_id": c.run_id,
        "cell": asdict(c),
        "fvu_base": base,
        "fvu_ft": ft if c.arm != "base" else base,
        "resid_max_abs_diff": {
            li: (0.0 if li not in adapted and int(li) < min(c.layers or (99,)) else 0.3)
            for li in layers
        },
        "gsm_correct": list((rng.random(N_GSM) < p_gsm).astype(int)),
        "mmlu_logprob": ["A" if rng.random() < acc_lp else "B" for _ in range(N_MMLU_LP)],
        "mmlu_logprob_gold": ["A"] * N_MMLU_LP,
        "mmlu_gen": [
            None if rng.random() < 0.5 else ("A" if rng.random() < 0.6 else "C")
            for _ in range(N_MMLU_GEN)
        ],
        "mmlu_gen_gold": ["A"] * N_MMLU_GEN,
    }
    if c.arm != "base":
        row["fvu_rand"] = rand
    return row


def _sweep(good: bool = True) -> tuple[list[cells.Cell], list[dict]]:  # type: ignore[type-arg]
    rng = np.random.default_rng(0)
    planned = cells.enumerate_cells(SCORES)
    return planned, [_row(c, rng, good) for c in planned]


def _divergence() -> dict:  # type: ignore[type-arg]
    p = [0.0004] * 3 + [0.6] * (cells.N_LAYERS - 3)
    return {
        m: {
            "p_by_layer": p,
            "peak_in_range_frequency": 0.9,
            "topk_sensitivity_done": True,
            "strata": "length x digit",
        }
        for m in cells.MODELS
    }


def test_refuses_a_partial_sweep() -> None:
    planned, rows = _sweep()
    with pytest.raises(RuntimeError, match="planned cells"):
        report.build_records(rows[:-1], planned, _divergence(), PROV, gsm_items_expected=N_GSM)


def test_one_self_consistent_record_per_claim_and_gate_step() -> None:
    planned, rows = _sweep()
    recs = report.build_records(rows, planned, _divergence(), PROV, gsm_items_expected=N_GSM)
    keys = [(r.row, r.observed.get("model", "")) for r in recs]
    gate_keys = {(cid, m) for cid, m in prereg.GATE_SEQUENCE}
    assert gate_keys <= set(keys)
    singles = {c.id for c in prereg.CLAIMS if c.multiplicity != "gatekeep"}
    assert singles <= {k for k, _ in keys}
    assert all(r.is_self_consistent() for r in recs)


def test_verdicts_follow_the_planted_truth() -> None:
    planned, rows = _sweep(good=True)
    good = {
        r.row: r.verdict
        for r in report.build_records(rows, planned, _divergence(), PROV, gsm_items_expected=N_GSM)
    }
    planned, rows = _sweep(good=False)
    bad = {
        r.row: r.verdict
        for r in report.build_records(rows, planned, _divergence(), PROV, gsm_items_expected=N_GSM)
    }
    assert good["S0"] == "pass" and good["S1a"] == "pass"
    assert bad["S1a"] == "fail"  # FVU jumps 0.2 at adapted layers


def test_controls_come_from_data_not_labels() -> None:
    planned, rows = _sweep()
    for r in rows:
        r.pop("fvu_rand", None)  # the random-LoRA null was never computed
    with pytest.raises(ValueError, match="spectrum-matched"):
        report.build_records(rows, planned, _divergence(), PROV, gsm_items_expected=N_GSM)


def test_full_test_set_control_needs_all_items() -> None:
    planned, rows = _sweep()
    with pytest.raises(ValueError, match="1319"):
        report.build_records(rows, planned, _divergence(), PROV)  # 60 items, 1319 expected
