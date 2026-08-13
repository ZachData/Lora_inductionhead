"""Closed-form and discrimination tests for src/indbw/schema.py.

PROJECT.md §9 / CLAUDE.md "Computable verdicts": a results record's
verdict must be exactly recomputable from its criteria and observed
values. The discrimination tests here are the ones that matter --
a schema that lets a wrong verdict through silently is exactly the
"correctly-functioning pipeline whose conclusion does not follow from
its own numbers" failure CLAUDE.md calls out as the one no unit test
normally catches.
"""

from __future__ import annotations

import json

import pytest

from indbw.schema import Criterion, ResultsRecord, load_records, record_from_dict

# --- Criterion.holds --------------------------------------------------


@pytest.mark.parametrize(
    "op,value,threshold,expected",
    [
        ("<=", 3, 3, True),
        ("<=", 4, 3, False),
        ("<", 2, 3, True),
        ("<", 3, 3, False),
        (">=", 3, 3, True),
        (">=", 2, 3, False),
        (">", 4, 3, True),
        (">", 3, 3, False),
        ("==", 1, 1, True),
        ("==", 0, 1, False),
    ],
)
def test_criterion_holds_exact_comparison(op, value, threshold, expected) -> None:
    c = Criterion(metric="x", op=op, threshold=threshold)
    assert c.holds({"x": value}) is expected


def test_criterion_holds_raises_on_missing_metric() -> None:
    c = Criterion(metric="bracket_width", op="<=", threshold=3)
    with pytest.raises(KeyError):
        c.holds({"other": 1})


# --- record construction / mandatory fields ----------------------------


def _record(**overrides) -> ResultsRecord:
    base = {
        "row": "G0",
        "null_tested": "no induction transition exists on the checkpoint grid",
        "criteria": (Criterion("found", "==", 1), Criterion("bracket_width", "<=", 3)),
        "observed": {"found": 1, "bracket_width": 2, "a_step": 512, "b_step": 2000},
        "verdict": "pass",
        "metric_version": "0.1.0",
        "git_sha": "deadbeef",
        "run_config_hash": "cfg123",
        "seed": 0,
        "checkpoint_revision": "all 154 grid steps",
        "eval_set_hash": "eval123",
        "torch_version": "2.0.0",
        "numpy_version": "2.0.0",
        "transformer_lens_version": "3.7.1",
        "wall_clock_s": 1234.5,
        "hardware": "aarch64/2cpu/1.8GB",
    }
    base.update(overrides)
    return ResultsRecord(**base)


def test_record_missing_field_raises() -> None:
    base = _record().to_dict()
    del base["seed"]
    with pytest.raises(TypeError):
        record_from_dict(base)


# --- recomputed_verdict / is_self_consistent: the silent-failure guard --


def test_verdict_recomputes_pass_when_all_criteria_hold() -> None:
    record = _record(verdict="pass")
    assert record.recomputed_verdict() == "pass"
    assert record.is_self_consistent()


def test_verdict_recomputes_fail_when_a_criterion_is_violated() -> None:
    # bracket_width=5 violates "<=3" even though "found" holds.
    record = _record(observed={"found": 1, "bracket_width": 5, "a_step": 512, "b_step": 3000})
    tampered_pass = _record(
        observed={"found": 1, "bracket_width": 5, "a_step": 512, "b_step": 3000},
        verdict="pass",
    )
    assert record.recomputed_verdict() == "fail"
    # A record whose stored verdict disagrees with recomputation must be
    # caught, not silently accepted -- this is the exact case tier 2.5
    # exists to reject.
    assert not tampered_pass.is_self_consistent()


def test_verdict_recomputes_fail_when_transition_not_found() -> None:
    record = _record(
        observed={"found": 0, "bracket_width": float("inf"), "a_step": None, "b_step": None},
        verdict="fail",
    )
    assert record.recomputed_verdict() == "fail"
    assert record.is_self_consistent()


# --- round trip: to_dict / to_json_line / record_from_dict --------------


def test_round_trip_through_json_preserves_verdict_and_criteria() -> None:
    record = _record()
    line = record.to_json_line()
    reconstructed = record_from_dict(json.loads(line))
    assert reconstructed == record
    assert reconstructed.is_self_consistent()


def test_load_records_reads_one_record_per_line(tmp_path) -> None:
    path = tmp_path / "g0.jsonl"
    r1 = _record(row="G0")
    r2 = _record(row="G0", seed=1)
    path.write_text(r1.to_json_line() + "\n" + r2.to_json_line() + "\n")
    loaded = load_records(path)
    assert loaded == [r1, r2]
