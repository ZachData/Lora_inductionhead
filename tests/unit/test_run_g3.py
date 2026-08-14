"""Tests for scripts/run_g3.py's verdict construction.

Same rationale as tests/unit/test_g0_sweep.py: run_g3.py is row
orchestration, not a metric, so it is outside the METRIC_VERSION
surface -- but the piece that turns G2's artifact into G3's *verdict*
is exactly what CLAUDE.md's "Computable verdicts" section calls the
single most important check in the repo, and G3 is a gate whose failure
stops the phase. The refusals matter as much as the happy path: a NaN
recovery silently becoming "fail" would stop the phase for the wrong
reason.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_g3

from indbw.schema import Criterion, ResultsRecord

RUN_HASH = "abc123def4567890"


def _g2_record(recovery: float) -> ResultsRecord:
    observed = {
        "reached_criterion_flag": 1.0,
        "final_recovery_canonical": recovery,
        "steps_run": 320.0,
        "wall_clock_s": 1800.0,
        "seconds_per_step": 5.625,
    }
    return ResultsRecord(
        row="G2",
        null_tested="...",
        criteria=(Criterion(metric="final_recovery_canonical", op=">=", threshold=0.80),),
        observed=observed,
        verdict="pass" if recovery >= 0.80 else "fail",
        metric_version="0.1.0",
        git_sha="f" * 40,
        run_config_hash=RUN_HASH,
        seed=0,
        checkpoint_revision="step 512 (A)",
        eval_set_hash="20ba57788b6cc830",
        torch_version="2.13.0+cpu",
        numpy_version="2.5.1",
        transformer_lens_version="3.7.1",
        wall_clock_s=1800.0,
        hardware="aarch64/aarch64",
    )


def _artifact(recovery: float, **overrides: Any) -> dict[str, Any]:
    artifact: dict[str, Any] = {
        "row": "G2",
        "config": {"rank": 64, "arm": "QK", "layer": 3, "head": 6},
        "run_config_hash": RUN_HASH,
        "reached_criterion": recovery >= 0.80,
        "steps_run": 320,
        "wall_clock_s": 1800.0,
        "final_recovery_canonical": recovery,
        "budget_exceeded": False,
    }
    artifact.update(overrides)
    return artifact


# ---------------------------------------------------------------------------
# 1. The verdict follows from the number
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recovery,expected",
    [
        (0.95, "pass"),
        (0.80, "pass"),  # the criterion is >=, so the boundary passes
        (0.7999, "fail"),
        (0.31, "fail"),
        (-0.2, "fail"),  # unclamped recovery below A is still just a fail
    ],
)
def test_verdict_matches_the_preregistered_criterion(recovery: float, expected: str) -> None:
    record = run_g3.build_g3_record(_g2_record(recovery), _artifact(recovery), "a" * 40)
    assert record.verdict == expected
    assert record.is_self_consistent()  # what tier 2.5 recomputes


def test_criterion_is_the_preregistered_threshold() -> None:
    # PROJECT.md §6's success band. If this ever reads anything but 0.80,
    # a threshold was edited -- which CLAUDE.md forbids outright.
    assert run_g3.CRITERION_R == 0.80
    record = run_g3.build_g3_record(_g2_record(0.9), _artifact(0.9), "a" * 40)
    assert record.criteria == (
        Criterion(metric="final_recovery_canonical", op=">=", threshold=0.80),
    )


def test_provenance_is_inherited_from_g2_except_the_git_sha() -> None:
    g2 = _g2_record(0.9)
    record = run_g3.build_g3_record(g2, _artifact(0.9), "a" * 40)
    assert record.row == "G3"
    assert record.git_sha == "a" * 40  # this commit, not G2's
    for field in (
        "run_config_hash",
        "seed",
        "checkpoint_revision",
        "eval_set_hash",
        "torch_version",
        "numpy_version",
        "transformer_lens_version",
        "hardware",
    ):
        assert getattr(record, field) == getattr(g2, field), field
    assert record.observed["rank"] == 64.0


# ---------------------------------------------------------------------------
# 2. The refusals -- each one prevents a wrong gate verdict
# ---------------------------------------------------------------------------


def test_non_finite_recovery_raises_rather_than_recording_a_failure() -> None:
    # The expensive false negative: `nan >= 0.80` is False, so a naive
    # path would record "the positive control failed" and stop the phase.
    nan = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        run_g3.build_g3_record(_g2_record(nan), _artifact(nan), "a" * 40)


def test_budget_exceeded_artifact_raises() -> None:
    with pytest.raises(ValueError, match="budget_exceeded"):
        run_g3.build_g3_record(_g2_record(0.9), _artifact(0.9, budget_exceeded=True), "a" * 40)


def test_record_and_artifact_disagreeing_about_recovery_raises() -> None:
    with pytest.raises(ValueError, match="disagree"):
        run_g3.build_g3_record(_g2_record(0.91), _artifact(0.42), "a" * 40)


def test_artifact_from_a_different_run_raises() -> None:
    with pytest.raises(ValueError, match="different runs"):
        run_g3.build_g3_record(
            _g2_record(0.9), _artifact(0.9, run_config_hash="0000000000000000"), "a" * 40
        )


def test_missing_recovery_field_raises() -> None:
    artifact = _artifact(0.9)
    del artifact["final_recovery_canonical"]
    with pytest.raises(ValueError, match="has no source"):
        run_g3.build_g3_record(_g2_record(0.9), artifact, "a" * 40)


# ---------------------------------------------------------------------------
# 3. Loading -- a missing input stops the row, it does not produce a verdict
# ---------------------------------------------------------------------------


def test_missing_g2_artifact_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no G2 run artifact"):
        run_g3.load_g2_artifact(tmp_path / "absent.json")


def test_missing_g2_record_exits(tmp_path: Path) -> None:
    with pytest.raises(SystemExit, match="no G2 results record"):
        run_g3.load_g2_record(tmp_path / "absent.jsonl")


def test_record_file_without_a_g2_row_exits(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    other = _g2_record(0.9)
    path.write_text(ResultsRecord(**{**other.to_dict(), "row": "G1"}).to_json_line() + "\n")
    with pytest.raises(SystemExit, match="no row=='G2' record"):
        run_g3.load_g2_record(path)


def test_load_g2_record_takes_the_most_recent(tmp_path: Path) -> None:
    path = tmp_path / "records.jsonl"
    path.write_text(_g2_record(0.10).to_json_line() + "\n" + _g2_record(0.95).to_json_line() + "\n")
    assert run_g3.load_g2_record(path).observed["final_recovery_canonical"] == 0.95
