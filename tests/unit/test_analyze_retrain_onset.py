"""Tests for the retrain-onset analysis' two pure functions.

Most of `analyze_retrain_onset.py` is plotting and printing, which
CLAUDE.md's TDD table says not to test. Two pieces are not:

`first_crossing` defines where the onset *is*. Every number the analysis
reports about timing goes through it, and an off-by-one returns a step
that exists, sits in the right range, and is wrong -- the silent-failure
shape the contract's kind 3 is about.

`load_probe_series` decides which of two trajectories the onset window is
read from. The run was resumed twice and probe.jsonl replays steps
852-868 and 1528-1540 from a *different* data stream; keeping the wrong
copy silently mixes trajectories inside the analysis window.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_retrain_onset import first_crossing, load_probe_series  # noqa: E402


# --- first_crossing ----------------------------------------------------


def test_returns_the_first_step_at_or_above_the_threshold() -> None:
    steps = [4, 8, 12, 16, 20]
    values = [0.0, 0.1, 0.3, 0.2, 0.9]
    assert first_crossing(steps, values, 0.3) == 12


def test_boundary_is_inclusive() -> None:
    """`>=`, matching gates.locate_transition and the retrain's own
    onset_pms trigger. Exclusive would shift every reported onset by one
    probe interval."""
    assert first_crossing([4, 8], [0.05, 0.9], 0.05) == 4


def test_returns_none_when_never_reached() -> None:
    """None, not the last step and not 0 -- both would read as a crossing
    that happened."""
    assert first_crossing([4, 8, 12], [0.0, 0.01, 0.02], 0.3) is None


def test_a_later_dip_below_threshold_does_not_move_the_crossing() -> None:
    """PMS is a finite-eval estimate and wobbles. The crossing is the
    first time it is reached, not the last."""
    assert first_crossing([4, 8, 12, 16], [0.0, 0.5, 0.1, 0.6], 0.3) == 8


def test_rejects_mismatched_lengths() -> None:
    with pytest.raises(ValueError):
        first_crossing([1, 2, 3], [0.1, 0.2], 0.1)


def test_rejects_empty_series() -> None:
    with pytest.raises(ValueError):
        first_crossing([], [], 0.1)


# --- load_probe_series -------------------------------------------------


def _write(tmp_path: Path, rows: list[dict[str, float]]) -> Path:
    p = tmp_path / "probe.jsonl"
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return p


def test_first_occurrence_wins_on_a_replayed_step(tmp_path: Path) -> None:
    """The real file replays steps after each resume with *different*
    values (different data stream). Session 1 is the clean pass over the
    onset bracket, so the first copy is the one kept.
    """
    path = _write(
        tmp_path,
        [
            {"step": 852, "pms": 0.11},  # session 1
            {"step": 856, "pms": 0.12},
            {"step": 852, "pms": 0.99},  # session 2 replay -- must lose
            {"step": 856, "pms": 0.98},
        ],
    )
    rows = load_probe_series(path)
    assert [r["step"] for r in rows] == [852, 856]
    assert rows[0]["pms"] == 0.11
    assert rows[1]["pms"] == 0.12


def test_output_is_sorted_by_step_even_when_the_file_is_not(tmp_path: Path) -> None:
    path = _write(tmp_path, [{"step": 900, "pms": 0.3}, {"step": 100, "pms": 0.1}])
    assert [r["step"] for r in load_probe_series(path)] == [100, 900]


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "probe.jsonl"
    p.write_text('{"step": 4, "pms": 0.1}\n\n{"step": 8, "pms": 0.2}\n')
    assert len(load_probe_series(p)) == 2


def test_raises_on_an_empty_file(tmp_path: Path) -> None:
    """An empty trace must not summarize to a clean-looking zero-row
    result -- the caller would report crossings of None and read it as
    'induction never formed'."""
    p = tmp_path / "probe.jsonl"
    p.write_text("")
    with pytest.raises(ValueError):
        load_probe_series(p)


# --- the committed trace itself ----------------------------------------


def test_the_real_trace_dedupes_to_a_strictly_increasing_grid() -> None:
    """Against the committed file: 381 raw rows contain two backward
    transitions (868->852, 1540->1528) from the two resumes, and must
    come back as a strictly increasing stride-4 grid."""
    path = REPO_ROOT / "data" / "retrain" / "cb25e3f6c2185c1e" / "probe.jsonl"
    if not path.exists():  # pragma: no cover - the trace is committed
        pytest.skip("retrain trace not present")
    rows = load_probe_series(path)
    steps = [r["step"] for r in rows]
    assert steps == sorted(set(steps))
    assert len(steps) < 381, "dedupe removed nothing; the replays are still in"
    assert steps[0] == 516
    assert steps[-1] == 2000
    # steps[:-1] against steps[1:] -- both length n-1, so strict=True still
    # guards a length bug. zip(steps, steps[1:], strict=True) raises by
    # construction, which is how this test failed on its first CI run.
    assert all(b - a == 4 for a, b in zip(steps[:-1], steps[1:], strict=True))


def test_the_real_trace_puts_onset_where_onset_bracket_json_says() -> None:
    """onset_bracket.json records onset_step 620 at pms >= 0.05, derived
    independently (by hand, on 2026-09-01) from the same file. This
    asserts the analysis reproduces that number rather than inventing a
    second, quietly different onset."""
    run = REPO_ROOT / "data" / "retrain" / "cb25e3f6c2185c1e"
    if not (run / "probe.jsonl").exists():  # pragma: no cover
        pytest.skip("retrain trace not present")
    rows = load_probe_series(run / "probe.jsonl")
    got = first_crossing([r["step"] for r in rows], [r["pms"] for r in rows], 0.05)
    recorded = json.loads((run / "onset_bracket.json").read_text())["onset_step"]
    assert got == recorded == 620
