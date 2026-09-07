"""Tests for the re-probe merge's dedup and coverage check.

Both functions fail silently and both failures produce a publishable
artifact.

`merge_shards` deduplicates by step. If it did not, a step probed twice
would appear twice in a curve that is then read as a time series, and
the duplicate would look like a flat segment rather than an error.

`coverage` is the only thing standing between a reclaimed shard and a
merged file that looks complete. Shards are cut round-robin, so losing
one removes every 7th checkpoint from an 82-point stride-4 curve -- the
gaps close up smoothly when plotted and nothing downstream would notice.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from merge_reprobe import coverage, merge_shards  # noqa: E402
from reprobe_retrain_checkpoints import bracket_steps, shard_steps  # noqa: E402


def _shard_file(tmp_path: Path, name: str, steps: list[int], tag: str = "a") -> Path:
    p = tmp_path / name
    p.write_text("".join(json.dumps({"step": s, "tag": tag}) + "\n" for s in steps))
    return p


# --- merge_shards ------------------------------------------------------


def test_merges_and_sorts_across_shards(tmp_path: Path) -> None:
    a = _shard_file(tmp_path, "reprobe_shard0.jsonl", [528, 556])
    b = _shard_file(tmp_path, "reprobe_shard1.jsonl", [532, 536])
    rows = merge_shards([a, b])
    assert [r["step"] for r in rows] == [528, 532, 536, 556]


def test_deduplicates_a_step_present_in_two_shards(tmp_path: Path) -> None:
    a = _shard_file(tmp_path, "reprobe_shard0.jsonl", [528, 532], tag="first")
    b = _shard_file(tmp_path, "reprobe_shard1.jsonl", [532, 536], tag="second")
    rows = merge_shards([a, b])
    assert [r["step"] for r in rows] == [528, 532, 536]
    assert sum(1 for r in rows if r["step"] == 532) == 1


def test_blank_lines_are_skipped(tmp_path: Path) -> None:
    p = tmp_path / "reprobe_shard0.jsonl"
    p.write_text('{"step": 528}\n\n{"step": 532}\n')
    assert len(merge_shards([p])) == 2


def test_no_files_gives_no_rows() -> None:
    assert merge_shards([]) == []


# --- coverage ----------------------------------------------------------


def test_full_bracket_reports_nothing_missing() -> None:
    rows = [{"step": s} for s in bracket_steps()]
    present, missing = coverage(rows)
    assert present == bracket_steps()
    assert missing == []


def test_a_lost_shard_is_reported_step_by_step() -> None:
    """The load-bearing case: shard 3 of 7 dies, and the survivors still
    plot as a smooth curve. coverage must name every step it took."""
    lost = set(shard_steps(bracket_steps(), 3, 7))
    rows = [{"step": s} for s in bracket_steps() if s not in lost]
    present, missing = coverage(rows)
    assert sorted(missing) == sorted(lost)
    assert len(missing) in (11, 12)
    assert set(present).isdisjoint(missing)


def test_missing_steps_are_reported_in_bracket_order() -> None:
    rows = [{"step": s} for s in bracket_steps() if s not in (556, 532)]
    _, missing = coverage(rows)
    assert missing == [532, 556]


def test_extra_steps_outside_the_bracket_do_not_mask_a_gap() -> None:
    """A row for a step that is not in the bracket -- 856, say, the
    contaminated one -- must not be counted toward coverage of a step
    that is genuinely missing."""
    rows = [{"step": s} for s in bracket_steps() if s != 620]
    rows.append({"step": 856})
    _, missing = coverage(rows)
    assert missing == [620], "a foreign step masked a real gap"


def test_empty_input_reports_the_whole_bracket_missing() -> None:
    present, missing = coverage([])
    assert present == []
    assert missing == bracket_steps()
