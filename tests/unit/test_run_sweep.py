"""Tests for scripts/run_sweep.py's guards.

The training loop itself is tested in test_train.py and the cell algebra
in test_sweep.py; what is left here is the part that decides *whether the
sweep may run at all* and *which controls it still owes*. Both are places
where a wrong answer is silent and expensive: starting the main sweep on
a failed gate burns the phase's whole compute budget on a question the
gate already answered, and mis-detecting an owed control lets a capacity
claim ship without h0_2 attached.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import run_sweep

from indbw.schema import Criterion, ResultsRecord, append_record
from indbw.sweep import DEFAULT_RANKS, CellConfig, lr_grid

GATE_FILES = {
    "G0": "g0_transition.jsonl",
    "G1": "g1_prerequisite.jsonl",
    "G2": "g2_cpu_timing.jsonl",
    "G3": "g3_positive_control.jsonl",
}


def _record(
    row: str,
    *,
    observed: dict[str, Any],
    criteria: tuple[Criterion, ...],
    verdict: str,
    run_config_hash: str = "0" * 16,
) -> ResultsRecord:
    return ResultsRecord(
        row=row,
        null_tested="...",
        criteria=criteria,
        observed=observed,
        verdict=verdict,  # type: ignore[arg-type]
        metric_version="0.1.0",
        git_sha="f" * 40,
        run_config_hash=run_config_hash,
        seed=0,
        checkpoint_revision="step 512 (A)",
        eval_set_hash="20ba57788b6cc830",
        torch_version="2.13.0+cpu",
        numpy_version="2.5.1",
        transformer_lens_version="3.7.1",
        wall_clock_s=1.0,
        hardware="aarch64/aarch64",
    )


def _gate_record(row: str, *, passed: bool = True, consistent: bool = True) -> ResultsRecord:
    observed = {"value": 1.0 if passed else 0.0}
    criteria = (Criterion(metric="value", op=">=", threshold=1.0),)
    verdict = "pass" if passed else "fail"
    if not consistent:
        verdict = "pass" if verdict == "fail" else "fail"
    return _record(row, observed=observed, criteria=criteria, verdict=verdict)


def _write_all_gates(results_dir: Path, **overrides: Any) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    for row, filename in GATE_FILES.items():
        kwargs = overrides.get(row, {})
        append_record(results_dir / filename, _gate_record(row, **kwargs))


# ---------------------------------------------------------------------------
# 1. The main sweep runs only after all gates pass (PROJECT.md §8)
# ---------------------------------------------------------------------------


def test_all_gates_passing_permits_the_sweep(tmp_path: Path) -> None:
    _write_all_gates(tmp_path)
    run_sweep.require_gates_passed(tmp_path)  # does not raise


@pytest.mark.parametrize("missing", sorted(GATE_FILES))
def test_a_missing_gate_record_blocks_the_sweep(tmp_path: Path, missing: str) -> None:
    _write_all_gates(tmp_path)
    (tmp_path / GATE_FILES[missing]).unlink()
    with pytest.raises(SystemExit, match=f"{missing} has no results record"):
        run_sweep.require_gates_passed(tmp_path)


@pytest.mark.parametrize("failed", sorted(GATE_FILES))
def test_a_failed_gate_blocks_the_sweep(tmp_path: Path, failed: str) -> None:
    # "A gate that fails stops the phase... Do not route around it."
    _write_all_gates(tmp_path, **{failed: {"passed": False}})
    with pytest.raises(SystemExit, match=f"{failed} verdict is 'fail'"):
        run_sweep.require_gates_passed(tmp_path)


def test_a_gate_whose_verdict_does_not_recompute_blocks_the_sweep(tmp_path: Path) -> None:
    # A record whose stored verdict contradicts its own numbers cannot
    # license anything downstream, even if it says "pass".
    _write_all_gates(tmp_path, G2={"passed": False, "consistent": False})
    with pytest.raises(SystemExit, match="not self-consistent"):
        run_sweep.require_gates_passed(tmp_path)


def test_a_file_with_no_matching_row_blocks_the_sweep(tmp_path: Path) -> None:
    _write_all_gates(tmp_path)
    (tmp_path / GATE_FILES["G3"]).write_text(_gate_record("G0").to_json_line() + "\n")
    with pytest.raises(SystemExit, match="no row=='G3' record"):
        run_sweep.require_gates_passed(tmp_path)


def test_the_latest_gate_record_is_the_one_that_counts(tmp_path: Path) -> None:
    # A re-run gate appends; the newest record is authoritative.
    _write_all_gates(tmp_path, G2={"passed": False})
    append_record(tmp_path / GATE_FILES["G2"], _gate_record("G2", passed=True))
    run_sweep.require_gates_passed(tmp_path)


# ---------------------------------------------------------------------------
# 2. Sharding
# ---------------------------------------------------------------------------


def _grid() -> tuple[CellConfig, ...]:
    from indbw.sweep import enumerate_cells

    return enumerate_cells(
        arms=("QK",),
        ranks=DEFAULT_RANKS,
        lrs_by_rank={r: lr_grid(3e-3) for r in DEFAULT_RANKS},
        seeds=(0, 1, 2),
        layer=3,
        head=6,
        alpha_over_r=1.0,
        max_steps=400,
        batch_size=4,
        T=128,
        d_vocab=50304,
        eval_seed=0,
        eval_n=512,
        criterion_r=0.80,
        a_step=512,
        b_step=2000,
    )


def test_shards_partition_the_grid_exactly() -> None:
    cells = _grid()
    shards = [run_sweep.shard(cells, f"{i}/7") for i in range(7)]
    flat = [c for s in shards for c in s]
    assert len(flat) == len(cells)
    assert {c.run_id for c in flat} == {c.run_id for c in cells}


def test_shards_are_disjoint() -> None:
    cells = _grid()
    seen: set[str] = set()
    for i in range(7):
        ids = {c.run_id for c in run_sweep.shard(cells, f"{i}/7")}
        assert not (ids & seen)
        seen |= ids


def test_a_shard_spans_the_rank_range() -> None:
    # Round-robin, not contiguous: a fleet that dies early leaves a
    # usable sweep rather than the low ranks only.
    ranks = {c.rank for c in run_sweep.shard(_grid(), "0/7")}
    assert len(ranks) >= 4


@pytest.mark.parametrize("spec", ["7/7", "-1/7", "0/0", "3/2"])
def test_an_invalid_shard_spec_exits(spec: str) -> None:
    with pytest.raises(SystemExit):
        run_sweep.shard(_grid(), spec)


# ---------------------------------------------------------------------------
# 3. M7: which ranks owe a 10x-steps arm
# ---------------------------------------------------------------------------


def _sweep_record(cell: CellConfig, recovery: float) -> ResultsRecord:
    from indbw.sweep import decision_band

    observed: dict[str, Any] = {
        "final_recovery": recovery,
        "band": decision_band(recovery),
        "arm": cell.arm,
        "variant": cell.variant,
        "rank": float(cell.rank),
        "lr": cell.lr,
        "train_seed": float(cell.train_seed),
    }
    return _record(
        "M1",
        observed=observed,
        criteria=(Criterion(metric="final_recovery", op=">=", threshold=0.80),),
        verdict="pass" if recovery >= 0.80 else "fail",
        run_config_hash=cell.run_id,
    )


def _record_rank(
    results_dir: Path, cells: tuple[CellConfig, ...], rank: int, recovery: float
) -> None:
    for cell in (c for c in cells if c.rank == rank):
        append_record(results_dir / "m1.jsonl", _sweep_record(cell, recovery))


def test_no_records_means_nothing_is_owed(tmp_path: Path) -> None:
    assert run_sweep.owed_control_cells("QK", _grid(), tmp_path) == ()


def test_a_wholly_failing_rank_owes_one_ten_x_steps_cell(tmp_path: Path) -> None:
    cells = _grid()
    _record_rank(tmp_path, cells, rank=2, recovery=0.2)
    owed = run_sweep.owed_control_cells("QK", cells, tmp_path)
    assert len(owed) == 1
    assert owed[0].rank == 2
    assert owed[0].variant == "ten_x_steps"
    assert owed[0].max_steps == cells[0].max_steps * 10


def test_a_succeeding_rank_owes_nothing(tmp_path: Path) -> None:
    cells = _grid()
    _record_rank(tmp_path, cells, rank=2, recovery=0.95)
    assert run_sweep.owed_control_cells("QK", cells, tmp_path) == ()


def test_an_ambiguous_rank_owes_nothing(tmp_path: Path) -> None:
    # 0.50 <= R < 0.80 is not a failure claim, so h0_2 is not owed.
    cells = _grid()
    _record_rank(tmp_path, cells, rank=2, recovery=0.65)
    assert run_sweep.owed_control_cells("QK", cells, tmp_path) == ()


def test_a_rank_with_one_success_owes_nothing(tmp_path: Path) -> None:
    cells = _grid()
    rank_cells = [c for c in cells if c.rank == 4]
    for cell in rank_cells[1:]:
        append_record(tmp_path / "m1.jsonl", _sweep_record(cell, 0.1))
    append_record(tmp_path / "m1.jsonl", _sweep_record(rank_cells[0], 0.9))
    assert run_sweep.owed_control_cells("QK", cells, tmp_path) == ()


def test_an_already_paid_control_is_not_owed_twice(tmp_path: Path) -> None:
    from indbw.sweep import ten_x_steps_cell

    cells = _grid()
    _record_rank(tmp_path, cells, rank=2, recovery=0.2)
    owed = run_sweep.owed_control_cells("QK", cells, tmp_path)
    append_record(tmp_path / "m1.jsonl", _sweep_record(ten_x_steps_cell(owed[0]), 0.2))
    assert run_sweep.owed_control_cells("QK", cells, tmp_path) == ()


def test_each_failing_rank_owes_its_own_control(tmp_path: Path) -> None:
    cells = _grid()
    for rank in (1, 2, 4):
        _record_rank(tmp_path, cells, rank=rank, recovery=0.1)
    owed = run_sweep.owed_control_cells("QK", cells, tmp_path)
    assert sorted(c.rank for c in owed) == [1, 2, 4]


def test_the_other_arm_s_records_do_not_create_a_debt(tmp_path: Path) -> None:
    cells = _grid()  # QK cells
    _record_rank(tmp_path, cells, rank=2, recovery=0.2)
    assert run_sweep.owed_control_cells("OV", cells, tmp_path) == ()
