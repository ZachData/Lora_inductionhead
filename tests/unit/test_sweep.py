"""Tests for indbw.sweep -- config-as-data, resumability, control gate.

Three distinct silent failures live here, and each gets its own section:

1. A run ID that does not actually track the config. Resumability is
   built on it, so a hash that ignores a field silently skips a cell
   that was never run -- and one that is unstable re-runs the whole
   sweep after a restart.
2. A decision band that rounds. PROJECT.md §6's ambiguous zone exists
   precisely so 0.6 is not called a success; the boundaries are tested
   exactly.
3. A control gate that passes a failure claim through. That is the
   rule CLAUDE.md calls out most emphatically, so the gate is tested
   against claims it must *reject*, not only ones it should allow.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from indbw.schema import Criterion, ResultsRecord, append_record
from indbw.sweep import (
    DEFAULT_RANKS,
    DEFAULT_SEEDS,
    FAILURE_R,
    MIN_LR_POINTS,
    SUCCESS_R,
    CellConfig,
    check_failure_claims,
    completed_run_ids,
    decision_band,
    enumerate_cells,
    lr_grid,
    pending_cells,
    ten_x_steps_cell,
)

BASE_KWARGS: dict[str, Any] = {
    "arm": "QK",
    "variant": "main",
    "layer": 3,
    "head": 6,
    "rank": 4,
    "alpha": 4.0,
    "lr": 3e-3,
    "train_seed": 0,
    "max_steps": 400,
    "batch_size": 4,
    "T": 128,
    "d_vocab": 50304,
    "eval_seed": 0,
    "eval_n": 512,
    "criterion_r": 0.80,
    "a_step": 512,
    "b_step": 2000,
}


def cell(**overrides: Any) -> CellConfig:
    return CellConfig(**{**BASE_KWARGS, **overrides})


# ---------------------------------------------------------------------------
# 1. Config as data: the hash *is* the run ID
# ---------------------------------------------------------------------------


def test_run_id_is_stable_across_identical_configs() -> None:
    # Resumability depends on this: an unstable hash re-runs a finished
    # sweep from scratch after every restart.
    assert cell().run_id == cell().run_id


def test_run_id_is_a_16_hex_digest() -> None:
    run_id = cell().run_id
    assert len(run_id) == 16
    int(run_id, 16)  # raises if not hex


@pytest.mark.parametrize("field", sorted(BASE_KWARGS))
def test_run_id_changes_when_any_field_changes(field: str) -> None:
    # The discrimination test that matters. A hash that ignored even one
    # field would silently treat two different runs as the same cell and
    # skip the second on resume.
    changes: dict[str, Any] = {
        "arm": "OV",
        "variant": "ten_x_steps",
        "layer": 4,
        "head": 7,
        "rank": 8,
        "alpha": 8.0,
        "lr": 1e-3,
        "train_seed": 1,
        "max_steps": 401,
        "batch_size": 8,
        "T": 64,
        "d_vocab": 50000,
        "eval_seed": 1,
        "eval_n": 256,
        "criterion_r": 0.5,
        "a_step": 256,
        "b_step": 4000,
    }
    assert cell().run_id != cell(**{field: changes[field]}).run_id


def test_run_id_is_independent_of_field_construction_order() -> None:
    reordered = {k: BASE_KWARGS[k] for k in reversed(list(BASE_KWARGS))}
    assert CellConfig(**reordered).run_id == cell().run_id


def test_config_round_trips_through_json() -> None:
    restored = CellConfig(**json.loads(json.dumps(cell().to_dict())))
    assert restored == cell()
    assert restored.run_id == cell().run_id


def test_config_is_frozen() -> None:
    from dataclasses import FrozenInstanceError

    with pytest.raises(FrozenInstanceError):
        cell().rank = 8  # type: ignore[misc]


@pytest.mark.parametrize(
    "override",
    [
        {"arm": "BOTH"},
        {"variant": "whatever"},
        {"rank": 0},
        {"alpha": 0.0},
        {"lr": 0.0},
        {"lr": -1e-3},
        {"lr": float("nan")},
        {"max_steps": 0},
        {"batch_size": 0},
        {"T": 1},
        {"d_vocab": 1},
        {"eval_n": 0},
        {"criterion_r": 0.0},
        {"criterion_r": 1.5},
        {"b_step": 512},  # not greater than a_step
        {"a_step": 3000},  # now exceeds b_step
    ],
)
def test_invalid_config_raises_at_construction(override: dict[str, Any]) -> None:
    with pytest.raises(ValueError):
        cell(**override)


def test_label_carries_the_run_id() -> None:
    c = cell()
    assert c.run_id in c.label
    assert "QK" in c.label and "r4" in c.label


# ---------------------------------------------------------------------------
# 2. The grid (PROJECT.md §8) and the alpha/r confound
# ---------------------------------------------------------------------------


def _grid(**overrides: Any) -> tuple[CellConfig, ...]:
    kwargs: dict[str, Any] = {
        "arms": ("QK", "OV"),
        "ranks": DEFAULT_RANKS,
        "lrs_by_rank": {r: lr_grid(3e-3) for r in DEFAULT_RANKS},
        "seeds": DEFAULT_SEEDS,
        "layer": 3,
        "head": 6,
        "alpha_over_r": 1.0,
        "max_steps": 400,
        "batch_size": 4,
        "T": 128,
        "d_vocab": 50304,
        "eval_seed": 0,
        "eval_n": 512,
        "criterion_r": 0.80,
        "a_step": 512,
        "b_step": 2000,
    }
    kwargs.update(overrides)
    return enumerate_cells(**kwargs)


def test_grid_size_matches_project_md_section_8() -> None:
    # 2 arms x 6 ranks x 3 lrs x 3 seeds = 108, the count run_g2.py's
    # CPU-vs-GPU projection is built on.
    assert len(_grid()) == 108


def test_every_cell_in_the_grid_has_a_distinct_run_id() -> None:
    cells = _grid()
    assert len({c.run_id for c in cells}) == len(cells)


def test_alpha_over_r_is_held_fixed_across_ranks() -> None:
    # PROJECT.md §8's confound guard: alpha/r constant means rank does
    # not silently rescale the effective lr on Delta W.
    for c in _grid(alpha_over_r=2.0):
        assert c.alpha / c.rank == pytest.approx(2.0, rel=1e-12)


def test_lr_is_swept_independently_at_every_rank() -> None:
    cells = _grid()
    for rank in DEFAULT_RANKS:
        lrs = {c.lr for c in cells if c.rank == rank and c.arm == "QK"}
        assert len(lrs) >= MIN_LR_POINTS, f"rank {rank} swept only {lrs}"


def test_grid_rejects_a_rank_with_no_lr_list() -> None:
    incomplete = {r: lr_grid(3e-3) for r in DEFAULT_RANKS if r != 8}
    with pytest.raises(ValueError, match="no entry for rank"):
        _grid(lrs_by_rank=incomplete)


def test_grid_rejects_too_few_lr_points() -> None:
    with pytest.raises(ValueError, match=f">= {MIN_LR_POINTS}"):
        _grid(lrs_by_rank={r: (3e-3, 1e-3) for r in DEFAULT_RANKS})


def test_grid_rejects_duplicate_lr_points() -> None:
    # Three "points" that are the same number is not a sweep.
    with pytest.raises(ValueError, match="duplicates"):
        _grid(lrs_by_rank={r: (3e-3, 3e-3, 3e-3) for r in DEFAULT_RANKS})


@pytest.mark.parametrize("empty", ["arms", "ranks", "seeds"])
def test_grid_rejects_an_empty_axis(empty: str) -> None:
    with pytest.raises(ValueError):
        _grid(**{empty: ()})


def test_lr_grid_is_geometric_and_centred() -> None:
    assert lr_grid(3e-3, 3, spread=3.0) == pytest.approx((1e-3, 3e-3, 9e-3), rel=1e-12)
    five = lr_grid(1.0, 5, spread=10.0)
    assert five == pytest.approx((1e-2, 1e-1, 1.0, 10.0, 100.0), rel=1e-12)


@pytest.mark.parametrize("args", [(3e-3, 2), (3e-3, 4), (0.0, 3), (-1.0, 3), (3e-3, 1)])
def test_lr_grid_rejects_a_degenerate_ladder(args: tuple[float, int]) -> None:
    with pytest.raises(ValueError):
        lr_grid(*args)


def test_lr_grid_rejects_a_spread_that_collapses_the_ladder() -> None:
    with pytest.raises(ValueError, match="spread"):
        lr_grid(3e-3, 3, spread=1.0)


def test_ten_x_steps_cell_multiplies_only_the_budget() -> None:
    base = cell()
    control = ten_x_steps_cell(base)
    assert control.max_steps == base.max_steps * 10
    assert control.variant == "ten_x_steps"
    assert control.run_id != base.run_id  # a distinct cell, separately recorded
    # Everything else is identical: h0_2 isolates the step budget alone.
    ignored = {"max_steps", "variant"}
    for field, value in base.to_dict().items():
        if field not in ignored:
            assert control.to_dict()[field] == value, field


# ---------------------------------------------------------------------------
# 3. Decision bands (PROJECT.md §6) -- ambiguous stays ambiguous
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "recovery,expected",
    [
        (1.2, "success"),
        (1.0, "success"),
        (0.80, "success"),  # >= 0.80
        (0.7999999, "ambiguous"),
        (0.65, "ambiguous"),
        (0.50, "ambiguous"),  # failure is R < 0.50, so 0.50 is not a failure
        (0.4999999, "failure"),
        (0.0, "failure"),
        (-0.3, "failure"),
    ],
)
def test_decision_band_boundaries_are_exact(recovery: float, expected: str) -> None:
    assert decision_band(recovery) == expected


def test_the_ambiguous_zone_is_never_rounded_into_a_verdict() -> None:
    # CLAUDE.md: "0.50 <= R < 0.80 is reported as ambiguous, never
    # rounded into a verdict." 0.79 must not become a success.
    assert decision_band(0.79) == "ambiguous"
    assert decision_band(0.51) == "ambiguous"


def test_thresholds_are_the_preregistered_ones() -> None:
    assert (SUCCESS_R, FAILURE_R) == (0.80, 0.50)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), float("-inf")])
def test_decision_band_raises_on_a_non_finite_recovery(bad: float) -> None:
    # `nan < 0.50` is False and `nan >= 0.80` is False, so a broken run
    # would otherwise be silently classified "ambiguous".
    with pytest.raises(ValueError, match="non-finite"):
        decision_band(bad)


# ---------------------------------------------------------------------------
# 4. Resumability
# ---------------------------------------------------------------------------


def _record(c: CellConfig, recovery: float) -> ResultsRecord:
    observed: dict[str, Any] = {
        "final_recovery": recovery,
        "arm": c.arm,
        "variant": c.variant,
        "rank": float(c.rank),
        "lr": c.lr,
        "train_seed": float(c.train_seed),
    }
    return ResultsRecord(
        row="M1",
        null_tested="R is flat in r (h0_1)",
        criteria=(Criterion(metric="final_recovery", op=">=", threshold=SUCCESS_R),),
        observed=observed,
        verdict="pass" if recovery >= SUCCESS_R else "fail",
        metric_version="0.1.0",
        git_sha="f" * 40,
        run_config_hash=c.run_id,
        seed=c.train_seed,
        checkpoint_revision="step 512 (A)",
        eval_set_hash="20ba57788b6cc830",
        torch_version="2.13.0+cpu",
        numpy_version="2.5.1",
        transformer_lens_version="3.7.1",
        wall_clock_s=12.0,
        hardware="aarch64/aarch64",
    )


def test_completed_run_ids_on_a_missing_directory_is_empty(tmp_path: Path) -> None:
    # First launch of a sweep is not an error.
    assert completed_run_ids(tmp_path / "absent") == set()


def test_completed_run_ids_reads_every_jsonl_in_the_directory(tmp_path: Path) -> None:
    a, b = cell(rank=1), cell(rank=2)
    append_record(tmp_path / "m1.jsonl", _record(a, 0.9))
    append_record(tmp_path / "m2.jsonl", _record(b, 0.3))
    assert completed_run_ids(tmp_path) == {a.run_id, b.run_id}


def test_pending_cells_skips_exactly_what_was_recorded(tmp_path: Path) -> None:
    cells = _grid()[:10]
    for c in cells[:4]:
        append_record(tmp_path / "m1.jsonl", _record(c, 0.9))
    pending = pending_cells(cells, completed_run_ids(tmp_path))
    assert pending == cells[4:]


def test_resuming_twice_is_idempotent(tmp_path: Path) -> None:
    # The actual failure mode: an instance stopped mid-sweep, restarted,
    # must neither redo finished cells nor skip unfinished ones.
    cells = _grid()[:6]
    append_record(tmp_path / "m1.jsonl", _record(cells[0], 0.9))
    first = pending_cells(cells, completed_run_ids(tmp_path))
    append_record(tmp_path / "m1.jsonl", _record(first[0], 0.4))
    second = pending_cells(cells, completed_run_ids(tmp_path))
    assert len(second) == len(first) - 1
    assert set(second) < set(first)


def test_pending_is_empty_once_every_cell_is_recorded(tmp_path: Path) -> None:
    cells = _grid()[:5]
    for c in cells:
        append_record(tmp_path / "m1.jsonl", _record(c, 0.9))
    assert pending_cells(cells, completed_run_ids(tmp_path)) == ()


# ---------------------------------------------------------------------------
# 5. The mandatory-control gate
# ---------------------------------------------------------------------------


def _failing_rank_records(
    *, n_lrs: int = MIN_LR_POINTS, with_ten_x: bool = True, recovery: float = 0.2
) -> list[ResultsRecord]:
    base = cell(rank=2)
    lrs = lr_grid(3e-3, 5)[:n_lrs]
    records = [_record(replace(base, lr=lr), recovery) for lr in lrs]
    if with_ten_x:
        records.append(_record(ten_x_steps_cell(base), recovery))
    return records


def test_a_fully_controlled_failure_claim_passes() -> None:
    assert check_failure_claims(_failing_rank_records()) == []


def test_a_failure_claim_without_the_ten_x_steps_arm_is_rejected() -> None:
    problems = check_failure_claims(_failing_rank_records(with_ten_x=False))
    assert len(problems) == 1
    assert "10x-steps arm" in problems[0] and "h0_2" in problems[0]


def test_a_failure_claim_on_too_few_lr_points_is_rejected() -> None:
    problems = check_failure_claims(_failing_rank_records(n_lrs=1))
    assert any("lr point" in p for p in problems)


def test_a_failure_claim_missing_both_controls_reports_both() -> None:
    problems = check_failure_claims(_failing_rank_records(n_lrs=2, with_ten_x=False))
    assert len(problems) == 2


def test_a_rank_that_succeeded_needs_no_controls() -> None:
    # Controls are owed by *failure* claims only; a successful rank with
    # a single lr point is not a violation.
    records = [_record(cell(rank=8), 0.95)]
    assert check_failure_claims(records) == []


def test_a_rank_with_one_success_among_failures_is_not_a_failure_claim() -> None:
    # If any lr reached criterion, the rank did not fail, so nothing is
    # owed -- and nothing may be reported as a capacity limit either.
    base = cell(rank=2)
    records = [
        _record(replace(base, lr=1e-3), 0.2),
        _record(replace(base, lr=3e-3), 0.9),
        _record(replace(base, lr=9e-3), 0.1),
    ]
    assert check_failure_claims(records) == []


def test_an_ambiguous_rank_is_not_a_failure_claim() -> None:
    base = cell(rank=2)
    records = [_record(replace(base, lr=lr), 0.65) for lr in lr_grid(3e-3)]
    assert check_failure_claims(records) == []


def test_arms_are_gated_independently() -> None:
    # A controlled failure on QK must not launder an uncontrolled one on OV.
    records = _failing_rank_records()
    ov = cell(arm="OV", rank=2)
    records += [_record(replace(ov, lr=lr), 0.2) for lr in lr_grid(3e-3)]
    problems = check_failure_claims(records)
    assert len(problems) == 1
    assert problems[0].startswith("OV rank 2")


def test_ranks_are_gated_independently() -> None:
    records = _failing_rank_records()
    r16 = cell(rank=16)
    records += [_record(replace(r16, lr=lr), 0.1) for lr in lr_grid(3e-3)]
    problems = check_failure_claims(records)
    assert len(problems) == 1
    assert "rank 16" in problems[0]


def test_non_sweep_records_are_ignored() -> None:
    # G0/G1/G2 records carry no arm/rank in `observed`; the gate must
    # not trip over them.
    g0 = ResultsRecord(
        row="G0",
        null_tested="...",
        criteria=(Criterion(metric="found", op="==", threshold=1),),
        observed={"found": 1, "bracket_width": 2},
        verdict="pass",
        metric_version="0.1.0",
        git_sha="f" * 40,
        run_config_hash="072e5df3ce3ddb31",
        seed=0,
        checkpoint_revision="all 154 grid steps",
        eval_set_hash="20ba57788b6cc830",
        torch_version="2.13.0+cpu",
        numpy_version="2.5.1",
        transformer_lens_version="3.7.1",
        wall_clock_s=34675.7,
        hardware="aarch64/aarch64",
    )
    assert check_failure_claims([g0]) == []
    assert check_failure_claims([g0, *_failing_rank_records()]) == []
