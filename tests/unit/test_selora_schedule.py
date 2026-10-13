"""Spec for selora.schedule: ordering the sweep and cutting it into runs of bounded length."""

from __future__ import annotations

import math

import numpy as np
import pytest

from indbw.selora import cells, preflight, schedule

SCORES = {m: list(np.linspace(0, 1, 26)) for m in cells.MODELS}
HOUR = 3600.0
SEC = 2.68  # s per optimiser step (preflight, data/selora/preflight.json)
N_TRAIN = 7473  # GSM8K train split


def _cs() -> list[cells.Cell]:
    return cells.enumerate_cells(SCORES)


def _cost(eval_s: float):  # type: ignore[no-untyped-def]
    return lambda c: schedule.cell_seconds(c.arm, c.epochs, SEC, N_TRAIN, eval_s)


# --- cost model -------------------------------------------------------------------------------


def test_train_steps_oracle() -> None:
    # ceil(7473 / (4*4)) = 468 steps per epoch; 3 epochs
    assert schedule.train_seconds("late", 3.0, SEC, N_TRAIN) == pytest.approx(1404 * SEC)
    assert 1404 == preflight.steps_per_cell(N_TRAIN, 4, 4, 3.0)


def test_base_cell_trains_nothing() -> None:
    assert schedule.train_seconds("base", 3.0, SEC, N_TRAIN) == 0.0
    assert schedule.cell_seconds("base", 3.0, SEC, N_TRAIN, 600.0) == 600.0


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_cost_inputs_must_be_positive_and_finite(bad: float) -> None:
    with pytest.raises(ValueError):
        schedule.train_seconds("late", 3.0, bad, N_TRAIN)
    with pytest.raises(ValueError):
        schedule.cell_seconds("late", 3.0, SEC, N_TRAIN, bad)


# --- ordering ----------------------------------------------------------------------------------


def test_order_is_a_deterministic_permutation() -> None:
    cs = _cs()
    out = schedule.order_cells(cs)
    assert sorted(c.run_id for c in out) == sorted(c.run_id for c in cs)
    assert [c.run_id for c in schedule.order_cells(list(reversed(cs)))] == [c.run_id for c in out]


def test_discordance_pair_runs_first() -> None:
    # STATE F2/F4: base-vs-late on the primary model is the first measurement, so it comes first
    out = schedule.order_cells(_cs())
    assert [(c.model, c.arm, c.seed) for c in out[:2]] == [
        (cells.PRIMARY_MODEL, "base", 0),
        (cells.PRIMARY_MODEL, "late", 0),
    ]


def test_random_draws_come_after_every_named_arm() -> None:
    out = schedule.order_cells(_cs())
    first_random = min(i for i, c in enumerate(out) if c.arm.startswith("random_"))
    assert all(c.arm.startswith("random_") for c in out[first_random:])
    assert first_random == 35 - 24


def test_s3a_cells_precede_sae_guided_arms() -> None:
    # S3a gates S3b, so the cells S3a reads (all/late x 3 seeds, primary) are scheduled first
    out = schedule.order_cells(_cs())
    s3a = {
        i
        for i, c in enumerate(out)
        if c.model == cells.PRIMARY_MODEL and c.arm in ("all", "late", "base")
    }
    guided = {i for i, c in enumerate(out) if c.arm in ("sae_top", "sae_bottom")}
    assert max(s3a) < min(guided)


# --- packing -----------------------------------------------------------------------------------


@pytest.mark.parametrize("eval_s", [900.0, 1800.0, 2700.0])
def test_every_cell_exactly_once_in_order(eval_s: float) -> None:
    ordered = schedule.order_cells(_cs())
    runs = schedule.pack_runs(ordered, _cost(eval_s), 8 * HOUR)
    assert [c.run_id for r in runs for c in r] == [c.run_id for c in ordered]
    assert all(r for r in runs)


@pytest.mark.parametrize("eval_s", [900.0, 1800.0, 2700.0])
def test_no_run_exceeds_budget_with_margin(eval_s: float) -> None:
    cost = _cost(eval_s)
    for r in schedule.pack_runs(schedule.order_cells(_cs()), cost, 8 * HOUR, margin=1.15):
        assert 1.15 * sum(cost(c) for c in r) <= 8 * HOUR


def test_uniform_cost_run_count_oracle() -> None:
    # n cells of cost c, budget B, margin m: floor(B/(m c)) per run, ceil(n/that) runs
    cs = [cells.Cell("m", f"a{i}", (), i) for i in range(35)]
    c, b, m = 1.2 * HOUR, 8 * HOUR, 1.15
    per_run = math.floor(b / (m * c))
    runs = schedule.pack_runs(cs, lambda _: c, b, margin=m)
    assert per_run == 5
    assert [len(r) for r in runs] == [per_run] * 7
    assert len(runs) == math.ceil(35 / per_run)


def test_longer_eval_never_means_fewer_runs() -> None:
    # discrimination: the packer must respond to the cost it is given
    ordered = schedule.order_cells(_cs())
    n = [len(schedule.pack_runs(ordered, _cost(e), 8 * HOUR)) for e in (900.0, 1800.0, 2700.0)]
    assert n == sorted(n) and n[0] < n[-1]


def test_cell_longer_than_a_run_raises_not_truncates() -> None:
    with pytest.raises(ValueError, match="does not fit"):
        schedule.pack_runs([cells.Cell("m", "a", (), 0)], lambda _: 9 * HOUR, 8 * HOUR)


def test_first_run_reserve_shrinks_only_the_first_run() -> None:
    cs = [cells.Cell("m", f"a{i}", (), i) for i in range(12)]
    c = 1.0 * HOUR
    plain = schedule.pack_runs(cs, lambda _: c, 8 * HOUR, margin=1.0)
    held = schedule.pack_runs(cs, lambda _: c, 8 * HOUR, margin=1.0, first_run_reserve_s=2 * HOUR)
    assert [len(r) for r in plain] == [8, 4]
    assert [len(r) for r in held] == [6, 6]


@pytest.mark.parametrize("bad", [0.0, -5.0, float("nan")])
def test_nonpositive_cost_raises(bad: float) -> None:
    with pytest.raises(ValueError):
        schedule.pack_runs([cells.Cell("m", "a", (), 0)], lambda _: bad, 8 * HOUR)


def test_empty_input_is_no_runs() -> None:
    assert schedule.pack_runs([], lambda _: HOUR, 8 * HOUR) == []


# --- the in-run stop rule ----------------------------------------------------------------------


def test_may_start_boundary() -> None:
    # elapsed 4h of 8h, margin 1.15: a 3.5h cell needs 4.025h -> refuse; a 3.4h cell needs 3.91h
    assert schedule.may_start(4 * HOUR, 3.4 * HOUR, 8 * HOUR, 1.15)
    assert not schedule.may_start(4 * HOUR, 3.5 * HOUR, 8 * HOUR, 1.15)


def test_may_start_never_true_past_the_budget() -> None:
    assert not schedule.may_start(8 * HOUR, 1.0, 8 * HOUR, 1.0)
    assert not schedule.may_start(9 * HOUR, 1.0, 8 * HOUR, 1.0)


# --- measuring the unmeasured part -------------------------------------------------------------


def _row(arm: str, wall: float) -> dict[str, object]:
    return {"cell": {"arm": arm, "epochs": 3.0}, "wall_clock_s": wall}


def test_measured_eval_recovers_planted_eval_time() -> None:
    train = schedule.train_seconds("late", 3.0, SEC, N_TRAIN)
    rows = [_row("base", 600.0), _row("late", train + 600.0), _row("all", train + 600.0)]
    assert schedule.measured_eval_s(rows, SEC, N_TRAIN) == pytest.approx(600.0)


def test_measured_eval_is_none_without_rows() -> None:
    assert schedule.measured_eval_s([], SEC, N_TRAIN) is None


def test_measured_eval_refuses_rows_without_timing() -> None:
    with pytest.raises(ValueError, match="wall_clock_s"):
        schedule.measured_eval_s([{"cell": {"arm": "base", "epochs": 3.0}}], SEC, N_TRAIN)


def test_measured_eval_refuses_wall_below_training_estimate() -> None:
    # training alone would take longer than the whole cell did: sec_per_step is wrong, say so
    with pytest.raises(ValueError, match="sec_per_step"):
        schedule.measured_eval_s([_row("late", 100.0)], SEC, N_TRAIN)
