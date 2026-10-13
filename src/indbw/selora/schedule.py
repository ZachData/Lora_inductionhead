"""Order the sweep and cut it into runs that each fit a fixed wall-clock budget.

A cell is not resumable mid-training, so a run that is stopped inside one loses it. The rule is
therefore never to start a cell that (with a safety margin) cannot finish inside the budget.
Pure functions only: no model, no network.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from indbw.selora import cells as C
from indbw.selora.preflight import steps_per_cell

BATCH_SIZE = 4  # TrainConfig defaults: one optimiser step is batch_size * grad_accum examples
GRAD_ACCUM = 4
GSM8K_TRAIN_N = 7473  # openai/gsm8k "main" train split; run_selora asserts the loaded data matches
MARGIN = 1.15  # estimates are not exact; a cell must fit with 15% to spare


def _positive(name: str, x: float) -> float:
    if not math.isfinite(x) or x <= 0:
        raise ValueError(f"{name} must be positive and finite, got {x!r}")
    return x


def train_seconds(arm: str, epochs: float, sec_per_step: float, n_train: int) -> float:
    _positive("sec_per_step", sec_per_step)
    if arm == "base":
        return 0.0
    return sec_per_step * steps_per_cell(n_train, BATCH_SIZE, GRAD_ACCUM, epochs)


def cell_seconds(
    arm: str, epochs: float, sec_per_step: float, n_train: int, eval_s: float
) -> float:
    """Training (analytic) plus everything else a cell does (SAE passes, GSM8K, MMLU): `eval_s`."""
    return train_seconds(arm, epochs, sec_per_step, n_train) + _positive("eval_s", eval_s)


def _tier(c: C.Cell) -> int:
    if c.model == C.PRIMARY_MODEL and (c.arm == "base" or (c.arm == "late" and c.seed == 0)):
        return 0  # base-vs-late: the discordance check and the cheapest kill (STATE F2, F4)
    if c.model == C.PRIMARY_MODEL and c.arm in ("all", "late"):
        return 1  # the rest of what S3a reads
    if c.model != C.PRIMARY_MODEL:
        return 2  # SAE model: S1a replication, S4
    if c.arm in ("sae_top", "sae_bottom"):
        return 3
    if c.arm.startswith("random_"):
        return 4  # S3b / S5 null draws: read only once every named arm exists
    return 5


def order_cells(cs: Sequence[C.Cell]) -> list[C.Cell]:
    """Deterministic order, independent of input order: what gates the rest comes first."""
    return sorted(cs, key=lambda c: (_tier(c), c.seed, c.arm, c.model, c.run_id))


def may_start(elapsed_s: float, cost_s: float, budget_s: float, margin: float = MARGIN) -> bool:
    return elapsed_s + margin * _positive("cost_s", cost_s) <= budget_s


def pack_runs(
    ordered: Sequence[C.Cell],
    cost_s: Callable[[C.Cell], float],
    budget_s: float,
    margin: float = MARGIN,
    first_run_reserve_s: float = 0.0,
) -> list[list[C.Cell]]:
    """Cut `ordered` (order preserved) into consecutive runs, each within `budget_s`.

    `first_run_reserve_s` is held back from run 1 only (work done before the first cell, e.g. the
    divergence stage). Raises if any single cell cannot fit in an empty run."""
    _positive("budget_s", budget_s)
    if not 0 <= first_run_reserve_s < budget_s:
        raise ValueError("first_run_reserve_s must be in [0, budget_s)")
    runs: list[list[C.Cell]] = []
    cur: list[C.Cell] = []
    cap = budget_s - first_run_reserve_s  # capacity of the run being filled
    used = 0.0
    for c in ordered:
        need = margin * _positive(f"cost of {c.arm}", cost_s(c))
        if need > (cap if not cur else budget_s) and (not cur or need > budget_s):
            raise ValueError(f"cell {c.model}/{c.arm}/{c.seed} does not fit a run")
        if used + need > cap:
            runs.append(cur)
            cur, used, cap = [], 0.0, budget_s
        cur.append(c)
        used += need
    if cur:
        runs.append(cur)
    return runs


def measured_eval_s(
    rows: Sequence[Mapping[str, Any]], sec_per_step: float, n_train: int
) -> float | None:
    """Median non-training time per cell from finished rows; None before any row exists.

    Raises on a row with no timing, or whose whole wall-clock is below the analytic training
    time (the step-time estimate is wrong; a negative eval time would plan runs that overrun)."""
    out: list[float] = []
    for r in rows:
        if "wall_clock_s" not in r:
            raise ValueError("row has no wall_clock_s: cannot measure eval time from it")
        cell = r["cell"]
        rest = float(r["wall_clock_s"]) - train_seconds(
            cell["arm"], float(cell["epochs"]), sec_per_step, n_train
        )
        if rest <= 0:
            raise ValueError(
                f"wall_clock_s {r['wall_clock_s']} is below the training estimate for "
                f"{cell['arm']}: sec_per_step={sec_per_step} is wrong"
            )
        out.append(rest)
    return statistics.median(out) if out else None
