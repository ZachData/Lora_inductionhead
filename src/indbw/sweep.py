"""Sweep cells: config-as-data, resumability, and the mandatory-control gate.

PROJECT.md §8's main sweep -- "Per arm: rank r in {1,2,4,8,16,32} x lr
(>=3 points, swept independently per rank) x 3 seeds" -- is 108 cells
before M3/M5-M8. This module is the part of that which is pure: what a
cell *is*, which cells still need running, and whether a failure claim
is allowed to be made yet. Running one belongs to scripts/run_sweep.py,
same split as gates.py vs. scripts/g0_sweep.py.

Three CLAUDE.md requirements are implemented here, none of which existed
anywhere in the repo before (close_g0.py, run_g1.py and run_g2.py each
inline their own ad-hoc config dict and sha256 call):

**Config as data.** "One frozen dataclass per run, serialized to JSON,
hashed. The hash *is* the run ID." `CellConfig.run_id` is that hash. It
goes into the results record's `run_config_hash` field, so a record
points back at the exact config that produced it and nothing else.

**Resumability.** "An idle-CPU alarm or the 4h hard cap can stop the
instance mid-sweep... On start, skip cells whose config hash already has
a record." `pending_cells` is that skip, driven by `completed_run_ids`
reading whatever is already in results/.

**The mandatory-control gate.** CLAUDE.md: "No failure claim without its
controls attached. A rank cell that failed must report the matched-lr
arm and the 10x-steps arm alongside it, or the result is about the
optimizer and cannot be reported as capacity." `check_failure_claims`
makes that a computation rather than a rule someone has to remember --
see its docstring for how each of the two controls is read.

Nothing here is a metric (no function reads a model), so this module is
outside the METRIC_VERSION surface, matching gates.py's precedent. The
§6 decision bands live here rather than in probes.py for the same
reason: they are pre-registered *decision thresholds* applied to a
metric, not a definition of one.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Literal

from indbw.schema import ResultsRecord, load_records

Arm = Literal["QK", "OV"]
ARMS: tuple[Arm, ...] = ("QK", "OV")

#: Which arm of the sweep a cell belongs to. "main" is the grid itself;
#: the other two are PROJECT.md §8/§9's mandatory controls, which exist
#: only to be attached to a failing main cell.
Variant = Literal["main", "ten_x_steps", "matched_norm_null"]
VARIANTS: tuple[Variant, ...] = ("main", "ten_x_steps", "matched_norm_null")

#: PROJECT.md §6, fixed before data. Never edit these here -- CLAUDE.md
#: forbids changing a pre-registered threshold outright; if one looks
#: wrong, say so and append to PROJECT.md §10.
SUCCESS_R = 0.80
FAILURE_R = 0.50

Band = Literal["success", "failure", "ambiguous"]

#: PROJECT.md §8's rank ladder and seed count.
DEFAULT_RANKS: tuple[int, ...] = (1, 2, 4, 8, 16, 32)
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
MIN_LR_POINTS = 3  # §8: "lr (>=3 points, swept independently per rank)"


@dataclass(frozen=True)
class CellConfig:
    """One sweep cell. Frozen, JSON-serializable, and hashed to its run ID.

    Every field is required: a config with a defaulted field is exactly
    the "I changed a default and no longer know which runs used it"
    failure CLAUDE.md's config-as-data section exists to kill.

    `alpha` is stored explicitly rather than derived at use time so the
    hash covers the value actually trained with; `enumerate_cells` is
    what sets it from a fixed alpha/r, and its docstring explains why.
    """

    arm: Arm
    variant: Variant
    layer: int
    head: int
    rank: int
    alpha: float
    lr: float
    train_seed: int
    max_steps: int
    batch_size: int
    T: int
    d_vocab: int
    eval_seed: int
    eval_n: int
    criterion_r: float
    a_step: int
    b_step: int

    def __post_init__(self) -> None:
        if self.arm not in ARMS:
            raise ValueError(f"unknown arm {self.arm!r}, expected one of {ARMS}")
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}, expected one of {VARIANTS}")
        if self.rank < 1:
            raise ValueError(f"rank must be >= 1, got {self.rank}")
        if self.alpha <= 0.0:
            raise ValueError(f"alpha must be > 0, got {self.alpha}")
        if not (self.lr > 0.0):  # also rejects nan
            raise ValueError(f"lr must be > 0, got {self.lr}")
        if self.max_steps < 1:
            raise ValueError(f"max_steps must be >= 1, got {self.max_steps}")
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}")
        if self.T < 2:
            raise ValueError(f"T must be >= 2, got {self.T}")
        if self.d_vocab < 2:
            raise ValueError(f"d_vocab must be >= 2, got {self.d_vocab}")
        if self.eval_n < 1:
            raise ValueError(f"eval_n must be >= 1, got {self.eval_n}")
        if not (0.0 < self.criterion_r <= 1.0):
            raise ValueError(f"criterion_r must be in (0, 1], got {self.criterion_r}")
        if self.b_step <= self.a_step:
            raise ValueError(f"b_step ({self.b_step}) must exceed a_step ({self.a_step})")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def run_id(self) -> str:
        """sha256 of the canonical JSON form, truncated. *This is the run ID.*

        Truncated to 16 hex characters, matching the width the existing
        G0/G1/G2 records already use, so the field stays comparable
        across rows. 64 bits over a sweep of ~10^2 cells has a collision
        probability around 10^-15; the risk that matters is not
        collision but a *silent* change to what is hashed -- see the
        module's own test file.
        """
        payload = json.dumps(self.to_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode()).hexdigest()[:16]

    @property
    def label(self) -> str:
        """Human-readable cell name for logs and filenames."""
        return (
            f"{self.arm}-r{self.rank}-lr{self.lr:g}-s{self.train_seed}-{self.variant}@{self.run_id}"
        )


def lr_grid(
    base_lr: float, n_points: int = MIN_LR_POINTS, spread: float = 3.0
) -> tuple[float, ...]:
    """Geometric lr ladder centred on `base_lr`, e.g. (base/3, base, base*3).

    PROJECT.md §8 requires ">=3 points" per rank but names no values, so
    the *shape* is fixed here and the *centre* is supplied by the caller
    -- scripts/run_sweep.py anchors it on the lr that G2's generous-rank
    run actually reached criterion with, rather than on a guess. An
    even `n_points` has no centre and is rejected.
    """
    if n_points < MIN_LR_POINTS:
        raise ValueError(f"PROJECT.md §8 requires >= {MIN_LR_POINTS} lr points, got {n_points}")
    if n_points % 2 == 0:
        raise ValueError(f"n_points must be odd so the ladder has a centre, got {n_points}")
    if not (base_lr > 0.0):
        raise ValueError(f"base_lr must be > 0, got {base_lr}")
    if spread <= 1.0:
        raise ValueError(f"spread must be > 1, got {spread} (a ladder needs distinct points)")
    half = n_points // 2
    return tuple(base_lr * spread**k for k in range(-half, half + 1))


def enumerate_cells(
    *,
    arms: Sequence[Arm],
    ranks: Sequence[int],
    lrs_by_rank: dict[int, Sequence[float]],
    seeds: Sequence[int],
    layer: int,
    head: int,
    alpha_over_r: float,
    max_steps: int,
    batch_size: int,
    T: int,
    d_vocab: int,
    eval_seed: int,
    eval_n: int,
    criterion_r: float,
    a_step: int,
    b_step: int,
    variant: Variant = "main",
) -> tuple[CellConfig, ...]:
    """The arm x rank x lr x seed grid, as `CellConfig`s in a stable order.

    `lrs_by_rank` must supply an independent lr list for *every* rank,
    each with at least MIN_LR_POINTS entries: sharing one lr list across
    ranks is the §8 confound this signature exists to make awkward.

    `alpha` is set per cell to `alpha_over_r * rank`, holding alpha/r
    fixed. PROJECT.md §8: "Rank changes the effective learning rate on
    Delta W. Either hold alpha/r fixed or sweep lr independently per
    rank. This is the single most likely way to produce a clean-looking
    curve that means nothing." §8 offers those as alternatives; this
    does both, because each leaves a different residual -- a fixed
    alpha/r still leaves the optimizer's own rank sensitivity, and a
    per-rank lr sweep still leaves the parameterization's scale drifting
    with r -- and doing both costs nothing beyond writing it down.
    """
    if not arms:
        raise ValueError("arms must be non-empty")
    if not ranks:
        raise ValueError("ranks must be non-empty")
    if not seeds:
        raise ValueError("seeds must be non-empty")
    if alpha_over_r <= 0.0:
        raise ValueError(f"alpha_over_r must be > 0, got {alpha_over_r}")
    missing = sorted(set(ranks) - set(lrs_by_rank))
    if missing:
        raise ValueError(
            f"lrs_by_rank has no entry for rank(s) {missing} -- PROJECT.md §8 requires lr "
            "swept independently per rank, so every rank needs its own list"
        )
    for rank, lrs in lrs_by_rank.items():
        if len(lrs) < MIN_LR_POINTS:
            raise ValueError(
                f"rank {rank} has {len(lrs)} lr point(s); PROJECT.md §8 requires >= {MIN_LR_POINTS}"
            )
        if len(set(lrs)) != len(lrs):
            raise ValueError(f"rank {rank}'s lr list has duplicates: {list(lrs)}")

    cells: list[CellConfig] = []
    for arm in arms:
        for rank in ranks:
            for lr in lrs_by_rank[rank]:
                for seed in seeds:
                    cells.append(
                        CellConfig(
                            arm=arm,
                            variant=variant,
                            layer=layer,
                            head=head,
                            rank=rank,
                            alpha=alpha_over_r * rank,
                            lr=lr,
                            train_seed=seed,
                            max_steps=max_steps,
                            batch_size=batch_size,
                            T=T,
                            d_vocab=d_vocab,
                            eval_seed=eval_seed,
                            eval_n=eval_n,
                            criterion_r=criterion_r,
                            a_step=a_step,
                            b_step=b_step,
                        )
                    )
    return tuple(cells)


def ten_x_steps_cell(cell: CellConfig) -> CellConfig:
    """The h0_2 / M7 control: the identical cell with 10x the step budget.

    PROJECT.md §6: "At r < r*, 10x steps closes the gap" is the null;
    h0_2 "carries the scientific content: reaches criterion slower ->
    optimization limit; plateaus below -> capacity limit."
    """
    return replace(cell, variant="ten_x_steps", max_steps=cell.max_steps * 10)


# ---------------------------------------------------------------------------
# Decision bands (PROJECT.md §6)
# ---------------------------------------------------------------------------


def decision_band(recovery: float) -> Band:
    """PROJECT.md §6: success R >= 0.80, failure R < 0.50, ambiguous between.

    "Ambiguous results stay ambiguous. 0.50 <= R < 0.80 is reported as
    ambiguous, never rounded into a verdict" (CLAUDE.md). Returning a
    third value rather than a bool is what makes rounding impossible at
    the type level.

    Takes the unclamped R (PROJECT.md §5 logs unclamped, clamps only for
    reporting); clamping cannot move a value across either threshold, so
    the band is the same either way. Non-finite input raises -- a NaN
    recovery is a broken run, and `nan < 0.50` being False would
    silently classify it as ambiguous.
    """
    if not math.isfinite(recovery):
        raise ValueError(f"decision_band received a non-finite recovery: {recovery}")
    if recovery >= SUCCESS_R:
        return "success"
    if recovery < FAILURE_R:
        return "failure"
    return "ambiguous"


# ---------------------------------------------------------------------------
# Resumability
# ---------------------------------------------------------------------------


def completed_run_ids(results_dir: Path) -> set[str]:
    """Every `run_config_hash` already recorded under `results_dir`.

    Reads every *.jsonl in the directory, not just this sweep's file: a
    cell recorded anywhere has been run, and re-running it would burn
    CPU to produce a duplicate. A missing directory yields an empty set
    rather than raising -- the first launch of a sweep is not an error.
    """
    results_dir = Path(results_dir)
    if not results_dir.exists():
        return set()
    done: set[str] = set()
    for path in sorted(results_dir.glob("*.jsonl")):
        for record in load_records(path):
            done.add(record.run_config_hash)
    return done


def pending_cells(cells: Iterable[CellConfig], done: set[str]) -> tuple[CellConfig, ...]:
    """`cells` minus those whose run ID already has a record, order preserved."""
    return tuple(cell for cell in cells if cell.run_id not in done)


# ---------------------------------------------------------------------------
# The mandatory-control gate (CLAUDE.md falsification rules, PROJECT.md §8/§9)
# ---------------------------------------------------------------------------


def _cell_key(record: ResultsRecord) -> tuple[str, int] | None:
    """(arm, rank) for a sweep record, or None if it is not one."""
    arm = record.observed.get("arm")
    rank = record.observed.get("rank")
    if arm is None or rank is None:
        return None
    return str(arm), int(rank)


def check_failure_claims(records: Sequence[ResultsRecord]) -> list[str]:
    """Problems with any capacity-failure claim in `records`; empty means OK.

    CLAUDE.md: "No failure claim without its controls attached. A rank
    cell that failed must report the matched-lr arm and the 10x-steps
    arm alongside it, or the result is about the optimizer and cannot be
    reported as capacity."

    An (arm, rank) is treated as *claiming a failure* when every `main`
    record at that (arm, rank) lands in the "failure" band -- if any
    lands in "success" or "ambiguous", the rank has not failed and needs
    no controls. For each such claim both controls must be present:

    - **The matched-lr arm** is read as §8's per-rank lr sweep itself:
      at least MIN_LR_POINTS *distinct* lr values recorded at that
      (arm, rank), all failing. A single "matched lr" run would be a
      weaker version of the same control, and the grid already pays for
      the stronger one. This reading is flagged in REVIEW.md.
    - **The 10x-steps arm** is a `ten_x_steps` record at that (arm,
      rank), which is h0_2 and has no substitute.

    Returns human-readable problems rather than raising, so a caller can
    report every unsupported claim at once instead of one per run.
    """
    by_key: dict[tuple[str, int], list[ResultsRecord]] = {}
    for record in records:
        key = _cell_key(record)
        if key is not None:
            by_key.setdefault(key, []).append(record)

    problems: list[str] = []
    for (arm, rank), group in sorted(by_key.items()):
        main = [r for r in group if r.observed.get("variant") == "main"]
        if not main:
            continue
        bands = [decision_band(float(r.observed["final_recovery"])) for r in main]
        if not all(band == "failure" for band in bands):
            continue  # not a failure claim; nothing to support

        distinct_lrs = {float(r.observed["lr"]) for r in main}
        if len(distinct_lrs) < MIN_LR_POINTS:
            problems.append(
                f"{arm} rank {rank} is recorded as a capacity failure on "
                f"{len(distinct_lrs)} lr point(s); PROJECT.md §8 requires >= "
                f"{MIN_LR_POINTS} swept independently at this rank before the "
                "result is about capacity rather than the optimizer"
            )
        if not any(r.observed.get("variant") == "ten_x_steps" for r in group):
            problems.append(
                f"{arm} rank {rank} is recorded as a capacity failure with no "
                "10x-steps arm (h0_2). Without it the cell cannot distinguish "
                "'reaches criterion slower' (optimization limit) from 'plateaus "
                "below' (capacity limit), and must not be reported as capacity"
            )
    return problems
