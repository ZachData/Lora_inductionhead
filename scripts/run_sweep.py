"""M1/M2: the main rank sweep, resumable, one record per cell.

PROJECT.md §8: "Main sweep, after all gates pass. Per arm: rank r in
{1,2,4,8,16,32} x lr (>=3 points, swept independently per rank) x 3
seeds." M1 is the QK arm (adapt W_Q only), M2 the OV arm (W_O only) --
PROJECT.md §3's one-matrix-per-form rule, enforced mechanically by
indbw.train's hook injection rather than by convention.

Cell definition, the run-ID hash, resumability and the control gate all
live in indbw.sweep; this script is orchestration only.

**It refuses to start until G2 and G3 have both passed.** §8 says the
main sweep runs "after all gates pass", and CLAUDE.md says a failed gate
stops the phase -- so that ordering is checked here against the actual
results records rather than left to whoever launches the instance.

**The lr grid is anchored, not guessed.** §8 requires >= 3 lr points per
rank but names no values. The ladder is centred on the lr that G2's
generous-rank run actually reached criterion with, read from its
artifact, so the sweep is built around a measured working point. If G2's
config ever changes, the ladder moves with it and every cell's run ID
changes -- which is correct: they are different runs.

Resumability is the whole point of the structure. Each cell appends its
record the moment it finishes, and a relaunch recomputes the pending set
from what is on disk. An idle-CPU alarm or the 4h cap costs one cell,
not the sweep.

Usage:
    python scripts/run_sweep.py --arm QK --dry-run
    python scripts/run_sweep.py --arm QK --budget-s 12600
    python scripts/run_sweep.py --arm QK --shard 2/7      # worker fleet
    python scripts/run_sweep.py --arm QK --controls       # M7: owed 10x-steps arms
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from indbw.schema import (
    METRIC_VERSION,
    Criterion,
    ResultsRecord,
    append_record,
    load_records,
)
from indbw.sweep import (
    DEFAULT_RANKS,
    DEFAULT_SEEDS,
    Arm,
    CellConfig,
    check_failure_claims,
    completed_run_ids,
    decision_band,
    enumerate_cells,
    lr_grid,
    pending_cells,
    ten_x_steps_cell,
)

RESULTS_DIR = REPO_ROOT / "results"
G2_ARTIFACT_PATH = REPO_ROOT / "data" / "g2_generous_rank_run.json"
SNAPSHOT_ROOT = REPO_ROOT / "data" / "sweep_snapshots"

ROW_FOR_ARM: dict[str, str] = {"QK": "M1", "OV": "M2"}
OUT_FOR_ARM: dict[str, Path] = {
    "QK": RESULTS_DIR / "m1_qk_rank_sweep.jsonl",
    "OV": RESULTS_DIR / "m2_ov_rank_sweep.jsonl",
}

# Fixed by G0/G1, not re-derived here (see run_g2.py's module docstring
# on why the recorded values are read rather than recomputed).
A_STEP = 512
B_STEP = 2000
INDUCTION_LAYER = 3
INDUCTION_HEAD = 6

T = 128  # PROJECT.md §5
D_VOCAB = 50304
EVAL_SEED = 0
FINAL_EVAL_N = 512  # §5's canonical N_eval, for the recorded observed value
TRAIN_EVAL_N = 128  # cheaper periodic checks during training
BATCH_SIZE = 4
ALPHA_OVER_R = 1.0  # holds alpha/r fixed across ranks (§8's confound guard)
CRITERION_R = 0.80  # §6, pre-registered


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def package_version(name: str) -> str:
    import importlib.metadata

    return importlib.metadata.version(name)


def require_gates_passed(results_dir: Path = RESULTS_DIR) -> None:
    """PROJECT.md §8: the main sweep runs only after all gates pass."""
    for row, filename in (
        ("G0", "g0_transition.jsonl"),
        ("G1", "g1_prerequisite.jsonl"),
        ("G2", "g2_cpu_timing.jsonl"),
        ("G3", "g3_positive_control.jsonl"),
    ):
        path = results_dir / filename
        if not path.exists():
            raise SystemExit(
                f"{row} has no results record at {path}. PROJECT.md §8 runs the main sweep "
                "only after all gates pass; close the gate first."
            )
        records = [r for r in load_records(path) if r.row == row]
        if not records:
            raise SystemExit(f"{path} contains no row=='{row}' record")
        latest = records[-1]
        if not latest.is_self_consistent():
            raise SystemExit(
                f"{row}'s record is not self-consistent (stored {latest.verdict!r}, recomputes "
                f"{latest.recomputed_verdict()!r}). Fix that before running anything on top of it."
            )
        if latest.verdict != "pass":
            raise SystemExit(
                f"{row} verdict is {latest.verdict!r}. A gate that fails stops the phase "
                "(CLAUDE.md) -- record the decision in PROJECT.md §10 rather than routing "
                "around it here."
            )


def base_lr_from_g2() -> float:
    """The lr G2's generous-rank run reached criterion with."""
    import json

    if not G2_ARTIFACT_PATH.exists():
        raise SystemExit(
            f"no G2 artifact at {G2_ARTIFACT_PATH}; the lr ladder is anchored on G2's "
            "working lr and must not be guessed."
        )
    artifact: dict[str, Any] = json.loads(G2_ARTIFACT_PATH.read_text())
    return float(artifact["config"]["lr"])


def build_cells(arm: Arm, base_lr: float) -> tuple[CellConfig, ...]:
    return enumerate_cells(
        arms=(arm,),
        ranks=DEFAULT_RANKS,
        lrs_by_rank={r: lr_grid(base_lr) for r in DEFAULT_RANKS},
        seeds=DEFAULT_SEEDS,
        layer=INDUCTION_LAYER,
        head=INDUCTION_HEAD,
        alpha_over_r=ALPHA_OVER_R,
        max_steps=_max_steps_from_g2(),
        batch_size=BATCH_SIZE,
        T=T,
        d_vocab=D_VOCAB,
        eval_seed=EVAL_SEED,
        eval_n=FINAL_EVAL_N,
        criterion_r=CRITERION_R,
        a_step=A_STEP,
        b_step=B_STEP,
    )


def _max_steps_from_g2() -> int:
    import json

    artifact: dict[str, Any] = json.loads(G2_ARTIFACT_PATH.read_text())
    return int(artifact["config"]["max_steps"])


def owed_control_cells(
    arm: Arm, cells: tuple[CellConfig, ...], results_dir: Path = RESULTS_DIR
) -> tuple[CellConfig, ...]:
    """M7: one 10x-steps cell per (arm, rank) whose main cells all failed.

    Runs after the main grid, because whether a rank owes a control is
    only knowable from its results. Ranks that succeeded or came back
    ambiguous owe nothing -- h0_2 exists to separate an optimization
    limit from a capacity limit, and neither of those is a claim yet.
    """
    records = [r for r in _all_records(results_dir) if r.observed.get("arm") == arm]
    owed: list[CellConfig] = []
    for rank in DEFAULT_RANKS:
        main = [
            r
            for r in records
            if r.observed.get("variant") == "main" and int(r.observed.get("rank", -1)) == rank
        ]
        if not main:
            continue
        if not all(decision_band(float(r.observed["final_recovery"])) == "failure" for r in main):
            continue
        if any(
            r.observed.get("variant") == "ten_x_steps" and int(r.observed.get("rank", -1)) == rank
            for r in records
        ):
            continue
        # One control per failing rank, at the grid's centre lr and seed 0.
        candidates = sorted(
            (c for c in cells if c.rank == rank and c.train_seed == DEFAULT_SEEDS[0]),
            key=lambda c: c.lr,
        )
        if not candidates:
            raise ValueError(
                f"rank {rank} has failing records but no cell in the current grid -- the "
                "grid and the records on disk describe different sweeps"
            )
        owed.append(ten_x_steps_cell(candidates[len(candidates) // 2]))
    return tuple(owed)


def _all_records(results_dir: Path = RESULTS_DIR) -> list[ResultsRecord]:
    records: list[ResultsRecord] = []
    if results_dir.exists():
        for path in sorted(results_dir.glob("*.jsonl")):
            records.extend(load_records(path))
    return records


def run_cell(cell: CellConfig, icl_a: float, icl_b: float, sha: str) -> ResultsRecord:
    """Train one cell and build its results record. Raises on a broken run."""
    import torch

    from indbw.evalset import build_eval_tokens
    from indbw.models import load_checkpoint
    from indbw.train import (
        LoRAFactors,
        TrainConfig,
        build_hooks,
        compute_recovery,
        train_lora,
    )

    model = load_checkpoint(cell.a_step, device="cpu")
    model.eval()
    try:
        config = TrainConfig(
            arm=cell.arm,
            layer=cell.layer,
            head=cell.head,
            rank=cell.rank,
            alpha=cell.alpha,
            lr=cell.lr,
            max_steps=cell.max_steps,
            batch_size=cell.batch_size,
            T=cell.T,
            d_vocab=cell.d_vocab,
            train_seed=cell.train_seed,
            icl_a=icl_a,
            icl_b=icl_b,
            criterion_r=cell.criterion_r,
            eval_every=20,
            eval_n=TRAIN_EVAL_N,
            eval_seed=cell.eval_seed,
            # No per-cell wall-clock cap beyond the sweep's own budget:
            # a cell that would exceed it is caught by the budget check
            # before it starts, not killed halfway through.
            max_wall_clock_s=float("inf"),
            snapshot_dir=SNAPSHOT_ROOT / cell.run_id,
            snapshot_every=100,
        )
        result = train_lora(model, config)

        # Canonical final measurement at §5's N_eval, not the cheaper
        # periodic eval used during training.
        eval_tokens = build_eval_tokens(cell.eval_n, cell.T, cell.eval_seed, cell.d_vocab)
        factors = LoRAFactors(
            B=torch.tensor(result.B, dtype=torch.float32),
            A=torch.tensor(result.A, dtype=torch.float32),
            alpha=cell.alpha,
        )
        hooks = build_hooks(cell.arm, cell.layer, cell.head, factors)
        final_recovery = compute_recovery(model, hooks, eval_tokens, cell.T, icl_a, icl_b)
    finally:
        del model

    observed: dict[str, Any] = {
        "final_recovery": final_recovery,
        "band": decision_band(final_recovery),  # raises on a non-finite R
        "arm": cell.arm,
        "variant": cell.variant,
        "rank": float(cell.rank),
        "lr": cell.lr,
        "train_seed": float(cell.train_seed),
        "alpha": cell.alpha,
        "steps_run": float(result.steps_run),
        "reached_criterion_flag": 1.0 if result.reached_criterion else 0.0,
        "wall_clock_s": result.wall_clock_s,
    }
    criteria = (Criterion(metric="final_recovery", op=">=", threshold=cell.criterion_r),)
    null = (
        "R is flat in r (h0_1)"
        if cell.variant == "main"
        else "at r < r*, 10x steps closes the gap (h0_2)"
    )
    return ResultsRecord(
        row=ROW_FOR_ARM[cell.arm],
        null_tested=null,
        criteria=criteria,
        observed=observed,
        verdict="pass" if all(c.holds(observed) for c in criteria) else "fail",
        metric_version=METRIC_VERSION,
        git_sha=sha,
        run_config_hash=cell.run_id,
        seed=cell.train_seed,
        checkpoint_revision=f"step {cell.a_step} (A)",
        eval_set_hash=_eval_set_hash(cell),
        torch_version=package_version("torch"),
        numpy_version=package_version("numpy"),
        transformer_lens_version=package_version("transformer_lens"),
        wall_clock_s=result.wall_clock_s,
        hardware=f"{platform.machine()}/{platform.processor() or 'unknown'}",
    )


def _eval_set_hash(cell: CellConfig) -> str:
    import hashlib
    import json

    payload = {"n_eval": cell.eval_n, "T": cell.T, "seed": cell.eval_seed}
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


def shard(cells: tuple[CellConfig, ...], spec: str) -> tuple[CellConfig, ...]:
    """`spec` is "i/n": every n-th cell starting at i. Round-robin, so a
    worker's slice spans the whole rank range rather than one corner of
    it -- a fleet that dies early then leaves a usable sweep instead of
    six ranks done and none of the rest."""
    index_s, _, total_s = spec.partition("/")
    index, total = int(index_s), int(total_s)
    if total < 1 or not (0 <= index < total):
        raise SystemExit(f"--shard must be i/n with 0 <= i < n, got {spec!r}")
    return tuple(c for k, c in enumerate(cells) if k % total == index)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=("QK", "OV"), required=True)
    parser.add_argument("--dry-run", action="store_true", help="print the plan, run nothing")
    parser.add_argument("--shard", default=None, help='worker slice, e.g. "2/7"')
    parser.add_argument(
        "--budget-s",
        type=float,
        default=12600.0,
        help="stop launching new cells past this wall-clock (default 3.5h, under the 4h cap)",
    )
    parser.add_argument(
        "--controls",
        action="store_true",
        help="run the 10x-steps arms owed by failing ranks (M7) instead of the main grid",
    )
    args = parser.parse_args(argv)
    arm: Arm = args.arm

    require_gates_passed()
    cells = build_cells(arm, base_lr_from_g2())
    if args.controls:
        cells = owed_control_cells(arm, cells)
        if not cells:
            print("no ranks currently owe a 10x-steps arm -- nothing to run")
            return 0
    if args.shard:
        cells = shard(cells, args.shard)

    done = completed_run_ids(RESULTS_DIR)
    pending = pending_cells(cells, done)
    print(f"{arm}: {len(cells)} cell(s) in scope, {len(cells) - len(pending)} already recorded")

    if args.dry_run:
        for cell in pending:
            print(f"  pending {cell.label}")
        return 0

    icl_a, icl_b = _icl_baselines()
    sha = git_sha()
    out_path = OUT_FOR_ARM[arm]
    t0 = time.time()
    # `ran` is the index of the cell about to start, which is exactly the
    # number already completed -- the budget message below relies on that.
    for ran, cell in enumerate(pending):
        elapsed = time.time() - t0
        if elapsed > args.budget_s:
            print(
                f"budget {args.budget_s:.0f}s reached after {ran} cell(s); "
                f"{len(pending) - ran} still pending. Relaunch to resume."
            )
            break
        print(f"[{elapsed:7.0f}s] running {cell.label}")
        record = run_cell(cell, icl_a, icl_b, sha)
        append_record(out_path, record)  # incremental: one record per cell
        print(
            f"           R={record.observed['final_recovery']:.4f} "
            f"({record.observed['band']}) in {record.wall_clock_s:.0f}s"
        )

    problems = check_failure_claims(_all_records())
    if problems:
        print("\nfailure claims not yet supported by their mandatory controls:")
        for problem in problems:
            print(f"  - {problem}")
        print("  run with --controls to fill the 10x-steps arms (M7).")
    return 0


def _icl_baselines() -> tuple[float, float]:
    from run_g2 import icl_baseline_from_g0  # same reader G2 used, not a second copy

    return icl_baseline_from_g0(A_STEP), icl_baseline_from_g0(B_STEP)


if __name__ == "__main__":
    raise SystemExit(main())
