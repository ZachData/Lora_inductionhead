"""Diagnostic (not a PROJECT.md gate or metric): does *any* learning rate
let the generous-rank QK-only arm move at all from checkpoint A?

G3 failed (results/g3_positive_control.jsonl, R=0.0117 vs 0.80) with a
loss curve that never moved across all 400 steps at lr=3e-3. Per
PROJECT.md §11 / REVIEW.md's 2026-08-14 entry, that flat curve is
consistent with either a genuine capacity limit or a mistuned/broken
optimization path, and the two are not distinguished by that run alone.
This script is the cheap way to tell them apart before trusting G3's
verdict: short runs (100 steps, not 400) at several lr values, same
rank/arm/checkpoint/objective as G2/G3, watching whether loss/R move at
all in the first few dozen steps.

Deliberately NOT run through indbw.sweep / scripts/run_sweep.py: this is
not the pre-registered M1/M2 rank sweep (which correctly refuses to
start until G2 and G3 both pass) and does not emit a schema-validated
PROJECT.md results record -- it's a diagnostic precondition for deciding
whether G3's own record should be trusted, re-run, or left as-is.

Usage (single cell, run directly):
    python scripts/diagnose_g3_lr.py --lr 0.01 --worker-id 3

Usage (manifest-driven, one cell per worker -- matches G0's pattern):
    python scripts/diagnose_g3_lr.py --manifest g3_lr_diag_manifest.json --worker-id 3
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

A_STEP = 512
INDUCTION_LAYER = 3
INDUCTION_HEAD = 6
RANK = 64  # matches G2/G3's generous-rank positive control exactly
ALPHA = 64.0
MAX_STEPS = 100  # not 400: only asking "does it move at all", not "does it converge"
BATCH_SIZE = 4
T = 128
D_VOCAB = 50304
TRAIN_SEED = 0  # matches G2/G3, so lr is the only thing that differs
EVAL_EVERY = 10
EVAL_N = 16  # matches run_g2.py's memory-safe TRAIN_EVAL_N fix
EVAL_SEED = 0
MAX_WALL_CLOCK_S = 900.0  # 15 min/cell -- generous for 100 steps at ~5.5s/step


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def icl_baseline_from_g0(step: int) -> float:
    for path in sorted((REPO_ROOT / "data").glob("g0_sweep*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["step"] == step:
                return float(rec["icl"])
    raise SystemExit(f"no G0 sweep record found for step {step}")


def run_one_cell(lr: float, worker_id: int, out_path: Path, max_steps: int = MAX_STEPS) -> None:
    from indbw.models import load_checkpoint
    from indbw.train import TrainConfig, TrainingBudgetExceeded, train_lora

    icl_a = icl_baseline_from_g0(A_STEP)
    icl_b = icl_baseline_from_g0(2000)

    model = load_checkpoint(A_STEP, device="cpu")
    model.eval()

    config = TrainConfig(
        arm="QK",
        layer=INDUCTION_LAYER,
        head=INDUCTION_HEAD,
        rank=RANK,
        alpha=ALPHA,
        lr=lr,
        max_steps=max_steps,
        batch_size=BATCH_SIZE,
        T=T,
        d_vocab=D_VOCAB,
        train_seed=TRAIN_SEED,
        icl_a=icl_a,
        icl_b=icl_b,
        criterion_r=0.80,
        eval_every=EVAL_EVERY,
        eval_n=EVAL_N,
        eval_seed=EVAL_SEED,
        max_wall_clock_s=MAX_WALL_CLOCK_S,
        snapshot_dir=None,
    )

    t0 = time.time()
    record: dict[str, object]
    try:
        result = train_lora(model, config)
        record = {
            "worker_id": worker_id,
            "lr": lr,
            "rank": RANK,
            "arm": "QK",
            "max_steps": max_steps,
            "steps_run": result.steps_run,
            "reached_criterion": result.reached_criterion,
            "final_recovery": result.final_recovery,
            "loss_history": result.loss_history,
            "recovery_history": result.recovery_history,
            "loss_first": result.loss_history[0] if result.loss_history else None,
            "loss_last": result.loss_history[-1] if result.loss_history else None,
            "budget_exceeded": False,
            "wall_clock_s": result.wall_clock_s,
        }
    except TrainingBudgetExceeded as exc:
        record = {
            "worker_id": worker_id,
            "lr": lr,
            "rank": RANK,
            "arm": "QK",
            "max_steps": max_steps,
            "budget_exceeded": True,
            "error": str(exc),
            "wall_clock_s": time.time() - t0,
        }
    except FloatingPointError as exc:
        # A non-finite loss is itself diagnostic -- e.g. lr too high --
        # record it rather than letting the worker crash with no output.
        record = {
            "worker_id": worker_id,
            "lr": lr,
            "rank": RANK,
            "arm": "QK",
            "max_steps": max_steps,
            "non_finite_loss": True,
            "error": str(exc),
            "wall_clock_s": time.time() - t0,
        }

    record["git_sha"] = git_sha()
    record["hardware"] = f"{platform.machine()}/{platform.processor() or 'unknown'}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"worker {worker_id} (lr={lr}): wrote {out_path}")
    if "loss_first" in record and "loss_last" in record:
        print(
            f"  loss {record['loss_first']:.4f} -> {record['loss_last']:.4f}, "
            f"final_recovery={record['final_recovery']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument(
        "--out",
        type=str,
        default=None,
        help="defaults to data/g3_lr_diag_worker<id>.jsonl",
    )
    parser.add_argument("--max-steps", type=int, default=MAX_STEPS)
    args = parser.parse_args()

    if args.manifest is not None:
        manifest = json.loads(Path(args.manifest).read_text())
        cell = next(c for c in manifest["workers"] if c["worker_id"] == args.worker_id)
        lr = float(cell["lr"])
    elif args.lr is not None:
        lr = args.lr
    else:
        raise SystemExit("must pass either --lr or --manifest")

    out_path = (
        Path(args.out)
        if args.out is not None
        else REPO_ROOT / "data" / f"g3_lr_diag_worker{args.worker_id}.jsonl"
    )
    run_one_cell(lr, args.worker_id, out_path, max_steps=args.max_steps)


if __name__ == "__main__":
    main()
