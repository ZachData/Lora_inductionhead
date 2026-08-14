"""G2: CPU wall-clock for one generous-rank run to criterion at 70m.

PROJECT.md §8 gate. Trains a single generous-rank (r = d_head = 64,
the point beyond which more rank cannot enlarge Delta M_QK = Delta W_Q
W_K^T since frozen W_K already caps that product's rank -- PROJECT.md
§3) QK-arm LoRA adapter from checkpoint A, timing how long it takes to
reach the pre-registered recovery criterion R >= 0.80 (PROJECT.md §6).
"This has never been measured; every schedule downstream assumes it."

This run is *also* exactly what G3 asks for ("generous rank on W_Q
alone... does R exceed 0.80 at all?") -- re-running the same expensive
generous-rank training a second time under a different row's name would
waste the wall-clock this row exists to conserve. So this script writes
the raw run artifact (data/g2_generous_rank_run.json: final recovery,
steps, wall-clock, full config+provenance) that scripts/run_g3.py reads
to emit its own, separately schema-validated results record without
retraining. Each row still gets its own commit, board update, and
results record -- only the underlying compute is shared, and that
sharing is logged in PROJECT.md §10.

Objective, arm, hook mechanics: indbw.train (see its module docstring
for the settled-here induction-objective decision, PROJECT.md §11).

ICL(A)/ICL(B) baselines for recovery are *not* recomputed here -- they
are read directly from G0's own sweep records (data/g0_sweep*.jsonl,
step 512 and step 2000), which already used the identical eval-set
construction and NLL windowing this module's compute_recovery uses
(indbw.train.second_copy_nll's docstring: the windowing was matched to
scripts/g0_sweep.py's exactly, for precisely this reuse). Recomputing
them here from a second, nominally-identical-but-not-provably-identical
code path would risk a silent mismatch; reading the already-committed
numbers is safer and cheaper.

CPU-vs-GPU decision rule (pre-registered here, before this run's
timing is known -- PROJECT.md gives no numeric threshold for this,
so one is operationalized now, same discipline as G0/G1's ambiguous
prose, logged in PROJECT.md §10): the main sweep is 108 cells (2 arms
x 6 ranks x >=3 lr points x 3 seeds, PROJECT.md §8) before M3/M5-M8.
Using the same N=7 worker fleet G0 used, wall-clock for the full sweep
is approximately ceil(108/7) * this-run's-wall-clock. If that projects
to more than 24 wall-clock hours, escalate to GPU; otherwise CPU is
judged affordable. This projection is a planning estimate, not a
falsification criterion, and is reported alongside the record rather
than folded into its verdict.

Usage:
    python scripts/run_g2.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_PATH = REPO_ROOT / "results" / "g2_cpu_timing.jsonl"
RAW_ARTIFACT_PATH = REPO_ROOT / "data" / "g2_generous_rank_run.json"
SNAPSHOT_DIR = REPO_ROOT / "data" / "g2_snapshots"

sys.path.insert(0, str(REPO_ROOT / "src"))

A_STEP = 512
B_STEP = 2000
INDUCTION_LAYER = 3
INDUCTION_HEAD = 6

# Training config. rank=64=d_head: generous (PROJECT.md §3 rank ceiling
# for a single-matrix QK delta), not swept -- this is a positive-control
# run, not the rank sweep (M1).
RANK = 64
ALPHA = 64.0  # alpha/r = 1, no extra LoRA scaling
LR = 3e-3
MAX_STEPS = 400
BATCH_SIZE = 4
T = 128  # PROJECT.md §5
D_VOCAB = 50304  # fixed for pythia-70m across checkpoints
TRAIN_SEED = 0
EVAL_SEED = 0  # matches G0/G1's canonical eval seed
CRITERION_R = 0.80  # PROJECT.md §6
MAX_WALL_CLOCK_S = 2700.0  # 45 min hard budget, independent of any OS-level cap
# 16, not 128: this box has 1.8GB RAM (§11). A first real attempt (no swap
# active) hit the OOM killer outright; a second attempt (swap active)
# survived load and a few clean training steps (~3.7s/step, confirmed by
# direct profiling) but then swap-thrashed into an unrecoverable D-state
# hang past 56 minutes wall-clock -- diagnosed to compute_recovery's
# internal eval batch_size=32 (8x train batch_size=4) periodically spiking
# peak memory during training, whose high-water-mark then persists (CPU
# torch's allocator doesn't return memory to the OS). Shrinking the cheap
# periodic-check eval set is a config-only mitigation; the canonical
# FINAL_EVAL_N=512 check below still runs once, at the end, unchanged.
TRAIN_EVAL_N = 16  # cheap periodic checks during training
FINAL_EVAL_N = 512  # PROJECT.md §5's canonical N_eval, for the recorded observed value


def icl_baseline_from_g0(step: int) -> float:
    """Read G0's own recorded ICL(step) -- same eval set, same NLL
    windowing as this script's compute_recovery (see module docstring)."""
    for path in sorted(DATA_DIR.glob("g0_sweep*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["step"] == step:
                return float(rec["icl"])
    raise SystemExit(f"no G0 sweep record found for step {step} under {DATA_DIR}")


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def package_version(name: str) -> str:
    import importlib.metadata

    return importlib.metadata.version(name)


def main() -> None:
    import torch

    from indbw.evalset import build_eval_tokens
    from indbw.models import load_checkpoint
    from indbw.schema import METRIC_VERSION, Criterion, ResultsRecord, append_record
    from indbw.train import (
        TrainConfig,
        TrainingBudgetExceeded,
        build_hooks,
        compute_recovery,
        train_lora,
    )

    icl_a = icl_baseline_from_g0(A_STEP)
    icl_b = icl_baseline_from_g0(B_STEP)
    print(f"ICL(A)={icl_a:.4f} (from G0, step {A_STEP}), ICL(B)={icl_b:.4f} (step {B_STEP})")

    model = load_checkpoint(A_STEP, device="cpu")
    model.eval()

    config = TrainConfig(
        arm="QK",
        layer=INDUCTION_LAYER,
        head=INDUCTION_HEAD,
        rank=RANK,
        alpha=ALPHA,
        lr=LR,
        max_steps=MAX_STEPS,
        batch_size=BATCH_SIZE,
        T=T,
        d_vocab=D_VOCAB,
        train_seed=TRAIN_SEED,
        icl_a=icl_a,
        icl_b=icl_b,
        criterion_r=CRITERION_R,
        eval_every=20,
        eval_n=TRAIN_EVAL_N,
        eval_seed=EVAL_SEED,
        max_wall_clock_s=MAX_WALL_CLOCK_S,
        snapshot_dir=SNAPSHOT_DIR,
        snapshot_every=100,
    )

    config_for_hash = {
        "arm": config.arm,
        "layer": config.layer,
        "head": config.head,
        "rank": config.rank,
        "alpha": config.alpha,
        "lr": config.lr,
        "max_steps": config.max_steps,
        "batch_size": config.batch_size,
        "T": config.T,
        "d_vocab": config.d_vocab,
        "train_seed": config.train_seed,
        "eval_seed": config.eval_seed,
        "criterion_r": config.criterion_r,
        "a_step": A_STEP,
        "b_step": B_STEP,
    }
    run_config_hash = hashlib.sha256(
        json.dumps(config_for_hash, sort_keys=True).encode()
    ).hexdigest()[:16]

    t0 = time.time()
    budget_exceeded = False
    try:
        result = train_lora(model, config)
    except TrainingBudgetExceeded as exc:
        budget_exceeded = True
        print(f"TRAINING BUDGET EXCEEDED: {exc}")
        wall_clock_s = time.time() - t0
        # No TrainResult on a budget raise -- record what we can and stop;
        # this itself is a valid (negative) G2 timing measurement.
        observed = {
            "reached_criterion_flag": 0.0,
            "final_recovery_canonical": float("nan"),
            "steps_run": -1.0,
            "wall_clock_s": wall_clock_s,
            "seconds_per_step": float("nan"),
        }
        _emit_budget_exceeded_record(observed, run_config_hash, icl_a, icl_b)
        return

    wall_clock_s = result.wall_clock_s

    # Canonical final measurement: full N_eval=512 eval set (PROJECT.md
    # §5), not the cheaper TRAIN_EVAL_N used for periodic checks during
    # training -- this is the number that goes in the results record.
    eval_tokens = build_eval_tokens(FINAL_EVAL_N, T, EVAL_SEED, D_VOCAB)
    from indbw.train import LoRAFactors

    factors = LoRAFactors(
        B=torch.tensor(result.B, dtype=torch.float32),
        A=torch.tensor(result.A, dtype=torch.float32),
        alpha=config.alpha,
    )
    hooks = build_hooks(config.arm, config.layer, config.head, factors)
    # batch_size=4, not compute_recovery's default 32: at T=128 (seq_len=256)
    # and d_vocab=50304, one batch's full-vocab logits tensor is
    # batch_size*256*50304*4 bytes -- 32 gives ~1.65GB, which alone exceeds
    # this box's 1.8GB RAM and swap-thrashed a real run to an unrecoverable
    # halt (PROJECT.md §10, 2026-08-14 G2 rerun). 4 keeps it to ~196MB.
    final_recovery = compute_recovery(model, hooks, eval_tokens, T, icl_a, icl_b, batch_size=4)
    del model

    observed = {
        "reached_criterion_flag": 1.0 if result.reached_criterion else 0.0,
        "final_recovery_canonical": final_recovery,
        "steps_run": float(result.steps_run),
        "wall_clock_s": wall_clock_s,
        "seconds_per_step": wall_clock_s / result.steps_run if result.steps_run else float("nan"),
    }
    criteria = (Criterion(metric="final_recovery_canonical", op=">=", threshold=CRITERION_R),)
    verdict = "pass" if all(c.holds(observed) for c in criteria) else "fail"

    n_worker_fleet = 7  # G0's precedent fleet size
    n_sweep_cells = 108  # PROJECT.md §8 main sweep, before M3/M5-M8
    projected_hours = -(-n_sweep_cells // n_worker_fleet) * wall_clock_s / 3600.0
    cpu_gpu_decision = "CPU" if projected_hours <= 24.0 else "GPU"

    record = ResultsRecord(
        row="G2",
        null_tested="the generous-rank (r=64) QK-only positive control from A does not reach "
        "recovery R>=0.80 within the step/wall-clock budget",
        criteria=criteria,
        observed=observed,
        verdict=verdict,
        metric_version=METRIC_VERSION,
        git_sha=git_sha(),
        run_config_hash=run_config_hash,
        seed=TRAIN_SEED,
        checkpoint_revision=f"step {A_STEP} (A)",
        eval_set_hash=hashlib.sha256(
            json.dumps({"n_eval": FINAL_EVAL_N, "T": T, "seed": EVAL_SEED}, sort_keys=True).encode()
        ).hexdigest()[:16],
        torch_version=package_version("torch"),
        numpy_version=package_version("numpy"),
        transformer_lens_version=package_version("transformer_lens"),
        wall_clock_s=wall_clock_s,
        hardware=f"{platform.machine()}/{platform.processor() or 'unknown'}",
    )
    append_record(OUT_PATH, record)

    raw_artifact = {
        "row": "G2",
        "config": config_for_hash,
        "run_config_hash": run_config_hash,
        "reached_criterion": result.reached_criterion,
        "steps_run": result.steps_run,
        "wall_clock_s": wall_clock_s,
        "final_recovery_canonical": final_recovery,
        "recovery_history": result.recovery_history,
        "icl_a": icl_a,
        "icl_b": icl_b,
        "budget_exceeded": budget_exceeded,
        "n_worker_fleet": n_worker_fleet,
        "n_sweep_cells": n_sweep_cells,
        "projected_full_sweep_hours": projected_hours,
        "cpu_gpu_decision": cpu_gpu_decision,
        "git_sha": record.git_sha,
    }
    RAW_ARTIFACT_PATH.write_text(json.dumps(raw_artifact, indent=2, sort_keys=True) + "\n")

    print(f"G2 verdict: {verdict}")
    print(
        f"  steps_run={result.steps_run} reached_criterion={result.reached_criterion} "
        f"final_recovery={final_recovery:.4f} wall_clock_s={wall_clock_s:.1f} "
        f"({observed['seconds_per_step']:.2f} s/step)"
    )
    print(
        f"  projected full-sweep wall-clock ({n_sweep_cells} cells, {n_worker_fleet}-way fleet): "
        f"{projected_hours:.1f}h -> {cpu_gpu_decision}"
    )
    print(f"  self-consistent: {record.is_self_consistent()}")
    print(f"  wrote {OUT_PATH}")
    print(f"  wrote raw artifact {RAW_ARTIFACT_PATH} (for scripts/run_g3.py to reuse)")


def _emit_budget_exceeded_record(
    observed: dict[str, float], run_config_hash: str, icl_a: float, icl_b: float
) -> None:
    from indbw.schema import METRIC_VERSION, Criterion, ResultsRecord, append_record

    criteria = (Criterion(metric="reached_criterion_flag", op=">=", threshold=1.0),)
    record = ResultsRecord(
        row="G2",
        null_tested="the generous-rank (r=64) QK-only positive control from A does not reach "
        "recovery R>=0.80 within the step/wall-clock budget",
        criteria=criteria,
        observed=observed,
        verdict="fail",
        metric_version=METRIC_VERSION,
        git_sha=git_sha(),
        run_config_hash=run_config_hash,
        seed=TRAIN_SEED,
        checkpoint_revision=f"step {A_STEP} (A)",
        eval_set_hash=hashlib.sha256(
            json.dumps({"n_eval": FINAL_EVAL_N, "T": T, "seed": EVAL_SEED}, sort_keys=True).encode()
        ).hexdigest()[:16],
        torch_version=package_version("torch"),
        numpy_version=package_version("numpy"),
        transformer_lens_version=package_version("transformer_lens"),
        wall_clock_s=observed["wall_clock_s"],
        hardware=f"{platform.machine()}/{platform.processor() or 'unknown'}",
    )
    append_record(OUT_PATH, record)
    print("G2 verdict: fail (budget exceeded)")
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
