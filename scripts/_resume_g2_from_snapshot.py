"""One-off: finish G2 by reusing the already-completed 400-step training run
whose factors are on disk at data/g2_snapshots/step_000400.npz (real training
compute already spent -- ~2198s -- CLAUDE.md's resumability principle says
not to discard it). The only thing that failed was the *final* canonical
(N_eval=512) recovery pass, which used compute_recovery's default
batch_size=32: at T=128 (seq_len=256) and d_vocab=50304, one batch's
full-vocab logits tensor alone is 32*256*50304*4 bytes =~ 1.65GB, which
exceeds this box's 1.8GB RAM outright and swap-thrashed the process to an
unrecoverable halt (killed manually after >15 min stuck past step 400/400).
batch_size=4 keeps that per-batch tensor to ~196MB.

This script is not part of the permanent pipeline -- scripts/run_g2.py is
patched separately (final compute_recovery call, batch_size=4) so a future
run doesn't hit this. This one just finishes emitting G2's already-earned
result without repeating 2198s of training.
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
SNAPSHOT_PATH = REPO_ROOT / "data" / "g2_snapshots" / "step_000400.npz"

sys.path.insert(0, str(REPO_ROOT / "src"))

A_STEP = 512
B_STEP = 2000
CRITERION_R = 0.80
T = 128
D_VOCAB = 50304
FINAL_EVAL_N = 512
TRAIN_SEED = 0
EVAL_SEED = 0
# Wall-clock actually observed for the training loop itself, read from
# scripts/run_g2.py's own stdout log (data/g2_snapshots step_000400.npz
# mtime landed at elapsed=2198s per the run's own progress print).
TRAINING_WALL_CLOCK_S = 2198.0
STEPS_RUN = 400
REACHED_CRITERION = False  # R plateaued ~0.013, never came close to 0.80


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def package_version(name: str) -> str:
    import importlib.metadata

    return importlib.metadata.version(name)


def icl_baseline_from_g0(step: int) -> float:
    for path in sorted(DATA_DIR.glob("g0_sweep*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["step"] == step:
                return float(rec["icl"])
    raise SystemExit(f"no G0 sweep record found for step {step}")


def main() -> None:
    import torch

    from indbw.evalset import build_eval_tokens
    from indbw.models import load_checkpoint
    from indbw.schema import METRIC_VERSION, Criterion, ResultsRecord, append_record
    from indbw.train import LoRAFactors, build_hooks, compute_recovery, load_snapshot

    icl_a = icl_baseline_from_g0(A_STEP)
    icl_b = icl_baseline_from_g0(B_STEP)

    snap = load_snapshot(SNAPSHOT_PATH)
    assert snap["step"] == 400
    assert snap["arm"] == "QK"
    assert snap["layer"] == 3
    assert snap["head"] == 6

    model = load_checkpoint(A_STEP, device="cpu")
    model.eval()

    factors = LoRAFactors(
        B=torch.tensor(snap["B"], dtype=torch.float32),
        A=torch.tensor(snap["A"], dtype=torch.float32),
        alpha=snap["alpha"],
    )
    hooks = build_hooks("QK", snap["layer"], snap["head"], factors)

    eval_tokens = build_eval_tokens(FINAL_EVAL_N, T, EVAL_SEED, D_VOCAB)
    t0 = time.time()
    final_recovery = compute_recovery(model, hooks, eval_tokens, T, icl_a, icl_b, batch_size=4)
    eval_wall_clock_s = time.time() - t0
    print(f"final canonical recovery: {final_recovery:.4f} (eval took {eval_wall_clock_s:.1f}s)")
    del model

    config_for_hash = {
        "arm": "QK",
        "layer": 3,
        "head": 6,
        "rank": 64,
        "alpha": 64.0,
        "lr": 3e-3,
        "max_steps": 400,
        "batch_size": 4,
        "T": T,
        "d_vocab": D_VOCAB,
        "train_seed": TRAIN_SEED,
        "eval_seed": EVAL_SEED,
        "criterion_r": CRITERION_R,
        "a_step": A_STEP,
        "b_step": B_STEP,
    }
    run_config_hash = hashlib.sha256(
        json.dumps(config_for_hash, sort_keys=True).encode()
    ).hexdigest()[:16]

    wall_clock_s = TRAINING_WALL_CLOCK_S  # the G2 timing measurement itself excludes the
    # one-off final-eval recovery from this repair script (that cost is an
    # artifact of the crash-recovery path, not of the training loop being timed)

    observed = {
        "reached_criterion_flag": 1.0 if REACHED_CRITERION else 0.0,
        "final_recovery_canonical": final_recovery,
        "steps_run": float(STEPS_RUN),
        "wall_clock_s": wall_clock_s,
        "seconds_per_step": wall_clock_s / STEPS_RUN,
    }
    criteria = (Criterion(metric="final_recovery_canonical", op=">=", threshold=CRITERION_R),)
    verdict = "pass" if all(c.holds(observed) for c in criteria) else "fail"

    n_worker_fleet = 7
    n_sweep_cells = 108
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
        "reached_criterion": REACHED_CRITERION,
        "steps_run": STEPS_RUN,
        "wall_clock_s": wall_clock_s,
        "final_recovery_canonical": final_recovery,
        "recovery_history": None,  # not recovered from the crashed process; see REVIEW.md
        "icl_a": icl_a,
        "icl_b": icl_b,
        "budget_exceeded": False,
        "n_worker_fleet": n_worker_fleet,
        "n_sweep_cells": n_sweep_cells,
        "projected_full_sweep_hours": projected_hours,
        "cpu_gpu_decision": cpu_gpu_decision,
        "git_sha": record.git_sha,
    }
    RAW_ARTIFACT_PATH.write_text(json.dumps(raw_artifact, indent=2, sort_keys=True) + "\n")

    print(f"G2 verdict: {verdict}")
    print(f"  final_recovery={final_recovery:.4f} steps_run={STEPS_RUN} wall_clock_s={wall_clock_s:.1f}")
    print(f"  projected full-sweep ({n_sweep_cells} cells, {n_worker_fleet}-way): "
          f"{projected_hours:.1f}h -> {cpu_gpu_decision}")
    print(f"  self-consistent: {record.is_self_consistent()}")
    print(f"  wrote {OUT_PATH} and {RAW_ARTIFACT_PATH}")


if __name__ == "__main__":
    main()
