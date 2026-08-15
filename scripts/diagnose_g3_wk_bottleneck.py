"""Diagnostic (not a PROJECT.md gate or metric, not a status-board row):
does letting $W_K$ adapt too -- not just $W_Q$ -- unlock recovery R past
the plateau G3 got stuck at?

G3 failed (results/g3_positive_control.jsonl, R=0.0117 vs 0.80). The
follow-up lr-diagnosis sweep (scripts/diagnose_g3_lr.py, PROJECT.md §10
2026-08-14) ruled out "objective/loop broken" -- every lr shows real,
lr-dependent learning -- but every lr and step count (100 and 400)
plateaus at R=0.01-0.015, two orders of magnitude below criterion.

PROJECT.md §11 (2026-08-14 "UPDATE") names a specific untested
explanation: the QK arm only trains Delta W_Q, so Delta M_QK's row
space is capped by W_K's own column space at checkpoint A -- and G1
already found that overlap to be a narrow pass (0.387 vs a 0.360 null,
margin 0.027). If W_K hasn't developed enough structure yet, no rank of
Delta W_Q could close the gap, however large -- a frozen-W_K ceiling,
not a rank/capacity limit.

This script runs the direct diagnostic: also let W_K adapt (generous
rank, same as W_Q) at the same checkpoint/head/objective. If R breaks
decisively past the ~0.01-0.015 plateau, that supports the W_K-bottleneck
explanation. If it still plateaus in the same place, that disfavors it
(the ceiling would have to be somewhere else -- e.g. genuinely the
target operator's structure, or still tuning, though the lr sweep
already argues against tuning).

Explicitly diagnostic, not a protocol change: PROJECT.md §3's "adapt
exactly one matrix per form" rule is untouched for G2/G3/M1/M2 and
whatever this finds does not change that rule by itself (REVIEW.md
2026-08-14 entry -- a human call on what to do with the result, which is
exactly why this run needed sign-off before executing). Uses
`indbw.train.qk_both_hooks`, a function that exists solely for this
probe and is never reachable from `build_hooks`/`TrainConfig`/`Arm`.
Deliberately NOT run through indbw.sweep / scripts/run_sweep.py, same
reasoning as diagnose_g3_lr.py: does not emit a schema-validated
PROJECT.md results record.

Usage:
    python scripts/diagnose_g3_wk_bottleneck.py
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

A_STEP = 512
INDUCTION_LAYER = 3
INDUCTION_HEAD = 6
RANK = 64  # matches G2/G3's generous-rank positive control exactly, for both W_Q and W_K
ALPHA = 64.0  # alpha/r = 1, matches G2/G3
LR = 1e-2  # best-performing lr from the lr-diagnosis sweep (data/g3_lr_diag_worker3.jsonl)
MAX_STEPS = 150  # more than the 100-step lr probe (some headroom), well under G3's 400
BATCH_SIZE = 4
T = 128
D_VOCAB = 50304
TRAIN_SEED = 0  # matches G2/G3/lr-diag, so this run differs only in which matrices adapt
EVAL_EVERY = 15
EVAL_N = 16  # matches run_g2.py's memory-safe TRAIN_EVAL_N fix (§10, 2026-08-14)
EVAL_SEED = 0
MAX_WALL_CLOCK_S = 1800.0  # 30 min hard budget, independent of any OS-level cap
OUT_PATH = REPO_ROOT / "data" / "g3_wk_bottleneck_diag.jsonl"


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


def main() -> None:
    from indbw.evalset import build_eval_tokens
    from indbw.models import load_checkpoint
    from indbw.train import (
        TrainingBudgetExceeded,
        _check_finite_loss,
        compute_recovery,
        freeze_base_model,
        init_lora_factors,
        qk_both_hooks,
        second_copy_nll,
    )

    icl_a = icl_baseline_from_g0(A_STEP)
    icl_b = icl_baseline_from_g0(2000)
    print(f"ICL(A)={icl_a:.4f}, ICL(B)={icl_b:.4f}")

    model = load_checkpoint(A_STEP, device="cpu")
    model.eval()
    freeze_base_model(model)

    d_model, d_head = int(model.cfg.d_model), int(model.cfg.d_head)
    q_factors = init_lora_factors(d_model, d_head, rank=RANK, alpha=ALPHA, seed=TRAIN_SEED)
    k_factors = init_lora_factors(d_model, d_head, rank=RANK, alpha=ALPHA, seed=TRAIN_SEED + 1)
    optimizer = torch.optim.Adam([q_factors.B, q_factors.A, k_factors.B, k_factors.A], lr=LR)
    hooks = qk_both_hooks(INDUCTION_LAYER, INDUCTION_HEAD, q_factors, k_factors)

    eval_tokens = build_eval_tokens(EVAL_N, T, EVAL_SEED, D_VOCAB)
    train_rng = np.random.default_rng(TRAIN_SEED + 1000)

    t0 = time.time()
    loss_history: list[float] = []
    recovery_history: list[tuple[int, float]] = []
    step = 0
    budget_exceeded = False
    non_finite = False
    try:
        for step in range(1, MAX_STEPS + 1):
            elapsed = time.time() - t0
            if elapsed > MAX_WALL_CLOCK_S:
                raise TrainingBudgetExceeded(
                    f"exceeded {MAX_WALL_CLOCK_S}s at step {step} (elapsed={elapsed:.1f}s)"
                )
            batch = build_eval_tokens(BATCH_SIZE, T, int(train_rng.integers(0, 2**31 - 1)), D_VOCAB)
            logits = model.run_with_hooks(batch, fwd_hooks=hooks, return_type="logits")
            loss = second_copy_nll(logits, batch, T).mean()
            _check_finite_loss(loss, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                r = compute_recovery(model, hooks, eval_tokens, T, icl_a, icl_b, batch_size=8)
                recovery_history.append((step, r))
                print(
                    f"  step {step}/{MAX_STEPS} loss={loss.item():.4f} R={r:.4f} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    except TrainingBudgetExceeded as exc:
        budget_exceeded = True
        print(f"BUDGET EXCEEDED: {exc}")
    except FloatingPointError as exc:
        non_finite = True
        print(f"NON-FINITE LOSS: {exc}")

    wall_clock_s = time.time() - t0
    final_recovery = recovery_history[-1][1] if recovery_history else float("nan")

    record = {
        "diagnostic": "g3_wk_bottleneck",
        "arm": "QK+K",  # not a real Arm value -- diagnostic label only
        "layer": INDUCTION_LAYER,
        "head": INDUCTION_HEAD,
        "rank_q": RANK,
        "rank_k": RANK,
        "alpha": ALPHA,
        "lr": LR,
        "checkpoint": f"step {A_STEP} (A)",
        "max_steps": MAX_STEPS,
        "steps_run": step,
        "wall_clock_s": wall_clock_s,
        "budget_exceeded": budget_exceeded,
        "non_finite_loss": non_finite,
        "final_recovery": final_recovery,
        "loss_first": loss_history[0] if loss_history else None,
        "loss_last": loss_history[-1] if loss_history else None,
        "loss_history": loss_history,
        "recovery_history": recovery_history,
        "git_sha": git_sha(),
        "hardware": f"{platform.machine()}/{platform.processor() or 'unknown'}",
        "comparison": {
            "g3_qk_only_final_recovery": 0.0117,
            "lr_diag_best_100step_recovery": 0.0095,  # lr=1e-2, data/g3_lr_diag_worker3.jsonl
        },
    }
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUT_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\nfinal_recovery={final_recovery:.4f} (QK-only G3 reference: 0.0117)")
    print(f"wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
