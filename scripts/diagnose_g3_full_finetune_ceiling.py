"""Diagnostic (not a PROJECT.md gate or metric, not a status-board row,
not a change to any protocol): the "sanity ceiling" -- can *anything*
move recovery R off the ~0.01-0.015 floor from checkpoint A, on the same
synthetic objective and a comparable step budget, if every parameter in
the model is trainable?

Four negative diagnostics in a row (lr sweep, W_K-bottleneck, three
alternative heads) all landed in the same plateau without explaining it
(PROJECT.md §10/§11, 2026-08-14/15). Two explanations remain live and
this session is not authorized to pick between them (CLAUDE.md reserves
falsification verdicts to humans):
  (a) a genuine capacity/structural limit under the single-matrix
      composition rule (§3/§8's named fallback branch: "too restrictive"),
  (b) something shared across every run so far that has nothing to do
      with which matrix is frozen -- the step budget, or the synthetic
      repeated-random objective's difficulty from this specific early
      checkpoint.

This is the maximally generous positive control that discriminates
between them: unfreeze *everything* (all weights, including LayerNorm
and embeddings -- explicitly not the M1-M8 protocol's rule, which is
deliberately violated here only because this is a diagnostic, never a
row whose result feeds §6/§7), full fine-tune from A on the identical
objective, comparable step budget to the other diagnostics.

  - If R still cannot move off the floor even with the entire model
    trainable, (a) is essentially ruled out -- no single-matrix
    composition-rule story can explain a result that persists when
    there is no composition rule constraining anything. The culprit
    would have to be (b): the objective or step budget.
  - If R moves substantially, that is real evidence for (a): the
    constraint (which matrices are frozen) is doing real work, and
    §8's named fallback ("composition rule too restrictive... fall
    back to per-circuit adaptation, downgrade the H2 claim") gains
    direct support.

No new hook mechanism: skips indbw.train.freeze_base_model entirely
(load_checkpoint's weights are trainable by default -- confirmed before
writing this script) and optimizes every model parameter directly with
Adam, reusing second_copy_nll/compute_recovery/build_eval_tokens
unchanged. compute_recovery is called with an empty hooks list (the
"delta" is now baked directly into the model's own weights, not
injected via a hook) -- this is the only new usage pattern, and it is a
degenerate case of an already-tested code path (run_with_hooks with no
hooks is a plain forward pass), not new mechanism requiring new tests.

The model's weights are mutated in place and the model is discarded at
the end of this script -- never saved, matching PROJECT.md §8's
"Storage: LoRA factors only, never a merged model" in spirit (there is
no merged-model artifact here at all, diagnostic or otherwise).

Runs two lr values (1e-4, 1e-5 -- standard full-fine-tune range, much
smaller than the LoRA runs' 1e-2 since gradients now flow directly into
raw weights rather than a zero-initialized low-rank factor) at 80 steps
each, matching the other diagnostics' cheap-probe philosophy.

Usage:
    python scripts/diagnose_g3_full_finetune_ceiling.py --lr 1e-4 --worker-id 0
    python scripts/diagnose_g3_full_finetune_ceiling.py --manifest g3_full_ft_manifest.json --worker-id 0
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
MAX_STEPS = 80
BATCH_SIZE = 4
T = 128
D_VOCAB = 50304
TRAIN_SEED = 0
EVAL_EVERY = 10
EVAL_N = 16
EVAL_SEED = 0
MAX_WALL_CLOCK_S = 1800.0  # 30 min hard budget per cell


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


def run_one_cell(lr: float, worker_id: int, out_path: Path) -> None:
    import numpy as np
    import torch

    from indbw.evalset import build_eval_tokens
    from indbw.models import load_checkpoint
    from indbw.train import (
        TrainingBudgetExceeded,
        _check_finite_loss,
        compute_recovery,
        second_copy_nll,
    )

    icl_a = icl_baseline_from_g0(A_STEP)
    icl_b = icl_baseline_from_g0(2000)
    print(f"ICL(A)={icl_a:.4f}, ICL(B)={icl_b:.4f}")

    model = load_checkpoint(A_STEP, device="cpu")
    model.eval()
    assert all(p.requires_grad for p in model.parameters()), (
        "expected all params trainable by default -- if load_checkpoint's "
        "behavior changed, this diagnostic silently degrades to a no-op"
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    eval_tokens = build_eval_tokens(EVAL_N, T, EVAL_SEED, D_VOCAB)
    train_rng = np.random.default_rng(TRAIN_SEED + 2000)

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
            logits = model(batch, return_type="logits")
            loss = second_copy_nll(logits, batch, T).mean()
            _check_finite_loss(loss, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

            if step % EVAL_EVERY == 0 or step == MAX_STEPS:
                r = compute_recovery(model, [], eval_tokens, T, icl_a, icl_b, batch_size=8)
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
        "diagnostic": "g3_full_finetune_ceiling",
        "worker_id": worker_id,
        "lr": lr,
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
            "best_head_choice_recovery": 0.0095,  # layer3/head6, data/g3_lr_diag_worker3.jsonl
        },
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")

    print(f"\nworker {worker_id} (lr={lr}): final_recovery={final_recovery:.4f}")
    print(f"wrote {out_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
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
        else REPO_ROOT / "data" / f"g3_full_ft_worker{args.worker_id}.jsonl"
    )
    run_one_cell(lr, args.worker_id, out_path)


if __name__ == "__main__":
    main()
