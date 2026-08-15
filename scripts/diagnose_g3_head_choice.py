"""Diagnostic (not a PROJECT.md gate or metric, not a status-board row):
is layer 3/head 6 -- the head G3 trains -- actually the right head to
target when training QK-only from checkpoint A?

PROJECT.md §11 (2026-08-15 UPDATE, after the W_K-bottleneck diagnostic
came back negative) raises this as the next untested candidate: layer
3/head 6 was chosen because it is checkpoint B's argmax-PMS head
(§10, 2026-08-13 decision) -- but B's winning head is the result of
~1500 steps of *full* pretraining (all weights, real text) after A, not
evidence about which head QK-only LoRA training (one head, one matrix,
frozen everything else, synthetic repeated-random data, <=400 steps)
would find easiest to push from A.

A quick look at the existing G0 sweep data (data/g0_sweep*.jsonl) before
running this: at every checkpoint before A, the argmax-PMS head bounces
between different heads at pure-noise scores (~0.006, indistinguishable)
-- [5,7] at step 0-16, [3,0] at 32-64, [4,0] at 128, [0,3] at 256 -- and
only at step 512 (A itself) does argmax land on [3,6], which is then the
head that explodes to PMS>0.9 by step 1000. That is mild evidence
*against* "wrong head" for natural full-pretraining dynamics. It says
nothing about QK-only LoRA training specifically, which is what this
script tests directly.

Same mechanism as G2/G3 (indbw.train.train_lora, arm="QK", qk_hooks) --
only `layer`/`head` vary. No new production code or tests needed: this
reuses an already-tested code path, just at different head indices, so
unlike the W_K-bottleneck diagnostic (which needed new hook-composition
logic) there is no new mechanism whose correctness needs separate
verification.

Candidate heads tested, each at the lr the earlier lr-diagnosis sweep
found best (1e-2), rank=64 (matches G2/G3), 100 steps (matches the lr
diagnosis's cheap-probe budget):
  - (2, 1): the prev-token head itself (G1's finding) -- structurally
    plausible if K-composition can be extended in place, though prior
    literature (Olsson et al. 2022) suggests prefix-matching and
    previous-token heads are usually distinct.
  - (3, 0): same layer as B's winner, different head index -- tests
    whether layer 3 in general is receptive independent of which head.
  - (4, 6): one layer deeper, same head index as B's winner -- tests
    depth vs. the specific head.

If any of these reaches meaningfully higher R than layer 3/head 6's own
~0.01 (100-step) plateau, that supports "wrong head chosen". If all
three plateau in the same place, that disfavors it and leaves the
plateau's cause still unexplained by any of the three ruled-out/
disfavored hypotheses so far (broken objective, insufficient rank,
frozen-W_K ceiling, wrong head).

Usage (single cell):
    python scripts/diagnose_g3_head_choice.py --layer 2 --head 1 --worker-id 0

Usage (manifest-driven):
    python scripts/diagnose_g3_head_choice.py --manifest g3_head_diag_manifest.json --worker-id 0
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
RANK = 64  # matches G2/G3's generous-rank positive control exactly
ALPHA = 64.0
LR = 1e-2  # best lr from the earlier lr-diagnosis sweep (data/g3_lr_diag_worker3.jsonl)
MAX_STEPS = 100  # matches the lr diagnosis's cheap-probe budget
BATCH_SIZE = 4
T = 128
D_VOCAB = 50304
TRAIN_SEED = 0  # matches G2/G3/lr-diag, so this differs only in which head is targeted
EVAL_EVERY = 10
EVAL_N = 16
EVAL_SEED = 0
MAX_WALL_CLOCK_S = 900.0


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


def run_one_cell(layer: int, head: int, worker_id: int, out_path: Path) -> None:
    from indbw.models import load_checkpoint
    from indbw.train import TrainConfig, TrainingBudgetExceeded, train_lora

    icl_a = icl_baseline_from_g0(A_STEP)
    icl_b = icl_baseline_from_g0(2000)

    model = load_checkpoint(A_STEP, device="cpu")
    model.eval()

    config = TrainConfig(
        arm="QK",
        layer=layer,
        head=head,
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
            "layer": layer,
            "head": head,
            "lr": LR,
            "rank": RANK,
            "arm": "QK",
            "max_steps": MAX_STEPS,
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
            "layer": layer,
            "head": head,
            "lr": LR,
            "rank": RANK,
            "arm": "QK",
            "max_steps": MAX_STEPS,
            "budget_exceeded": True,
            "error": str(exc),
            "wall_clock_s": time.time() - t0,
        }
    except FloatingPointError as exc:
        record = {
            "worker_id": worker_id,
            "layer": layer,
            "head": head,
            "lr": LR,
            "rank": RANK,
            "arm": "QK",
            "max_steps": MAX_STEPS,
            "non_finite_loss": True,
            "error": str(exc),
            "wall_clock_s": time.time() - t0,
        }

    record["git_sha"] = git_sha()
    record["hardware"] = f"{platform.machine()}/{platform.processor() or 'unknown'}"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"worker {worker_id} (layer={layer}, head={head}): wrote {out_path}")
    if "loss_first" in record and "loss_last" in record:
        print(
            f"  loss {record['loss_first']:.4f} -> {record['loss_last']:.4f}, "
            f"final_recovery={record['final_recovery']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--layer", type=int, default=None)
    parser.add_argument("--head", type=int, default=None)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.manifest is not None:
        manifest = json.loads(Path(args.manifest).read_text())
        cell = next(c for c in manifest["workers"] if c["worker_id"] == args.worker_id)
        layer, head = int(cell["layer"]), int(cell["head"])
    elif args.layer is not None and args.head is not None:
        layer, head = args.layer, args.head
    else:
        raise SystemExit("must pass either --layer/--head or --manifest")

    out_path = (
        Path(args.out)
        if args.out is not None
        else REPO_ROOT / "data" / f"g3_head_diag_worker{args.worker_id}.jsonl"
    )
    run_one_cell(layer, head, args.worker_id, out_path)


if __name__ == "__main__":
    main()
