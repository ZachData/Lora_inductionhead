"""Diagnostic (not a PROJECT.md gate or metric, not a status-board row):
three more axes for the G3-plateau investigation, user-directed
2026-08-15 follow-up to the full-fine-tune sanity-ceiling diagnostic.

Five diagnostics so far each varied one axis while holding four others
fixed (checkpoint=A, objective=synthetic repeated-random,
layer=3/head=6, <=400 steps) and each disfavored its targeted
explanation: broken objective/loop, insufficient rank, frozen-$W_K$
ceiling, wrong target head, and (via full fine-tune) the single-matrix
composition rule itself. Three axes were *never* varied by any of them:
the objective's own difficulty, the step budget, and the checkpoint.
This script runs one cell per remaining axis, each changing exactly one
thing relative to G3's own setup (rank=64, lr=1e-2 -- the
lr-diagnosis's best -- layer 3/head 6, QK arm):

  - `natural_text`: swaps the synthetic repeated-random training signal
    for real text (wikitext-2-raw-v1, `indbw.natural_text`) at the same
    400-step budget as G3 itself. Recovery R is still evaluated on the
    *standard* synthetic eval set via `indbw.train.compute_recovery`,
    unchanged, so R stays directly comparable to G3/every prior
    diagnostic (PROJECT.md §11: "evaluate on the other as a held-out
    check"). Tests whether the synthetic objective itself, not rank/head/
    matrix/checkpoint, is the hard part.

  - `more_steps`: same as G3 exactly (synthetic objective, checkpoint A,
    layer 3/head 6) but 2000 steps instead of 400 -- five times G3's own
    budget, and the first test past the 100/150/400 range every prior
    diagnostic stayed within. Tests whether the plateau breaks given
    enough steps.

  - `diff_checkpoint`: same as G3 exactly but starts from step 256
    instead of step 512 (A) -- the next grid point back, still safely
    pre-transition (max_pms=0.0054, essentially identical noise floor to
    A's own 0.0058; data/g0_sweep*.jsonl). R is still measured against
    the *official* A/B baselines (icl_a from step 512, icl_b from step
    2000) -- both step-256 and step-512's own baseline ICL are close to
    each other and near-chance, so this baseline choice does not
    manufacture an easy win. Tests whether A's specific state (e.g. its
    narrow G1 W_K-overlap margin, 0.387 vs a 0.360 null) is what's hard,
    as opposed to any pre-transition checkpoint being equally hard.
    (Step 1000, the *other* neighboring grid point, was checked first and
    rejected: its own max_pms is already 0.91 and ICL 9.55 -- essentially
    post-transition already, PROJECT.md §2's B-like regime, not a
    comparable pre-transition control.)

Explicitly diagnostic: does not change PROJECT.md §3/§8's protocol
(`train.py`'s objective stays synthetic; the official A/B checkpoints,
step 512/2000, are untouched), does not emit a schema-validated
PROJECT.md results record, and is deliberately NOT run through
indbw.sweep/scripts/run_sweep.py -- same reasoning as every earlier G3
diagnostic script.

Usage (single cell, run directly):
    python scripts/diagnose_g3_next.py --variant natural_text --worker-id 0

Usage (manifest-driven, one cell per worker):
    python scripts/diagnose_g3_next.py --manifest g3_next_diag_manifest.json --worker-id 0
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Literal

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

Variant = Literal["natural_text", "more_steps", "diff_checkpoint"]
VARIANTS: tuple[Variant, ...] = ("natural_text", "more_steps", "diff_checkpoint")

A_STEP = 512
B_STEP = 2000
DIFF_CHECKPOINT_STEP = 256
INDUCTION_LAYER = 3
INDUCTION_HEAD = 6
RANK = 64  # matches G2/G3's generous-rank positive control exactly
ALPHA = 64.0  # alpha/r = 1, matches G2/G3
LR = 1e-2  # best lr from the lr-diagnosis sweep (data/g3_lr_diag_worker3.jsonl)
T = 128
D_VOCAB = 50304
TRAIN_SEED = 0  # matches G2/G3/every prior diagnostic
EVAL_N = 16  # memory-safe, matches every prior diagnostic
EVAL_SEED = 0
CRITERION_R = 0.80

# Per-variant step budget / wall-clock budget.
MORE_STEPS_MAX_STEPS = 2000
STANDARD_MAX_STEPS = 400  # matches G3's own full budget, for direct comparability
NATURAL_TEXT_EVAL_EVERY = 20
MORE_STEPS_EVAL_EVERY = 50
STANDARD_EVAL_EVERY = 20

NATURAL_TEXT_MAX_WALL_CLOCK_S = 3600.0
MORE_STEPS_MAX_WALL_CLOCK_S = 13500.0  # 3.75h, ~2000 steps at ~5.5s/step + margin
DIFF_CHECKPOINT_MAX_WALL_CLOCK_S = 3600.0

BATCH_SIZE = 4
NATURAL_TEXT_SEQ_LEN = 2 * T  # matches the synthetic objective's own sequence length


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


def _base_record(variant: Variant, worker_id: int, extra: dict[str, object]) -> dict[str, object]:
    record: dict[str, object] = {
        "diagnostic": "g3_next",
        "variant": variant,
        "worker_id": worker_id,
        "layer": INDUCTION_LAYER,
        "head": INDUCTION_HEAD,
        "rank": RANK,
        "lr": LR,
        "arm": "QK",
        "git_sha": git_sha(),
        "hardware": f"{platform.machine()}/{platform.processor() or 'unknown'}",
        "comparison": {
            "g3_qk_only_final_recovery": 0.0117,  # 400 steps, lr=3e-3, checkpoint A
            "lr_diag_best_100step_recovery": 0.0095,  # lr=1e-2, data/g3_lr_diag_worker3.jsonl
        },
    }
    record.update(extra)
    return record


def run_more_steps(worker_id: int, out_path: Path) -> None:
    from indbw.models import load_checkpoint
    from indbw.train import TrainConfig, TrainingBudgetExceeded, train_lora

    icl_a = icl_baseline_from_g0(A_STEP)
    icl_b = icl_baseline_from_g0(B_STEP)

    model = load_checkpoint(A_STEP, device="cpu")
    model.eval()

    config = TrainConfig(
        arm="QK",
        layer=INDUCTION_LAYER,
        head=INDUCTION_HEAD,
        rank=RANK,
        alpha=ALPHA,
        lr=LR,
        max_steps=MORE_STEPS_MAX_STEPS,
        batch_size=BATCH_SIZE,
        T=T,
        d_vocab=D_VOCAB,
        train_seed=TRAIN_SEED,
        icl_a=icl_a,
        icl_b=icl_b,
        criterion_r=CRITERION_R,
        eval_every=MORE_STEPS_EVAL_EVERY,
        eval_n=EVAL_N,
        eval_seed=EVAL_SEED,
        max_wall_clock_s=MORE_STEPS_MAX_WALL_CLOCK_S,
        snapshot_dir=None,
    )

    t0 = time.time()
    try:
        result = train_lora(model, config)
        record = _base_record(
            "more_steps",
            worker_id,
            {
                "checkpoint": f"step {A_STEP} (A)",
                "max_steps": MORE_STEPS_MAX_STEPS,
                "steps_run": result.steps_run,
                "reached_criterion": result.reached_criterion,
                "final_recovery": result.final_recovery,
                "loss_history": result.loss_history,
                "recovery_history": result.recovery_history,
                "loss_first": result.loss_history[0] if result.loss_history else None,
                "loss_last": result.loss_history[-1] if result.loss_history else None,
                "budget_exceeded": False,
                "non_finite_loss": False,
                "wall_clock_s": result.wall_clock_s,
            },
        )
    except TrainingBudgetExceeded as exc:
        record = _base_record(
            "more_steps",
            worker_id,
            {
                "checkpoint": f"step {A_STEP} (A)",
                "max_steps": MORE_STEPS_MAX_STEPS,
                "budget_exceeded": True,
                "non_finite_loss": False,
                "error": str(exc),
                "wall_clock_s": time.time() - t0,
            },
        )
    except FloatingPointError as exc:
        record = _base_record(
            "more_steps",
            worker_id,
            {
                "checkpoint": f"step {A_STEP} (A)",
                "max_steps": MORE_STEPS_MAX_STEPS,
                "budget_exceeded": False,
                "non_finite_loss": True,
                "error": str(exc),
                "wall_clock_s": time.time() - t0,
            },
        )

    _write(record, out_path, worker_id)


def run_diff_checkpoint(worker_id: int, out_path: Path) -> None:
    from indbw.models import load_checkpoint
    from indbw.train import TrainConfig, TrainingBudgetExceeded, train_lora

    # R is still measured against the *official* A/B baselines -- this
    # variant asks "is starting from a different pre-transition checkpoint
    # easier", not "redefine R around a new baseline".
    icl_a = icl_baseline_from_g0(A_STEP)
    icl_b = icl_baseline_from_g0(B_STEP)

    model = load_checkpoint(DIFF_CHECKPOINT_STEP, device="cpu")
    model.eval()

    config = TrainConfig(
        arm="QK",
        layer=INDUCTION_LAYER,
        head=INDUCTION_HEAD,
        rank=RANK,
        alpha=ALPHA,
        lr=LR,
        max_steps=STANDARD_MAX_STEPS,
        batch_size=BATCH_SIZE,
        T=T,
        d_vocab=D_VOCAB,
        train_seed=TRAIN_SEED,
        icl_a=icl_a,
        icl_b=icl_b,
        criterion_r=CRITERION_R,
        eval_every=STANDARD_EVAL_EVERY,
        eval_n=EVAL_N,
        eval_seed=EVAL_SEED,
        max_wall_clock_s=DIFF_CHECKPOINT_MAX_WALL_CLOCK_S,
        snapshot_dir=None,
    )

    t0 = time.time()
    try:
        result = train_lora(model, config)
        record = _base_record(
            "diff_checkpoint",
            worker_id,
            {
                "checkpoint": f"step {DIFF_CHECKPOINT_STEP}",
                "max_steps": STANDARD_MAX_STEPS,
                "steps_run": result.steps_run,
                "reached_criterion": result.reached_criterion,
                "final_recovery": result.final_recovery,
                "loss_history": result.loss_history,
                "recovery_history": result.recovery_history,
                "loss_first": result.loss_history[0] if result.loss_history else None,
                "loss_last": result.loss_history[-1] if result.loss_history else None,
                "budget_exceeded": False,
                "non_finite_loss": False,
                "wall_clock_s": result.wall_clock_s,
            },
        )
    except TrainingBudgetExceeded as exc:
        record = _base_record(
            "diff_checkpoint",
            worker_id,
            {
                "checkpoint": f"step {DIFF_CHECKPOINT_STEP}",
                "max_steps": STANDARD_MAX_STEPS,
                "budget_exceeded": True,
                "non_finite_loss": False,
                "error": str(exc),
                "wall_clock_s": time.time() - t0,
            },
        )
    except FloatingPointError as exc:
        record = _base_record(
            "diff_checkpoint",
            worker_id,
            {
                "checkpoint": f"step {DIFF_CHECKPOINT_STEP}",
                "max_steps": STANDARD_MAX_STEPS,
                "budget_exceeded": False,
                "non_finite_loss": True,
                "error": str(exc),
                "wall_clock_s": time.time() - t0,
            },
        )

    _write(record, out_path, worker_id)


def run_natural_text(worker_id: int, out_path: Path) -> None:
    import torch

    from indbw.evalset import build_eval_tokens
    from indbw.models import load_checkpoint
    from indbw.natural_text import (
        load_wikitext_tokens,
        natural_induction_nll,
        sample_natural_batches,
    )
    from indbw.train import (
        TrainingBudgetExceeded,
        _check_finite_loss,
        build_hooks,
        compute_recovery,
        factor_shapes,
        freeze_base_model,
        init_lora_factors,
    )

    icl_a = icl_baseline_from_g0(A_STEP)
    icl_b = icl_baseline_from_g0(B_STEP)

    model = load_checkpoint(A_STEP, device="cpu")
    model.eval()
    freeze_base_model(model)

    print("tokenizing wikitext-2-raw-v1 (test split)...", flush=True)
    corpus_tokens = load_wikitext_tokens(model.tokenizer, split="test")
    print(f"corpus: {corpus_tokens.shape[0]} tokens", flush=True)

    d_out, d_in = factor_shapes(model, "QK")
    factors = init_lora_factors(d_out, d_in, RANK, ALPHA, seed=TRAIN_SEED)
    optimizer = torch.optim.Adam([factors.B, factors.A], lr=LR)
    hooks = build_hooks("QK", INDUCTION_LAYER, INDUCTION_HEAD, factors)

    # Recovery R is evaluated on the *standard* synthetic eval set,
    # unchanged -- only the training signal is natural text (module
    # docstring, PROJECT.md §11 "evaluate on the other as a held-out
    # check").
    eval_tokens = build_eval_tokens(EVAL_N, T, EVAL_SEED, D_VOCAB)
    train_rng_seed = TRAIN_SEED + 1

    t0 = time.time()
    loss_history: list[float] = []
    recovery_history: list[tuple[int, float]] = []
    step = 0
    budget_exceeded = False
    non_finite = False
    try:
        for step in range(1, STANDARD_MAX_STEPS + 1):
            elapsed = time.time() - t0
            if elapsed > NATURAL_TEXT_MAX_WALL_CLOCK_S:
                raise TrainingBudgetExceeded(
                    f"exceeded {NATURAL_TEXT_MAX_WALL_CLOCK_S}s at step {step} "
                    f"(elapsed={elapsed:.1f}s)"
                )
            batch, mask = sample_natural_batches(
                corpus_tokens, BATCH_SIZE, NATURAL_TEXT_SEQ_LEN, seed=train_rng_seed + step
            )
            logits = model.run_with_hooks(batch, fwd_hooks=hooks, return_type="logits")
            loss = natural_induction_nll(logits, batch, mask).mean()
            _check_finite_loss(loss, step)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(float(loss.item()))

            if step % NATURAL_TEXT_EVAL_EVERY == 0 or step == STANDARD_MAX_STEPS:
                r = compute_recovery(model, hooks, eval_tokens, T, icl_a, icl_b, batch_size=8)
                recovery_history.append((step, r))
                print(
                    f"  step {step}/{STANDARD_MAX_STEPS} loss={loss.item():.4f} R={r:.4f} "
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

    record = _base_record(
        "natural_text",
        worker_id,
        {
            "checkpoint": f"step {A_STEP} (A)",
            "corpus": "wikitext-2-raw-v1/test",
            "corpus_n_tokens": int(corpus_tokens.shape[0]),
            "max_steps": STANDARD_MAX_STEPS,
            "steps_run": step,
            "budget_exceeded": budget_exceeded,
            "non_finite_loss": non_finite,
            "final_recovery": final_recovery,
            "loss_first": loss_history[0] if loss_history else None,
            "loss_last": loss_history[-1] if loss_history else None,
            "loss_history": loss_history,
            "recovery_history": recovery_history,
            "wall_clock_s": wall_clock_s,
        },
    )
    _write(record, out_path, worker_id)


def _write(record: dict[str, object], out_path: Path, worker_id: int) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(json.dumps(record) + "\n")
    print(f"worker {worker_id} ({record['variant']}): wrote {out_path}")
    if record.get("loss_first") is not None:
        print(
            f"  loss {record['loss_first']:.4f} -> {record['loss_last']:.4f}, "
            f"final_recovery={record['final_recovery']:.4f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker-id", type=int, required=True)
    parser.add_argument("--variant", type=str, default=None, choices=VARIANTS)
    parser.add_argument("--manifest", type=str, default=None)
    parser.add_argument("--out", type=str, default=None)
    args = parser.parse_args()

    if args.manifest is not None:
        manifest = json.loads(Path(args.manifest).read_text())
        cell = next(c for c in manifest["workers"] if c["worker_id"] == args.worker_id)
        variant: Variant = cell["variant"]
    elif args.variant is not None:
        variant = args.variant  # type: ignore[assignment]
    else:
        raise SystemExit("must pass either --variant or --manifest")

    out_path = (
        Path(args.out)
        if args.out is not None
        else REPO_ROOT / "data" / f"g3_next_diag_worker{args.worker_id}.jsonl"
    )

    if variant == "natural_text":
        run_natural_text(args.worker_id, out_path)
    elif variant == "more_steps":
        run_more_steps(args.worker_id, out_path)
    elif variant == "diff_checkpoint":
        run_diff_checkpoint(args.worker_id, out_path)
    else:
        raise SystemExit(f"unknown variant {variant!r}, expected one of {VARIANTS}")


if __name__ == "__main__":
    main()
