"""G0: locate the induction transition on the Pythia-70m checkpoint grid.

PROJECT.md §8 gate. Sweeps max-over-heads PMS and ICL over the full
checkpoint grid, appending one JSONL record per checkpoint to
results/g0_sweep.jsonl (CLAUDE.md resumability: a checkpoint already
recorded is skipped on restart). Once the grid (or --limit prefix of it)
is covered, runs indbw.gates.locate_transition over the accumulated
series and prints the verdict.

Eval set: N_eval repeated-random sequences of period T (PROJECT.md §5),
fixed seed, held fixed across all checkpoints.

Each checkpoint's HF download is cleared after use (own HF_HOME pointed
at a scratch dir, wiped per step) to keep disk bounded across a 154-step
sweep. Memory is bounded by processing the eval set in small batches
under torch.no_grad, with per-layer attention patterns consumed via a
forward hook and discarded immediately rather than retained for the
whole forward pass -- pythia-70m's baseline (torch + transformers
imports + model weights) already leaves little headroom on this box's
~1.8GB RAM (PROJECT.md §11), and retaining all 6 layers' patterns at
once was enough to force swap thrashing.

Usage:
    python scripts/g0_sweep.py [--n-eval 512] [--T 128] [--seed 0]
                                [--limit N] [--batch-size 4]
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
RESULTS_PATH = REPO_ROOT / "results" / "g0_sweep.jsonl"
SCRATCH_HF_HOME = Path("/home/ubuntu/.cache/g0_hf_scratch")

sys.path.insert(0, str(REPO_ROOT / "src"))


def build_eval_tokens(n_eval: int, T: int, seed: int, d_vocab: int) -> torch.Tensor:
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    first_half = rng.integers(0, d_vocab, size=(n_eval, T))
    sequences = np.concatenate([first_half, first_half], axis=1)
    return torch.from_numpy(sequences).long()


def load_existing_steps() -> dict[int, dict]:
    if not RESULTS_PATH.exists():
        return {}
    records: dict[int, dict] = {}
    for line in RESULTS_PATH.read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        records[rec["step"]] = rec
    return records


def append_record(record: dict) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS_PATH.open("a") as f:
        f.write(json.dumps(record) + "\n")
        f.flush()
        os.fsync(f.fileno())


def sweep_one_checkpoint(step: int, eval_tokens: torch.Tensor, T: int, batch_size: int) -> dict:
    import numpy as np
    import torch

    from indbw.models import load_checkpoint
    from indbw.probes import icl_score, prefix_matching_score

    t0 = time.time()
    model = load_checkpoint(step, device="cpu")
    model.eval()
    print(f"  [step {step}] model loaded ({time.time() - t0:.1f}s)", flush=True)
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    pms_sum = np.zeros((n_layers, n_heads))
    nll_first_batches: list[np.ndarray] = []
    nll_second_batches: list[np.ndarray] = []

    # Process-and-discard per layer via fwd hooks, instead of run_with_cache
    # (which retains all n_layers pattern tensors simultaneously -- on this
    # box's ~1.8GB RAM that peak was enough to force heavy swapping and
    # stall a single checkpoint past 500s, PROJECT.md §11). A hook fires as
    # each layer's pattern is produced and nothing else in the loop holds a
    # reference afterward, so peak attention-pattern memory drops from
    # n_layers tensors to one.
    def make_pattern_hook(layer: int):  # type: ignore[no-untyped-def]
        def hook(pattern: torch.Tensor, hook: object) -> torch.Tensor:
            pattern_np = pattern.detach().numpy()  # [b, heads, 2T, 2T]
            for head in range(n_heads):
                for bi in range(pattern_np.shape[0]):
                    pms_sum[layer, head] += prefix_matching_score(pattern_np[bi, head], T)
            return pattern

        return hook

    fwd_hooks = [
        (f"blocks.{layer}.attn.hook_pattern", make_pattern_hook(layer)) for layer in range(n_layers)
    ]

    n = eval_tokens.shape[0]
    n_batches = -(-n // batch_size)
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = eval_tokens[start : start + batch_size]
            logits = model.run_with_hooks(batch, fwd_hooks=fwd_hooks)
            log_probs = torch.log_softmax(logits[:, :-1, :].float(), dim=-1)
            targets = batch[:, 1:]
            token_logp = log_probs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
            nll = (-token_logp).numpy()  # [b, 2T - 1]
            nll_first_batches.append(nll[:, : T - 1].mean(axis=1))
            nll_second_batches.append(nll[:, T:].mean(axis=1))
            del logits, log_probs, token_logp
            print(
                f"  [step {step}] batch {start // batch_size + 1}/{n_batches} "
                f"done ({time.time() - t0:.1f}s elapsed)",
                flush=True,
            )

    del model
    nll_first = np.concatenate(nll_first_batches)
    nll_second = np.concatenate(nll_second_batches)
    icl = icl_score(nll_first, nll_second)
    pms_mean = pms_sum / n

    return {
        "step": step,
        "max_pms": float(pms_mean.max()),
        "max_pms_layer_head": [int(x) for x in np.unravel_index(pms_mean.argmax(), pms_mean.shape)],
        "icl": float(icl),
        "wall_clock_s": time.time() - t0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-eval", type=int, default=512)
    parser.add_argument("--T", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--limit", type=int, default=None, help="only sweep the first N grid steps")
    parser.add_argument("--batch-size", type=int, default=4)
    args = parser.parse_args()

    SCRATCH_HF_HOME.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(SCRATCH_HF_HOME)

    from indbw.models import checkpoint_steps

    steps = checkpoint_steps()
    if args.limit is not None:
        steps = steps[: args.limit]

    existing = load_existing_steps()
    print(
        f"{len(existing)}/{len(steps)} checkpoints already recorded in {RESULTS_PATH}", flush=True
    )

    eval_tokens = None
    for step in steps:
        if step in existing:
            continue
        if eval_tokens is None:
            # d_vocab is fixed for pythia-70m across checkpoints (50304); avoid
            # loading a model just to read it.
            eval_tokens = build_eval_tokens(args.n_eval, args.T, args.seed, d_vocab=50304)
        print(f"[step {step}] loading + evaluating...", flush=True)
        record = sweep_one_checkpoint(step, eval_tokens, args.T, args.batch_size)
        append_record(record)
        print(
            f"[step {step}] max_pms={record['max_pms']:.4f} icl={record['icl']:.4f} "
            f"({record['wall_clock_s']:.1f}s)",
            flush=True,
        )
        shutil.rmtree(SCRATCH_HF_HOME, ignore_errors=True)
        SCRATCH_HF_HOME.mkdir(parents=True, exist_ok=True)

    print("Sweep complete.", flush=True)


if __name__ == "__main__":
    main()
