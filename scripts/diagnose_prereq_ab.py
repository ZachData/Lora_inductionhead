"""Prerequisite contrast: the full head map at A and at B.

G1 (`scripts/run_g1.py`) asked whether A clears an *absolute* bar --
does some head score >= 0.3 on previous-token attention, and does its
output subspace overlap the candidate induction head's frozen W_K by
more than chance. A passed both, but narrowly: prev-token score 0.356
against a 0.3 threshold, overlap 0.387 against a 0.360 null (margin
0.027). PROJECT.md 11 flagged that margin at the time as "a real
alternative explanation to a capacity limit" if the QK arm later
struggled. It then struggled, and seven G3 diagnostics searched the
optimization axis instead.

G1 never measured the same quantities at B. That is the missing
contrast: an absolute bar says whether the prerequisite is *present*,
not whether it is *as strong as the checkpoint where induction
actually works*. `compute_prev_token_scores` was already parameterized
by step; only `main()` hardcoded A.

The 2026-09-03 reachability graft makes the contrast worth having.
Grafting B's entire layer-3 block into A -- head, MLP, LayerNorms --
leaves R at 0.0077, so whatever is missing sits *upstream of layer 3*.
The previous-token head G1 found is at layer 2. That is upstream.

This walks the circuit in both directions at both checkpoints, in one
forward pass each:

  backwards  prev-token score for every (layer, head) -- who supplies
             the prerequisite, and how strongly
  forwards   PMS for every (layer, head) -- who consumes it and
             actually does prefix matching
  the join   k_composition_overlap between each checkpoint's own
             prev-token head and its own induction-head W_K

Both probes are `indbw.probes`' existing, metric-hashed definitions
applied over the head grid; nothing new is defined here, so these
numbers are directly comparable to the committed G1 record.

Diagnostic only. No results record, no status-board row, no change to
PROJECT.md 6's thresholds or 8's gates -- what to conclude is a human
call (CLAUDE.md falsification discipline).

Usage:
    python scripts/diagnose_prereq_ab.py
"""

from __future__ import annotations

import json
import platform
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reused verbatim from G1 so the eval set, batching, and metric path are
# byte-identical to the committed record. Re-deriving any of them here
# would make A's new numbers incomparable to A's old ones, which is the
# entire point of the exercise.
from run_g1 import (  # noqa: E402
    D_VOCAB,
    N_EVAL,
    SEED,
    T,
    build_eval_tokens,
    git_sha,
    induction_head_from_g0,
)

A_STEP, B_STEP = 512, 2000
OUT_PATH = REPO_ROOT / "data" / "prereq_ab_diag.jsonl"


def head_maps(step: int) -> tuple[Any, Any, Any, Any]:
    """prev-token score and PMS for every (layer, head) at `step`.

    One forward pass over G1's fixed eval set, hooking every layer's
    attention pattern and scoring it with both probes. Returns
    (prev_token[n_layers, n_heads], pms[n_layers, n_heads], W_O, W_K).
    """
    import numpy as np
    import torch

    from indbw.models import load_checkpoint
    from indbw.probes import prefix_matching_score, prev_token_score

    model = load_checkpoint(step, device="cpu")
    model.eval()
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    eval_tokens = build_eval_tokens(N_EVAL, T, SEED, D_VOCAB)

    prev_sum = np.zeros((n_layers, n_heads))
    pms_sum = np.zeros((n_layers, n_heads))

    def make_hook(layer: int):  # type: ignore[no-untyped-def]
        def hook(pattern: torch.Tensor, hook: object) -> torch.Tensor:
            patt = pattern.detach().numpy()  # [b, heads, seq, seq]
            for head in range(n_heads):
                for bi in range(patt.shape[0]):
                    prev_sum[layer, head] += prev_token_score(patt[bi, head])
                    pms_sum[layer, head] += prefix_matching_score(patt[bi, head], T)
            return pattern

        return hook

    fwd_hooks = [
        (f"blocks.{layer}.attn.hook_pattern", make_hook(layer)) for layer in range(n_layers)
    ]

    # return_type=None skips the unembedding matmul: only attention
    # patterns are needed here, and full [b, seq, vocab] logits are
    # enough to OOM a 1.8GB box (PROJECT.md 11, and G1's own note).
    batch_size = 4
    n = eval_tokens.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            model.run_with_hooks(eval_tokens[start : start + batch_size], fwd_hooks=fwd_hooks,
                                 return_type=None)

    W_O = model.W_O.detach().numpy().copy()
    W_K = model.W_K.detach().numpy().copy()
    del model

    prev = prev_sum / n
    pms = pms_sum / n

    assert_grid_informative("prev_token", prev, step)
    assert_grid_informative("pms", pms, step)
    return prev, pms, W_O, W_K


def assert_grid_informative(name: str, grid: Any, step: int) -> None:
    """Raise unless `grid` could actually have come from a real forward pass.

    CLAUDE.md's dominant-risk note: "a readout that returns a constant,
    or empty, or the same number for every input looks completely fine
    in logs and poisons every downstream claim." Both failures this
    guards are exactly that shape. A hook registered on a layer that
    never fires leaves its row at zero; a whole grid of zeros is the
    signature of hooks that never fired at all, and would read as "no
    head does prefix matching anywhere" -- a plausible, publishable, and
    completely wrong sentence. Non-finite values do the same via NaN
    propagation. Neither crashes anything on its own.
    """
    import numpy as np

    if grid.size == 0:
        raise ValueError(f"{name} grid at step {step} is empty")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} grid at step {step} contains non-finite values")
    if float(np.max(grid) - np.min(grid)) < 1e-12:
        raise ValueError(
            f"{name} grid at step {step} is constant ({float(np.min(grid))}) across all "
            f"{grid.size} heads -- hooks likely never fired; refusing to report it"
        )


def main() -> None:
    import numpy as np

    from indbw.gates import k_composition_overlap, locate_prev_token_head

    ind_layer, ind_head = induction_head_from_g0(B_STEP)
    print(f"induction head from G0 at B: layer {ind_layer}, head {ind_head}\n", flush=True)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, step in (("A", A_STEP), ("B", B_STEP)):
        t0 = time.time()
        prev, pms, W_O, W_K = head_maps(step)
        wall = time.time() - t0

        prev_res = locate_prev_token_head(prev, threshold=0.3)
        # Each checkpoint is scored against its *own* prev-token head and
        # its *own* W_K. Holding A's head fixed while varying the weights
        # would confound "the prerequisite got stronger" with "it moved."
        overlap = k_composition_overlap(
            W_O[prev_res.layer, prev_res.head],
            W_K[ind_layer, ind_head],
            n_null_draws=100,
            percentile=95.0,
            seed=SEED,
        )

        order = np.argsort(prev.ravel())[::-1][:3]
        top_prev = [
            {"layer": int(i // prev.shape[1]), "head": int(i % prev.shape[1]),
             "score": float(prev.ravel()[i])}
            for i in order
        ]
        order = np.argsort(pms.ravel())[::-1][:3]
        top_pms = [
            {"layer": int(i // pms.shape[1]), "head": int(i % pms.shape[1]),
             "score": float(pms.ravel()[i])}
            for i in order
        ]

        rec = {
            "diagnostic": "prereq_ab_contrast",
            "checkpoint": label,
            "step": step,
            "induction_layer": ind_layer,
            "induction_head": ind_head,
            "prev_token_layer": prev_res.layer,
            "prev_token_head": prev_res.head,
            "prev_token_score": prev_res.score,
            "prev_token_found": prev_res.found,
            "induction_head_pms": float(pms[ind_layer, ind_head]),
            "induction_head_prev_token": float(prev[ind_layer, ind_head]),
            "overlap_ratio": overlap.ratio,
            "overlap_null_95th": overlap.null_percentile_value,
            "overlap_margin": overlap.ratio - overlap.null_percentile_value,
            "overlap_significant": overlap.significant,
            "top3_prev_token": top_prev,
            "top3_pms": top_pms,
            "prev_token_grid": prev.tolist(),
            "pms_grid": pms.tolist(),
            "n_eval": N_EVAL,
            "T": T,
            "eval_seed": SEED,
            "wall_clock_s": wall,
            "git_sha": git_sha(),
            "hardware": f"{platform.machine()}/{platform.processor() or platform.machine()}",
        }
        rows.append(rec)
        with OUT_PATH.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()

        print(f"[{label}] step {step}  ({wall:.0f}s)", flush=True)
        print(f"  prev-token head : layer {prev_res.layer} head {prev_res.head} "
              f"score {prev_res.score:.4f} (found={prev_res.found})", flush=True)
        print(f"  induction head  : ({ind_layer},{ind_head}) PMS {pms[ind_layer, ind_head]:.4f}",
              flush=True)
        print(f"  overlap         : {overlap.ratio:.4f} vs null {overlap.null_percentile_value:.4f}"
              f"  margin {overlap.ratio - overlap.null_percentile_value:+.4f}", flush=True)
        print(f"  top prev-token  : {top_prev}", flush=True)
        print(f"  top PMS         : {top_pms}\n", flush=True)

    a, b = rows
    print("=" * 64)
    print(f"{'quantity':34s} {'A':>12s} {'B':>12s}")
    print("-" * 64)
    for key in ("prev_token_score", "induction_head_pms", "overlap_ratio", "overlap_margin"):
        print(f"{key:34s} {a[key]:12.4f} {b[key]:12.4f}")
    print(f"{'prev-token head (layer,head)':34s} "
          f"{str((a['prev_token_layer'], a['prev_token_head'])):>12s} "
          f"{str((b['prev_token_layer'], b['prev_token_head'])):>12s}")
    print(f"\nwrote {OUT_PATH}")


if __name__ == "__main__":
    main()
