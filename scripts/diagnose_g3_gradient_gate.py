"""Is the QK arm's gradient gated at A? Copying score and OV value spread.

Authorized 2026-09-07 (PROJECT.md 10). Diagnostic only: no results
record, no status-board row, no change to 6's thresholds or 8's gates.
G3 stays failed and M1-M4 stay blocked -- this is not a route around the
gate, it is a measurement of a precondition the gate's interpretation
already assumed.

WHY THIS, AFTER NINE DIAGNOSTICS
-------------------------------
Eight of the nine varied an optimization axis (lr, rank, frozen W_K,
target head, full fine-tune, step budget) or a graft. None asked whether
the QK arm had a usable gradient at A to begin with.

docs/mathematics.md makes that a forward-only question:

  Thm 17.1   dL/ds_ij = alpha_ij <g_i, z_j - zbar_i>, exactly.
  Cor 17.2   so |dL/ds| <= ||g_i|| * sigma_OV(i), where sigma_OV(i) =
             max_j ||z_j - zbar_i|| is the head's OV *value spread* --
             a property of the head at A, independent of r.
  Cor 17.3   spread is necessary but not sufficient. Descent moves
             attention toward positions with low <g_i, z_j>; for that to
             build induction the head's OV must carry the attended
             token's identity to the logits, i.e. a nonzero *copying
             score*. Spread without copying moves the loss and not R.
  Prop 17.5  rank cannot repair a vanishing gradient -- every rank's
             gradient is a linear image of the full one. A rank sweep in
             a gated regime returns R(r) flat at every r, which is the
             literal statement of h0_1 by a mechanism that has nothing
             to do with intrinsic dimension.

That last point is why this is worth measuring before M1/M2 rather than
after: it decides whether those rows measure capacity or measure a
closed gate.

**No record in results/ or data/ reports a copying score at A for head
(3,6) or for any head.** probes.copying_score has only ever run on toy
fixtures. This closes that.

WHAT IS MEASURED
----------------
  copying score   every (layer, head) at A and at B, using the hashed
                  5 definition evaluated in token chunks (see
                  copying_score_chunked -- the unchunked call needs
                  10.1 GB in one allocation and cannot run anywhere).
  value spread    sigma_OV(i) over query positions for the candidate
                  head, at A and at B, on G1's fixed eval set.

The authorized question is the candidate head at A. The head grid and
the B contrast are the discrimination context that makes a single
number readable at all: CLAUDE.md's TDD contract, kind 3, and the
precedent diagnose_prereq_ab.py set by measuring at both checkpoints
rather than reporting A alone against a bar.

Both outcomes are informative and neither is a verdict. A near-zero
copying score at A would make the nine diagnostics consistent as
instances of one mechanism, and bears on the 11 "lumped metric used as
a single-component gate" inconsistency between 4 and 6/8 -- which it
does NOT resolve; that is flagged there as a human call. A healthy
copying score rules the mechanism out and leaves the plateau
unexplained. Which, if either, is a human call (CLAUDE.md 9).

Usage:
    python scripts/diagnose_g3_gradient_gate.py [--out PATH] [--chunk N]
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

# Reused verbatim from G1 so the eval set is byte-identical to the
# committed records and to diagnose_prereq_ab.py's contrast.
from run_g1 import (  # noqa: E402
    D_VOCAB,
    N_EVAL,
    SEED,
    T,
    build_eval_tokens,
    induction_head_from_g0,
)

A_STEP, B_STEP = 512, 2000
DEFAULT_OUT = REPO_ROOT / "data" / "g3_gradient_gate_diag.jsonl"

# Query positions sampled for the value-spread readout. sigma_OV(i) is a
# per-query quantity (Cor 17.2) and the eval set has 2T = 256 positions
# per sequence; the full distribution over a few sequences is plenty to
# tell a closed gate from an open one, and keeps the [seq, seq] pattern
# and [seq, d] value arrays small.
SPREAD_SEQUENCES = 8


def ov_matrix(W_V: np.ndarray, W_O: np.ndarray) -> np.ndarray:
    """M_OV = W_O^T W_V^T, the composed OV form of PROJECT.md 3.

    W_V is [d_model, d_head] and W_O is [d_head, d_model] in
    TransformerLens's layout, so the head's action on a residual row
    vector x is (x @ W_V) @ W_O. As a matrix acting on a column vector
    that is (W_V W_O)^T = W_O^T W_V^T -- which is 3's expression, and
    the transpose matters: the tests pin it against the explicit
    vector application because an identity-weight oracle cannot.
    """
    if W_V.ndim != 2 or W_O.ndim != 2:
        raise ValueError(f"W_V and W_O must be 2D, got {W_V.shape} and {W_O.shape}")
    if W_V.shape[1] != W_O.shape[0]:
        raise ValueError(f"inner dims disagree: W_V {W_V.shape}, W_O {W_O.shape}")
    if W_V.shape[0] != W_O.shape[1]:
        raise ValueError(f"M_OV must be square: W_V {W_V.shape}, W_O {W_O.shape}")
    return np.asarray((W_V @ W_O).T)


def copying_score_chunked(
    W_U: np.ndarray, M_OV: np.ndarray, W_E: np.ndarray, chunk: int = 512
) -> float:
    """PROJECT.md 5's copying score, evaluated over token blocks.

    Identical definition to `probes.copying_score` -- fraction of vocab
    tokens t with argmax_s (W_U M_OV e_t)_s == t -- and the tests assert
    exact agreement with it at every chunk size. This exists only
    because the hashed function allocates the whole [vocab, vocab] logit
    matrix at once: at pythia-70m's vocab that is 50304^2 * 4 = 10.1 GB,
    which no instance this project may launch can hold. probes.py is
    deliberately NOT modified -- it is inside METRIC_MODULES, so
    touching it would force a METRIC_VERSION bump and stale every
    committed record for a change with no effect on any number.

    The argmax is over the full vocabulary for every token. Taking it
    within a chunk instead returns ~0.0 for a real model, which is
    indistinguishable from the finding this diagnostic is looking for;
    that is the failure the tests spend most of their length on.
    """
    if W_U.ndim != 2 or M_OV.ndim != 2 or W_E.ndim != 2:
        raise ValueError("copying_score_chunked requires 2D W_U, M_OV, W_E")
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")
    vocab, d_u = W_U.shape
    if M_OV.shape[0] != M_OV.shape[1]:
        raise ValueError(f"M_OV must be square, got {M_OV.shape}")
    if d_u != M_OV.shape[0] or M_OV.shape[1] != W_E.shape[1]:
        raise ValueError(f"dim mismatch: W_U {W_U.shape}, M_OV {M_OV.shape}, W_E {W_E.shape}")
    if W_E.shape[0] != vocab:
        raise ValueError(f"W_U and W_E must share vocab, got {vocab} and {W_E.shape[0]}")
    if vocab < 2:
        raise ValueError("copying_score_chunked requires vocab >= 2")
    # Same guard as the hashed metric: an all-zero OV makes every logit
    # row identical, argmax returns 0 for every token, and the score is
    # a small believable 1/vocab for a circuit that does not exist.
    if not np.any(M_OV):
        raise ValueError("copying_score_chunked: M_OV is all-zero — the OV circuit is undefined")

    # logits[t, s] = (W_U @ M_OV @ e_t)[s]; e_t = W_E[t].
    right = M_OV.T @ W_U.T  # [d, vocab], formed once
    n_correct = 0
    for start in range(0, vocab, chunk):
        stop = min(start + chunk, vocab)
        block = W_E[start:stop] @ right  # [c, vocab]
        # argmax over the vocabulary axis, compared against the token's
        # own *global* id -- not its offset within the block.
        n_correct += int(np.count_nonzero(np.argmax(block, axis=1) == np.arange(start, stop)))
        del block
    value = n_correct / vocab
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"copying_score_chunked out of range: {value}")
    return value


def copying_score_factored(
    W_U: np.ndarray, W_V: np.ndarray, W_O: np.ndarray, W_E: np.ndarray, chunk: int = 512
) -> float:
    """The same score again, routed through the rank-d_head factorization.

    M_OV = W_O^T W_V^T has rank <= d_head, so the [vocab, vocab] logit
    matrix factors through a 64-dimensional inner product rather than a
    512-dimensional one:

        logits[t, s] = W_U[s] . (W_O^T W_V^T e_t)
                     = (W_U[s] W_O^T) . (W_V^T e_t)
                     = L[s] . V[t],   L = W_U W_O^T,  V = W_E W_V

    Both L and V are [vocab, d_head]. That is 8x fewer FLOPs at
    pythia-70m's shapes and it is the difference between fitting inside a
    worker's 3-hour cap and not: the composed path costs ~2.6e12 FLOPs per
    head, ~130 s on a t4g.small, ~3.5 h over 48 heads at two checkpoints.

    Arithmetically identical to `copying_score_chunked` -- the tests assert
    exact agreement -- but *not* bit-identical in floating point, since the
    two associations round differently. Computed in float64 for that
    reason: the model's weights are float32, and at float32 a near-tie in a
    head with no copying structure could resolve differently between the
    two paths. float64 puts the disagreement ~1e-13 below any real margin.
    """
    if W_V.ndim != 2 or W_O.ndim != 2:
        raise ValueError(f"W_V and W_O must be 2D, got {W_V.shape} and {W_O.shape}")
    if not np.any(W_V) or not np.any(W_O):
        raise ValueError("copying_score_factored: W_V or W_O is all-zero — OV is undefined")
    # Deliberately reuses ov_matrix's shape checks rather than restating
    # them, so the two paths cannot drift apart on what they accept.
    ov_matrix(W_V, W_O)

    L = np.asarray(W_U, dtype=np.float64) @ np.asarray(W_O, dtype=np.float64).T  # [vocab, d_head]
    V = np.asarray(W_E, dtype=np.float64) @ np.asarray(W_V, dtype=np.float64)  # [vocab, d_head]
    if L.shape != V.shape:
        raise ValueError(f"factored shapes disagree: L {L.shape}, V {V.shape}")
    vocab = L.shape[0]
    if chunk < 1:
        raise ValueError(f"chunk must be >= 1, got {chunk}")

    n_correct = 0
    for start in range(0, vocab, chunk):
        stop = min(start + chunk, vocab)
        block = V[start:stop] @ L.T  # [c, vocab]
        n_correct += int(np.count_nonzero(np.argmax(block, axis=1) == np.arange(start, stop)))
        del block
    value = n_correct / vocab
    if not (0.0 <= value <= 1.0):
        raise ValueError(f"copying_score_factored out of range: {value}")
    return value


def assert_grid_informative(name: str, grid: np.ndarray, step: int) -> None:
    """Refuse to report a readout that cannot have come from a real pass.

    Same contract and rationale as diagnose_prereq_ab.py's guard of the
    same name. A constant grid across all 48 heads reads as a clean
    finding ("no head copies anywhere") and is the signature of a probe
    that never ran.
    """
    if grid.size == 0:
        raise ValueError(f"{name} grid at step {step} is empty")
    if not np.all(np.isfinite(grid)):
        raise ValueError(f"{name} grid at step {step} contains non-finite values")
    if float(np.max(grid) - np.min(grid)) < 1e-12:
        raise ValueError(
            f"{name} grid at step {step} is constant ({float(np.min(grid))}) across all "
            f"{grid.size} heads — probe likely never ran; refusing to report it"
        )


def measure(step: int, chunk: int, ind_layer: int, ind_head: int) -> dict[str, Any]:
    """Copying score for every head, and value spread for the candidate."""
    import torch

    from indbw.existence import value_spread
    from indbw.models import load_checkpoint

    model = load_checkpoint(step, device="cpu")
    model.eval()
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads

    # W_V/W_O are small (6 MB each across all heads) and are needed on
    # both sides, so they are taken now. W_U and W_E are vocab-sized
    # (103 MB each) and are taken *after* the forward pass, below.
    W_V_all = model.W_V.detach().numpy().copy()
    W_O_all = model.W_O.detach().numpy().copy()

    # --- value spread for the candidate head, FIRST ------------------
    # Ordered before the copying grid so the model can be freed before
    # the vocab-sized allocations start. Measured separately: the
    # forward pass peaks ~1.34 GB (diagnose_prereq_ab.py's recorded RSS)
    # and the copying loop ~0.50 GB; run concurrently they exceed a
    # t4g.small's 1.8 GB and fall into swap, run in sequence neither
    # does. This is the cheapest available way to keep CLAUDE.md's
    # "smallest thing that works" honest rather than sizing up.
    M_OV_cand = ov_matrix(W_V_all[ind_layer, ind_head], W_O_all[ind_layer, ind_head])
    eval_tokens = build_eval_tokens(N_EVAL, T, SEED, D_VOCAB)[:SPREAD_SEQUENCES]

    captured: dict[str, np.ndarray] = {}

    def pattern_hook(pattern: torch.Tensor, hook: object) -> torch.Tensor:
        captured["attn"] = pattern.detach().numpy()[:, ind_head].copy()  # [b, seq, seq]
        return pattern

    def resid_hook(x: torch.Tensor, hook: object) -> torch.Tensor:
        # ln1.hook_normalized fires three times per block (once per Q/K/V
        # projection) and all three firings are bit-identical -- pinned
        # by tests/unit/test_train.py since 2026-09-05. Overwriting is
        # therefore safe, which was previously true by accident.
        captured["resid"] = x.detach().numpy().copy()  # [b, seq, d_model]
        return x

    with torch.no_grad():
        model.run_with_hooks(
            eval_tokens,
            fwd_hooks=[
                (f"blocks.{ind_layer}.attn.hook_pattern", pattern_hook),
                (f"blocks.{ind_layer}.ln1.hook_normalized", resid_hook),
            ],
            return_type=None,
        )

    # Vocab-sized weights taken only now, so they never coexist with the
    # forward pass's activations. W_U is [d_model, vocab] and W_E is
    # [vocab, d_model] in TL; the hashed metric wants both [vocab, d_model].
    W_U = model.W_U.detach().numpy().T.copy()
    W_E = model.W_E.detach().numpy().copy()
    del model

    if "attn" not in captured or "resid" not in captured:
        raise ValueError("value-spread hooks never fired; refusing to report a spread")

    attn, resid = captured["attn"], captured["resid"]
    spreads: list[float] = []
    write_norms: list[float] = []
    for b in range(attn.shape[0]):
        # z_j = M_OV x_j for every key position j.
        z = resid[b] @ (M_OV_cand.T)  # [seq, d_model]
        write_norms.append(float(np.mean(np.linalg.norm(z, axis=1))))
        for i in range(attn.shape[1]):
            row = attn[b, i].astype(np.float64)
            row = row / row.sum()  # renormalize float32 softmax for the 1e-6 guard
            spreads.append(value_spread(row, z.astype(np.float64)))

    spread_arr = np.asarray(spreads)
    if not np.all(np.isfinite(spread_arr)):
        raise ValueError(f"non-finite value spread at step {step}")
    # clear(), not `del captured` — the name is closed over by the hooks
    # above, and deleting it makes those closures reference an unbound
    # name. Clearing drops the array references, which is the point.
    captured.clear()
    del attn, resid

    # --- copying score for every head, model now freed ---------------
    copy_grid = np.zeros((n_layers, n_heads))
    for layer in range(n_layers):
        for head in range(n_heads):
            # Factored path: 8x cheaper than composing M_OV first, and
            # pinned to the composed path (which is itself pinned to the
            # hashed metric) by tests. The composed path at 48 heads x 2
            # checkpoints does not fit the worker's 3-hour cap.
            copy_grid[layer, head] = copying_score_factored(
                W_U, W_V_all[layer, head], W_O_all[layer, head], W_E, chunk=chunk
            )
        print(
            f"  step {step} layer {layer} copying: "
            f"{np.array2string(copy_grid[layer], precision=5)}",
            flush=True,
        )
    assert_grid_informative("copying_score", copy_grid, step)

    return {
        "step": step,
        "copying_grid": copy_grid.tolist(),
        "candidate_copying": float(copy_grid[ind_layer, ind_head]),
        "max_copying": float(copy_grid.max()),
        "argmax_copying": [
            int(np.argmax(copy_grid) // n_heads),
            int(np.argmax(copy_grid) % n_heads),
        ],
        "spread_mean": float(spread_arr.mean()),
        "spread_max": float(spread_arr.max()),
        "spread_min": float(spread_arr.min()),
        "spread_median": float(np.median(spread_arr)),
        "mean_write_norm": float(np.mean(write_norms)),
        "n_spread_samples": int(spread_arr.size),
    }


def sync_to_s3(path: Path, bucket: str) -> None:
    """Best-effort backstop, same contract as g0_sweep.py's: never raises.

    Written after each checkpoint rather than at the end -- the exact
    gap that cost a 45-minute job to a spot reclaim on 2026-09-05
    (REVIEW.md, fixed for diagnose_g3_reachability.py in 60e28d8).
    """
    if not bucket:
        return
    try:
        subprocess.run(
            ["aws", "s3", "cp", str(path), f"s3://{bucket}/g3_gradient_gate/{path.name}"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate: a lost backup is not a lost cell
        print(f"WARNING: S3 sync of {path.name} failed ({exc}); continuing locally", flush=True)


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--chunk", type=int, default=512, help="vocab tokens per logit block")
    parser.add_argument("--steps", type=int, nargs="+", default=[A_STEP, B_STEP])
    args = parser.parse_args()

    bucket = os.environ.get("G0_S3_BUCKET", "")
    ind_layer, ind_head = induction_head_from_g0(B_STEP)
    print(f"candidate induction head from G0 at B: layer {ind_layer}, head {ind_head}", flush=True)
    print(f"chunk={args.chunk} steps={args.steps}\n", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    sha = git_sha()
    for step in args.steps:
        t0 = time.time()
        rec = measure(step, args.chunk, ind_layer, ind_head)
        rec.update(
            {
                "diagnostic": "g3_gradient_gate",
                "checkpoint": "A" if step == A_STEP else ("B" if step == B_STEP else str(step)),
                "induction_layer": ind_layer,
                "induction_head": ind_head,
                "n_eval": N_EVAL,
                "spread_sequences": SPREAD_SEQUENCES,
                "T": T,
                "eval_seed": SEED,
                "chunk": args.chunk,
                "git_sha": sha,
                "hardware": platform.machine(),
                "wall_clock_s": time.time() - t0,
            }
        )
        # Append-and-sync per checkpoint: a reclaim costs one checkpoint,
        # not the run (CLAUDE.md, "results must be written incrementally").
        with args.out.open("a") as fh:
            fh.write(json.dumps(rec, sort_keys=True) + "\n")
        sync_to_s3(args.out, bucket)
        print(
            f"\nstep {step}: candidate copying={rec['candidate_copying']:.5f} "
            f"max={rec['max_copying']:.5f} at {rec['argmax_copying']} | "
            f"spread mean={rec['spread_mean']:.4f} max={rec['spread_max']:.4f} "
            f"| write norm={rec['mean_write_norm']:.4f} "
            f"({rec['wall_clock_s']:.0f}s)\n",
            flush=True,
        )

    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
