"""G3-plateau diagnostic #7 (+ #8, localization): is the QK arm's
criterion reachable at all, and if grafting block 3 alone isn't enough,
where outside it does the missing structure live?

Six optimization diagnostics (lr, rank, frozen-W_K, target head, full
fine-tune, objective/steps) came back negative, all sharing the same
R ~= 0.01 floor. Diagnostic #7 asked the prior question constructively:
does a solution exist inside the QK arm's search space at all? Checkpoint
B *has* the induction circuit, so grafting B's own weights into A is a
known-good witness, no training involved.

  qk_q      W_Q[3,6]  <- B        the exact object the QK arm trains.
                                  An upper bound on what *any* Delta W_Q
                                  can reach from A: if this doesn't move
                                  R, no rank and no step budget can.
  qk_qk     W_Q,W_K   <- B        the full prefix-matching half.
  ov        W_V,W_O   <- B        the full copying half.
  head      all four  <- B        the whole induction head.
  layer_host  block 3 <- B        head + MLP + LN of the host block.
  full      everything <- B       sanity: must give R == 1 by construction.

Result (PROJECT.md §10, 2026-09-03): `qk_q` = -0.0006, `head` = 0.0046,
`layer_host` = 0.0077 -- the same ~0.01 floor as every optimization
diagnostic -- while `full` = 1.0004. The missing structure is outside
block 3 entirely, and nothing between `layer_host` and `full` was
measured. Diagnostic #8 below localizes that gap, guided by the prereq
A/B contrast (PROJECT.md §10, 2026-09-03): checkpoint A's previous-token
head (layer 2, head 1) is 2.6-15x weaker than B's, and it is upstream of
block 3 -- the most concrete lead available for where to look first.

  layer_host_plus_prevtok_head   block 3 + layer2/head1 <- B  sharpest
                                  test of "the layer-2 prerequisite is
                                  the bottleneck" as a standalone claim.
  layer_host_plus_block2         block 3 + block 2 (all of it) <- B
  layer_host_plus_pre            block 3 + blocks 0,1,2 <- B
  layer_host_plus_post           block 3 + blocks 4,5 <- B
  layer_host_plus_embed_unembed  block 3 + embed/unembed <- B
  layer_host_plus_ln_final       block 3 + ln_final <- B

Each localization cell starts from clean A and grafts block 3 (as
`layer_host` does) plus exactly the named extra component -- not
cumulative across cells, so cells can be read independently and run in
any order or subset.

Reported per cell alongside R: PMS on head (3,6) and the NLL_first /
NLL_second decomposition R is built from, so a cell that moves prefix
matching without moving ICL is visible as such rather than collapsing
into one number (PROJECT.md §4: "a lumped metric cannot tell you which
was the bottleneck" -- G3's criterion is exactly such a lumped metric).

Diagnostic only. No protocol change, no results record, no status-board
row: PROJECT.md §6's thresholds and §8's gates are untouched, and what to
conclude from this is a human call (CLAUDE.md falsification discipline).
PROJECT.md §11 (2026-09-03 "UPDATE") flags the localization cells as
needing sign-off before being run against real checkpoints -- this
change adds the cells and their tests only; nothing here launches a
worker or touches data/g3_reachability_diag.jsonl.

REVIEW.md (2026-09-05): a 10-cell job on this script previously lost an
entire ~45-minute run to a mid-run spot reclaim, because the only sync
point was after every requested cell finished -- each cell's JSON line
was flushed to local disk immediately, but nothing shipped it off-box
before the whole command exited. Now synced to S3 after every cell
(same best-effort, non-raising pattern as g0_sweep.py's sync_to_s3),
so a reclaim costs at most the in-flight cell, not the run -- matching
CLAUDE.md's resumability principle instead of relying on hand-sharding
into one-cell-per-worker jobs to get the same property operationally.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from indbw.evalset import build_eval_tokens
from indbw.probes import icl_score, prefix_matching_score, recovery
from indbw.train import first_copy_nll, second_copy_nll

REPO_ROOT = Path(__file__).resolve().parents[1]

A_STEP, B_STEP = 512, 2000
LAYER, HEAD = 3, 6
T = 128
D_VOCAB = 50304
EVAL_SEED = 0
# ICL(A)/ICL(B) as measured by the G0 sweep and reused by G2/G3 as the
# recovery baselines -- not recomputed here, so R is on exactly the same
# scale as every prior diagnostic (data/g2_generous_rank_run.json).
ICL_A = -0.024705959483981133
ICL_B = 11.512047719210386

# Which parameter tensors each graft cell copies from B into A. Names are
# HookedTransformer state-dict keys; `W_Q` etc. are per-layer tensors of
# shape [n_heads, ...], sliced to HEAD where the cell is head-scoped.
HEAD_SCOPED = {
    "qk_q": ["W_Q", "b_Q"],
    "qk_qk": ["W_Q", "b_Q", "W_K", "b_K"],
    "ov": ["W_V", "b_V", "W_O"],
    "head": ["W_Q", "b_Q", "W_K", "b_K", "W_V", "b_V", "W_O"],
}

# Localization cells (diagnostic #8): each grafts block LAYER (as
# layer_host does) plus exactly the named extra component. The prev-token
# head location (layer 2, head 1) matches G1's finding (PROJECT.md §10,
# 2026-08-13/2026-09-03), not the general HEAD/LAYER constants above.
PREVTOK_LAYER, PREVTOK_HEAD = 2, 1
LOCALIZATION_EXTRA_LAYERS = {
    "layer_host_plus_block2": (2,),
    "layer_host_plus_pre": (0, 1, 2),
    "layer_host_plus_post": (4, 5),
}
LOCALIZATION_GLOBAL_KEYS = {
    "layer_host_plus_embed_unembed": ["embed.W_E", "unembed.W_U", "unembed.b_U"],
    "layer_host_plus_ln_final": ["ln_final.w", "ln_final.b"],
}
LOCALIZATION_PREVTOK_HEAD_CELLS = ("layer_host_plus_prevtok_head",)

CELLS = [
    "base_a",
    "qk_q",
    "qk_qk",
    "ov",
    "head",
    "layer_host",
    "full",
    "base_b",
    *LOCALIZATION_EXTRA_LAYERS,
    *LOCALIZATION_GLOBAL_KEYS,
    *LOCALIZATION_PREVTOK_HEAD_CELLS,
]


def sync_to_s3(path: Path, bucket: str) -> None:
    """Best-effort push of `path` to S3, keyed by filename -- same
    contract as g0_sweep.py's sync_to_s3: a durability backstop against
    the instance dying before its final git push, never the source of
    truth, and it must never raise (a transient S3/network failure is a
    lost backup, not a lost cell). Empty bucket is a no-op.
    """
    if not bucket:
        return
    try:
        subprocess.run(
            ["aws", "s3", "cp", str(path), f"s3://{bucket}/g3_reachability/{path.name}"],
            check=True,
            capture_output=True,
            timeout=30,
        )
    except Exception as exc:  # noqa: BLE001 -- deliberate, see docstring: must never propagate
        print(f"WARNING: S3 sync of {path.name} failed ({exc}); continuing locally", flush=True)


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def graft(
    model_dst: Any,
    sd_src: dict[str, torch.Tensor],
    cell: str,
    layer: int = LAYER,
    head: int = HEAD,
) -> None:
    """Mutate `model_dst`'s weights in place, copying in `sd_src`'s values
    for exactly the tensors (and, where head-scoped, the exactly one head
    slice) that `cell` names. Everything else is left untouched.

    The "left untouched" half is the load-bearing half and the reason
    tests/unit/test_diagnose_g3_reachability.py checks it tensor by
    tensor: a graft that copied a whole [n_heads, ...] tensor instead of
    one head's slice, or copied nothing at all, would still produce a
    perfectly plausible-looking R and PMS. Nothing in the reported
    numbers would reveal it.
    """
    if cell not in CELLS:
        raise ValueError(f"unknown cell {cell!r}, expected one of {CELLS}")
    if cell == "base_a":
        return
    if cell in ("full", "base_b"):
        model_dst.load_state_dict(sd_src)
        return
    sd_dst = model_dst.state_dict()
    if cell == "layer_host":
        _copy_block(sd_dst, sd_src, layer)
        return
    if cell in LOCALIZATION_EXTRA_LAYERS:
        _copy_block(sd_dst, sd_src, layer)
        for extra_layer in LOCALIZATION_EXTRA_LAYERS[cell]:
            _copy_block(sd_dst, sd_src, extra_layer)
        return
    if cell in LOCALIZATION_GLOBAL_KEYS:
        _copy_block(sd_dst, sd_src, layer)
        for key in LOCALIZATION_GLOBAL_KEYS[cell]:
            sd_dst[key].copy_(sd_src[key])
        return
    if cell in LOCALIZATION_PREVTOK_HEAD_CELLS:
        _copy_block(sd_dst, sd_src, layer)
        for short in HEAD_SCOPED["head"]:
            key = f"blocks.{PREVTOK_LAYER}.attn.{short}"
            sd_dst[key][PREVTOK_HEAD].copy_(sd_src[key][PREVTOK_HEAD])
        return
    for short in HEAD_SCOPED[cell]:
        key = f"blocks.{layer}.attn.{short}"
        # W_O is [n_heads, d_head, d_model]; every other tensor here is
        # [n_heads, d_model, d_head] or [n_heads, d_head]. All are indexed
        # by head on dim 0, so one slice rule covers them.
        sd_dst[key][head].copy_(sd_src[key][head])


def _copy_block(
    sd_dst: dict[str, torch.Tensor], sd_src: dict[str, torch.Tensor], block: int
) -> None:
    """Copy every tensor belonging to `blocks.{block}.` -- attn (all
    heads), MLP, and both LayerNorms -- from source into destination.
    """
    prefix = f"blocks.{block}."
    for k, v in sd_src.items():
        if k.startswith(prefix):
            sd_dst[k].copy_(v)


@torch.no_grad()
def evaluate(
    model: Any,
    tokens: torch.Tensor,
    batch_size: int,
    layer: int = LAYER,
    head: int = HEAD,
    period: int = T,
    icl_a: float = ICL_A,
    icl_b: float = ICL_B,
) -> dict[str, float]:
    """R, PMS on (`layer`, `head`), and the NLL halves R is built from.

    Returning the halves alongside R is the point, not decoration: R is a
    lumped metric (PROJECT.md §4) and a graft that installs prefix
    matching without installing copying moves PMS a long way while
    leaving ICL, and therefore R, flat. Reporting only R would render
    that case indistinguishable from a graft that did nothing at all --
    which is precisely the ambiguity six prior G3 diagnostics ran into.
    """
    n = tokens.shape[0]
    nll_first = np.empty(n)
    nll_second = np.empty(n)
    pms_vals: list[float] = []
    pattern_name = f"blocks.{layer}.attn.hook_pattern"
    for start in range(0, n, batch_size):
        batch = tokens[start : start + batch_size]
        logits, cache = model.run_with_cache(
            batch, names_filter=[pattern_name], return_type="logits"
        )
        nll_first[start : start + batch.shape[0]] = first_copy_nll(logits, batch, period).numpy()
        nll_second[start : start + batch.shape[0]] = second_copy_nll(logits, batch, period).numpy()
        patt = cache[pattern_name][:, head].to(torch.float64).numpy()  # [b, 2T, 2T]
        pms_vals.extend(prefix_matching_score(p, period) for p in patt)
        del cache
    icl = icl_score(nll_first, nll_second)
    return {
        "recovery": recovery(icl, icl_a, icl_b),
        "icl": icl,
        "nll_first": float(nll_first.mean()),
        "nll_second": float(nll_second.mean()),
        "pms": float(np.mean(pms_vals)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cells", nargs="*", default=CELLS, choices=CELLS)
    ap.add_argument("--n-eval", type=int, default=128)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--out", default="data/g3_reachability_diag.jsonl")
    args = ap.parse_args()

    from indbw.models import load_checkpoint

    print(f"loading A (step {A_STEP}) and B (step {B_STEP})...", flush=True)
    model_a = load_checkpoint(A_STEP, device="cpu")
    model_b = load_checkpoint(B_STEP, device="cpu")
    sd_b = {k: v.clone() for k, v in model_b.state_dict().items()}
    sd_a_pristine = {k: v.clone() for k, v in model_a.state_dict().items()}
    del model_b

    tokens = build_eval_tokens(args.n_eval, T, EVAL_SEED, D_VOCAB)
    out_path = REPO_ROOT / args.out
    sha = git_sha()

    for cell in args.cells:
        model_a.load_state_dict(sd_a_pristine)  # every cell starts from clean A
        graft(model_a, sd_b, cell)
        t0 = time.time()
        obs = evaluate(model_a, tokens, args.batch_size)
        rec: dict[str, Any] = {
            "diagnostic": "g3_reachability_graft",
            "cell": cell,
            "grafted_from": f"step {B_STEP} (B)" if cell != "base_a" else "(none)",
            "layer": LAYER,
            "head": HEAD,
            "n_eval": args.n_eval,
            "T": T,
            "eval_seed": EVAL_SEED,
            "wall_clock_s": time.time() - t0,
            "git_sha": sha,
            "hardware": f"{platform.machine()}/{platform.processor() or platform.machine()}",
            **obs,
        }
        print(
            f"  {cell:8s} R={obs['recovery']:8.4f}  PMS={obs['pms']:.4f}  "
            f"ICL={obs['icl']:7.3f}  nll1={obs['nll_first']:6.3f} "
            f"nll2={obs['nll_second']:6.3f}  ({rec['wall_clock_s']:.0f}s)",
            flush=True,
        )
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("a") as f:
            f.write(json.dumps(rec) + "\n")
            f.flush()
            os.fsync(f.fileno())
        sync_to_s3(out_path, os.environ.get("G0_S3_BUCKET", ""))

    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
