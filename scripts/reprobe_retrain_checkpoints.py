"""Re-probe the retrain's onset checkpoints at N_eval=512, all heads.

Authorized 2026-09-07. Forward-only: no training, no GPU, no new metric.
Every readout is an existing hashed function from `indbw.probes` applied
to a checkpoint that already exists. No verdict is produced or implied.

WHY
---
The 2026-09-01 retrain (PROJECT.md 10, 2026-09-07) left 82 checkpoints
at stride 4 over steps 528-852, bracketing induction onset at step 620.
Its own probe trace has two defects that block downstream use, and
neither is fixed by retraining:

  n_eval=128, not 5's N_eval=512
      Every committed record (G0, G1, G2, G3) uses 512. The trace's
      numbers therefore carry ~2x the sampling noise and are not
      directly comparable to any of them. D1 compares subspaces against
      a pretraining delta and would inherit that mismatch.

  head (3,6) only
      11 (2026-09-03) found three layer-3 heads with prefix matching at
      B -- (3,6) 0.9652, (3,1) 0.8993, (3,0) 0.5094 -- and asks whether
      induction there is carried by one head or three. The trace cannot
      answer it: it logged one head.

Both are properties of the *probe*, not of the weights, so re-probing
the existing checkpoints fixes them at zero GPU cost. This is the cheap
path; a contiguous re-run costs ~9.4 GPU-hours and would produce a
different trajectory (Adam is cold-started at 512 and the batch order is
not Pythia's), so it answers a different question -- a second sample --
rather than a cleaner version of this one.

WHAT IT DOES NOT FIX
--------------------
Steps 856 and 860 are excluded: onset_bracket.json records that session
2's resume overwrote session 1's files at those paths, so they belong to
a different trajectory. The seams at 852 and 1528 are untouched by this
script and by anything downstream of it, because the onset window ends
at 852.

OBSERVABILITY
-------------
This project cannot read a worker's console (`ec2:GetConsoleOutput` is
denied to the role, HANDOFF.md), so a worker that dies mid-run has
historically left nothing behind but the absence of a commit. Every
checkpoint here appends to a log AND syncs both the log and the results
to S3 before moving on, so progress is inspectable live and a reclaim
costs one checkpoint rather than the shard. A top-level handler logs and
syncs the traceback before re-raising.

Usage:
    python scripts/reprobe_retrain_checkpoints.py --shard 0 --n-shards 7
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

MODEL_NAME = "EleutherAI/pythia-70m"
RUN_ID = "cb25e3f6c2185c1e"
S3_CKPT_PREFIX = f"s3://research-vm-shared-176048535722/retrain/{RUN_ID}/ckpt"

# The clean onset bracket. onset_bracket.json: session 1 wrote 528-860,
# of which 856 and 860 were later overwritten by session 2's resume and
# are a different trajectory. 528-852 inclusive at stride 4 is 82 files.
BRACKET_FIRST, BRACKET_LAST, BRACKET_STRIDE = 528, 852, 4

# 5's canonical eval set. Not configurable: the entire point is to match
# the committed records, and a flag inviting a smaller value would make
# that silently optional.
N_EVAL, T, EVAL_SEED = 512, 128, 0


def bracket_steps() -> list[int]:
    return list(range(BRACKET_FIRST, BRACKET_LAST + 1, BRACKET_STRIDE))


def shard_steps(steps: list[int], shard: int, n_shards: int) -> list[int]:
    """Steps assigned to `shard`, round-robin.

    Round-robin rather than contiguous blocks so that a shard that dies
    leaves gaps spread across the bracket instead of removing a
    contiguous window -- with stride-4 data an interleaved gap is still
    a usable curve, a missing block is not.

    Every step lands in exactly one shard, which is what the tests pin:
    an off-by-one here silently drops or double-counts checkpoints, and
    a duplicated step would be written twice with the same run key and
    look like a resumability success rather than a bug.
    """
    if n_shards < 1:
        raise ValueError(f"n_shards must be >= 1, got {n_shards}")
    if not 0 <= shard < n_shards:
        raise ValueError(f"shard must be in [0, {n_shards}), got {shard}")
    return [s for i, s in enumerate(steps) if i % n_shards == shard]


class Logger:
    """Append-and-sync log. The only channel out of a worker this project
    can read while it is still running."""

    def __init__(self, path: Path, bucket: str, prefix: str) -> None:
        self.path = path
        self.bucket = bucket
        self.prefix = prefix
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str, sync: bool = False) -> None:
        stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        line = f"{stamp}  {msg}"
        print(line, flush=True)
        with self.path.open("a") as fh:
            fh.write(line + "\n")
        if sync:
            sync_to_s3(self.path, self.bucket, self.prefix)


def sync_to_s3(path: Path, bucket: str, prefix: str) -> None:
    """Best-effort, never raises -- same contract as g0_sweep.py's."""
    if not bucket or not path.exists():
        return
    try:
        subprocess.run(
            ["aws", "s3", "cp", str(path), f"s3://{bucket}/{prefix}/{path.name}"],
            check=True,
            capture_output=True,
            timeout=60,
        )
    except Exception as exc:  # noqa: BLE001 -- a lost backup is not a lost cell
        print(f"WARNING: S3 sync of {path.name} failed ({exc})", flush=True)


def fetch_checkpoint(step: int, dest: Path, log: Logger) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    uri = f"{S3_CKPT_PREFIX}/step_{step:06d}.pt"
    t0 = time.time()
    subprocess.run(["aws", "s3", "cp", uri, str(dest)], check=True, capture_output=True)
    log(f"  fetched {uri} ({dest.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
    return dest


def build_model(config: Any) -> Any:
    from transformers import GPTNeoXForCausalLM

    # eager attention is required for output_attentions; sdpa silently
    # returns attentions=None, which probe code then reads as an empty
    # pattern rather than an error.
    config._attn_implementation = "eager"
    return GPTNeoXForCausalLM(config)


def probe_all_heads(model: Any, eval_tokens: Any, batch_size: int = 2) -> dict[str, Any]:
    """PMS and prev-token score for every (layer, head), plus the ICL
    decomposition, off one pass over 5's fixed eval set.

    Mirrors `retrain.probe_induction`'s reading of the HF model -- same
    hashed probes, same eval set, same NLL windowing -- but returns the
    whole head grid instead of one cell. probe_induction is left
    untouched: the retrain harness depends on it.
    """
    import torch

    from indbw.probes import icl_score, prefix_matching_score, prev_token_score
    from indbw.train import first_copy_nll, second_copy_nll

    n_layers = model.config.num_hidden_layers
    n_heads = model.config.num_attention_heads
    n = eval_tokens.shape[0]
    pms_sum = np.zeros((n_layers, n_heads))
    prev_sum = np.zeros((n_layers, n_heads))
    nll_first = np.empty(n)
    nll_second = np.empty(n)

    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = eval_tokens[start : start + batch_size]
            out = model(batch, output_attentions=True)
            if out.attentions is None or len(out.attentions) != n_layers:
                raise ValueError(
                    f"expected {n_layers} attention tensors, got "
                    f"{None if out.attentions is None else len(out.attentions)} — "
                    "attn_implementation must be 'eager'"
                )
            logits = out.logits.float().cpu()
            tok = batch.cpu()
            nll_first[start : start + tok.shape[0]] = first_copy_nll(logits, tok, T).numpy()
            nll_second[start : start + tok.shape[0]] = second_copy_nll(logits, tok, T).numpy()
            for layer in range(n_layers):
                patt = out.attentions[layer].double().cpu().numpy()  # [b, heads, s, s]
                for head in range(n_heads):
                    for bi in range(patt.shape[0]):
                        pms_sum[layer, head] += prefix_matching_score(patt[bi, head], T)
                        prev_sum[layer, head] += prev_token_score(patt[bi, head])
            del out, logits

    pms = pms_sum / n
    prev = prev_sum / n
    for name, grid in (("pms", pms), ("prev_token", prev)):
        if not np.all(np.isfinite(grid)):
            raise ValueError(f"{name} grid has non-finite entries")
        if float(np.max(grid) - np.min(grid)) < 1e-12:
            raise ValueError(
                f"{name} grid is constant ({float(np.min(grid))}) across all "
                f"{grid.size} heads — the probe did not run; refusing to report it"
            )
    return {
        "pms_grid": pms.tolist(),
        "prev_token_grid": prev.tolist(),
        "icl": icl_score(nll_first, nll_second),
        "nll_first": float(nll_first.mean()),
        "nll_second": float(nll_second.mean()),
    }


def git_sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--shard", type=int, required=True)
    ap.add_argument("--n-shards", type=int, required=True)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    bucket = os.environ.get("G0_S3_BUCKET", "")
    prefix = "reprobe"
    out = args.out or REPO_ROOT / "data" / f"reprobe_shard{args.shard}.jsonl"
    log = Logger(REPO_ROOT / "data" / f"reprobe_shard{args.shard}.log", bucket, prefix)

    mine = shard_steps(bracket_steps(), args.shard, args.n_shards)
    log(
        f"shard {args.shard}/{args.n_shards}: {len(mine)} checkpoints {mine[:3]}...{mine[-1]}", True
    )

    # Resumability: skip steps already in the output (CLAUDE.md).
    done: set[int] = set()
    if out.exists():
        with out.open() as fh:
            done = {json.loads(line)["step"] for line in fh if line.strip()}
        log(f"resuming: {len(done)} already done")

    try:
        from transformers import AutoConfig

        from indbw.evalset import build_eval_tokens
        import torch

        config = AutoConfig.from_pretrained(MODEL_NAME, revision="step512")
        eval_tokens = build_eval_tokens(N_EVAL, T, EVAL_SEED, int(config.vocab_size))
        log(
            f"eval set {tuple(eval_tokens.shape)} seed={EVAL_SEED}; config {MODEL_NAME}@step512",
            True,
        )
        sha = git_sha()
        tmp = REPO_ROOT / "data" / f"_ckpt_shard{args.shard}.pt"

        for i, step in enumerate(mine):
            if step in done:
                log(f"[{i + 1}/{len(mine)}] step {step}: already done, skipping")
                continue
            t0 = time.time()
            log(f"[{i + 1}/{len(mine)}] step {step}: starting")
            fetch_checkpoint(step, tmp, log)
            state = torch.load(tmp, map_location="cpu", weights_only=True)
            model = build_model(config)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                raise ValueError(f"state_dict mismatch at {step}: {missing=} {unexpected=}")
            del state
            tmp.unlink(missing_ok=True)

            rec = probe_all_heads(model, eval_tokens, args.batch_size)
            del model
            rec.update(
                {
                    "diagnostic": "reprobe_retrain_checkpoints",
                    "run_id": RUN_ID,
                    "step": step,
                    "n_eval": N_EVAL,
                    "T": T,
                    "eval_seed": EVAL_SEED,
                    "shard": args.shard,
                    "n_shards": args.n_shards,
                    "git_sha": sha,
                    "hardware": platform.machine(),
                    "wall_clock_s": time.time() - t0,
                }
            )
            with out.open("a") as fh:
                fh.write(json.dumps(rec, sort_keys=True) + "\n")
            sync_to_s3(out, bucket, prefix)
            g = np.asarray(rec["pms_grid"])
            am = int(np.argmax(g))
            log(
                f"  step {step}: max PMS {g.max():.4f} at "
                f"({am // g.shape[1]},{am % g.shape[1]}), (3,6)={g[3][6]:.4f}, "
                f"ICL={rec['icl']:.4f}  [{rec['wall_clock_s']:.0f}s]",
                True,
            )
        log(f"shard {args.shard} COMPLETE: {len(mine)} checkpoints", True)
    except Exception:
        log("FAILED with traceback:\n" + traceback.format_exc(), True)
        raise


if __name__ == "__main__":
    main()
