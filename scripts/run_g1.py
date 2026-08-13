"""G1: prerequisite check at checkpoint A (PROJECT.md §8).

Two independent checks, both at A's frozen weights:
  1. Does a previous-token head exist (prev-token score >= 0.3)?
  2. Does that head's output subspace overlap the candidate induction
     head's frozen W_K by more than chance (indbw.gates.k_composition_
     overlap's random-subspace null)?

The candidate induction head is identified from B's PMS, not A's -- A
is pre-transition by definition (PROJECT.md §2: "no head above PMS
0.1"), so A's own argmax-PMS head is noise, not a circuit. B's
argmax-PMS head (already computed by G0, data/g0_sweep*.jsonl) is the
head whose frozen-at-A W_K this check is actually about: it is the one
any QK-arm run would train Delta W_Q around.

Single checkpoint, one forward pass with the same fixed eval set as
G0 (PROJECT.md §5) -- "minutes," not a sweep.

Usage:
    python scripts/run_g1.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_PATH = REPO_ROOT / "results" / "g1_prerequisite.jsonl"

sys.path.insert(0, str(REPO_ROOT / "src"))

# Must match scripts/g0_sweep.py's defaults -- PROJECT.md §5's fixed,
# pre-registered eval set, held identical across every gate/row.
N_EVAL = 512
T = 128
SEED = 0
D_VOCAB = 50304  # fixed for pythia-70m across checkpoints


def build_eval_tokens(n_eval: int, T: int, seed: int, d_vocab: int) -> torch.Tensor:
    import numpy as np
    import torch

    rng = np.random.default_rng(seed)
    first_half = rng.integers(0, d_vocab, size=(n_eval, T))
    sequences = np.concatenate([first_half, first_half], axis=1)
    return torch.from_numpy(sequences).long()


def induction_head_from_g0(step: int) -> tuple[int, int]:
    """Read the argmax-PMS (layer, head) G0 already recorded for `step`."""
    for path in sorted(DATA_DIR.glob("g0_sweep*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec["step"] == step:
                layer, head = rec["max_pms_layer_head"]
                return int(layer), int(head)
    raise SystemExit(f"no G0 sweep record found for step {step} under {DATA_DIR}")


def compute_prev_token_scores(step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as np
    import torch

    from indbw.models import load_checkpoint
    from indbw.probes import prev_token_score

    model = load_checkpoint(step, device="cpu")
    model.eval()
    n_layers, n_heads = model.cfg.n_layers, model.cfg.n_heads
    eval_tokens = build_eval_tokens(N_EVAL, T, SEED, D_VOCAB)

    score_sum = np.zeros((n_layers, n_heads))

    def make_hook(layer: int):  # type: ignore[no-untyped-def]
        def hook(pattern: torch.Tensor, hook: object) -> torch.Tensor:
            pattern_np = pattern.detach().numpy()  # [b, heads, seq, seq]
            for head in range(n_heads):
                for bi in range(pattern_np.shape[0]):
                    score_sum[layer, head] += prev_token_score(pattern_np[bi, head])
            return pattern

        return hook

    fwd_hooks = [
        (f"blocks.{layer}.attn.hook_pattern", make_hook(layer)) for layer in range(n_layers)
    ]

    batch_size = 4
    n = eval_tokens.shape[0]
    with torch.no_grad():
        for start in range(0, n, batch_size):
            batch = eval_tokens[start : start + batch_size]
            # return_type=None skips the unembedding matmul entirely -- G1
            # only needs attention patterns, and computing full [b, seq,
            # vocab] logits every batch was enough to OOM this box's
            # ~1.8GB RAM (PROJECT.md §11), unlike g0_sweep.py which
            # actually needs the logits for ICL.
            model.run_with_hooks(batch, fwd_hooks=fwd_hooks, return_type=None)

    W_O = model.W_O.detach().numpy().copy()  # [n_layers, n_heads, d_head, d_model]
    W_K = model.W_K.detach().numpy().copy()  # [n_layers, n_heads, d_model, d_head]
    del model
    return score_sum / n, W_O, W_K


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def package_version(name: str) -> str:
    import importlib.metadata

    return importlib.metadata.version(name)


def main() -> None:
    from indbw.gates import k_composition_overlap, locate_prev_token_head
    from indbw.schema import METRIC_VERSION, Criterion, ResultsRecord, append_record

    a_step = 512  # PROJECT.md status board: G0 closed, A = step 512
    ind_layer, ind_head = induction_head_from_g0(2000)  # B = step 2000

    t0 = time.time()
    prev_token_scores, W_O, W_K = compute_prev_token_scores(a_step)
    wall_clock_s = time.time() - t0

    prev_result = locate_prev_token_head(prev_token_scores, threshold=0.3)
    overlap_result = k_composition_overlap(
        W_O[prev_result.layer, prev_result.head],
        W_K[ind_layer, ind_head],
        n_null_draws=100,
        percentile=95.0,
        seed=SEED,
    )

    observed = {
        "prev_token_layer": prev_result.layer,
        "prev_token_head": prev_result.head,
        "prev_token_score": prev_result.score,
        "induction_layer": ind_layer,
        "induction_head": ind_head,
        "overlap_ratio": overlap_result.ratio,
        "overlap_null_95th_percentile": overlap_result.null_percentile_value,
        "overlap_margin": overlap_result.ratio - overlap_result.null_percentile_value,
    }
    criteria = (
        Criterion(metric="prev_token_score", op=">=", threshold=0.3),
        Criterion(metric="overlap_margin", op=">", threshold=0.0),
    )
    verdict = "pass" if all(c.holds(observed) for c in criteria) else "fail"

    config = {
        "a_step": a_step,
        "induction_head_source_step": 2000,
        "n_eval": N_EVAL,
        "T": T,
        "seed": SEED,
        "n_null_draws": 100,
        "null_percentile": 95.0,
    }
    run_config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    eval_set_hash = hashlib.sha256(
        json.dumps({"n_eval": N_EVAL, "T": T, "seed": SEED}, sort_keys=True).encode()
    ).hexdigest()[:16]

    record = ResultsRecord(
        row="G1",
        null_tested="no previous-token head exists at A (score < 0.3), "
        "or the candidate induction head's frozen W_K has no more overlap "
        "with the previous-token head's output subspace than a random subspace of matching rank",
        criteria=criteria,
        observed=observed,
        verdict=verdict,
        metric_version=METRIC_VERSION,
        git_sha=git_sha(),
        run_config_hash=run_config_hash,
        seed=SEED,
        checkpoint_revision=f"step {a_step} (A)",
        eval_set_hash=eval_set_hash,
        torch_version=package_version("torch"),
        numpy_version=package_version("numpy"),
        transformer_lens_version=package_version("transformer_lens"),
        wall_clock_s=wall_clock_s,
        hardware=f"{platform.machine()}/{platform.processor() or 'unknown'}",
    )

    append_record(OUT_PATH, record)
    print(f"G1 verdict: {verdict}")
    print(
        f"  prev-token head: layer={prev_result.layer} head={prev_result.head} "
        f"score={prev_result.score:.4f} (found={prev_result.found})"
    )
    print(
        f"  induction head (from B): layer={ind_layer} head={ind_head} "
        f"overlap_ratio={overlap_result.ratio:.4f} "
        f"null_95th={overlap_result.null_percentile_value:.4f} "
        f"(significant={overlap_result.significant})"
    )
    print(f"  self-consistent: {record.is_self_consistent()}")
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
