"""How much training separates A from B, and how much G3 actually spent.

Answers a question none of the seven G3 diagnostics asked: not "did the
optimizer find the update?" but "was there remotely enough training for
it to?" Everything here is arithmetic over published constants and
already-committed run configs -- no model, no network, no GPU.

Pythia's own numbers (Biderman et al. 2023, PROJECT.md §12): every
pythia-70m step is 1024 sequences x 2048 tokens, and PROJECT.md §2's
A and B sit 1488 such steps apart. The comparison this makes possible is
the compute-budget ratio between the pretraining run that *did* form the
induction head and the LoRA run that was asked to reinstall it.

Two honest caveats on reading the ratio, both of which cut against
treating it as a bound:

  - Pythia's tokens buy general LM improvement, of which the induction
    circuit is a small fraction (PROJECT.md §1.1). A gradient aimed
    directly at an induction objective should be far more
    token-efficient, so the ratio is not "how much G3 needs".
  - Conversely G3 trains ~37k parameters, not 70M, and from a
    checkpoint that already has the prerequisite prev-token head (G1).

What the ratio does establish is scale: whether G3 was short by a factor
of two or by three orders of magnitude. That distinction decides whether
"more steps" is a plausible explanation for the plateau at all, and it
is not currently written down anywhere in the repo.

FLOPs use the standard 6ND forward+backward estimate (6 x params x
tokens), reported both on all 70M parameters and on the ~19M
non-embedding parameters, since the two conventions differ by 3.7x and
the literature is not consistent about which it means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]

# Pythia-70m pretraining constants (Biderman et al. 2023).
PYTHIA_SEQS_PER_STEP = 1024
PYTHIA_SEQ_LEN = 2048
PYTHIA_TOKENS_PER_STEP = PYTHIA_SEQS_PER_STEP * PYTHIA_SEQ_LEN

# PROJECT.md §2 / results/g0_transition.jsonl.
A_STEP, B_STEP = 512, 2000
# The next grid point after A. The transition is bracketed by A and this,
# not by A and B -- Pythia's grid jumps 512 -> 1000, so "bracket width 2
# checkpoints" (G0) is 488 pretraining steps wide in token terms.
FIRST_POST_A_STEP = 1000

N_PARAMS_TOTAL = 70_426_624
N_PARAMS_NON_EMBEDDING = 6 * 12 * 512**2  # 6 layers x 12 d_model^2 per layer

G2_RUN_ARTIFACT = REPO_ROOT / "data" / "g2_generous_rank_run.json"


def flops_6nd(n_params: int, n_tokens: float) -> float:
    """Standard forward+backward training-FLOP estimate."""
    return 6.0 * n_params * n_tokens


def per_token_flops(
    d_model: int = 512,
    n_layers: int = 6,
    d_vocab: int = 50304,
    seq_len: int = 2048,
    causal: bool = True,
) -> dict[str, float]:
    """Forward+backward FLOPs per token, split into the three terms that
    actually matter at pythia-70m's shape.

    `flops_6nd` with the full 70.4M parameter count is the convention the
    literature uses, and at this shape it is ~23% high. The reason is
    worth knowing before sizing a GPU run: 6ND assumes the weight matmuls
    dominate, and here they are only a third of the work. d_model is 512
    against a 50304 vocab, so the unembedding projection alone is ~45% of
    every token's cost; and d_model is 512 against a 2048 sequence, so
    attention's seq^2 term -- which 6ND ignores entirely -- is another
    ~12%. A GPU estimate built on 6ND alone is therefore both wrong and
    wrong in the direction that hides which kernel to optimize.

    `causal=True` (the default) halves the attention term, since a
    decoder computes only the lower triangle. This affects MFU and
    achieved-FLOP/s figures but *not* any wall-clock projection, which is
    extrapolated from measured tokens/s and never passes through a FLOP
    count.

    Ratios in `analyze` are unaffected either way, since both sides of
    every ratio use the same formula.
    """
    n_body = n_layers * 12 * d_model * d_model
    n_embed = d_vocab * d_model
    body = 3 * 2 * n_body
    # A decoder attends only to positions <= its own, so on average each
    # query touches seq/2 keys, not seq. Counting the full square
    # overstates the attention term by 2x -- and it is not a bookkeeping
    # nicety: SDPA and FlashAttention with is_causal skip the upper
    # triangle rather than computing and masking it, so the work really
    # is halved. Left switchable because the full-square count is what a
    # bidirectional encoder would cost, and because an implementation
    # that masks after the fact does pay the full price.
    attention = 3 * 2 * 2 * seq_len * d_model * n_layers
    if causal:
        attention /= 2
    unembedding = 3 * 2 * n_embed
    return {
        "body_matmuls": float(body),
        "attention_seq_squared": float(attention),
        "unembedding": float(unembedding),
        "total": float(body + attention + unembedding),
    }


def lora_trainable_params(rank: int, d_model: int = 512, d_head: int = 64) -> int:
    """B [d_model, r] + A [r, d_head] for the QK arm (indbw.train)."""
    return rank * d_model + rank * d_head


def analyze(g2: dict[str, Any]) -> dict[str, Any]:
    cfg = g2["config"]
    steps, batch, period = g2["steps_run"], cfg["batch_size"], cfg["T"]

    g3_tokens = steps * batch * 2 * period
    g3_tokens_per_step = batch * 2 * period
    # Only second-copy targets carry loss (indbw.train.second_copy_nll),
    # which is the quantity actually supplying gradient signal.
    g3_loss_positions = steps * batch * (period - 1)

    pythia_a_to_next = (FIRST_POST_A_STEP - A_STEP) * PYTHIA_TOKENS_PER_STEP
    pythia_a_to_b = (B_STEP - A_STEP) * PYTHIA_TOKENS_PER_STEP

    return {
        "pythia": {
            "tokens_per_step": PYTHIA_TOKENS_PER_STEP,
            "steps_A_to_next_grid_point": FIRST_POST_A_STEP - A_STEP,
            "steps_A_to_B": B_STEP - A_STEP,
            "tokens_A_to_next_grid_point": pythia_a_to_next,
            "tokens_A_to_B": pythia_a_to_b,
            "flops_A_to_next_grid_point_total_params": flops_6nd(N_PARAMS_TOTAL, pythia_a_to_next),
            "flops_A_to_next_grid_point_non_embedding": flops_6nd(
                N_PARAMS_NON_EMBEDDING, pythia_a_to_next
            ),
            "flops_A_to_B_total_params": flops_6nd(N_PARAMS_TOTAL, pythia_a_to_b),
        },
        "g3": {
            "steps": steps,
            "batch_size": batch,
            "seq_len": 2 * period,
            "tokens_per_step": g3_tokens_per_step,
            "tokens_total": g3_tokens,
            "loss_bearing_positions": g3_loss_positions,
            "trainable_params": lora_trainable_params(cfg["rank"]),
            "flops_total_params": flops_6nd(N_PARAMS_TOTAL, g3_tokens),
            "seconds_per_step_measured": g2["wall_clock_s"] / steps,
        },
        "ratios": {
            "tokens_pythia_bracket_over_g3": pythia_a_to_next / g3_tokens,
            "tokens_pythia_A_to_B_over_g3": pythia_a_to_b / g3_tokens,
            "batch_tokens_pythia_over_g3": PYTHIA_TOKENS_PER_STEP / g3_tokens_per_step,
            "flops_pythia_bracket_over_g3": pythia_a_to_next / g3_tokens,
        },
        "cpu_cost_to_close_the_token_gap_hours": (
            pythia_a_to_next / g3_tokens * g2["wall_clock_s"] / 3600.0
        ),
    }


def format_report(a: dict[str, Any]) -> str:
    p, g, r = a["pythia"], a["g3"], a["ratios"]
    return "\n".join(
        [
            "Pythia-70m pretraining, A (step 512) -> next grid point (step 1000):",
            f"  steps                     {p['steps_A_to_next_grid_point']:>15,}",
            f"  tokens                    {p['tokens_A_to_next_grid_point']:>15,}"
            f"  ({p['tokens_A_to_next_grid_point'] / 1e9:.2f}B)",
            f"  FLOPs (6ND, N=70.4M)      {p['flops_A_to_next_grid_point_total_params']:>15.2e}",
            f"  FLOPs (6ND, N=18.9M)      {p['flops_A_to_next_grid_point_non_embedding']:>15.2e}",
            "",
            f"Pythia-70m, full A -> B span ({p['steps_A_to_B']:,} steps):",
            f"  tokens                    {p['tokens_A_to_B']:>15,}"
            f"  ({p['tokens_A_to_B'] / 1e9:.2f}B)",
            f"  FLOPs (6ND, N=70.4M)      {p['flops_A_to_B_total_params']:>15.2e}",
            "",
            "G3 / G2 generous-rank LoRA run (data/g2_generous_rank_run.json):",
            f"  steps                     {g['steps']:>15,}",
            f"  tokens/step               {g['tokens_per_step']:>15,}"
            f"  (batch {g['batch_size']} x seq_len {g['seq_len']})",
            f"  tokens total              {g['tokens_total']:>15,}",
            f"  loss-bearing positions    {g['loss_bearing_positions']:>15,}",
            f"  trainable params          {g['trainable_params']:>15,}",
            f"  FLOPs (6ND, N=70.4M)      {g['flops_total_params']:>15.2e}",
            f"  measured s/step           {g['seconds_per_step_measured']:>15.2f}",
            "",
            "Ratios (pretraining : G3):",
            f"  tokens, A -> next grid pt {r['tokens_pythia_bracket_over_g3']:>15,.0f}x",
            f"  tokens, A -> B            {r['tokens_pythia_A_to_B_over_g3']:>15,.0f}x",
            f"  tokens per optimizer step {r['batch_tokens_pythia_over_g3']:>15,.0f}x",
            "",
            f"Matching the A -> next-grid-point token budget at G3's measured "
            f"{g['seconds_per_step_measured']:.1f} s/step",
            f"and batch size would take {a['cpu_cost_to_close_the_token_gap_hours']:,.0f} "
            "CPU-hours on the box G2 was timed on.",
        ]
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/training_budget_accounting.json")
    args = ap.parse_args()

    g2 = json.loads(G2_RUN_ARTIFACT.read_text())
    analysis = analyze(g2)
    print(format_report(analysis))

    out_path = REPO_ROOT / args.out
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n")
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
