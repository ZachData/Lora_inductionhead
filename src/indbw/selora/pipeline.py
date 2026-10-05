"""SAE-side measurement of one model: per-prompt FVU at every layer against base reference stats."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from indbw.selora import generate, probes
from indbw.selora.cells import Cell
from indbw.selora.sae import JumpReLUSAE, content_mask, fvu_per_seq, sq_dev_per_seq


def reference_stats(
    base_model: Any, ids: torch.Tensor, attn: torch.Tensor, layers: Sequence[int]
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer (mu, den) from the *base* model: the shared FVU scale for base and fine-tuned."""
    mask = content_mask(attn)
    cap = probes.capture_resid(base_model, ids, attn, layers)
    out = {}
    for li in layers:
        x = cap[li].double()
        mu = x[mask].mean(0)
        out[li] = (mu, sq_dev_per_seq(x, mu, mask))
    return out


def fvu_by_layer(
    model: Any,
    saes: Mapping[int, JumpReLUSAE],
    ids: torch.Tensor,
    attn: torch.Tensor,
    ref: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
) -> dict[int, torch.Tensor]:
    mask = content_mask(attn)
    cap = probes.capture_resid(model, ids, attn, list(saes))
    return {li: fvu_per_seq(cap[li].double(), saes[li], mask, *ref[li]) for li in saes}


@dataclass(frozen=True)
class CellData:
    """Everything a cell needs that is not the model: frozen, hashed upstream as the eval set."""

    train_ids: list[list[int]]
    pad_id: int
    sae_ids: torch.Tensor  # [n, T] held-out SAE prompts (BOS first), same for every cell
    sae_attn: torch.Tensor
    gsm_questions: list[str]
    gsm_golds: list[float]
    mmlu_prompts: list[str]
    mmlu_gold: list[str]
    gen_tokens_gsm: int = 256
    gen_tokens_mmlu: int = 32


def _fvu_lists(f: Mapping[int, torch.Tensor]) -> dict[str, list[float]]:
    return {str(k): [float(x) for x in v] for k, v in f.items()}


def run_cell(
    cell: Cell,
    model: Any,
    tok: Any,
    saes: Mapping[int, JumpReLUSAE],
    data: CellData,
    wall_budget_s: float,
) -> dict[str, Any]:
    """Train one adapter and take every measurement the claims need. `model` is consumed.

    Order matters: base-model reference stats and base FVU are taken *before* the adapter
    is attached, because attaching mutates the model in place.
    """
    from peft import get_peft_model

    from indbw.selora.placement import lora_config
    from indbw.selora.random_lora import delta_norms, set_matched_norm_random
    from indbw.selora.train import TrainConfig, train_lora

    layers = sorted(saes)
    ref = reference_stats(model, data.sae_ids, data.sae_attn, layers)
    base_fvu = fvu_by_layer(model, saes, data.sae_ids, data.sae_attn, ref)
    base_resid = probes.capture_resid(model, data.sae_ids, data.sae_attn, layers)
    base_resid = {k: v.clone() for k, v in base_resid.items()}

    row: dict[str, Any] = {
        "run_id": cell.run_id,
        "cell": asdict(cell),
        "fvu_base": _fvu_lists(base_fvu),
    }
    if cell.arm == "base":
        pm: Any = model
        row["loss_first"] = row["loss_last"] = None
    else:
        pm = get_peft_model(model, lora_config(cell.layers, cell.r, cell.alpha))
        losses = train_lora(
            pm,
            data.train_ids,
            data.pad_id,
            TrainConfig(
                lr=cell.lr, epochs=cell.epochs, seed=cell.seed, wall_clock_budget_s=wall_budget_s
            ),
        )
        row["loss_first"], row["loss_last"] = losses[0], losses[-1]

    ft_resid = probes.capture_resid(pm, data.sae_ids, data.sae_attn, layers)
    row["resid_max_abs_diff"] = {
        str(li): float((ft_resid[li] - base_resid[li]).abs().max()) for li in layers
    }
    row["fvu_ft"] = _fvu_lists(fvu_by_layer(pm, saes, data.sae_ids, data.sae_attn, ref))
    row["gsm_correct"] = generate.gsm8k_correct_flags(
        pm, tok, data.gsm_questions, data.gsm_golds, data.gen_tokens_gsm
    )
    row["mmlu_logprob"] = generate.mmlu_logprob_preds(pm, tok, data.mmlu_prompts)
    row["mmlu_gen"] = generate.mmlu_generation_preds(
        pm, tok, data.mmlu_prompts, data.gen_tokens_mmlu
    )
    row["mmlu_gold"] = data.mmlu_gold
    if cell.arm != "base":
        set_matched_norm_random(pm, delta_norms(pm), seed=cell.seed + 7919)
        row["fvu_rand"] = _fvu_lists(fvu_by_layer(pm, saes, data.sae_ids, data.sae_attn, ref))
    return row
