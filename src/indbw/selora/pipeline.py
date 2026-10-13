"""SAE-side measurement of one model: per-prompt FVU at every layer against base reference stats."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import torch

from indbw.selora import generate, probes
from indbw.selora.cells import Cell
from indbw.selora.sae import JumpReLUSAE, content_mask, fvu_per_seq, sq_dev_per_seq


TOKEN_BUDGET = 2000  # padded tokens per forward pass: bounds activation memory on a 10 GB card


def _batches(attn: torch.Tensor, budget: int) -> list[tuple[torch.Tensor, int]]:
    """Index batches over prompts sorted by length, each with its trim length.

    Padding is on the right and attention is causal, so trimming a batch to its longest real prompt
    changes no per-sequence value; it only removes padding compute. Sorting keeps batches tight."""
    lens = attn.sum(1)
    order = torch.argsort(lens, stable=True).cpu()
    lens = lens.cpu()
    out: list[tuple[torch.Tensor, int]] = []
    i = 0
    while i < len(order):
        j = i + 1
        while j < len(order) and (j + 1 - i) * int(lens[order[j]]) <= budget:
            j += 1
        idx = order[i:j]
        out.append((idx, int(lens[idx].max())))
        i = j
    return out


def reference_stats(
    base_model: Any,
    ids: torch.Tensor,
    attn: torch.Tensor,
    layers: Sequence[int],
    budget: int = TOKEN_BUDGET,
) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
    """Per-layer (mu, den) from the *base* model: the shared FVU scale for base and fine-tuned.

    Two passes over batches: mu is the mean over every content token of every prompt, so it needs
    the whole set before any per-sequence deviation can be taken."""
    dev = next(base_model.parameters()).device
    ids, attn = ids.to(dev), attn.to(dev)
    mask = content_mask(attn)
    batches = _batches(attn, budget)
    total: dict[int, torch.Tensor] = {}
    count = 0
    for idx, T in batches:
        cap = probes.capture_resid(base_model, ids[idx, :T], attn[idx, :T], list(layers))
        m = mask[idx, :T]
        count += int(m.sum())
        for li in layers:
            x = cap[li].double()[m].sum(0)
            total[li] = x if li not in total else total[li] + x
    mus = {li: total[li] / count for li in layers}
    dens = {li: torch.zeros(len(ids), dtype=torch.float64, device=mus[li].device) for li in layers}
    for idx, T in batches:
        cap = probes.capture_resid(base_model, ids[idx, :T], attn[idx, :T], list(layers))
        for li in layers:
            dens[li][idx.to(dens[li].device)] = sq_dev_per_seq(
                cap[li].double(), mus[li], mask[idx, :T]
            )
    return {li: (mus[li], dens[li]) for li in layers}


def fvu_by_layer(
    model: Any,
    saes: Mapping[int, JumpReLUSAE],
    ids: torch.Tensor,
    attn: torch.Tensor,
    ref: Mapping[int, tuple[torch.Tensor, torch.Tensor]],
    budget: int = TOKEN_BUDGET,
) -> dict[int, torch.Tensor]:
    dev = next(model.parameters()).device
    ids, attn = ids.to(dev), attn.to(dev)
    mask = content_mask(attn)
    out = {li: torch.zeros(len(ids), dtype=torch.float64, device=ref[li][1].device) for li in saes}
    for idx, T in _batches(attn, budget):
        cap = probes.capture_resid(model, ids[idx, :T], attn[idx, :T], list(saes))
        for li in saes:
            mu, den = ref[li]
            di = idx.to(den.device)
            out[li][di] = fvu_per_seq(cap[li].double(), saes[li], mask[idx, :T], mu, den[di])
        del cap
    return out


@dataclass(frozen=True)
class CellData:
    """Everything a cell needs that is not the model: frozen, hashed upstream as the eval set."""

    train_ids: list[list[int]]
    pad_id: int
    sae_ids: torch.Tensor  # [n, T] held-out SAE prompts (BOS first), same for every cell
    sae_attn: torch.Tensor
    gsm_questions: list[str]
    gsm_golds: list[float]
    mmlu_logprob_prompts: list[str]  # full MMLU test set: one forward pass each
    mmlu_logprob_gold: list[str]
    mmlu_gen_prompts: list[str]  # MMLU_GEN_N subset: generation is the expensive protocol
    mmlu_gen_gold: list[str]
    train_ids_shuffled: list[list[int]] | None = None  # shuffled-answer arm only
    gen_tokens_gsm: int = 256
    gen_tokens_mmlu: int = 32


def max_abs_resid_diff(
    pm: Any, ids: torch.Tensor, attn: torch.Tensor, layers: Sequence[int], budget: int = TOKEN_BUDGET
) -> dict[str, float]:
    """Per-layer max |resid(adapted) - resid(base)|: proves the adapter changed the stream.

    Base residuals come from the same weights with the adapter disabled, batch by batch, so no
    full-set copy is held (26 layers x all prompts does not fit in memory). A model with no
    adapter (the base arm) is compared with itself and gives exactly 0."""
    dev = next(pm.parameters()).device
    ids, attn = ids.to(dev), attn.to(dev)
    out = {str(li): 0.0 for li in layers}
    for idx, T in _batches(attn, budget):
        ft = probes.capture_resid(pm, ids[idx, :T], attn[idx, :T], list(layers))
        if hasattr(pm, "disable_adapter"):
            with pm.disable_adapter():
                base = probes.capture_resid(pm, ids[idx, :T], attn[idx, :T], list(layers))
        else:
            base = probes.capture_resid(pm, ids[idx, :T], attn[idx, :T], list(layers))
        for li in layers:
            out[str(li)] = max(out[str(li)], float((ft[li] - base[li]).abs().max()))
    return out


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
    from indbw.selora.random_lora import set_spectrum_matched_random
    from indbw.selora.train import TrainConfig, seed_everything, train_lora

    layers = sorted(saes)
    ref = reference_stats(model, data.sae_ids, data.sae_attn, layers)
    base_fvu = fvu_by_layer(model, saes, data.sae_ids, data.sae_attn, ref)

    row: dict[str, Any] = {
        "run_id": cell.run_id,
        "cell": asdict(cell),
        "fvu_base": _fvu_lists(base_fvu),
    }
    if cell.arm == "base":
        pm: Any = model
        row["loss_first"] = row["loss_last"] = None
    else:
        seed_everything(cell.seed)  # PEFT draws lora_A's init here, before train_lora seeds
        pm = get_peft_model(model, lora_config(cell.layers, cell.r, cell.alpha))
        train_ids = data.train_ids
        if cell.arm == "shuffled_answers":
            if data.train_ids_shuffled is None:
                raise ValueError("shuffled_answers arm needs CellData.train_ids_shuffled")
            train_ids = data.train_ids_shuffled
        losses = train_lora(
            pm,
            train_ids,
            data.pad_id,
            TrainConfig(
                lr=cell.lr, epochs=cell.epochs, seed=cell.seed, wall_clock_budget_s=wall_budget_s
            ),
        )
        row["loss_first"], row["loss_last"] = losses[0], losses[-1]

    row["resid_max_abs_diff"] = max_abs_resid_diff(pm, data.sae_ids, data.sae_attn, layers)
    row["fvu_ft"] = _fvu_lists(fvu_by_layer(pm, saes, data.sae_ids, data.sae_attn, ref))
    row["gsm_correct"] = generate.gsm8k_correct_flags(
        pm, tok, data.gsm_questions, data.gsm_golds, data.gen_tokens_gsm
    )
    row["mmlu_logprob"] = generate.mmlu_logprob_preds(pm, tok, data.mmlu_logprob_prompts)
    row["mmlu_logprob_gold"] = data.mmlu_logprob_gold
    row["mmlu_gen"] = generate.mmlu_generation_preds(
        pm, tok, data.mmlu_gen_prompts, data.gen_tokens_mmlu
    )
    row["mmlu_gen_gold"] = data.mmlu_gen_gold
    if cell.arm != "base":
        set_spectrum_matched_random(pm, seed=cell.seed + 7919)
        row["fvu_rand"] = _fvu_lists(fvu_by_layer(pm, saes, data.sae_ids, data.sae_attn, ref))
    return row
