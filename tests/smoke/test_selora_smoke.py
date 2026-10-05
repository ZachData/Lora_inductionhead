"""Smoke: whole selora path on a random 4-layer Gemma2. Asserts nothing about science.

load tiny model -> capture base residuals -> inject LoRA on late layers -> 5 training steps
-> capture again -> every probe -> adjudicate -> schema round-trip.
"""

from __future__ import annotations

import json

import numpy as np
import torch
from peft import get_peft_model

from indbw.schema import record_from_dict
from indbw.selora import analysis, evals, prereg, probes
from indbw.selora.adjudicate import adjudicate
from indbw.selora.placement import lora_config
from indbw.selora.random_lora import delta_norms, set_matched_norm_random
from indbw.selora.sae import content_mask, fvu_per_seq, sq_dev_per_seq
from indbw.selora.train import TrainConfig, train_lora


def test_selora_end_to_end(tiny_gemma2, tiny_sae_factory) -> None:  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(3, 64, (6, 12), generator=g)
    ids[:, 0] = 2
    attn = torch.ones_like(ids)
    layers = [0, 1, 2, 3]
    saes = {li: tiny_sae_factory(seed=li) for li in layers}

    base = probes.capture_resid(tiny_gemma2, ids, attn, layers)
    base = {k: v.clone() for k, v in base.items()}
    clean_ce = probes.spliced_ce_per_seq(tiny_gemma2, ids, attn, None, None)

    pm = get_peft_model(tiny_gemma2, lora_config((2, 3), r=2, alpha=4, dropout=0.0))
    train_lora(pm, ids.tolist(), 0, TrainConfig(lr=1e-2, batch_size=2, grad_accum=1, max_steps=5))
    ft = probes.capture_resid(pm, ids, attn, layers)

    mask = content_mask(attn)
    cells = {}
    for li in layers:
        mu = base[li][mask].mean(0).double()
        den = sq_dev_per_seq(base[li].double(), mu, mask)
        cells[f"L{li}"] = (
            fvu_per_seq(base[li].double(), saes[li], mask, mu, den).numpy(),
            fvu_per_seq(ft[li].double(), saes[li], mask, mu, den).numpy(),
        )
    probes.spliced_ce_per_seq(pm, ids, attn, 3, saes[3])
    assert clean_ce.shape == (6,)

    set_matched_norm_random(pm, delta_norms(pm), seed=1)

    rng = np.random.default_rng(0)
    acts = saes[2].encode(ft[2][:, -1].double()).numpy()
    probes.divergence_with_null(acts, np.array([0, 0, 0, 1, 1, 1]), 5, 20, rng)

    evals.mmlu_summary([evals.parse_choice("B"), None], ["B", "A"])
    obs = analysis.s0_observed(base[0].numpy(), ft[0].numpy())
    rec = adjudicate("S0", obs, ["late-layer arm"], {"smoke": 1}, 0, "tiny-random", "none", 0.0)
    again = record_from_dict(json.loads(rec.to_json_line()))
    assert again.is_self_consistent() and again.row in {c.id for c in prereg.CLAIMS}
    assert len(cells) == 4


def test_run_cell_end_to_end_on_tiny_model(tiny_gemma2, tiny_sae_factory, stub_tokenizer) -> None:  # type: ignore[no-untyped-def]
    from indbw.selora import generate
    from indbw.selora.cells import Cell
    from indbw.selora.pipeline import CellData, run_cell

    g = torch.Generator().manual_seed(0)
    sae_ids = torch.randint(3, 64, (6, 10), generator=g)
    sae_ids[:, 0] = 2
    data = CellData(
        train_ids=torch.randint(3, 64, (8, 12), generator=g).tolist(),
        pad_id=0,
        sae_ids=sae_ids,
        sae_attn=torch.ones_like(sae_ids),
        gsm_questions=["1+1?", "2+3?"],
        gsm_golds=[2.0, 5.0],
        mmlu_prompts=[generate.mmlu_prompt("q", ["a", "b", "c", "d"])] * 3,
        mmlu_gold=["A", "B", "C"],
        gen_tokens_gsm=3,
        gen_tokens_mmlu=2,
    )
    saes = {li: tiny_sae_factory(seed=li) for li in range(4)}
    cell = Cell("tiny", "late", (2, 3), 0, lr=1e-2, r=2, alpha=4, epochs=0.5)
    row = run_cell(cell, tiny_gemma2, stub_tokenizer, saes, data, wall_budget_s=60.0)
    assert row["run_id"] == cell.run_id
    assert row["resid_max_abs_diff"]["0"] == 0.0 and row["resid_max_abs_diff"]["1"] == 0.0
    assert row["resid_max_abs_diff"]["2"] > 0.0 and row["resid_max_abs_diff"]["3"] > 0.0
    assert (
        set(row["fvu_ft"]) == set(row["fvu_base"]) == set(row["fvu_rand"]) == {"0", "1", "2", "3"}
    )
    assert row["fvu_base"]["0"] == row["fvu_ft"]["0"]  # unadapted layer: identical FVU, to the bit
    json.dumps(row)  # row must be JSON-serialisable for the append-only results file
