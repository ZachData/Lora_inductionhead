"""Selective-LoRA x SAE-validity runner (PROJECT.md section 13).

  python scripts/run_selora.py plan                       # enumerate cells; no network, no model
  python scripts/run_selora.py divergence --model <id> --out data/selora/divergence_<id with / -> _>.json
  python scripts/run_selora.py cell --div-dir data/selora [--limit N]
  python scripts/run_selora.py adjudicate --div-dir data/selora --eval-set-hash <h>   # once, complete sweep

`divergence` and `cell` refuse to run while prereg.PREREG_STATUS == "draft". Results are appended
one row per cell to data/selora/cells.jsonl and skipped on restart (config hash = run id).
NOT yet exercised against real checkpoints: this sandbox has no route to huggingface.co. The
measurement path (`pipeline.run_cell`) is exercised on a random 4-layer Gemma2 in
tests/smoke/test_selora_smoke.py; the dataset / checkpoint loaders in this file are not.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from indbw.selora import cells as C
from indbw.selora import prereg

OUT = Path("data/selora")
MMLU_SEED = 0
SAE_PROMPTS_PER_CLASS = 512


def _load_model(name: str):  # type: ignore[no-untyped-def]
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        name, torch_dtype=dtype, attn_implementation="eager"
    )
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    return tok, model


def _load_saes(layers: list[int]):  # type: ignore[no-untyped-def]
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files

    from indbw.selora.sae import load_sae_npz

    files = list_repo_files(C.SAE_REPO)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    out = {}
    for li in layers:
        path = hf_hub_download(C.SAE_REPO, C.pick_sae_path(files, li))
        sae = load_sae_npz(path, dtype=torch.float32)
        out[li] = type(sae)(
            *(t.to(dev) for t in (sae.W_enc, sae.W_dec, sae.b_enc, sae.b_dec, sae.threshold))
        )
    return out


def cmd_plan(_: argparse.Namespace) -> None:
    fake = {m: list(np.linspace(0, 1, C.N_LAYERS)) for m in C.MODELS}  # placeholder scores
    cs = C.enumerate_cells(fake)
    print(
        f"{len(cs)} cells ({len(cs) // len(C.MODELS)} per model); prereg {prereg.prereg_hash()} "
        f"status={prereg.PREREG_STATUS}"
    )
    print("sae_top / sae_bottom layer sets above are placeholders until `divergence` is run")


def _prompt_sets() -> dict[str, list[str]]:
    from datasets import load_dataset

    gsm = load_dataset("openai/gsm8k", "main", split="test")
    mmlu = load_dataset("cais/mmlu", "all", split="validation")
    rng = np.random.default_rng(MMLU_SEED)
    pick = lambda n, k: [int(i) for i in rng.permutation(n)[:k]]  # noqa: E731
    math = [gsm[i]["question"] for i in pick(len(gsm), SAE_PROMPTS_PER_CLASS)]
    general = [mmlu[i]["question"] for i in pick(len(mmlu), SAE_PROMPTS_PER_CLASS)]
    return {"math": math, "general": general}


def cmd_divergence(a: argparse.Namespace) -> None:
    """Per-layer top-k divergence with a length x digit stratified permutation null, S2b's
    prompt-level peak bootstrap, and the top-100 sensitivity run. Output is what
    `report.build_records` and `cells.enumerate_cells` read."""
    from indbw.selora import analysis, probes

    prereg.require_registered()
    tok, model = _load_model(a.model)
    saes = _load_saes(list(range(C.N_LAYERS)))
    sets = _prompt_sets()
    texts = sets["math"] + sets["general"]
    labels = np.array([1] * len(sets["math"]) + [0] * len(sets["general"]))
    lengths = np.array([len(tok(t).input_ids) for t in texts])
    digits = np.array([sum(ch.isdigit() for ch in t) for t in texts])
    strata = probes.make_strata(lengths, digits)
    last: dict[int, list[np.ndarray]] = {li: [] for li in range(C.N_LAYERS)}
    for t in texts:
        enc = tok(t, return_tensors="pt").to(model.device)
        cap = probes.capture_resid(
            model, enc.input_ids, enc.attention_mask, list(range(C.N_LAYERS))
        )
        for li in last:
            last[li].append(saes[li].encode(cap[li][:, -1].float())[0].detach().cpu().numpy())
    acts = [np.stack(last[li]) for li in range(C.N_LAYERS)]
    rng = np.random.default_rng(0)
    res = {
        k: [
            probes.divergence_with_null(x, labels, k, prereg.N_PERM_DIVERGENCE, rng, strata=strata)
            for x in acts
        ]
        for k in (prereg.TOPK, 100)
    }
    main = res[prereg.TOPK]
    null_means = np.array([r["null_mean"] for r in main])
    s2b = analysis.s2b_observed(acts, labels, null_means, prereg.TOPK, strata=strata)
    out = {
        "model": a.model,
        "topk": prereg.TOPK,
        "strata": "length x digit",
        "p_by_layer": [float(r["p_value"]) for r in main],
        "excess_by_layer": [float(r["excess"]) for r in main],
        "peak_in_range_frequency": s2b["peak_in_range_frequency"],
        "topk_sensitivity_done": True,
        "excess_by_layer_top100": [float(r["excess"]) for r in res[100]],
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=1))


def _cell_data(tok):  # type: ignore[no-untyped-def]
    from datasets import load_dataset

    from indbw.selora import evals, generate
    from indbw.selora.pipeline import CellData

    gsm_tr = load_dataset("openai/gsm8k", "main", split="train")
    gsm_te = load_dataset("openai/gsm8k", "main", split="test")
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    gen_idx = np.random.default_rng(MMLU_SEED).permutation(len(mmlu))[: prereg.MMLU_GEN_N]

    def encode(pairs):  # type: ignore[no-untyped-def]
        return [
            tok(generate.gsm8k_train_text(q, ans)).input_ids + [tok.eos_token_id]
            for q, ans in pairs
        ]

    pairs = [(x["question"], x["answer"]) for x in gsm_tr]
    shuffled = generate.shuffle_answers([q for q, _ in pairs], [ans for _, ans in pairs], seed=0)
    prompts = _prompt_sets()
    tok.padding_side = "right"  # content_mask assumes BOS at position 0
    enc = tok(prompts["math"] + prompts["general"], return_tensors="pt", padding=True)

    def mmlu_p(i: int) -> str:
        return generate.mmlu_prompt(mmlu[i]["question"], mmlu[i]["choices"])

    return CellData(
        train_ids=encode(pairs),
        train_ids_shuffled=encode(shuffled) if C.INCLUDE_SHUFFLED else None,
        pad_id=tok.pad_token_id,
        sae_ids=enc.input_ids,
        sae_attn=enc.attention_mask,
        gsm_questions=[x["question"] for x in gsm_te],
        gsm_golds=[evals.gsm8k_gold(x["answer"]) for x in gsm_te],
        mmlu_logprob_prompts=[mmlu_p(i) for i in range(len(mmlu))],
        mmlu_logprob_gold=["ABCD"[mmlu[i]["answer"]] for i in range(len(mmlu))],
        mmlu_gen_prompts=[mmlu_p(int(i)) for i in gen_idx],
        mmlu_gen_gold=["ABCD"[mmlu[int(i)]["answer"]] for i in gen_idx],
    )


def _scores(div_dir: Path) -> dict[str, list[float]]:
    out = {}
    for m in C.MODELS:
        d = json.loads((div_dir / f"divergence_{m.replace('/', '_')}.json").read_text())
        out[m] = d["excess_by_layer"]
    return out


def cmd_cell(a: argparse.Namespace) -> None:
    from indbw.selora.pipeline import run_cell

    prereg.require_registered()
    todo = C.pending(C.enumerate_cells(_scores(Path(a.div_dir))), OUT / "cells.jsonl")
    print(f"{len(todo)} cells pending")
    cache: dict[str, Any] = {}
    for cell in todo[: a.limit]:
        t0 = time.monotonic()
        tok, model = _load_model(cell.model)  # fresh weights per cell: run_cell consumes the model
        if "data" not in cache:
            cache["data"] = _cell_data(tok)
            cache["saes"] = _load_saes(list(range(C.N_LAYERS)))
        data, saes = cache["data"], cache["saes"]
        row = run_cell(cell, model, tok, saes, data, wall_budget_s=6 * 3600.0)
        row["wall_clock_s"] = time.monotonic() - t0
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "cells.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()


def cmd_adjudicate(a: argparse.Namespace) -> None:
    """Once, on the complete sweep: rows -> results/selora.jsonl (verdicts recomputed by CI)."""
    from indbw.schema import append_record
    from indbw.selora import report

    prereg.require_registered()
    div_dir = Path(a.div_dir)
    planned = C.enumerate_cells(_scores(div_dir))
    rows = [json.loads(x) for x in (OUT / "cells.jsonl").read_text().splitlines() if x.strip()]
    divergence = {
        m: json.loads((div_dir / f"divergence_{m.replace('/', '_')}.json").read_text())
        for m in C.MODELS
    }
    prov = report.Provenance(
        config={"cells": [c.run_id for c in planned], "prereg": prereg.prereg_hash()},
        seed=0,
        checkpoint_revision=",".join(C.MODELS),
        eval_set_hash=a.eval_set_hash,
        wall_clock_s=float(sum(r.get("wall_clock_s", 0.0) for r in rows)),
    )
    for rec in report.build_records(rows, planned, divergence, prov):
        append_record("results/selora.jsonl", rec)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("plan").set_defaults(fn=cmd_plan)
    d = sub.add_parser("divergence")
    d.add_argument("--model", required=True, choices=list(C.MODELS))
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_divergence)
    c = sub.add_parser("cell")
    c.add_argument("--div-dir", required=True)
    c.add_argument("--limit", type=int, default=None)
    c.set_defaults(fn=cmd_cell)
    j = sub.add_parser("adjudicate")
    j.add_argument("--div-dir", required=True)
    j.add_argument("--eval-set-hash", required=True)
    j.set_defaults(fn=cmd_adjudicate)
    args = ap.parse_args()
    args.fn(args)
