"""Selective-LoRA x SAE-validity runner (PROJECT.md section 13).

  python scripts/run_selora.py plan                       # enumerate cells; no network, no model
  python scripts/run_selora.py divergence --out data/selora/divergence_<model>.json
  python scripts/run_selora.py cell --scores data/selora/divergence_<model>.json [--limit N]

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
GSM_SEED, MMLU_N, MMLU_SEED = 0, 500, 0
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


def _length_bins(lengths: np.ndarray, n_bins: int = 5) -> np.ndarray:
    return np.digitize(lengths, np.quantile(lengths, np.linspace(0, 1, n_bins + 1)[1:-1]))


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
    """Per-layer top-k divergence (math vs general) with a length-stratified permutation null."""
    from indbw.selora import probes

    prereg.require_registered()
    tok, model = _load_model(a.model)
    saes = _load_saes(list(range(C.N_LAYERS)))
    sets = _prompt_sets()
    texts = sets["math"] + sets["general"]
    labels = np.array([1] * len(sets["math"]) + [0] * len(sets["general"]))
    lengths = np.array([len(tok(t).input_ids) for t in texts])
    strata = _length_bins(lengths)
    last = {li: [] for li in range(C.N_LAYERS)}
    for t in texts:
        enc = tok(t, return_tensors="pt").to(model.device)
        cap = probes.capture_resid(
            model, enc.input_ids, enc.attention_mask, list(range(C.N_LAYERS))
        )
        for li in last:
            last[li].append(saes[li].encode(cap[li][:, -1].float())[0].detach().cpu().numpy())
    rng = np.random.default_rng(0)
    out = {}
    for li in range(C.N_LAYERS):
        r = probes.divergence_with_null(
            np.stack(last[li]), labels, prereg.TOPK, 2000, rng, strata=strata
        )
        out[li] = {k: float(v) for k, v in r.items() if k != "null"}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(
        json.dumps({"model": a.model, "topk": prereg.TOPK, "layers": out}, indent=1)
    )


def _cell_data(tok):  # type: ignore[no-untyped-def]
    from datasets import load_dataset

    from indbw.selora import evals, generate

    gsm_tr = load_dataset("openai/gsm8k", "main", split="train")
    gsm_te = load_dataset("openai/gsm8k", "main", split="test")
    mmlu = load_dataset("cais/mmlu", "all", split="test")
    idx = np.random.default_rng(MMLU_SEED).permutation(len(mmlu))[:MMLU_N]
    train_ids = [
        tok(generate.gsm8k_train_text(x["question"], x["answer"])).input_ids + [tok.eos_token_id]
        for x in gsm_tr
    ]
    prompts = _prompt_sets()
    enc = tok(prompts["math"] + prompts["general"], return_tensors="pt", padding=True)
    from indbw.selora.pipeline import CellData

    return CellData(
        train_ids=train_ids,
        pad_id=tok.pad_token_id,
        sae_ids=enc.input_ids,
        sae_attn=enc.attention_mask,
        gsm_questions=[x["question"] for x in gsm_te],
        gsm_golds=[evals.gsm8k_gold(x["answer"]) for x in gsm_te],
        mmlu_prompts=[
            generate.mmlu_prompt(mmlu[int(i)]["question"], mmlu[int(i)]["choices"]) for i in idx
        ],
        mmlu_gold=["ABCD"[mmlu[int(i)]["answer"]] for i in idx],
    )


def cmd_cell(a: argparse.Namespace) -> None:
    from indbw.selora.pipeline import run_cell

    prereg.require_registered()
    scores = json.loads(Path(a.scores).read_text())
    todo = C.pending(C.enumerate_cells(scores), OUT / "cells.jsonl")
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


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="stage", required=True)
    sub.add_parser("plan").set_defaults(fn=cmd_plan)
    d = sub.add_parser("divergence")
    d.add_argument("--model", required=True, choices=list(C.MODELS))
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_divergence)
    c = sub.add_parser("cell")
    c.add_argument("--scores", required=True)
    c.add_argument("--limit", type=int, default=None)
    c.set_defaults(fn=cmd_cell)
    args = ap.parse_args()
    args.fn(args)
