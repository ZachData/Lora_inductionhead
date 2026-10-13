"""Selective-LoRA x SAE-validity runner (PROJECT.md section 13).

  python scripts/run_selora.py plan [--budget-hours 8]    # enumerate cells and cut them into runs
  python scripts/run_selora.py divergence --model <id> --out data/selora/divergence_<id with / -> _>.json
  python scripts/run_selora.py cell --div-dir data/selora [--budget-hours 8] [--limit N]
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
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from indbw.selora import cells as C
from indbw.selora import prereg, schedule

OUT = Path("data/selora")
MMLU_SEED = 0
SAE_PROMPTS_PER_CLASS = 512


def _free_gpu(*_: Any) -> None:
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


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


class _LazySAEs(Mapping):  # type: ignore[type-arg]
    """SAEs held on CPU; one layer is moved to the GPU at a time (26 fp32 SAEs are ~7.8 GB,
    which does not fit beside the model on a 10 GB card). Values are unchanged, only placement."""

    def __init__(self, cpu: dict) -> None:  # type: ignore[type-arg]
        self._cpu = cpu
        self._dev = "cuda" if __import__("torch").cuda.is_available() else "cpu"
        self._live: tuple[int, Any] | None = None

    def __getitem__(self, li: int):  # type: ignore[no-untyped-def]
        if self._live is None or self._live[0] != li:
            sae = self._cpu[li]
            self._live = (
                li,
                type(sae)(
                    *(
                        t.to(self._dev)
                        for t in (sae.W_enc, sae.W_dec, sae.b_enc, sae.b_dec, sae.threshold)
                    )
                ),
            )
        return self._live[1]

    def __iter__(self):  # type: ignore[no-untyped-def]
        return iter(sorted(self._cpu))

    def __len__(self) -> int:
        return len(self._cpu)


def _load_saes(layers: list[int]):  # type: ignore[no-untyped-def]
    import torch
    from huggingface_hub import hf_hub_download, list_repo_files

    from indbw.selora.sae import load_sae_npz

    files = list_repo_files(C.SAE_REPO)
    out = {}
    for li in layers:
        path = hf_hub_download(C.SAE_REPO, C.pick_sae_path(files, li))
        out[li] = load_sae_npz(path, dtype=torch.float32)
    return _LazySAEs(out)


def _rows() -> list[dict[str, Any]]:
    p = OUT / "cells.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()] if p.exists() else []


def _sec_per_step(a: argparse.Namespace) -> float:
    if a.sec_per_step is not None:
        return float(a.sec_per_step)
    return float(json.loads((OUT / "preflight.json").read_text())["sec_per_step"])


def cmd_plan(a: argparse.Namespace) -> None:
    """Pending cells in execution order, cut into runs of at most --budget-hours."""
    try:
        scores, placeholder = _scores(OUT), False
    except FileNotFoundError:
        scores, placeholder = {m: list(np.linspace(0, 1, C.N_LAYERS)) for m in C.MODELS}, True
    cs = C.enumerate_cells(scores)
    print(
        f"{len(cs)} cells ({sum(c.arm != 'base' for c in cs)} trained); prereg {prereg.prereg_hash()} "
        f"status={prereg.PREREG_STATUS}"
    )
    if placeholder:
        print("sae_top / sae_bottom layer sets are placeholders until `divergence` is run")
    todo = schedule.order_cells(C.pending(cs, OUT / "cells.jsonl"))
    sec = _sec_per_step(a)
    n = schedule.GSM8K_TRAIN_N
    measured = schedule.measured_eval_s(_rows(), sec, n)
    if a.eval_minutes is not None:
        evals, how = [a.eval_minutes * 60.0], "--eval-minutes"
    elif measured is not None:
        evals, how = [measured], "measured from cells.jsonl"
    else:
        evals, how = [900.0, 1800.0, 2700.0], "UNMEASURED: scenarios, replan after the first cell"
    reserve = a.first_run_reserve_hours * 3600.0 if not _rows() else 0.0
    budget = a.budget_hours * 3600.0
    for ev in evals:
        runs = schedule.pack_runs(
            todo,
            lambda c, ev=ev: schedule.cell_seconds(c.arm, c.epochs, sec, n, ev),
            budget,
            first_run_reserve_s=reserve,
        )
        print(
            f"\n{len(todo)} pending cells, eval {ev / 60:.0f} min/cell ({how}), "
            f"{sec:.2f} s/step, {a.budget_hours:g} h budget, margin {schedule.MARGIN}: {len(runs)} runs"
        )
        for i, r in enumerate(runs, 1):
            h = sum(schedule.cell_seconds(c.arm, c.epochs, sec, n, ev) for c in r) / 3600.0
            first = f"{r[0].arm}/{r[0].seed}"
            last = f"{r[-1].arm}/{r[-1].seed}"
            print(f"  run {i}: {len(r):2d} cells  {h:4.1f} h  {first} .. {last}")


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
    """Run pending cells in schedule order until --budget-hours would be exceeded.

    A cell cannot be resumed mid-training, so one is started only if its estimated cost (analytic
    training time + eval time, measured from rows already written) fits the time left."""
    from indbw.selora.pipeline import run_cell

    t_start = time.monotonic()
    prereg.require_registered()
    todo = schedule.order_cells(
        C.pending(C.enumerate_cells(_scores(Path(a.div_dir))), OUT / "cells.jsonl")
    )
    print(f"{len(todo)} cells pending")
    sec, n, budget = _sec_per_step(a), schedule.GSM8K_TRAIN_N, a.budget_hours * 3600.0
    rows = _rows()
    cache: dict[str, Any] = {}
    done = 0
    for cell in todo[: a.limit]:
        elapsed = time.monotonic() - t_start
        eval_s = schedule.measured_eval_s(rows, sec, n)
        if eval_s is None and a.eval_minutes is not None:
            eval_s = a.eval_minutes * 60.0
        if eval_s is not None:
            need = schedule.cell_seconds(cell.arm, cell.epochs, sec, n, eval_s)
            if not schedule.may_start(elapsed, need, budget):
                print(
                    f"stop: {cell.arm}/{cell.seed} needs ~{need / 3600:.2f} h x {schedule.MARGIN}, "
                    f"{(budget - elapsed) / 3600:.2f} h left of {a.budget_hours:g}"
                )
                break
        t0 = time.monotonic()
        tok, model = _load_model(cell.model)  # fresh weights per cell: run_cell consumes the model
        if "data" not in cache:
            cache["data"] = _cell_data(tok)
            cache["saes"] = _load_saes(list(range(C.N_LAYERS)))
            if len(cache["data"].train_ids) != n:
                raise ValueError(
                    f"train set has {len(cache['data'].train_ids)} items, schedule assumes {n}"
                )
        data, saes = cache["data"], cache["saes"]
        row = run_cell(
            cell, model, tok, saes, data, wall_budget_s=budget - (time.monotonic() - t_start)
        )
        row["wall_clock_s"] = time.monotonic() - t0
        OUT.mkdir(parents=True, exist_ok=True)
        with (OUT / "cells.jsonl").open("a") as f:
            f.write(json.dumps(row, sort_keys=True) + "\n")
            f.flush()
        rows.append(row)
        done += 1
        print(
            f"{cell.arm}/{cell.seed}: {row['wall_clock_s'] / 3600:.2f} h  "
            f"({done} this run, {len(todo) - done} pending)"
        )
        del model, row
        _free_gpu()


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

    def timing(p: argparse.ArgumentParser) -> None:
        p.add_argument("--budget-hours", type=float, default=8.0)
        p.add_argument(
            "--sec-per-step", type=float, default=None, help="default: data/selora/preflight.json"
        )
        p.add_argument(
            "--eval-minutes",
            type=float,
            default=None,
            help="non-training minutes per cell; default: measured",
        )

    pl = sub.add_parser("plan")
    timing(pl)
    pl.add_argument(
        "--first-run-reserve-hours",
        type=float,
        default=1.0,
        help="held back from run 1 for the divergence stage (unmeasured assumption)",
    )
    pl.set_defaults(fn=cmd_plan)
    d = sub.add_parser("divergence")
    d.add_argument("--model", required=True, choices=list(C.MODELS))
    d.add_argument("--out", required=True)
    d.set_defaults(fn=cmd_divergence)
    c = sub.add_parser("cell")
    c.add_argument("--div-dir", required=True)
    c.add_argument("--limit", type=int, default=None)
    timing(c)
    c.set_defaults(fn=cmd_cell)
    j = sub.add_parser("adjudicate")
    j.add_argument("--div-dir", required=True)
    j.add_argument("--eval-set-hash", required=True)
    j.set_defaults(fn=cmd_adjudicate)
    args = ap.parse_args()
    args.fn(args)
