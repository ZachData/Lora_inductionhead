"""Preflight for Extension S (STATE.md D1). UNVERIFIED: needs huggingface.co and a GPU.

Runs once before any sweep, and stops on the first failing section:
  1. checkpoints and all 26 Gemma Scope SAEs download; SAE shapes match the model
  2. base-model FVU per layer is sane on 64 SAE prompts (wrong hook / wrong SAE shows here)
  3. a 20-step LoRA timing probe on the late arm -> projected sweep hours
  4. one full cell on a 64-item slice; its row must be populated
  5. measured quantities vs prereg.ASSUMPTIONS (if off, the power analysis no longer applies)

Writes data/selora/preflight.json. Exit code 1 on any problem.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_selora as R  # noqa: E402

from indbw.selora import cells as C  # noqa: E402
from indbw.selora import preflight as P  # noqa: E402
from indbw.selora import prereg  # noqa: E402


def main() -> int:
    from peft import get_peft_model

    from indbw.selora.pipeline import fvu_by_layer, reference_stats, run_cell
    from indbw.selora.placement import lora_config
    from indbw.selora.train import TrainConfig, train_lora

    report: dict[str, object] = {}
    problems: list[str] = []
    model_id = C.MODELS[1]
    tok, model = R._load_model(model_id)
    saes = R._load_saes(list(range(C.N_LAYERS)))
    for li, sae in saes.items():
        problems += [f"layer {li}: {p}" for p in P.check_sae(sae)]
    if problems:
        return _finish(report, problems)

    data = R._cell_data(tok)
    ids, attn = data.sae_ids[:64].to(model.device), data.sae_attn[:64].to(model.device)
    ref = reference_stats(model, ids, attn, list(range(C.N_LAYERS)))
    fvu = {li: float(v.mean()) for li, v in fvu_by_layer(model, saes, ids, attn, ref).items()}
    report["base_fvu"] = fvu
    problems += P.check_base_fvu(fvu)

    from indbw.selora.placement import named_sets

    late = named_sets(C.N_LAYERS, C.N_ADAPT)["late"]
    pm = get_peft_model(model, lora_config(late))
    t0 = time.monotonic()
    train_lora(pm, data.train_ids, data.pad_id, TrainConfig(max_steps=20))
    sec = (time.monotonic() - t0) / 20
    steps = P.steps_per_cell(len(data.train_ids), 4, 4, 3.0)
    n_cells = len(C.enumerate_cells({m: [0.0] * C.N_LAYERS for m in C.MODELS}))
    report["sec_per_step"] = sec
    report["sweep_hours_1gpu"] = P.estimate_sweep_hours(sec, steps, n_cells, 1)

    tok, model = R._load_model(model_id)
    small = type(data)(
        **{
            **data.__dict__,
            "gsm_questions": data.gsm_questions[:64],
            "gsm_golds": data.gsm_golds[:64],
            "train_ids": data.train_ids[:64],
        }
    )
    row = run_cell(C.Cell(model_id, "late", late, 0, epochs=1.0), model, tok, saes, small, 3600.0)
    problems += P.check_populated(
        row, ("fvu_ft", "fvu_rand", "gsm_correct", "mmlu_logprob", "mmlu_gen")
    )
    acc = float(np.mean(row["gsm_correct"]))
    report["gsm8k_accuracy_64"] = acc
    problems += P.compare_assumptions({"gsm8k_accuracy": acc})
    return _finish(report, problems)


def _finish(report: dict[str, object], problems: list[str]) -> int:
    report["problems"] = problems
    report["prereg_hash"] = prereg.prereg_hash()
    out = Path("data/selora/preflight.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1))
    for p in problems:
        print("PREFLIGHT:", p)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
