"""Untrained-model GSM8K accuracy on the full test set (STATE.md F2). UNVERIFIED: needs hub + GPU.

The quantity prereg.ASSUMPTIONS["gsm8k_accuracy"] stands for, measured through the same prompt,
generation and scoring path the sweep uses, with no adapter. Writes data/selora/gsm8k_baseline.json.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import run_selora as R  # noqa: E402

from indbw.selora import cells as C  # noqa: E402
from indbw.selora import evals, generate  # noqa: E402


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    p = k / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (c - h) / d, (c + h) / d


def main() -> None:
    from datasets import load_dataset

    tok, model = R._load_model(C.MODELS[1])
    te = load_dataset("openai/gsm8k", "main", split="test")
    qs = [x["question"] for x in te]
    golds = [evals.gsm8k_gold(x["answer"]) for x in te]
    flags = generate.gsm8k_correct_flags(model, tok, qs, golds, 256)
    n, k = len(flags), sum(flags)
    lo, hi = wilson(k, n)
    out = {
        "n": n,
        "correct": k,
        "accuracy": k / n,
        "wilson95": [lo, hi],
        "first64_accuracy": sum(flags[:64]) / 64,
        "model": C.MODELS[1],
        "adapter": None,
    }
    Path("data/selora").mkdir(parents=True, exist_ok=True)
    Path("data/selora/gsm8k_baseline.json").write_text(json.dumps(out, indent=1))
    print(out)


if __name__ == "__main__":
    main()
