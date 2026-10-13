"""GSM8K and MMLU scoring: answer parsing and the two MMLU protocols.

The MMLU question (does the model *know* vs does it *format*) is answered by
scoring the same model two ways: log-probability over the four option letters
(format-free) and free generation parsed for a letter. These functions are
the parsers; they are metric-defining and carry discrimination tests.
"""

from __future__ import annotations

import re
from collections.abc import Sequence

_NUM = re.compile(r"-?\d[\d,]*\.?\d*")


def _norm_num(s: str) -> float | None:
    s = s.replace(",", "").replace("$", "").strip().rstrip(".")
    try:
        return float(s)
    except ValueError:
        return None


def gsm8k_gold(answer_field: str) -> float:
    """Gold is the number after the final '####' in the dataset's answer field."""
    if "####" not in answer_field:
        raise ValueError("gold answer has no '####' marker")
    tail = answer_field.rsplit("####", 1)[1].split()
    g = _norm_num(tail[0]) if tail else None
    if g is None:
        raise ValueError("could not parse gold number")
    return g


def gsm8k_pred(text: str) -> float | None:
    """Number after the last '####' if present, else the last number in the text; else None."""
    if "####" in text:
        m = _NUM.search(text.rsplit("####", 1)[1])
        return _norm_num(m.group(0)) if m else None
    nums = _NUM.findall(text)
    return _norm_num(nums[-1]) if nums else None


def gsm8k_correct(pred: float | None, gold: float) -> bool:
    return pred is not None and abs(pred - gold) < 1e-6


_LEAD = re.compile(r"^\s*\(?([ABCD])(?![A-Za-z])")
_ANS = re.compile(r"answer\s*(?:is|:)\s*\(?([ABCD])(?![A-Za-z])", re.IGNORECASE)


def parse_choice(text: str) -> str | None:
    """A letter A-D if the generation leads with one or states 'answer is X'; else None."""
    m = _LEAD.match(text)
    if m:
        return m.group(1)
    m = _ANS.search(text)
    return m.group(1).upper() if m else None


def mmlu_summary(preds: Sequence[str | None], gold: Sequence[str]) -> dict[str, float]:
    """acc_all counts unparseable as wrong; acc_parseable conditions on a letter being present."""
    if len(preds) != len(gold) or len(gold) == 0:
        raise ValueError("preds and gold must be equal-length and non-empty")
    parsed = [(p, g) for p, g in zip(preds, gold, strict=True) if p is not None]
    n = len(gold)
    correct = sum(p == g for p, g in parsed)
    return {
        "acc_all": correct / n,
        "parse_rate": len(parsed) / n,
        "acc_parseable": (correct / len(parsed)) if parsed else float("nan"),
        "n": float(n),
    }


def logprob_choice(option_logprobs: Sequence[float]) -> str:
    """argmax over four option-letter log-probs (a tie raises: a tie is a broken readout)."""
    if len(option_logprobs) != 4:
        raise ValueError("need exactly four option log-probabilities")
    best = max(option_logprobs)
    if sum(lp == best for lp in option_logprobs) > 1:
        raise ValueError("tied option log-probabilities")
    return "ABCD"[list(option_logprobs).index(best)]
