"""Spec for selora.evals: parsers must discriminate right from wrong and parseable from not."""

from __future__ import annotations

import math

import pytest

from indbw.selora import evals


def test_gsm8k_gold_and_pred() -> None:
    assert evals.gsm8k_gold("blah\n#### 1,234") == 1234.0
    assert evals.gsm8k_pred("so 3+4=7 and the total is\n#### 18") == 18.0
    assert evals.gsm8k_pred("working... the answer is $1,200.50.") == 1200.5  # last-number fallback
    assert evals.gsm8k_pred("no digits here") is None
    assert evals.gsm8k_pred("#### ") is None


def test_gsm8k_marker_beats_trailing_numbers() -> None:
    assert evals.gsm8k_pred("#### 5\nextra 99") == 5.0


def test_gsm8k_correct_discriminates() -> None:
    assert evals.gsm8k_correct(18.0, 18.0)
    assert not evals.gsm8k_correct(17.0, 18.0)
    assert not evals.gsm8k_correct(None, 18.0)


def test_gold_without_marker_raises() -> None:
    with pytest.raises(ValueError):
        evals.gsm8k_gold("no marker")


@pytest.mark.parametrize(
    "text,expected",
    [
        ("B", "B"),
        ("  (C) because", "C"),
        ("The answer is D.", "D"),
        ("answer: a", "A"),
        ("Let me think step by step. First, 2+2=4. #### 4", None),  # GSM8K-style output: no letter
        ("Because the Answer", None),  # 'A' as a word start must not parse
        ("", None),
    ],
)
def test_parse_choice(text: str, expected: str | None) -> None:
    assert evals.parse_choice(text) == expected


def test_mmlu_summary_separates_forgetting_from_formatting() -> None:
    gold = ["A"] * 10
    formatting = [None] * 5 + ["A"] * 5  # half unparseable, parseable all right
    s = evals.mmlu_summary(formatting, gold)
    assert s["acc_all"] == 0.5 and s["parse_rate"] == 0.5 and s["acc_parseable"] == 1.0
    forgetting = ["B"] * 10
    f = evals.mmlu_summary(forgetting, gold)
    assert f["acc_all"] == 0.0 and f["parse_rate"] == 1.0 and f["acc_parseable"] == 0.0
    assert math.isnan(evals.mmlu_summary([None] * 10, gold)["acc_parseable"])


def test_mmlu_summary_guards() -> None:
    with pytest.raises(ValueError):
        evals.mmlu_summary(["A"], ["A", "B"])
    with pytest.raises(ValueError):
        evals.mmlu_summary([], [])


def test_logprob_choice() -> None:
    assert evals.logprob_choice([-2.0, -0.1, -3.0, -4.0]) == "B"
    with pytest.raises(ValueError):
        evals.logprob_choice([-1.0, -1.0, -2.0, -3.0])  # tie
    with pytest.raises(ValueError):
        evals.logprob_choice([0.0, 0.0])
