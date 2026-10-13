"""Prompt formats and the two evaluation harnesses (GSM8K generation; MMLU logprob + generation).

The training text and the evaluation prompt share one format, defined here once,
so a format mismatch cannot masquerade as forgetting.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import torch

from indbw.selora import evals

LETTERS = ("A", "B", "C", "D")


def gsm8k_prompt(question: str) -> str:
    return f"Question: {question}\nAnswer:"


def gsm8k_train_text(question: str, answer: str) -> str:
    return f"{gsm8k_prompt(question)} {answer}"


def mmlu_prompt(question: str, choices: Sequence[str]) -> str:
    if len(choices) != 4:
        raise ValueError("MMLU items have exactly four choices")
    body = "\n".join(f"{letter}. {c}" for letter, c in zip(LETTERS, choices, strict=True))
    return f"{question}\n{body}\nAnswer:"


def option_token_ids(tok: Any) -> list[int]:
    ids = [tok.encode(f" {letter}", add_special_tokens=False)[-1] for letter in LETTERS]
    if len(set(ids)) != 4:
        raise ValueError("option letters do not map to four distinct tokens")
    return ids


def _encode_batch(
    tok: Any, prompts: Sequence[str], device: Any, side: str
) -> dict[str, torch.Tensor]:
    tok.padding_side = side
    enc = tok(list(prompts), return_tensors="pt", padding=True)
    return {k: v.to(device) for k, v in enc.items()}


def _length_batches(
    tok: Any, prompts: Sequence[str], max_bs: int, budget: int
) -> list[list[int]]:
    """Prompt indices grouped by length, each batch within `max_bs` rows and `budget` padded
    tokens. Memory only: a prediction depends on its own prompt, and callers restore item order."""
    lens = [len(x) for x in tok(list(prompts))["input_ids"]]
    order = sorted(range(len(prompts)), key=lambda i: lens[i])
    batches: list[list[int]] = []
    for i in order:
        if batches and len(batches[-1]) < max_bs and (len(batches[-1]) + 1) * lens[i] <= budget:
            batches[-1].append(i)
        else:
            batches.append([i])
    return batches


TOKEN_BUDGET = 3000  # padded prompt tokens per forward pass on a 10 GB card (eager attention)


@torch.no_grad()
def mmlu_logprob_preds(
    model: Any, tok: Any, prompts: Sequence[str], batch_size: int = 8
) -> list[str]:
    """Format-free MMLU: argmax over the four option-letter next-token log-probs."""
    opt = option_token_ids(tok)
    device = next(model.parameters()).device
    out: list[str] = [""] * len(prompts)
    for idx in _length_batches(tok, prompts, batch_size, TOKEN_BUDGET):
        enc = _encode_batch(tok, [prompts[i] for i in idx], device, "left")
        # Option logits in fp32 from the final (post-norm) hidden state. bf16 logits are coarse
        # enough to tie two letters exactly; the argmax over four options needs neither the
        # vocabulary normaliser nor the (monotone) final softcap, so neither is computed.
        h = model(**enc, output_hidden_states=True, logits_to_keep=1).hidden_states[-1][:, -1]
        w = model.get_output_embeddings().weight[opt]
        lp = h.float() @ w.float().T
        for i, row in zip(idx, lp, strict=True):
            out[i] = evals.logprob_choice(row.tolist())
    return out


@torch.no_grad()
def generate_texts(
    model: Any, tok: Any, prompts: Sequence[str], max_new_tokens: int, batch_size: int = 8
) -> list[str]:
    device = next(model.parameters()).device
    out: list[str] = [""] * len(prompts)
    for idx in _length_batches(tok, prompts, batch_size, TOKEN_BUDGET):
        enc = _encode_batch(tok, [prompts[i] for i in idx], device, "left")
        gen = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
        new = gen[:, enc["input_ids"].shape[1] :]
        for i, text in zip(idx, tok.batch_decode(new, skip_special_tokens=True), strict=True):
            out[i] = text
    return out


def mmlu_generation_preds(
    model: Any, tok: Any, prompts: Sequence[str], max_new_tokens: int = 32
) -> list[str | None]:
    return [evals.parse_choice(t) for t in generate_texts(model, tok, prompts, max_new_tokens)]


def gsm8k_correct_flags(
    model: Any,
    tok: Any,
    questions: Sequence[str],
    golds: Sequence[float],
    max_new_tokens: int = 256,
) -> list[int]:
    """0/1 per item, in item order (paired across arms)."""
    texts = generate_texts(model, tok, [gsm8k_prompt(q) for q in questions], max_new_tokens)
    return [
        int(evals.gsm8k_correct(evals.gsm8k_pred(t), g)) for t, g in zip(texts, golds, strict=True)
    ]


def shuffle_answers(
    questions: Sequence[str], answers: Sequence[str], seed: int
) -> list[tuple[str, str]]:
    """Pair each question with another question's answer (a seeded derangement).

    The shuffled-answer control arm: same tokens, same answer format, no question-answer
    signal. Separates "learned GSM8K" from "saw GSM8K-shaped text".
    """
    n = len(questions)
    if n != len(answers) or n < 2:
        raise ValueError("need >= 2 equal-length questions and answers")
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n)
    shift = int(rng.integers(1, n))
    target = perm[(np.arange(n) + shift) % n]  # cyclic shift of a random order: no fixed points
    deranged = np.empty(n, dtype=int)
    deranged[perm] = target
    return [(questions[i], answers[int(deranged[i])]) for i in range(n)]
