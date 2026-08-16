"""Invariants for indbw.natural_text over random input (CLAUDE.md's
property-test tier). Closed-form/discrimination point checks live in
tests/unit/test_natural_text.py; this file checks they hold generically.
"""

from __future__ import annotations

import numpy as np
import torch
from hypothesis import given
from hypothesis import strategies as st

from indbw.natural_text import induction_position_mask, natural_induction_nll

_lengths = st.integers(min_value=1, max_value=40)
_seeds = st.integers(min_value=0, max_value=2**31 - 1)
_vocabs = st.integers(min_value=2, max_value=6)  # small vocab -> real repeats occur


@given(seed=_seeds, length=_lengths, vocab=_vocabs)
def test_mask_position_zero_always_false(seed: int, length: int, vocab: int) -> None:
    tokens = np.random.default_rng(seed).integers(0, vocab, size=length)
    mask = induction_position_mask(tokens)
    assert not bool(mask[0])


@given(seed=_seeds, length=_lengths, vocab=_vocabs)
def test_mask_true_implies_earlier_occurrence(seed: int, length: int, vocab: int) -> None:
    tokens = np.random.default_rng(seed).integers(0, vocab, size=length)
    mask = induction_position_mask(tokens)
    for t in range(1, length):
        if mask[t]:
            assert tokens[t - 1] in tokens[: t - 1]
        else:
            assert tokens[t - 1] not in tokens[: t - 1]


@given(seed=_seeds, length=st.integers(min_value=3, max_value=20), vocab=_vocabs)
def test_nll_unaffected_by_perturbing_non_eligible_logits(
    seed: int, length: int, vocab: int
) -> None:
    rng = np.random.default_rng(seed)
    tokens_np = rng.integers(0, vocab, size=length)
    mask_np = induction_position_mask(tokens_np)
    if not mask_np.any():
        return  # nothing to test -- natural_induction_nll's own guard covers this case
    tokens = torch.from_numpy(tokens_np).long().unsqueeze(0)
    mask = torch.from_numpy(mask_np).unsqueeze(0)

    torch_gen = torch.Generator().manual_seed(seed)
    logits_a = torch.randn(1, length, vocab, generator=torch_gen)
    logits_b = logits_a.clone()
    eligible = mask[0, 1:]
    # Perturb only positions that predict a non-eligible target -- must not
    # change the result, since natural_induction_nll's mean is restricted
    # to the eligible predicting positions.
    for i in range(length - 1):
        if not eligible[i]:
            logits_b[0, i] += 1000.0 * (i + 1)

    nll_a = natural_induction_nll(logits_a, tokens, mask)
    nll_b = natural_induction_nll(logits_b, tokens, mask)
    torch.testing.assert_close(nll_a, nll_b)
