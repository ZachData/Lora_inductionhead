"""Unit tests for indbw.natural_text -- the natural-text induction
objective used only by the G3-plateau diagnostic (PROJECT.md §11,
user-directed 2026-08-15 follow-up: is the *synthetic* repeated-random
objective itself unusually hard, independent of which matrix/head/rank/
checkpoint every prior G3 diagnostic varied?). Not part of the M1-M8
protocol or METRIC_VERSION's hashed surface (train.py's objective stays
synthetic repeated-random per PROJECT.md §10, 2026-08-14) -- these are
closed-form oracles and discrimination guards for a new diagnostic-only
readout, per CLAUDE.md's TDD contract ("every readout... must be shown to
return different answers on a known-positive and a known-negative").
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from indbw.natural_text import (
    induction_position_mask,
    natural_induction_nll,
    sample_natural_batches,
)


class TestInductionPositionMask:
    def test_oracle_hand_constructed_sequence(self) -> None:
        # tokens[3] (value 3) is a repeat of tokens[1] -- eligible only at
        # t=4, where the *predecessor* token (tokens[3]=3) has itself
        # already appeared earlier (at index 1, within tokens[0:3]).
        tokens = np.array([5, 3, 7, 3, 9])
        mask = induction_position_mask(tokens)
        np.testing.assert_array_equal(mask, [False, False, False, False, True])

    def test_position_zero_always_false(self) -> None:
        tokens = np.array([1, 1, 1, 1])
        assert not bool(induction_position_mask(tokens)[0])

    def test_output_length_matches_input(self) -> None:
        tokens = np.arange(17)
        assert induction_position_mask(tokens).shape == (17,)

    def test_rejects_non_1d_input(self) -> None:
        with pytest.raises(ValueError):
            induction_position_mask(np.zeros((2, 2)))

    def test_discrimination_repeated_pattern_vs_random(self) -> None:
        # A clean repeat: [1,2,3,4] followed by the same [1,2,3,4] -- every
        # position in the second copy after the first is induction-eligible
        # (matches second_copy_nll's own seam-exclusion convention: the
        # seam position itself, predicting the *first* token of the
        # repeat, is not eligible, since its predecessor has no earlier
        # occurrence yet).
        repeated = np.array([1, 2, 3, 4, 1, 2, 3, 4])
        mask_repeated = induction_position_mask(repeated)
        assert mask_repeated[5:].all()
        assert not mask_repeated[:5].any()

        # A random sequence over a large vocabulary essentially never
        # repeats a token by chance at this length -- near-zero eligible
        # positions, the "known negative" this guard exists to catch.
        rng = np.random.default_rng(0)
        random_seq = rng.integers(0, 50_000, size=64)
        mask_random = induction_position_mask(random_seq)
        assert mask_random.sum() <= 1  # allow for a rare chance collision


class TestNaturalInductionNll:
    def _one_hot_logits(
        self, targets: torch.Tensor, vocab: int, scale: float = 30.0
    ) -> torch.Tensor:
        batch, L = targets.shape
        logits = torch.zeros(batch, L, vocab)
        logits.scatter_(2, targets.unsqueeze(-1), scale)
        return logits

    def test_oracle_near_zero_when_eligible_positions_predicted_correctly(self) -> None:
        vocab = 10
        tokens = torch.tensor([[1, 2, 3, 2, 4]])  # tokens[3]=2 repeats tokens[1]=2
        mask = torch.from_numpy(induction_position_mask(tokens[0].numpy())).unsqueeze(0)
        assert mask.tolist() == [[False, False, False, False, True]]

        # logits at position L-2=3 must predict tokens[4]=4 to hit the one
        # eligible position (t=4); everything else can be garbage.
        logits = torch.randn(1, 5, vocab) * 5
        logits[0, 3] = self._one_hot_logits(torch.tensor([[4]]), vocab)[0, 0]
        nll = natural_induction_nll(logits, tokens, mask)
        assert nll.shape == (1,)
        assert nll.item() < 1e-3

    def test_invariant_to_logits_at_non_eligible_positions(self) -> None:
        vocab = 12
        tokens = torch.tensor([[1, 2, 3, 2, 4]])
        mask = torch.from_numpy(induction_position_mask(tokens[0].numpy())).unsqueeze(0)

        logits_a = torch.randn(1, 5, vocab)
        logits_a[0, 3] = self._one_hot_logits(torch.tensor([[4]]), vocab)[0, 0]
        logits_b = logits_a.clone()
        # Perturb every non-eligible-position's logits arbitrarily -- the
        # eligible position (index 3, predicting token index 4) is the
        # only one masked True, so the result must be unchanged.
        logits_b[0, 0] += 100.0
        logits_b[0, 1] -= 50.0
        logits_b[0, 2] *= 0.0

        nll_a = natural_induction_nll(logits_a, tokens, mask)
        nll_b = natural_induction_nll(logits_b, tokens, mask)
        torch.testing.assert_close(nll_a, nll_b)

    def test_raises_on_all_false_mask(self) -> None:
        vocab = 5
        tokens = torch.tensor([[1, 2, 3, 4]])
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        logits = torch.randn(1, 4, vocab)
        with pytest.raises(ValueError, match="eligible"):
            natural_induction_nll(logits, tokens, mask)

    def test_rejects_shape_mismatch(self) -> None:
        logits = torch.randn(1, 4, 5)
        tokens = torch.tensor([[1, 2, 3]])  # wrong length
        mask = torch.ones(1, 4, dtype=torch.bool)
        with pytest.raises(ValueError):
            natural_induction_nll(logits, tokens, mask)


class TestSampleNaturalBatches:
    def test_basic_shapes_on_repetitive_corpus(self) -> None:
        corpus = np.tile(np.arange(20), 20)  # 400 tokens, heavily repetitive
        tokens, mask = sample_natural_batches(corpus, n=6, seq_len=40, seed=0)
        assert tokens.shape == (6, 40)
        assert mask.shape == (6, 40)
        assert mask.dtype == torch.bool
        assert mask.any(dim=1).all()  # every sampled window has >=1 eligible position

    def test_deterministic_given_seed(self) -> None:
        corpus = np.tile(np.arange(20), 20)
        t1, m1 = sample_natural_batches(corpus, n=4, seq_len=30, seed=7)
        t2, m2 = sample_natural_batches(corpus, n=4, seq_len=30, seed=7)
        torch.testing.assert_close(t1, t2)
        torch.testing.assert_close(m1, m2)

    def test_corpus_shorter_than_seq_len_raises(self) -> None:
        with pytest.raises(ValueError):
            sample_natural_batches(np.arange(10), n=2, seq_len=20, seed=0)

    def test_no_repeats_in_corpus_raises_after_retries(self) -> None:
        # Strictly increasing tokens never repeat within any window, so no
        # window can ever have an eligible position -- the retry budget
        # must be exhausted and this must raise, not silently return an
        # all-False-mask batch that would later crash training deep inside
        # a run (CLAUDE.md's silent-failure guard).
        corpus = np.arange(500)
        with pytest.raises(RuntimeError, match="induction-eligible"):
            sample_natural_batches(corpus, n=1, seq_len=50, seed=0)
