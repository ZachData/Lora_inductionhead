"""Repeated-random eval-set construction, shared across gates/rows.

PROJECT.md §5's fixed protocol: T tokens sampled uniformly from vocab,
concatenated to length 2T, N_eval such sequences, fixed seed.

`scripts/g0_sweep.py` and `scripts/run_g1.py` each carry their own inline
copy of this (frozen, already-closed gates -- not touched here to avoid
any regression risk on results already committed). This module exists so
`train.py` and `scripts/run_g2.py` (and everything downstream) share one
definition instead of adding a third copy.
"""

from __future__ import annotations

import numpy as np
import torch


def build_eval_tokens(n_eval: int, T: int, seed: int, d_vocab: int) -> torch.Tensor:
    """[n_eval, 2T] long tensor: T uniform-random tokens, repeated once."""
    if n_eval < 1:
        raise ValueError(f"n_eval must be >= 1, got {n_eval}")
    if T < 1:
        raise ValueError(f"T must be >= 1, got {T}")
    if d_vocab < 2:
        raise ValueError(f"d_vocab must be >= 2, got {d_vocab}")
    rng = np.random.default_rng(seed)
    first_half = rng.integers(0, d_vocab, size=(n_eval, T))
    sequences = np.concatenate([first_half, first_half], axis=1)
    return torch.from_numpy(sequences).long()
