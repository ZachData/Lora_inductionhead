"""Natural-text induction objective.

Diagnostic-only, for the G3-plateau investigation (PROJECT.md §11 /
REVIEW.md, 2026-08-15 user-directed follow-up to five negative
diagnostics that each varied a different axis -- objective correctness,
rank, which matrix, which head -- while holding the synthetic
repeated-random *objective itself* fixed throughout. This module gives
the diagnostic script a natural-text alternative to train against, so
that axis can finally be varied too. Recovery R is still evaluated with
`indbw.train.compute_recovery` on the standard synthetic eval set,
unchanged -- only the training signal changes, keeping R comparable
across every prior G3 diagnostic (PROJECT.md §11: "evaluate on the other
as a held-out check").

Not part of the M1-M8 protocol: `train.py`'s objective stays synthetic
repeated-random (PROJECT.md §10, 2026-08-14 decision), and this module
is never imported by `train.py`, `sweep.py`, or `run_sweep.py`. Not part
of METRIC_VERSION's hashed surface (`metric_hash.py` only covers
algebra.py/probes.py) -- this defines a diagnostic training signal, not
one of PROJECT.md §5's metrics.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn.functional as F


def induction_position_mask(tokens: np.ndarray) -> np.ndarray:
    """[seq_len] bool array. `mask[t]` (t >= 1) is True iff `tokens[t-1]`
    occurs somewhere in `tokens[0 : t-1]` -- i.e. predicting `tokens[t]`
    from the representation at position `t-1` is an induction-eligible
    prediction: an earlier occurrence of the same preceding token exists,
    so a prefix-matching-and-copy strategy has something to copy from.
    `mask[0]` is always False (no earlier context to compare against).

    This is the natural-text analogue of `indbw.train.second_copy_nll`'s
    eligible positions: on a repeated-random sequence built by
    `indbw.evalset.build_eval_tokens`, this mask is True on exactly the
    same positions `second_copy_nll` scores (verified by
    `tests/unit/test_natural_text.py`'s discrimination test) -- the seam
    position itself (predicting the first token of a repeat) is excluded
    in both, since that token's predecessor has no earlier occurrence yet.
    """
    tokens = np.asarray(tokens)
    if tokens.ndim != 1:
        raise ValueError(f"expected a 1D token array, got shape {tokens.shape}")
    length = tokens.shape[0]
    mask = np.zeros(length, dtype=bool)
    seen: set[int] = set()
    for t in range(1, length):
        prev_tok = int(tokens[t - 1])
        mask[t] = prev_tok in seen
        seen.add(prev_tok)
    return mask


def natural_induction_nll(
    logits: torch.Tensor, tokens: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    """Per-example mean cross-entropy NLL of predicting `tokens[:, t]`
    from `logits[:, t-1, :]`, restricted to positions where `mask[:, t]`
    is True (`induction_position_mask`). An example with zero eligible
    positions raises rather than silently contributing a 0/0 "loss" that
    would look like a valid near-zero number (CLAUDE.md's silent-failure
    guard -- an empty mean here would be indistinguishable from perfect
    prediction).

    logits: [batch, L, vocab]. tokens: [batch, L] long. mask: [batch, L]
    bool. Returns [batch].
    """
    if logits.ndim != 3:
        raise ValueError(f"expected logits [batch, L, vocab], got shape {tuple(logits.shape)}")
    batch, length, vocab = logits.shape
    if tuple(tokens.shape) != (batch, length):
        raise ValueError(
            f"tokens shape {tuple(tokens.shape)} must match logits' [batch, L]={(batch, length)}"
        )
    if tuple(mask.shape) != (batch, length):
        raise ValueError(
            f"mask shape {tuple(mask.shape)} must match logits' [batch, L]={(batch, length)}"
        )
    eligible = mask[:, 1:]
    if not eligible.any(dim=1).all():
        empty = (~eligible.any(dim=1)).nonzero(as_tuple=True)[0].tolist()
        raise ValueError(f"examples {empty} have zero induction-eligible positions")

    pred_logits = logits[:, :-1, :]  # position t-1 predicts token t, t in [1, L-1]
    targets = tokens[:, 1:]

    nll = F.cross_entropy(
        pred_logits.reshape(-1, vocab), targets.reshape(-1), reduction="none"
    ).reshape(batch, length - 1)
    nll = nll.masked_fill(~eligible, 0.0)
    counts = eligible.sum(dim=1)
    return nll.sum(dim=1) / counts


def sample_natural_batches(
    corpus_tokens: np.ndarray, n: int, seq_len: int, seed: int, max_retries: int = 50
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample `n` random contiguous windows of length `seq_len` from a 1D
    pre-tokenized natural-text corpus, each paired with its
    `induction_position_mask`. Returns `(tokens [n, seq_len] long,
    mask [n, seq_len] bool)`.

    A window with zero eligible positions is resampled (up to
    `max_retries` attempts) rather than included -- `natural_induction_nll`
    would raise on it anyway, and failing here at data-construction time
    is more diagnosable than failing on some arbitrary future training
    step deep inside a run.
    """
    corpus_tokens = np.asarray(corpus_tokens)
    if corpus_tokens.ndim != 1:
        raise ValueError(f"expected a 1D corpus array, got shape {corpus_tokens.shape}")
    if n < 1:
        raise ValueError(f"n must be >= 1, got {n}")
    if seq_len < 2:
        raise ValueError(f"seq_len must be >= 2, got {seq_len}")
    if corpus_tokens.shape[0] < seq_len:
        raise ValueError(f"corpus length {corpus_tokens.shape[0]} < seq_len {seq_len}")

    rng = np.random.default_rng(seed)
    max_start = corpus_tokens.shape[0] - seq_len
    out_tokens = np.empty((n, seq_len), dtype=np.int64)
    out_mask = np.empty((n, seq_len), dtype=bool)
    for i in range(n):
        for _attempt in range(max_retries):
            start = int(rng.integers(0, max_start + 1))
            window = corpus_tokens[start : start + seq_len]
            window_mask = induction_position_mask(window)
            if window_mask.any():
                break
        else:
            raise RuntimeError(
                f"could not find a window with any induction-eligible position after "
                f"{max_retries} attempts (corpus too short/non-repetitive for seq_len={seq_len})"
            )
        out_tokens[i] = window
        out_mask[i] = window_mask
    return torch.from_numpy(out_tokens).long(), torch.from_numpy(out_mask)


def load_wikitext_tokens(tokenizer: Any, split: str = "train") -> np.ndarray:
    """Tokenize HuggingFace's `wikitext-2-raw-v1` `split` into one flat 1D
    token array via `tokenizer` (e.g. a loaded `HookedTransformer`'s own
    `.tokenizer`). Orchestration only -- network/disk I/O via the
    `datasets` library, same "delegate rather than duplicate" rationale as
    `indbw.models`' checkpoint loader, and likewise not unit-tested here
    (no network access from `tests/unit`); exercised directly by whichever
    diagnostic script calls it.
    """
    from datasets import load_dataset  # type: ignore[import-untyped]

    # "wikitext" (the legacy loading-script repo) is no longer resolvable
    # under datasets>=5's script-loading removal; "Salesforce/wikitext" is
    # the same data, parquet-backed, under HF's current dataset-loading path.
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split=split)
    text = "\n".join(row["text"] for row in ds if row["text"].strip())
    ids = tokenizer(text, return_tensors=None)["input_ids"]
    return np.asarray(ids, dtype=np.int64)
