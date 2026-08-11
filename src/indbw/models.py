"""Pythia checkpoint loader and local cache.

PROJECT.md §2. Target: EleutherAI/pythia-70m by checkpoint revision.

Checkpoint validation and revision naming are TransformerLens's own
(loading_from_pretrained.get_checkpoint_labels / PYTHIA_CHECKPOINTS) —
delegated to, not duplicated here, so this module can't silently drift
from what HookedTransformer.from_pretrained actually accepts. Caching
is HuggingFace's standard local hub cache (persists across runs by
default); no bespoke cache layer.
"""

from __future__ import annotations

from typing import Any

from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_checkpoint_labels

MODEL_NAME = "EleutherAI/pythia-70m"


def checkpoint_steps() -> list[int]:
    """The full Pythia training-step grid for MODEL_NAME (PROJECT.md §2)."""
    labels, label_type = get_checkpoint_labels(MODEL_NAME)
    if label_type != "step":
        raise ValueError(f"expected step-labelled checkpoints for {MODEL_NAME}, got {label_type!r}")
    return labels


def load_checkpoint(step: int, **kwargs: Any) -> HookedTransformer:
    """Load MODEL_NAME at the given training step.

    step must be on the Pythia checkpoint grid (checkpoint_steps());
    validated before any network/HF call. Extra kwargs (e.g. device)
    are forwarded to HookedTransformer.from_pretrained.
    """
    valid_steps = checkpoint_steps()
    if step not in valid_steps:
        raise ValueError(f"step {step} is not on the Pythia checkpoint grid for {MODEL_NAME}")
    return HookedTransformer.from_pretrained(MODEL_NAME, checkpoint_value=step, **kwargs)
