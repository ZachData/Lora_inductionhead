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

import huggingface_hub.utils._headers as _hf_headers
from transformer_lens import HookedTransformer
from transformer_lens.loading_from_pretrained import get_checkpoint_labels

MODEL_NAME = "EleutherAI/pythia-70m"

# Work around an unpatched bug in transformer_lens's pythia-checkpoint
# loading path (loading_from_pretrained.py's get_pretrained_state_dict,
# still present as of 3.7.1, the latest release as of writing): it calls
# AutoModelForCausalLM.from_pretrained(..., token=os.environ.get("HF_TOKEN",
# "")) unconditionally, unlike its sibling branches (e.g. stanford-crfm)
# which guard with `if len(token) > 0 else None`. With no HF_TOKEN set,
# huggingface_hub's get_token_to_send (utils/_headers.py) only special-cases
# None/True/False -- an explicit "" is returned as-is and turned into an
# `Authorization: Bearer ` header. The old requests-based huggingface_hub
# sent that malformed header without complaint; huggingface_hub's current
# httpx backend validates headers strictly and raises LocalProtocolError
# ("Illegal header value b'Bearer '"), which broke every checkpointed load.
# Normalizing a falsy-but-non-None token to None matches build_hf_headers'
# own documented "False or None -> no token" contract, so this only changes
# behavior in the exact case transformer_lens already intended: no real
# token was ever supplied.
_original_get_token_to_send = _hf_headers.get_token_to_send


def _get_token_to_send_treating_empty_as_none(token: bool | str | None) -> str | None:
    return _original_get_token_to_send(token) or None


_hf_headers.get_token_to_send = _get_token_to_send_treating_empty_as_none


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
