"""Shared fixtures. Unit tests use tiny models from here — never
instantiate a model inline in a test (CLAUDE.md, "Fixtures").
"""

from __future__ import annotations

import pytest
from transformer_lens import HookedTransformer, HookedTransformerConfig


@pytest.fixture
def tiny_model() -> HookedTransformer:
    """A randomly-initialized 2-layer HookedTransformer, small enough for
    fast unit tests -- CLAUDE.md's smoke-test spec (kind 5) names exactly
    this shape ("a randomly-initialized 2-layer model"). `cfg.seed` makes
    weight init deterministic across runs.
    """
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        d_head=16,
        n_heads=2,
        n_ctx=64,
        d_vocab=50,
        act_fn="relu",
        normalization_type="LN",
        seed=0,
    )
    model = HookedTransformer(cfg)
    model.eval()
    return model
