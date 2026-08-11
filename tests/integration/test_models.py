"""Real-checkpoint tests for src/indbw/models.py.

PROJECT.md §2. Slow — downloads from HuggingFace on first run, then
hits the local HF cache. Not on the push path (CLAUDE.md tier 3).
"""

from __future__ import annotations

import torch

from indbw.models import load_checkpoint


def test_load_checkpoint_step_zero_matches_pythia_70m_architecture() -> None:
    # PROJECT.md §2: 6 layers, d_model=512, 8 heads, d_head=64.
    model = load_checkpoint(0, device="cpu")
    assert model.cfg.n_layers == 6
    assert model.cfg.d_model == 512
    assert model.cfg.n_heads == 8
    assert model.cfg.d_head == 64


def test_load_checkpoint_step_zero_runs_a_forward_pass() -> None:
    model = load_checkpoint(0, device="cpu")
    tokens = torch.randint(0, model.cfg.d_vocab, (1, 8))
    with torch.no_grad():
        logits = model(tokens)
    assert logits.shape == (1, 8, model.cfg.d_vocab)
    assert torch.isfinite(logits).all()
