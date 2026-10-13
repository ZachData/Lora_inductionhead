"""Shared fixtures. Unit tests use tiny models from here — never
instantiate a model inline in a test (CLAUDE.md, "Fixtures").
"""

from __future__ import annotations

import pytest
import torch
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


def _make_tiny_gemma2():  # type: ignore[no-untyped-def]
    """A randomly-initialized 4-layer Gemma2 (HF) for the selective-LoRA extension.

    float32, eager attention (Gemma2 softcapping), seeded. Never a real checkpoint.
    """
    import torch
    from transformers import Gemma2Config, Gemma2ForCausalLM

    torch.manual_seed(0)
    cfg = Gemma2Config(
        vocab_size=64,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        max_position_embeddings=64,
        sliding_window=16,
        pad_token_id=0,
        bos_token_id=2,
        eos_token_id=1,
        attn_implementation="eager",
    )
    model = Gemma2ForCausalLM(cfg)
    model.eval()
    return model


@pytest.fixture
def tiny_gemma2():  # type: ignore[no-untyped-def]
    """A randomly-initialized 4-layer Gemma2 (HF), float32, eager attention, seeded."""
    return _make_tiny_gemma2()


@pytest.fixture
def tiny_gemma2_factory():  # type: ignore[no-untyped-def]
    """Callable returning a fresh, identically-initialized tiny Gemma2 each call."""
    return _make_tiny_gemma2


@pytest.fixture
def tiny_sae_factory():  # type: ignore[no-untyped-def]
    """Callable (d_model, d_sae, seed) -> random float64 JumpReLU SAE."""
    import torch

    from indbw.selora.sae import JumpReLUSAE

    def make(d_model: int = 32, d_sae: int = 64, seed: int = 0) -> JumpReLUSAE:
        g = torch.Generator().manual_seed(seed)
        dt = torch.float64
        return JumpReLUSAE(
            W_enc=torch.randn(d_model, d_sae, generator=g, dtype=dt) * 0.2,
            W_dec=torch.randn(d_sae, d_model, generator=g, dtype=dt) * 0.2,
            b_enc=torch.zeros(d_sae, dtype=dt),
            b_dec=torch.zeros(d_model, dtype=dt),
            threshold=torch.full((d_sae,), 0.05, dtype=dt),
        )

    return make


class _Tok:
    """Char-level stand-in tokenizer over a 64-id vocabulary (pad=0, bos=2)."""

    padding_side = "left"
    pad_token_id = 0

    def encode(self, s: str, add_special_tokens: bool = True) -> list[int]:
        return [3 + (ord(c) % 60) for c in s]

    def __call__(self, texts, return_tensors="pt", padding=True):  # type: ignore[no-untyped-def]
        seqs = [[2] + self.encode(t) for t in texts]
        n = max(map(len, seqs))
        ids = torch.zeros(len(seqs), n, dtype=torch.long)
        mask = torch.zeros(len(seqs), n, dtype=torch.long)
        for i, s in enumerate(seqs):
            if self.padding_side == "left":
                ids[i, n - len(s) :], mask[i, n - len(s) :] = torch.tensor(s), 1
            else:
                ids[i, : len(s)], mask[i, : len(s)] = torch.tensor(s), 1
        return {"input_ids": ids, "attention_mask": mask}

    def batch_decode(self, ids, skip_special_tokens=True):  # type: ignore[no-untyped-def]
        return ["".join(chr(65 + (int(t) % 26)) for t in row) for row in ids]


@pytest.fixture
def stub_tokenizer():  # type: ignore[no-untyped-def]
    """Char-level stand-in tokenizer for the tiny Gemma2 (no real tokenizer is ever loaded)."""

    return _Tok()
