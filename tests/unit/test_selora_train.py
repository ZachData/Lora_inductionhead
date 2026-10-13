"""Spec for selora.train: the loop trains, and its fail-fast guards actually fire."""

from __future__ import annotations

import pytest
import torch
from peft import get_peft_model

from indbw.selora.placement import lora_config
from indbw.selora.train import TrainConfig, collate, train_lora


def _examples(n: int = 12, t: int = 12) -> list[list[int]]:
    g = torch.Generator().manual_seed(0)
    base = torch.randint(3, 64, (3, t), generator=g)
    return [base[i % 3].tolist() for i in range(n)]  # 3 distinct sequences: learnable


def test_collate_masks_padding_in_labels() -> None:
    b = collate([[5, 6, 7], [5, 6]], pad_id=0)
    assert b["labels"][1, 2].item() == -100 and b["attention_mask"][1].tolist() == [1, 1, 0]


def test_training_lowers_loss_and_moves_only_lora(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    pm = get_peft_model(tiny_gemma2, lora_config((0, 1, 2, 3), r=4, alpha=8, dropout=0.0))
    frozen_before = {n: p.detach().clone() for n, p in pm.named_parameters() if not p.requires_grad}
    losses = train_lora(
        pm, _examples(), 0, TrainConfig(lr=1e-2, batch_size=2, grad_accum=1, max_steps=60)
    )
    # Attention-only rank-4 LoRA on a random 4-layer model memorising 3 sequences: loss falls
    # ~10% in 60 steps (measured once at authoring); 5% is the bar, far above step noise.
    assert losses[-1] < losses[0] * 0.95
    for n, p in pm.named_parameters():
        if n in frozen_before:
            assert torch.equal(p, frozen_before[n]), f"frozen parameter {n} changed"


def test_nan_loss_raises(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    pm = get_peft_model(tiny_gemma2, lora_config((3,), r=2, alpha=4, dropout=0.0))
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if "lora_B" in n:
                p.fill_(float("nan"))
    with pytest.raises(FloatingPointError):
        train_lora(pm, _examples(), 0, TrainConfig(batch_size=2, grad_accum=1, max_steps=2))


def test_wall_clock_budget_raises(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    pm = get_peft_model(tiny_gemma2, lora_config((3,), r=2, alpha=4, dropout=0.0))
    with pytest.raises(TimeoutError):
        train_lora(
            pm,
            _examples(),
            0,
            TrainConfig(batch_size=2, grad_accum=1, max_steps=50, wall_clock_budget_s=0.0),
        )


def test_no_trainable_params_raises(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    for p in tiny_gemma2.parameters():
        p.requires_grad_(False)
    with pytest.raises(ValueError):
        train_lora(tiny_gemma2, _examples(), 0, TrainConfig(max_steps=1))


def test_batch_order_depends_only_on_seed_and_shape() -> None:
    from indbw.selora.train import batch_order

    a = batch_order(10, 2, 2, 7, seed=3)
    assert a == batch_order(10, 2, 2, 7, seed=3) and a != batch_order(10, 2, 2, 7, seed=4)
    assert len(a) == 7 * 2 and all(len(b) == 2 for b in a)
    flat = [i for b in a[:5] for i in b]
    assert sorted(flat) == list(range(10))  # first epoch is a permutation


def test_two_arms_see_identical_batches_for_the_same_seed(tiny_gemma2_factory) -> None:  # type: ignore[no-untyped-def]
    # Common random numbers: arm differences must not include data-order noise.
    seen: dict[str, list[list[int]]] = {"a": [], "b": []}
    for key, layers in (("a", (0, 1)), ("b", (2, 3))):
        m = tiny_gemma2_factory()
        pm = get_peft_model(m, lora_config(layers, r=2, alpha=4, dropout=0.0))
        m.model.embed_tokens.register_forward_pre_hook(
            lambda _m, inp, k=key: seen[k].append(inp[0].tolist())
        )
        train_lora(pm, _examples(), 0, TrainConfig(batch_size=2, grad_accum=2, max_steps=3, seed=9))
    assert seen["a"] == seen["b"] and len(seen["a"]) == 6
