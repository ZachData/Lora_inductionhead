"""Spec for selora.placement and selora.random_lora."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from peft import get_peft_model

from indbw.selora import placement
from indbw.selora.random_lora import delta_norms, set_matched_norm_random


def test_named_sets_for_gemma_2b() -> None:
    s = placement.named_sets(26, 13)
    assert s["late"] == tuple(range(13, 26))
    assert s["early"] == tuple(range(13))
    assert s["all"] == tuple(range(26))


def test_sae_guided_sets_pick_extremes() -> None:
    scores = np.arange(26, dtype=float)[::-1]  # layer 0 highest
    g = placement.sae_guided_sets(scores, 13)
    assert g["sae_top"] == tuple(range(13))
    assert g["sae_bottom"] == tuple(range(13, 26))
    with pytest.raises(ValueError):
        placement.sae_guided_sets(np.array([1.0, np.nan]), 1)


def test_random_sets_distinct_sized_seeded_and_exclude_named() -> None:
    named = set(placement.named_sets(26, 13).values())
    a = placement.random_sets(26, 13, 24, seed=7, exclude=named)
    b = placement.random_sets(26, 13, 24, seed=7, exclude=named)
    assert a == b and len(set(a)) == 24
    assert all(len(x) == 13 and list(x) == sorted(x) and not set(x) - set(range(26)) for x in a)
    assert not set(a) & named
    assert a != placement.random_sets(26, 13, 24, seed=8, exclude=named)


def test_random_sets_cannot_exceed_population() -> None:
    with pytest.raises(ValueError):
        placement.random_sets(4, 2, 7, seed=0, exclude=set())  # C(4,2)=6


def test_random_layer_means_are_unbiased() -> None:
    # Each layer should land in a random 13-of-26 set half the time: guards a biased sampler.
    sets = placement.random_sets(26, 13, 400, seed=0, exclude=set())
    freq = np.bincount(np.concatenate([np.array(s) for s in sets]), minlength=26) / 400
    # binomial sd = 0.025; 5 sd covers all 26 layers comfortably
    assert np.abs(freq - 0.5).max() < 0.125


def _adapted(model, layers):  # type: ignore[no-untyped-def]
    return get_peft_model(model, placement.lora_config(layers, r=2, alpha=4, dropout=0.0))


def test_lora_attaches_only_to_chosen_layers_and_freezes_everything_else(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    pm = _adapted(tiny_gemma2, (2, 3))
    lora_names = [n for n, _ in pm.named_modules() if n.endswith("lora_A")]
    assert lora_names and all(".layers.2." in n or ".layers.3." in n for n in lora_names)
    assert len(lora_names) == 2 * 4  # two layers x q,k,v,o
    trainable = {n for n, p in pm.named_parameters() if p.requires_grad}
    assert trainable and all("lora_" in n for n in trainable)  # LN, embeddings, MLP frozen


def test_matched_norm_random_matches_norms_and_changes_direction(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    pm = _adapted(tiny_gemma2, (1, 2))
    g = torch.Generator().manual_seed(0)
    with torch.no_grad():  # stand-in for "trained": B starts at zero, so make it nonzero
        for n, p in pm.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)
    target = delta_norms(pm)
    before = {
        n: (
            m.scaling["default"] * (m.lora_B["default"].weight @ m.lora_A["default"].weight)
        ).clone()
        for n, m in pm.named_modules()
        if hasattr(m, "lora_B") and "default" in m.lora_B
    }
    set_matched_norm_random(pm, target, seed=1)
    after_norms = delta_norms(pm)
    for k, v in target.items():
        assert after_norms[k] == pytest.approx(v, rel=1e-5)  # float32 factor rescale
    for n, m in pm.named_modules():
        if n in before:
            now = m.scaling["default"] * (m.lora_B["default"].weight @ m.lora_A["default"].weight)
            cos = torch.nn.functional.cosine_similarity(now.flatten(), before[n].flatten(), dim=0)
            assert abs(float(cos.detach())) < 0.9
    a1 = (
        next(m for m in pm.modules() if hasattr(m, "lora_A") and "default" in m.lora_A)
        .lora_A["default"]
        .weight.detach()
        .clone()
    )
    set_matched_norm_random(pm, target, seed=2)
    a2 = (
        next(m for m in pm.modules() if hasattr(m, "lora_A") and "default" in m.lora_A)
        .lora_A["default"]
        .weight.detach()
    )
    assert not torch.equal(a1, a2)  # a different seed is a different draw


def test_matched_norm_requires_all_modules(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    pm = _adapted(tiny_gemma2, (3,))
    with pytest.raises(KeyError):
        set_matched_norm_random(pm, {}, seed=0)
    with pytest.raises(ValueError):
        delta_norms(torch.nn.Linear(2, 2))  # no LoRA modules: refuse, don't return {}
