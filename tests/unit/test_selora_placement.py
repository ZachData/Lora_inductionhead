"""Spec for selora.placement and selora.random_lora."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from peft import get_peft_model

from indbw.selora import placement
from indbw.selora.random_lora import delta_norms, set_spectrum_matched_random


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


def _set_trained(pm, seed: int = 0) -> None:  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(seed)
    with torch.no_grad():  # stand-in for "trained": B starts at zero, so make it nonzero
        for n, p in pm.named_parameters():
            if "lora_" in n:
                p.copy_(torch.randn(p.shape, generator=g) * 0.3)


def _deltas(pm) -> dict[str, torch.Tensor]:  # type: ignore[no-untyped-def]
    return {
        n: (m.scaling["default"] * (m.lora_B["default"].weight @ m.lora_A["default"].weight))
        .detach()
        .double()
        for n, m in pm.named_modules()
        if hasattr(m, "lora_B") and "default" in m.lora_B
    }


def test_spectrum_matched_random_keeps_singular_values_and_changes_subspace(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    pm = _adapted(tiny_gemma2, (1, 2))
    _set_trained(pm)
    before = _deltas(pm)
    set_spectrum_matched_random(pm, seed=1)
    after = _deltas(pm)
    r = 2
    for n, d0 in before.items():
        s0 = torch.linalg.svdvals(d0)[:r]
        s1 = torch.linalg.svdvals(after[n])[:r]
        torch.testing.assert_close(s1, s0, rtol=1e-5, atol=1e-7)  # float32 factors
        u0 = torch.linalg.svd(d0).U[:, :r]
        u1 = torch.linalg.svd(after[n]).U[:, :r]
        overlap = float((u0.T @ u1).pow(2).sum()) / r  # chance level ~ r / d_out
        assert overlap < 0.6, f"{n}: random subspace overlaps trained one ({overlap:.2f})"
    assert delta_norms(pm).keys() == {k for k in before}


def test_spectrum_matched_random_is_seeded(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    pm = _adapted(tiny_gemma2, (3,))
    _set_trained(pm)
    snapshot = {n: p.detach().clone() for n, p in pm.named_parameters() if "lora_" in n}
    set_spectrum_matched_random(pm, seed=5)
    a = _deltas(pm)
    with torch.no_grad():
        for n, p in pm.named_parameters():
            if n in snapshot:
                p.copy_(snapshot[n])
    set_spectrum_matched_random(pm, seed=5)
    for n, d in _deltas(pm).items():
        torch.testing.assert_close(d, a[n])


def test_delta_norms_refuses_a_model_without_lora() -> None:
    with pytest.raises(ValueError):
        delta_norms(torch.nn.Linear(2, 2))
