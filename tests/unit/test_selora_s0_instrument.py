"""S0's instrument oracle: capture is bit-identical below the first adapted layer, and differs at/above it.

Both halves are needed: 'identical below' alone would also hold for a capture that ignores the adapter.
"""

from __future__ import annotations

import torch
from peft import get_peft_model

from indbw.selora import probes
from indbw.selora.placement import lora_config
from indbw.selora.train import TrainConfig, train_lora


def test_zero_difference_below_first_adapted_layer_and_nonzero_above(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(3, 64, (4, 10), generator=g)
    ids[:, 0] = 2
    attn = torch.ones_like(ids)
    base = {
        k: v.clone() for k, v in probes.capture_resid(tiny_gemma2, ids, attn, [0, 1, 2, 3]).items()
    }
    pm = get_peft_model(tiny_gemma2, lora_config((2, 3), r=2, alpha=4, dropout=0.0))
    train_lora(pm, ids.tolist(), 0, TrainConfig(lr=1e-2, batch_size=2, grad_accum=1, max_steps=5))
    ft = probes.capture_resid(pm, ids, attn, [0, 1, 2, 3])
    for li in (0, 1):
        assert torch.equal(base[li], ft[li]), f"layer {li} changed below the first adapted layer"
    for li in (2, 3):
        assert (base[li] - ft[li]).abs().max() > 0, f"layer {li} unchanged: adapter not applied"
