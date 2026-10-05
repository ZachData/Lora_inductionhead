"""Matched-norm random LoRA: the null for "the SAE shift is specific to training".

Same layers, same modules, same per-module ||Delta W||_F as a trained adapter,
but random direction. If a trained adapter shifts SAE reconstruction error
no more than this does, the shift is a norm effect, not a learned one.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import torch


def _lora_modules(model: torch.nn.Module) -> Iterator[tuple[str, Any]]:
    for name, mod in model.named_modules():
        if hasattr(mod, "lora_A") and "default" in getattr(mod, "lora_A", {}):
            yield name, mod


def delta_norms(model: torch.nn.Module) -> dict[str, float]:
    """||scale * B @ A||_F for every LoRA module, keyed by module name."""
    out: dict[str, float] = {}
    for name, mod in _lora_modules(model):
        a = mod.lora_A["default"].weight.detach().double()
        b = mod.lora_B["default"].weight.detach().double()
        out[name] = float((mod.scaling["default"] * (b @ a)).norm())
    if not out:
        raise ValueError("model has no LoRA modules")
    return out


@torch.no_grad()
def set_matched_norm_random(
    model: torch.nn.Module, target_norms: dict[str, float], seed: int
) -> None:
    """Overwrite every LoRA (A, B) with Gaussian factors rescaled to the target norm."""
    g = torch.Generator().manual_seed(seed)
    seen = set()
    for name, mod in _lora_modules(model):
        if name not in target_norms:
            raise KeyError(f"no target norm for module {name}")
        a = mod.lora_A["default"].weight
        b = mod.lora_B["default"].weight
        a.copy_(torch.randn(a.shape, generator=g, dtype=torch.float64).to(a.dtype))
        b.copy_(torch.randn(b.shape, generator=g, dtype=torch.float64).to(b.dtype))
        cur = (mod.scaling["default"] * (b.double() @ a.double())).norm()
        b.mul_((target_norms[name] / cur).to(b.dtype))
        seen.add(name)
    if seen != set(target_norms):
        raise KeyError(f"modules not matched: {sorted(set(target_norms) ^ seen)}")
