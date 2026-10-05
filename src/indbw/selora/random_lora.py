"""Spectrum-matched random LoRA: the null for "the SAE shift is specific to training".

Same layers, modules and singular values as a trained adapter, random singular vectors.
If a trained adapter shifts SAE reconstruction error no more than this does, the shift
is a property of the update's size and spectrum, not of what it learned.
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


def _orthonormal(rows: int, cols: int, g: torch.Generator) -> torch.Tensor:
    q, r = torch.linalg.qr(torch.randn(rows, cols, generator=g, dtype=torch.float64))
    out: torch.Tensor = q * torch.sign(torch.diagonal(r))  # Haar-uniform columns
    return out


@torch.no_grad()
def set_spectrum_matched_random(model: torch.nn.Module, seed: int) -> None:
    """Replace every adapter's Delta W = U S V^T by Q_u S Q_v^T with Haar-random Q_u, Q_v.

    Same layers, modules, rank, Frobenius norm *and* singular values as the trained update;
    only the directions are random. Stricter than matching the norm alone, which lets a
    random direction differ from the trained one in spectrum as well as in orientation.
    """
    g = torch.Generator().manual_seed(seed)
    n = 0
    for _name, mod in _lora_modules(model):
        a = mod.lora_A["default"].weight  # [r, d_in]
        b = mod.lora_B["default"].weight  # [d_out, r]
        scale = float(mod.scaling["default"])
        r = a.shape[0]
        sv = torch.linalg.svdvals(scale * (b.double() @ a.double()))[:r]
        qu = _orthonormal(b.shape[0], r, g)
        qv = _orthonormal(a.shape[1], r, g)
        root = torch.sqrt(sv / scale)
        b.copy_((qu * root).to(b.dtype))
        a.copy_((root[:, None] * qv.T).to(a.dtype))
        n += 1
    if n == 0:
        raise ValueError("model has no LoRA modules")
