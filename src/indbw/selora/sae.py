"""JumpReLU sparse autoencoder (Gemma Scope format) and reconstruction metrics.

Gemma Scope `params.npz` holds `W_enc [d_model, d_sae]`, `W_dec [d_sae, d_model]`,
`b_enc [d_sae]`, `b_dec [d_model]`, `threshold [d_sae]`. Neuronpedia indexes
and serves the same SAEs; the weights themselves live in the
`google/gemma-scope-2b-pt-res` repository (one npz per layer / width / L0).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


@dataclass(frozen=True)
class JumpReLUSAE:
    W_enc: torch.Tensor
    W_dec: torch.Tensor
    b_enc: torch.Tensor
    b_dec: torch.Tensor
    threshold: torch.Tensor

    def __post_init__(self) -> None:
        d_model, d_sae = self.W_enc.shape
        if self.W_dec.shape != (d_sae, d_model):
            raise ValueError(f"W_dec {tuple(self.W_dec.shape)} does not match W_enc")
        if self.b_enc.shape != (d_sae,) or self.threshold.shape != (d_sae,):
            raise ValueError("b_enc / threshold must have shape [d_sae]")
        if self.b_dec.shape != (d_model,):
            raise ValueError("b_dec must have shape [d_model]")

    @property
    def d_model(self) -> int:
        return int(self.W_enc.shape[0])

    @property
    def d_sae(self) -> int:
        return int(self.W_enc.shape[1])

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"x last dim {x.shape[-1]} != d_model {self.d_model}")
        pre = x.to(self.W_enc.dtype) @ self.W_enc + self.b_enc
        return pre * (pre > self.threshold)

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return acts @ self.W_dec + self.b_dec

    def reconstruct(self, x: torch.Tensor) -> torch.Tensor:
        return self.decode(self.encode(x))


def load_sae_npz(path: str | Path, dtype: torch.dtype = torch.float32) -> JumpReLUSAE:
    """Load one Gemma Scope layer file. Raises on missing keys rather than defaulting."""
    z = np.load(path)
    missing = {"W_enc", "W_dec", "b_enc", "b_dec", "threshold"} - set(z.files)
    if missing:
        raise KeyError(f"{path}: missing {sorted(missing)}")
    t = {k: torch.as_tensor(z[k], dtype=dtype) for k in z.files}
    return JumpReLUSAE(t["W_enc"], t["W_dec"], t["b_enc"], t["b_dec"], t["threshold"])


def _check_mask(x: torch.Tensor, mask: torch.Tensor) -> None:
    if x.ndim != 3:
        raise ValueError(f"x must be [batch, seq, d_model], got {tuple(x.shape)}")
    if mask.shape != x.shape[:2]:
        raise ValueError(f"mask {tuple(mask.shape)} != x[:2] {tuple(x.shape[:2])}")
    if not torch.isfinite(x).all():
        raise ValueError("x contains NaN/Inf")
    if not bool((mask.sum(dim=1) > 0).all()):
        raise ValueError("a sequence has no valid (non-BOS, non-pad) tokens")


def content_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """Boolean mask of tokens that count: real tokens excluding position 0 (BOS).

    Gemma's BOS residual has a norm orders of magnitude above other tokens
    and is reconstructed very differently; including it dominates every
    average, so it is excluded by construction.
    """
    m = attention_mask.bool().clone()
    m[:, 0] = False
    return m


def sq_error_per_seq(x: torch.Tensor, sae: JumpReLUSAE, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence sum over valid tokens of ||x - sae(x)||^2, shape [batch]."""
    _check_mask(x, mask)
    xf = x.to(sae.W_enc.dtype)
    err = ((xf - sae.reconstruct(xf)) ** 2).sum(-1)
    return (err * mask).sum(dim=1)


def sq_dev_per_seq(x: torch.Tensor, mu: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Per-sequence sum over valid tokens of ||x - mu||^2 (the FVU denominator)."""
    _check_mask(x, mask)
    dev = ((x.to(mu.dtype) - mu) ** 2).sum(-1)
    return (dev * mask).sum(dim=1)


def fvu_per_seq(
    x: torch.Tensor, sae: JumpReLUSAE, mask: torch.Tensor, mu: torch.Tensor, den: torch.Tensor
) -> torch.Tensor:
    """Fraction of variance unexplained per sequence: sq_error / den.

    `den` is passed in, not recomputed from `x`: for a base-vs-fine-tuned
    comparison it must be the *base* model's `sq_dev_per_seq` on the same
    prompts, so the two FVUs share a scale and their difference is purely
    a change in reconstruction error. Raises on a zero/negative denominator
    (a constant residual stream makes FVU meaningless, not 0 or 1).
    """
    if bool((den <= 0).any()):
        raise ValueError("non-positive FVU denominator: activations are constant")
    return sq_error_per_seq(x, sae, mask) / den


def l0_per_seq(x: torch.Tensor, sae: JumpReLUSAE, mask: torch.Tensor) -> torch.Tensor:
    """Mean number of active features per valid token, per sequence."""
    _check_mask(x, mask)
    active = (sae.encode(x) > 0).sum(-1).to(torch.float64)
    return (active * mask).sum(dim=1) / mask.sum(dim=1)
