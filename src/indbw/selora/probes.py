"""SAE-facing probes: residual capture, divergence statistic with a permutation null,
spliced cross-entropy.

Definitions here are metric-defining (hashed under SELORA_METRIC_VERSION).
"""

from __future__ import annotations

from collections.abc import Sequence
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from indbw.selora.sae import JumpReLUSAE, content_mask
from indbw.selora.stats import rank_pvalue


def _layer_list(model: Any) -> torch.nn.ModuleList:
    inner = model.get_base_model() if hasattr(model, "get_base_model") else model
    layers = inner.model.layers
    assert isinstance(layers, torch.nn.ModuleList)
    return layers


def _out_tensor(out: object) -> torch.Tensor:
    return out[0] if isinstance(out, tuple) else out  # type: ignore[return-value]


@torch.no_grad()
def capture_resid(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layers: Sequence[int],
) -> dict[int, torch.Tensor]:
    """Residual stream *after* decoder layer l, float32, for each l in `layers`.

    Captured with a forward hook on the layer, not `output_hidden_states`:
    HF's last hidden state is post-final-norm, so index L would silently be
    a different quantity from the pre-norm stream Gemma Scope was trained on.
    """
    store: dict[int, torch.Tensor] = {}
    handles = []
    mods = _layer_list(model)
    for li in layers:
        handles.append(
            mods[li].register_forward_hook(
                lambda _m, _i, out, li=li: store.__setitem__(li, _out_tensor(out).detach().float())
            )
        )
    try:
        model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        for h in handles:
            h.remove()
    if set(store) != set(layers):
        raise RuntimeError(f"hooks did not fire for layers {sorted(set(layers) - set(store))}")
    return store


@contextmanager
def _splice(model: Any, layer: int, sae: JumpReLUSAE, replace: torch.Tensor) -> Iterator[None]:
    """Swap decoder-layer output for its SAE reconstruction at positions where `replace` is True."""

    def hook(_m: Any, _i: Any, out: Any) -> Any:
        h = _out_tensor(out)
        rec = sae.reconstruct(h.float()).to(h.dtype)
        new = torch.where(replace.unsqueeze(-1), rec, h)
        return (new,) + tuple(out[1:]) if isinstance(out, tuple) else new

    handle = _layer_list(model)[layer].register_forward_hook(hook)
    try:
        yield
    finally:
        handle.remove()


@torch.no_grad()
def spliced_ce_per_seq(
    model: torch.nn.Module,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    layer: int | None,
    sae: JumpReLUSAE | None,
) -> torch.Tensor:
    """Mean next-token CE per sequence, optionally with the SAE spliced at `layer`.

    BOS position is never replaced. `layer=None` is the clean forward pass.
    Delta-CE = spliced - clean is the downstream-faithfulness metric.
    """
    replace = content_mask(attention_mask)

    def run() -> torch.Tensor:
        logits = model(input_ids=input_ids, attention_mask=attention_mask).logits.float()
        ce = F.cross_entropy(logits[:, :-1].transpose(1, 2), input_ids[:, 1:], reduction="none")
        m = attention_mask[:, 1:].float()
        return (ce * m).sum(1) / m.sum(1)

    if layer is None:
        return run()
    if sae is None:
        raise ValueError("sae is required when layer is given")
    with _splice(model, layer, sae, replace):
        return run()


def welch_effects(acts: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """Per-feature Welch t between label 1 and label 0. Dead/constant features -> 0."""
    a = np.asarray(acts, dtype=np.float64)
    y = np.asarray(labels).astype(bool)
    n1, n0 = int(y.sum()), int((~y).sum())
    if n1 < 2 or n0 < 2:
        raise ValueError(f"need >=2 prompts per class, got {n1}/{n0}")
    x1, x0 = a[y], a[~y]
    se2 = x1.var(0, ddof=1) / n1 + x0.var(0, ddof=1) / n0
    diff = x1.mean(0) - x0.mean(0)
    pos = se2[se2 > 0]
    if pos.size == 0:
        return np.zeros_like(diff)
    # A feature with zero within-class variance but a mean gap is a perfect separator, not a
    # dead feature: floor the variance (relative, so it is scale-free) rather than return 0.
    floor = 1e-6 * float(np.median(pos))
    return np.asarray(diff / np.sqrt(np.maximum(se2, floor)))


def topk_mean_abs(effects: np.ndarray, k: int) -> float:
    if k < 1 or k > effects.size:
        raise ValueError(f"k={k} out of range for {effects.size} features")
    return float(np.sort(np.abs(effects))[-k:].mean())


def _check_acts(acts: np.ndarray, labels: np.ndarray) -> None:
    if acts.ndim != 2 or labels.shape != (acts.shape[0],):
        raise ValueError(f"acts {acts.shape} / labels {labels.shape} mismatch")
    if not np.all(np.isfinite(acts)):
        raise ValueError("acts contain NaN/Inf")
    if np.ptp(acts) == 0:
        raise ValueError("acts are constant: divergence is undefined, not zero")


def divergence_with_null(
    acts: np.ndarray,
    labels: np.ndarray,
    k: int,
    n_perm: int,
    rng: np.random.Generator,
    strata: np.ndarray | None = None,
    chunk: int = 200,
) -> dict[str, float | np.ndarray]:
    """Top-k mean |Welch t| between two prompt classes, against a label-permutation null.

    The raw top-k statistic is positive even when the classes are identical
    (selection of the k largest of d_sae noisy effects), and its bias
    differs by layer through sparsity. Only `excess = observed - null mean`
    and the permutation p-value mean anything. Labels are permuted within
    `strata` (e.g. prompt-length bins) when given, so a length difference
    cannot masquerade as a content difference.
    """
    acts = np.asarray(acts, dtype=np.float32)
    y = np.asarray(labels).astype(bool)
    _check_acts(acts, y)
    obs = topk_mean_abs(welch_effects(acts, y), k)
    n = acts.shape[0]
    strata_arr = np.zeros(n, dtype=int) if strata is None else np.asarray(strata)
    null = np.empty(n_perm)
    for start in range(0, n_perm, chunk):
        stop = min(start + chunk, n_perm)
        perms = np.empty((stop - start, n), dtype=bool)
        for b in range(stop - start):
            p = y.copy()
            for s in np.unique(strata_arr):
                idx = np.flatnonzero(strata_arr == s)
                p[idx] = y[rng.permutation(idx)]
            perms[b] = p
        for b in range(stop - start):
            null[start + b] = topk_mean_abs(welch_effects(acts, perms[b]), k)
    return {
        "observed": obs,
        "null_mean": float(null.mean()),
        "excess": obs - float(null.mean()),
        "p_value": rank_pvalue(obs, null, "greater"),
        "null": null,
    }


def peak_layer_stability(
    excess_by_layer: np.ndarray,
    lo: int,
    hi: int,
    rng: np.random.Generator,
    noise_sd: np.ndarray,
    n_boot: int,
) -> float:
    """Fraction of parametric-bootstrap draws whose argmax layer lies in [lo, hi].

    `excess_by_layer[l]` is the observed excess, `noise_sd[l]` its standard
    error (from a prompt-level bootstrap done by the caller). Draws are
    Gaussian around the observed excess. Pre-registered use: the old
    "peak at layer 8" claim is supported only if this is >= 0.8.
    """
    ex = np.asarray(excess_by_layer, dtype=np.float64)
    sd = np.asarray(noise_sd, dtype=np.float64)
    if ex.shape != sd.shape or ex.ndim != 1:
        raise ValueError("excess_by_layer and noise_sd must be matching 1-D arrays")
    draws = ex + sd * rng.standard_normal((n_boot, ex.size))
    peaks = draws.argmax(axis=1)
    return float(np.mean((peaks >= lo) & (peaks <= hi)))
