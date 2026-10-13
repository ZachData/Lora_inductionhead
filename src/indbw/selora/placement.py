"""Layer-set construction for the placement arms, and the PEFT config.

Random arms are uniform draws of `n_adapt`-subsets of the layers, distinct
from each other and from every named contiguous / SAE-guided set. That
matters statistically: a named set fixed *in advance* is one more exchangeable
draw under "placement is irrelevant", so its rank among the random draws is
an exact test (stats.rank_pvalue). One random draw, as in the original report,
supports no test at all.
"""

from __future__ import annotations

from math import comb

import numpy as np

ATTN_TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj")


def named_sets(n_layers: int, n_adapt: int) -> dict[str, tuple[int, ...]]:
    if not 0 < n_adapt <= n_layers:
        raise ValueError("n_adapt must be in [1, n_layers]")
    return {
        "all": tuple(range(n_layers)),
        "late": tuple(range(n_layers - n_adapt, n_layers)),
        "early": tuple(range(n_adapt)),
    }


def sae_guided_sets(scores: np.ndarray, n_adapt: int) -> dict[str, tuple[int, ...]]:
    """Top / bottom `n_adapt` layers by a per-layer score computed *before* training."""
    s = np.asarray(scores, dtype=np.float64)
    if s.ndim != 1 or not np.all(np.isfinite(s)):
        raise ValueError("scores must be a finite 1-D array")
    order = np.argsort(s, kind="stable")
    return {
        "sae_top": tuple(sorted(int(i) for i in order[-n_adapt:])),
        "sae_bottom": tuple(sorted(int(i) for i in order[:n_adapt])),
    }


def random_sets(
    n_layers: int, n_adapt: int, n_draws: int, seed: int, exclude: set[tuple[int, ...]]
) -> list[tuple[int, ...]]:
    if n_draws > comb(n_layers, n_adapt) - len(exclude):
        raise ValueError("more random draws requested than distinct subsets exist")
    rng = np.random.default_rng(seed)
    seen = set(exclude)
    out: list[tuple[int, ...]] = []
    while len(out) < n_draws:
        cand = tuple(sorted(int(i) for i in rng.choice(n_layers, size=n_adapt, replace=False)))
        if cand not in seen:
            seen.add(cand)
            out.append(cand)
    return out


def lora_config(layers: tuple[int, ...], r: int = 8, alpha: int = 32, dropout: float = 0.05):  # type: ignore[no-untyped-def]
    """PEFT config: attention projections only, on exactly `layers`. LN/embeddings untouched."""
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=list(ATTN_TARGETS),
        layers_to_transform=list(layers),
        task_type="CAUSAL_LM",
    )
