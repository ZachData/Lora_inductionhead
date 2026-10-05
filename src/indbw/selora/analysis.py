"""From raw per-prompt / per-item measurements to each claim's `observed` dict.

One function per claim. Every function returns exactly the metric names that
claim's criteria reference (prereg.CLAIMS), computed with the cluster-level
tests in stats.py, so `adjudicate` can never be handed a hand-typed number.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from indbw.selora import prereg, stats

N_PERM = 4000
N_BOOT = 4000


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def s0_observed(resid_base_below: np.ndarray, resid_ft_below: np.ndarray) -> dict[str, float]:
    """Residual streams below the first adapted layer must be bit-identical."""
    a, b = np.asarray(resid_base_below), np.asarray(resid_ft_below)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("residual arrays must be non-empty and equal-shaped")
    return {"max_abs_resid_diff_below_first_adapted": float(np.max(np.abs(a - b)))}


def s1a_observed(
    cells: Mapping[str, tuple[np.ndarray, np.ndarray]], seed: int = 0
) -> dict[str, float]:
    """cells[key] = (fvu_base, fvu_ft), each [n_prompts], paired by prompt.

    H0 per cell: mean(fvu_ft - fvu_base) >= FVU_MARGIN. Holm across all cells.
    """
    if not cells:
        raise ValueError("no cells")
    rng = _rng(seed)
    keys = sorted(cells)
    ps, uppers = [], []
    for k in keys:
        base, ft = (np.asarray(x, dtype=float) for x in cells[k])
        if base.shape != ft.shape:
            raise ValueError(f"{k}: unpaired arrays")
        ps.append(stats.sign_flip_pvalue(ft - base, N_PERM, rng, "less", shift=prereg.FVU_MARGIN))
        uppers.append(stats.one_sided_upper_bound(ft, N_BOOT, rng))
    adj = stats.holm(np.array(ps))
    return {
        "n_cells_holm_p_above_alpha": float(np.sum(adj > prereg.ALPHA)),
        "max_fvu_ft_upper95": float(max(uppers)),
        "n_cells": float(len(keys)),
    }


def s1b_observed(
    cells: Mapping[str, tuple[np.ndarray, np.ndarray]], seed: int = 0
) -> dict[str, float]:
    """cells[key] = (delta_fvu_trained, delta_fvu_random) per prompt. H0: trained <= random."""
    if not cells:
        raise ValueError("no cells")
    rng = _rng(seed)
    ps = [
        stats.sign_flip_pvalue(
            np.asarray(cells[k][0], float) - np.asarray(cells[k][1], float), N_PERM, rng, "greater"
        )
        for k in sorted(cells)
    ]
    adj = stats.holm(np.array(ps))
    return {
        "n_cells_holm_p_above_alpha": float(np.sum(adj > prereg.ALPHA)),
        "n_cells": float(len(ps)),
    }


def s2a_observed(p_by_layer: Sequence[float]) -> dict[str, float]:
    adj = stats.holm(np.asarray(p_by_layer, dtype=float))
    return {"n_layers_holm_p_below_alpha": float(np.sum(adj < prereg.ALPHA))}


def s2b_observed(
    excess_by_layer: np.ndarray, se_by_layer: np.ndarray, seed: int = 0
) -> dict[str, float]:
    lo, hi = prereg.PEAK_RANGE
    from indbw.selora.probes import peak_layer_stability

    return {
        "peak_in_range_frequency": peak_layer_stability(
            excess_by_layer, lo, hi, _rng(seed), se_by_layer, N_BOOT
        )
    }


def s3a_observed(
    item_correct_late: np.ndarray, item_correct_all: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """Arrays [n_seeds, n_items] of 0/1 correctness, items paired across arms and seeds.

    `min_seed_lower95_diff`: the *worst* seed's one-sided 95% lower bound on
    (late - all). The p-value tests the seed-averaged item differences
    against the margin; `holm_p` here is the single-test p (the family S3 is
    Holm-adjusted by the caller across S3a/S3b).
    """
    late, allc = np.asarray(item_correct_late, float), np.asarray(item_correct_all, float)
    if late.shape != allc.shape or late.ndim != 2:
        raise ValueError("expected equal-shaped [n_seeds, n_items] arrays")
    rng = _rng(seed)
    lows = [
        stats.one_sided_lower_bound(late[s] - allc[s], N_BOOT, rng) for s in range(late.shape[0])
    ]
    p = stats.sign_flip_pvalue(
        (late - allc).mean(axis=0), N_PERM, rng, "greater", shift=-prereg.ACC_MARGIN
    )
    return {"min_seed_lower95_diff": float(min(lows)), "holm_p": p, "n_seeds": float(late.shape[0])}


def rank_claim_observed(observed: float, random_draws: np.ndarray) -> dict[str, float]:
    """S3b / S5: the named arm's score versus the 24 random placements (one-sided)."""
    return {
        "holm_p": stats.rank_pvalue(observed, random_draws, "greater"),
        "n_reference": float(len(random_draws)),
    }


def s4_observed(
    parseable_correct: np.ndarray, logprob_correct: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """parseable_correct: 0/1 over items with a letter; logprob_correct: 0/1 over all items."""
    rng = _rng(seed)
    return {
        "acc_parseable_lower95": stats.one_sided_lower_bound(
            np.asarray(parseable_correct, float), N_BOOT, rng
        ),
        "acc_logprob_lower95": stats.one_sided_lower_bound(
            np.asarray(logprob_correct, float), N_BOOT, rng
        ),
    }
