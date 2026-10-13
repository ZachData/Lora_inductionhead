"""From raw per-prompt / per-item measurements to each claim's `observed` dict.

One function per claim (or per step of a gatekeeping family). Each returns exactly the
metric names its claim's criteria reference, computed with the cluster-level tests in
stats.py, so `adjudicate` is never handed a hand-typed number. Raw step functions return
`p_raw`; only `gatekeep` produces `p_adjusted`, so a raw p cannot reach a verdict.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence

import numpy as np

from indbw.selora import prereg, stats


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


def require_complete(done_run_ids: Iterable[str], planned_run_ids: Sequence[str]) -> None:
    """Refuse to adjudicate on a partial sweep (STATE.md practice 10)."""
    done = set(done_run_ids)
    missing = [r for r in planned_run_ids if r not in done]
    if missing:
        raise RuntimeError(
            f"{len(missing)} of {len(planned_run_ids)} planned cells have no row; "
            "verdicts are computed once, on the complete sweep"
        )


def s0_observed(resid_base_below: np.ndarray, resid_ft_below: np.ndarray) -> dict[str, float]:
    """Residual streams below the first adapted layer must be bit-identical."""
    a, b = np.asarray(resid_base_below), np.asarray(resid_ft_below)
    if a.shape != b.shape or a.size == 0:
        raise ValueError("residual arrays must be non-empty and equal-shaped")
    return {"max_abs_resid_diff_below_first_adapted": float(np.max(np.abs(a - b)))}


def _paired(
    cells: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> list[tuple[np.ndarray, np.ndarray]]:
    if not cells:
        raise ValueError("no cells")
    out = []
    for k in sorted(cells):
        a, b = (np.asarray(x, dtype=float) for x in cells[k])
        if a.shape != b.shape:
            raise ValueError(f"{k}: unpaired arrays")
        out.append((a, b))
    return out


def s1a_observed(
    cells: Mapping[str, tuple[np.ndarray, np.ndarray]], seed: int = 0
) -> dict[str, float]:
    """cells[key] = (fvu_base, fvu_ft) per prompt. Intersection-union: each cell at alpha.

    H0 per cell: mean(fvu_ft - fvu_base) >= FVU_MARGIN, studentized bootstrap. The claim
    needs every cell rejected, so no multiplicity correction applies (Berger 1982).
    """
    rng = _rng(seed)
    ps, uppers = [], []
    for base, ft in _paired(cells):
        ps.append(
            stats.bootstrap_t_pvalue(ft - base, prereg.FVU_MARGIN, prereg.N_BOOT, rng, "less")
        )
        uppers.append(stats.one_sided_upper_bound(ft, prereg.N_BOOT, rng))
    return {
        "n_cells_p_above_alpha": float(np.sum(np.array(ps) > prereg.ALPHA)),
        "max_fvu_ft_upper95": float(max(uppers)),
        "n_cells": float(len(ps)),
    }


def s1b_observed(
    cells: Mapping[str, tuple[np.ndarray, np.ndarray]], seed: int = 0
) -> dict[str, float]:
    """cells[key] = (delta_fvu_trained, delta_fvu_random). H0: trained <= random. IUT."""
    rng = _rng(seed)
    ps = [stats.sign_flip_pvalue(t - r, prereg.N_PERM, rng, "greater") for t, r in _paired(cells)]
    return {
        "n_cells_p_above_alpha": float(np.sum(np.array(ps) > prereg.ALPHA)),
        "n_cells": float(len(ps)),
    }


def seed_noise_floor(
    cells: Mapping[str, tuple[np.ndarray, np.ndarray]], seed: int = 0
) -> dict[str, float]:
    """cells[key] = (fvu_ft seed a, fvu_ft seed b), same arm. Control for S1a.

    The upper 95% bound on |mean seed-to-seed FVU difference|, worst cell. If it reaches
    FVU_MARGIN, S1a's margin is inside training noise and its verdict is uninformative.
    """
    rng = _rng(seed)
    ups = [
        max(
            stats.one_sided_upper_bound(a - b, prereg.N_BOOT, rng),
            stats.one_sided_upper_bound(b - a, prereg.N_BOOT, rng),
        )
        for a, b in _paired(cells)
    ]
    floor = float(max(ups))
    return {"noise_floor_upper95": floor, "margin_resolvable": float(floor < prereg.FVU_MARGIN)}


def s2a_observed(p_by_layer: Sequence[float]) -> dict[str, float]:
    adj = stats.holm(np.asarray(p_by_layer, dtype=float))
    return {"n_layers_holm_p_below_alpha": float(np.sum(adj < prereg.ALPHA))}


def s2b_observed(
    acts_by_layer: Sequence[np.ndarray],
    labels: np.ndarray,
    null_means: np.ndarray,
    k: int,
    seed: int = 0,
    strata: np.ndarray | None = None,
) -> dict[str, float]:
    from indbw.selora.probes import peak_layer_bootstrap

    lo, hi = prereg.PEAK_RANGE
    return {
        "peak_in_range_frequency": peak_layer_bootstrap(
            acts_by_layer, labels, null_means, k, 1000, _rng(seed), lo, hi, strata
        )
    }


def s3a_step(late: np.ndarray, allc: np.ndarray, seed: int = 0) -> dict[str, float]:
    """[n_seeds, n_items] 0/1 correctness, paired by item and seed. Pooled over seeds."""
    a, b = np.asarray(late, float), np.asarray(allc, float)
    if a.shape != b.shape or a.ndim != 2:
        raise ValueError("expected equal-shaped [n_seeds, n_items] arrays")
    return s3a_from_diffs(a - b, seed)


def s3a_from_diffs(diffs: np.ndarray, seed: int = 0) -> dict[str, float]:
    """Crossed (seed x item) bootstrap of mean(late - all): one-sided 95% lower bound and the
    bootstrap p for H0: mean <= -ACC_MARGIN."""
    d = np.asarray(diffs, float)
    boots = stats.crossed_bootstrap_means(d, prereg.N_BOOT, _rng(seed))
    k = int(np.sum(boots <= -prereg.ACC_MARGIN))
    return {
        "pooled_lower95_diff": float(np.quantile(boots, prereg.ALPHA)),
        "p_raw": (1 + k) / (prereg.N_BOOT + 1),
        "n_seeds": float(d.shape[0]),
    }


def rank_step(
    named_by_seed: Mapping[int, float], random_scores: np.ndarray, random_seed: int
) -> dict[str, float]:
    """Named arm vs random placements, using only the named arm's score at the random draws' seed.

    A multi-seed mean is less noisy than a single-seed draw, which breaks exchangeability.
    """
    if random_seed not in named_by_seed:
        raise KeyError(f"named arm has no score at seed {random_seed}")
    return {
        "p_raw": stats.rank_pvalue(float(named_by_seed[random_seed]), random_scores, "greater"),
        "n_reference": float(len(random_scores)),
    }


def gatekeep(
    steps: Mapping[tuple[str, str], dict[str, float]],
) -> dict[tuple[str, str], dict[str, float]]:
    """Fixed-sequence adjustment over prereg.GATE_SEQUENCE; adds `p_adjusted` to each step."""
    if set(steps) != set(prereg.GATE_SEQUENCE):
        raise ValueError(f"steps must match the registered sequence {prereg.GATE_SEQUENCE}")
    raw = np.array([steps[k]["p_raw"] for k in prereg.GATE_SEQUENCE])
    adj = stats.fixed_sequence_adjust(raw)
    return {
        k: {**steps[k], "p_adjusted": float(a)}
        for k, a in zip(prereg.GATE_SEQUENCE, adj, strict=True)
    }


def s5_observed(
    named_by_seed: Mapping[int, float], random_scores: np.ndarray, random_seed: int
) -> dict[str, float]:
    r = rank_step(named_by_seed, random_scores, random_seed)
    return {"p_value": r["p_raw"], "n_reference": r["n_reference"]}


def s4_observed(
    parseable_correct: np.ndarray, logprob_correct: np.ndarray, seed: int = 0
) -> dict[str, float]:
    """parseable_correct: 0/1 over generated items with a letter; logprob_correct: full test set."""
    rng = _rng(seed)
    return {
        "acc_parseable_lower95": stats.one_sided_lower_bound(
            np.asarray(parseable_correct, float), prereg.N_BOOT, rng
        ),
        "acc_logprob_lower95": stats.one_sided_lower_bound(
            np.asarray(logprob_correct, float), prereg.N_BOOT, rng
        ),
    }
