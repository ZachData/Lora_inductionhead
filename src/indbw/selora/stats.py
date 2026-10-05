"""Inference primitives for the selective-LoRA extension.

Everything here is a pure numpy function with a stated resolution floor.
The two rules that matter (PROJECT.md section 13.4):

* The unit of resampling is the *prompt* (or test item), never the token.
  Tokens inside a sequence are correlated; treating them as independent
  shrinks every interval and manufactures significance.
* A permutation or rank p-value is never 0. Its floor is 1/(n+1), and a
  reference set too small to reach alpha is a design error that
  `min_attainable_p` exposes before any compute is spent.
"""

from __future__ import annotations

import math
from typing import Literal

import numpy as np

Side = Literal["greater", "less", "two-sided"]


def _check_finite(x: np.ndarray, name: str) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    if x.ndim != 1:
        raise ValueError(f"{name} must be 1-D, got shape {x.shape}")
    if x.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"{name} contains NaN/Inf")
    return x


def min_attainable_p(n_reference: int) -> float:
    """Smallest p a rank / permutation test with `n_reference` draws can return."""
    if n_reference < 1:
        raise ValueError("need at least one reference draw")
    return 1.0 / (n_reference + 1)


def rank_pvalue(observed: float, reference: np.ndarray, side: Side = "greater") -> float:
    """Exchangeable-rank p-value: (1 + #{reference >= observed}) / (n + 1).

    Valid when `observed` is one more draw from the same distribution as
    `reference` under the null (e.g. a layer set fixed in advance versus
    uniformly random layer sets). Never 0.
    """
    ref = _check_finite(reference, "reference")
    if not math.isfinite(observed):
        raise ValueError("observed is not finite")
    n = ref.size
    if side == "greater":
        k = int(np.sum(ref >= observed))
    elif side == "less":
        k = int(np.sum(ref <= observed))
    elif side == "two-sided":
        centre = float(np.mean(ref))
        k = int(np.sum(np.abs(ref - centre) >= abs(observed - centre)))
    else:
        raise ValueError(f"unknown side {side!r}")
    return (1 + k) / (n + 1)


def sign_flip_pvalue(
    diffs: np.ndarray,
    n_perm: int,
    rng: np.random.Generator,
    side: Side = "greater",
    shift: float = 0.0,
) -> float:
    """Paired sign-flip test of H0: the (diffs - shift) are symmetric about 0.

    `shift` turns it into a margin test: for non-inferiority with margin
    delta pass diffs=(a-b) and shift=-delta, side="greater" (H0: mean <= -delta).
    Rows are clusters (prompts / test items). Valid under symmetry of the
    paired differences about the null value; bootstrap CIs are reported
    beside it because symmetry is an assumption, not a fact.
    """
    d = _check_finite(diffs, "diffs") - shift
    n = d.size
    obs = float(d.mean())
    signs = rng.choice(np.array([-1.0, 1.0]), size=(n_perm, n))
    null = (signs * d).mean(axis=1)
    if side == "greater":
        k = int(np.sum(null >= obs))
    elif side == "less":
        k = int(np.sum(null <= obs))
    elif side == "two-sided":
        k = int(np.sum(np.abs(null) >= abs(obs)))
    else:
        raise ValueError(f"unknown side {side!r}")
    return (1 + k) / (n_perm + 1)


def cluster_bootstrap_ci(
    values: np.ndarray,
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> tuple[float, float, float]:
    """Percentile bootstrap of the mean over clusters: (mean, lo, hi), two-sided 1-alpha."""
    v = _check_finite(values, "values")
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    means = v[idx].mean(axis=1)
    return (
        float(v.mean()),
        float(np.quantile(means, alpha / 2)),
        float(np.quantile(means, 1 - alpha / 2)),
    )


def one_sided_lower_bound(
    values: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05
) -> float:
    """One-sided (1-alpha) lower confidence bound on the cluster mean."""
    v = _check_finite(values, "values")
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    return float(np.quantile(v[idx].mean(axis=1), alpha))


def one_sided_upper_bound(
    values: np.ndarray, n_boot: int, rng: np.random.Generator, alpha: float = 0.05
) -> float:
    """One-sided (1-alpha) upper confidence bound on the cluster mean."""
    v = _check_finite(values, "values")
    idx = rng.integers(0, v.size, size=(n_boot, v.size))
    return float(np.quantile(v[idx].mean(axis=1), 1 - alpha))


def holm(pvalues: np.ndarray) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (FWER control, no independence assumption)."""
    p = _check_finite(pvalues, "pvalues")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p)
    adj_sorted = np.maximum.accumulate((m - np.arange(m)) * p[order])
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


def benjamini_hochberg(pvalues: np.ndarray) -> np.ndarray:
    """BH adjusted p-values (FDR control; valid under independence / PRDS)."""
    p = _check_finite(pvalues, "pvalues")
    if np.any((p < 0) | (p > 1)):
        raise ValueError("p-values must lie in [0, 1]")
    m = p.size
    order = np.argsort(p)
    ranked = p[order] * m / (np.arange(m) + 1)
    adj_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(m)
    out[order] = np.minimum(adj_sorted, 1.0)
    return out


def mcnemar_exact_pvalue(only_a: int, only_b: int) -> float:
    """Two-sided exact McNemar test from the discordant counts of paired binary outcomes."""
    if only_a < 0 or only_b < 0:
        raise ValueError("counts must be non-negative")
    n = only_a + only_b
    if n == 0:
        return 1.0
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return float(min(1.0, 2 * tail))
