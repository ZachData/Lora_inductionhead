"""Invariants of selora.stats / placement under random input (hypothesis)."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from indbw.selora import placement, stats

pvals = st.lists(st.floats(0, 1), min_size=1, max_size=30).map(np.array)


@given(pvals)
def test_holm_dominates_raw_and_bh_and_is_bounded(p: np.ndarray) -> None:
    h, b = stats.holm(p), stats.benjamini_hochberg(p)
    assert np.all(h >= p - 1e-15) and np.all(h <= 1.0)
    assert np.all(h >= b - 1e-12)  # FWER control is never less conservative than FDR
    order = np.argsort(p)
    assert np.all(np.diff(h[order]) >= -1e-12)  # monotone in the raw p


@given(st.lists(st.floats(-5, 5), min_size=1, max_size=40).map(np.array), st.floats(-6, 6))
def test_rank_pvalue_in_floor_and_one(ref: np.ndarray, obs: float) -> None:
    for side in ("greater", "less", "two-sided"):
        p = stats.rank_pvalue(obs, ref, side)  # type: ignore[arg-type]
        assert stats.min_attainable_p(len(ref)) <= p <= 1.0


@settings(max_examples=30, deadline=None)
@given(st.integers(0, 10_000), st.integers(1, 6))
def test_random_sets_valid(seed: int, n_draws: int) -> None:
    sets = placement.random_sets(8, 4, n_draws, seed, exclude={(0, 1, 2, 3)})
    assert len(set(sets)) == n_draws and (0, 1, 2, 3) not in sets
    assert all(len(s) == 4 and len(set(s)) == 4 and min(s) >= 0 and max(s) < 8 for s in sets)


@settings(max_examples=30, deadline=None)
@given(st.lists(st.floats(-10, 10), min_size=2, max_size=40).map(np.array), st.floats(-3, 3))
def test_bootstrap_ci_orders_and_contains_mean_for_symmetric_shift(x: np.ndarray, c: float) -> None:
    rng = np.random.default_rng(0)
    m, lo, hi = stats.cluster_bootstrap_ci(x, 200, rng)
    assert lo <= hi
    _, lo2, hi2 = stats.cluster_bootstrap_ci(x + c, 200, np.random.default_rng(0))
    np.testing.assert_allclose([lo2, hi2], [lo + c, hi + c], atol=1e-9)  # translation equivariance
