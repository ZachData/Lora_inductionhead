"""Oracles for the stats added in v0.2: bootstrap-t, crossed bootstrap, fixed sequence."""

from __future__ import annotations

import numpy as np
import pytest

from indbw.selora import stats


def test_bootstrap_t_discriminates() -> None:
    rng = np.random.default_rng(0)
    far = rng.normal(0.0, 0.01, 300)
    near = rng.normal(0.05, 0.01, 300)
    assert stats.bootstrap_t_pvalue(far, 0.05, 2000, rng, "less") == pytest.approx(1 / 2001)
    assert stats.bootstrap_t_pvalue(near, 0.05, 2000, rng, "less") > 0.05
    assert stats.bootstrap_t_pvalue(far + 0.1, 0.05, 2000, rng, "greater") == pytest.approx(
        1 / 2001
    )


def test_bootstrap_t_constant_input_raises() -> None:
    with pytest.raises(ValueError, match="constant"):
        stats.bootstrap_t_pvalue(np.full(20, 0.1), 0.0, 100, np.random.default_rng(0))


def test_crossed_bootstrap_degenerate_and_pairing() -> None:
    rng = np.random.default_rng(1)
    d = np.full((3, 50), 0.2)
    m = stats.crossed_bootstrap_means(d, 200, rng)
    np.testing.assert_allclose(m, 0.2, rtol=1e-12)
    with pytest.raises(ValueError):
        stats.crossed_bootstrap_means(np.zeros(5), 10, rng)


def test_crossed_bootstrap_is_wider_than_item_only_when_seeds_disagree() -> None:
    rng = np.random.default_rng(2)
    seed_effects = np.array([-0.05, 0.0, 0.05])[:, None]
    d = seed_effects + rng.normal(0, 0.3, (3, 1319))
    crossed = stats.crossed_bootstrap_means(d, 2000, rng).std()
    items_only = stats.cluster_bootstrap_ci(d.mean(0), 2000, rng)
    assert crossed > (items_only[2] - items_only[1]) / 3.92  # wider than item-only SE


def test_fixed_sequence_adjust_guards() -> None:
    with pytest.raises(ValueError):
        stats.fixed_sequence_adjust(np.array([0.1, 1.5]))
