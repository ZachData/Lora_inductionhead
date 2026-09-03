"""The silent-failure guard in scripts/diagnose_prereq_ab.py.

`head_maps` itself loads a real checkpoint and so belongs to tier 3, but
its guard does not, and the guard is the only new logic in that script --
everything else orchestrates already-tested probes. It is worth testing
precisely because it is the thing standing between "the hooks never
fired" and a grid of zeros being reported as "no head anywhere does
prefix matching," which is a plausible-looking and completely wrong
claim that nothing else in the pipeline would catch.

Discrimination is the point (CLAUDE.md, "silent-failure guards"): the
guard must reject degenerate grids *and* accept a real one. A guard that
raises on everything is as useless as one that raises on nothing.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from diagnose_prereq_ab import assert_grid_informative  # noqa: E402


def test_all_zero_grid_raises() -> None:
    """Hooks that never fired leave the accumulator at exactly zero."""
    with pytest.raises(ValueError, match="constant"):
        assert_grid_informative("pms", np.zeros((6, 8)), 512)


def test_constant_nonzero_grid_raises() -> None:
    """A probe returning the same number regardless of input is the same
    failure wearing a more plausible value."""
    with pytest.raises(ValueError, match="constant"):
        assert_grid_informative("prev_token", np.full((6, 8), 0.37), 512)


def test_nan_grid_raises() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        grid = np.zeros((6, 8))
        grid[2, 1] = np.nan
        assert_grid_informative("pms", grid, 512)


def test_inf_grid_raises() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        grid = np.linspace(0, 1, 48).reshape(6, 8)
        grid[0, 0] = np.inf
        assert_grid_informative("pms", grid, 512)


def test_empty_grid_raises() -> None:
    with pytest.raises(ValueError, match="empty"):
        assert_grid_informative("pms", np.zeros((0, 0)), 512)


def test_realistic_varying_grid_passes() -> None:
    """The known-positive half of the discrimination pair: a grid shaped
    like a real one -- one strong head, everything else near chance --
    must pass untouched."""
    grid = np.full((6, 8), 0.02)
    grid[2, 1] = 0.356  # G1's actual previous-token head at A
    assert_grid_informative("prev_token", grid, 512) is None


def test_barely_varying_grid_passes() -> None:
    """The threshold is for *identical* values, not for small ones. A
    real grid whose heads genuinely differ by little is still real, and
    widening this bound to catch it would suppress true results."""
    grid = np.full((6, 8), 0.02)
    grid[0, 0] = 0.02 + 1e-9
    assert_grid_informative("pms", grid, 512) is None
