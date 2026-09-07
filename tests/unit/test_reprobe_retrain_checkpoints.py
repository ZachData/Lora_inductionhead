"""Tests for the re-probe worker's sharding and bracket definition.

Most of this script is orchestration over already-tested probes, which
CLAUDE.md's TDD table says not to test. Two pieces are definitions, and
both fail silently:

`bracket_steps` fixes *which* checkpoints are the scientific payload.
onset_bracket.json says session 2's resume overwrote 856 and 860, so
those two hold a different trajectory. Including them puts foreign
weights in the middle of an onset curve and nothing downstream would
show it.

`shard_steps` decides coverage. A shard that drops a step leaves a hole
that looks like a spot reclaim; one that double-covers writes the same
step twice, which reads as a resumability success rather than a bug.
Neither raises, and the merged curve looks plausible either way.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reprobe_retrain_checkpoints import bracket_steps, shard_steps  # noqa: E402


# --- the bracket -------------------------------------------------------


def test_bracket_is_the_82_clean_checkpoints() -> None:
    steps = bracket_steps()
    assert steps[0] == 528
    assert steps[-1] == 852
    assert len(steps) == 82
    assert all(b - a == 4 for a, b in zip(steps[:-1], steps[1:], strict=True))


def test_bracket_excludes_the_two_contaminated_steps() -> None:
    """856 and 860 were session 1 files overwritten by session 2's resume
    (onset_bracket.json). They are a different trajectory and must not
    appear in an onset curve."""
    steps = bracket_steps()
    assert 856 not in steps
    assert 860 not in steps


def test_bracket_contains_the_onset_step() -> None:
    """Onset is step 620. A bracket that misses it is measuring the wrong
    window, which no downstream assertion would catch."""
    assert 620 in bracket_steps()


# --- sharding ----------------------------------------------------------


@pytest.mark.parametrize("n_shards", [1, 2, 3, 5, 7, 8, 13])
def test_shards_exactly_partition_the_bracket(n_shards: int) -> None:
    """Every step in exactly one shard: union is the bracket, and the
    shards are pairwise disjoint. Checked at shard counts that do and do
    not divide 82."""
    steps = bracket_steps()
    shards = [shard_steps(steps, i, n_shards) for i in range(n_shards)]
    union: list[int] = []
    for s in shards:
        union.extend(s)
    assert sorted(union) == steps
    assert len(union) == len(set(union)), "a step was assigned to two shards"


@pytest.mark.parametrize("n_shards", [3, 7, 13])
def test_shards_are_balanced_to_within_one(n_shards: int) -> None:
    sizes = [len(shard_steps(bracket_steps(), i, n_shards)) for i in range(n_shards)]
    assert max(sizes) - min(sizes) <= 1


def test_sharding_is_round_robin_not_contiguous() -> None:
    """Interleaved, so a dead shard leaves gaps spread through the curve
    rather than deleting a contiguous window. With stride-4 data a
    scattered gap is still usable; a missing block is not."""
    steps = bracket_steps()
    s0 = shard_steps(steps, 0, 4)
    assert s0[:3] == [528, 544, 560]
    assert s0[1] - s0[0] == 16


def test_single_shard_is_the_whole_bracket() -> None:
    assert shard_steps(bracket_steps(), 0, 1) == bracket_steps()


@pytest.mark.parametrize("shard,n_shards", [(-1, 4), (4, 4), (5, 4), (0, 0), (0, -1)])
def test_rejects_out_of_range_shard_arguments(shard: int, n_shards: int) -> None:
    """A shard index past the end silently returns [] -- a worker that
    does nothing, pushes an empty result, and terminates looking exactly
    like a success."""
    with pytest.raises(ValueError):
        shard_steps(bracket_steps(), shard, n_shards)


def test_empty_step_list_yields_empty_shards() -> None:
    assert shard_steps([], 0, 3) == []
