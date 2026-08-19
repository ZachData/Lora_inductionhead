"""Oracles for the A->B training-budget arithmetic.

Borderline under CLAUDE.md's "when to write a test" table -- this is an
analysis script, which the table says to skip. It gets a test anyway for
one reason: the ratio it computes is about to be an input to a decision
about whether to spend GPU budget, an arithmetic slip in it would be
completely silent (every wrong version still prints a plausible
multiple), and every quantity involved has a closed form to check
against. That is three of the four "write a test" columns.

Only the arithmetic is pinned. Nothing here asserts what the ratio
*should* be -- that is the finding, not a regression baseline.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import analyze_training_budget as budget


@pytest.fixture
def g2_stub() -> dict:
    """A stub with round numbers, so every expected value below is one a
    reader can verify by hand rather than by rerunning the code."""
    return {
        "steps_run": 100,
        "wall_clock_s": 1000.0,
        "config": {"batch_size": 2, "T": 10, "rank": 64},
    }


def test_flops_6nd_matches_its_closed_form() -> None:
    assert budget.flops_6nd(1_000, 2_000) == 6.0 * 1_000 * 2_000


def test_lora_trainable_params_matches_the_two_factor_shapes() -> None:
    # QK arm: B is [d_model, r], A is [r, d_head] (indbw.train shape convention)
    assert budget.lora_trainable_params(rank=64, d_model=512, d_head=64) == 64 * 512 + 64 * 64
    assert budget.lora_trainable_params(rank=1, d_model=512, d_head=64) == 512 + 64


def test_pythia_tokens_per_step_is_seqs_times_seq_len() -> None:
    assert budget.PYTHIA_TOKENS_PER_STEP == 1024 * 2048


def test_g3_token_count_counts_the_whole_repeated_sequence(g2_stub: dict) -> None:
    """seq_len is 2T, not T: a repeated-random example is the first copy
    concatenated with itself, and the forward pass costs all of it."""
    g3 = budget.analyze(g2_stub)["g3"]
    assert g3["seq_len"] == 20
    assert g3["tokens_per_step"] == 2 * 2 * 10
    assert g3["tokens_total"] == 100 * 2 * 2 * 10


def test_loss_bearing_positions_exclude_the_seam(g2_stub: dict) -> None:
    """T-1 per example, not T -- second_copy_nll drops the seam position
    (indbw.train). Counting T would overstate the gradient signal by a
    factor that shrinks with T and so would hide at large T."""
    g3 = budget.analyze(g2_stub)["g3"]
    assert g3["loss_bearing_positions"] == 100 * 2 * (10 - 1)


def test_bracket_span_uses_the_next_grid_point_not_b(g2_stub: dict) -> None:
    """The transition is bracketed by A and the *next* checkpoint on the
    grid (512 -> 1000). Measuring the span to B (step 2000) instead would
    overstate it 3x -- and B is where the circuit has stabilized, not
    where it formed."""
    p = budget.analyze(g2_stub)["pythia"]
    assert p["steps_A_to_next_grid_point"] == 1000 - 512
    assert p["steps_A_to_B"] == 2000 - 512
    assert p["tokens_A_to_next_grid_point"] == 488 * 1024 * 2048
    assert p["tokens_A_to_B"] > p["tokens_A_to_next_grid_point"]


def test_ratio_is_pretraining_over_g3(g2_stub: dict) -> None:
    """Direction guard. An inverted ratio is the single most likely
    silent error here, and it would read as 'G3 trained 2500x *more*
    than pretraining' -- absurd on inspection, but only if someone
    inspects it."""
    a = budget.analyze(g2_stub)
    expected = a["pythia"]["tokens_A_to_next_grid_point"] / a["g3"]["tokens_total"]
    assert a["ratios"]["tokens_pythia_bracket_over_g3"] == pytest.approx(expected, rel=1e-12)
    assert a["ratios"]["tokens_pythia_bracket_over_g3"] > 1.0


def test_report_renders_without_error(g2_stub: dict) -> None:
    text = budget.format_report(budget.analyze(g2_stub))
    assert "Ratios (pretraining : G3)" in text
