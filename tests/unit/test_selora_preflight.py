"""Spec for selora.preflight: each check passes a good input and names the problem in a bad one."""

from __future__ import annotations

import math

import pytest
import torch

from indbw.selora import preflight, prereg


def test_check_sae_shapes(tiny_sae_factory) -> None:  # type: ignore[no-untyped-def]
    sae = tiny_sae_factory(d_model=32, d_sae=64)
    assert preflight.check_sae(sae, d_model=32, d_sae=64) == []
    assert any("d_model" in p for p in preflight.check_sae(sae, d_model=2304, d_sae=64))
    dead = type(sae)(sae.W_enc, torch.zeros_like(sae.W_dec), sae.b_enc, sae.b_dec, sae.threshold)
    assert any("W_dec" in p for p in preflight.check_sae(dead, d_model=32, d_sae=64))


def test_check_base_fvu_range() -> None:
    assert preflight.check_base_fvu({0: 0.05, 1: 0.1}) == []
    probs = preflight.check_base_fvu({0: 0.05, 1: 0.9, 2: float("nan")})
    assert (
        len(probs) == 2
        and any("layer 1" in p for p in probs)
        and any("layer 2" in p for p in probs)
    )


def test_check_populated_flags_empty_constant_and_missing_fields() -> None:
    good = {"fvu_ft": {"0": [0.1, 0.2]}, "gsm_correct": [0, 1, 1], "mmlu_logprob": ["A", "B"]}
    keys = ("fvu_ft", "gsm_correct", "mmlu_logprob")
    assert preflight.check_populated(good, keys) == []
    bad = {"fvu_ft": {"0": [0.1, 0.1, 0.1]}, "gsm_correct": []}
    probs = preflight.check_populated(bad, keys)
    assert any("constant" in p for p in probs)
    assert any("gsm_correct" in p and "empty" in p for p in probs)
    assert any("mmlu_logprob" in p and "missing" in p for p in probs)


def test_steps_and_sweep_hours() -> None:
    # 7473 examples, batch 4 x accum 4, 3 epochs -> ceil(7473/16) * 3 = 468 * 3 = 1404 steps
    assert preflight.steps_per_cell(7473, 4, 4, 3.0) == 1404
    h = preflight.estimate_sweep_hours(sec_per_step=1.0, steps=1404, n_cells=80, n_parallel=4)
    assert math.isclose(h, 1404 * 80 / 4 / 3600)
    with pytest.raises(ValueError):
        preflight.estimate_sweep_hours(0.0, 10, 1, 1)


def test_compare_assumptions_names_the_drifted_quantity() -> None:
    a = prereg.ASSUMPTIONS
    assert preflight.compare_assumptions({"gsm8k_accuracy": a["gsm8k_accuracy"]}) == []
    probs = preflight.compare_assumptions({"gsm8k_accuracy": a["gsm8k_accuracy"] * 3})
    assert len(probs) == 1 and "gsm8k_accuracy" in probs[0]
    with pytest.raises(KeyError):
        preflight.compare_assumptions({"not_an_assumption": 1.0})
