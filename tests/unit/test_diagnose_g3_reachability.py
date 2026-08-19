"""Tests for the reachability-graft diagnostic's weight surgery.

Same rationale as tests/unit/test_run_g3.py: this is a script, not a
metric, so it is outside the METRIC_VERSION surface -- but `graft` is a
textbook silent failure (CLAUDE.md TDD contract, kind 3). Every wrong
version of it still returns an R and a PMS in exactly the plausible
range: grafting a whole [n_heads, ...] tensor instead of one head's
slice, grafting nothing at all, or grafting the wrong tensor all produce
numbers that look like a result. The only thing that distinguishes them
is which parameters actually changed, so that is what is asserted here --
tensor by tensor, on both sides: what must change, and what must not.

The diagnostic's entire point is to decide whether G3's plateau is an
optimization failure or an unreachable criterion. A graft bug would
answer that question confidently and wrongly, which is the most
expensive mistake available here.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from transformer_lens import HookedTransformer, HookedTransformerConfig

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import diagnose_g3_reachability as diag

LAYER, HEAD = 1, 1


def _model(seed: int) -> HookedTransformer:
    cfg = HookedTransformerConfig(
        n_layers=2,
        d_model=32,
        d_head=16,
        n_heads=2,
        n_ctx=64,
        d_vocab=50,
        act_fn="relu",
        normalization_type="LN",
        seed=seed,
    )
    model = HookedTransformer(cfg)
    # HookedTransformer zero-initializes every bias, so two toy models
    # built from different seeds still share identical b_Q/b_K/b_V --
    # and a graft of those tensors would be undetectable, silently
    # weakening the "touches exactly its own tensors" assertion below to
    # cover only the weights. Real Pythia checkpoints have distinct
    # nonzero biases; give the fixture the same property rather than
    # exempting biases from the test.
    gen = torch.Generator().manual_seed(1000 + seed)
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name.rsplit(".", 1)[-1].startswith("b_"):
                param.copy_(torch.randn(param.shape, generator=gen) * 0.1)
    model.eval()
    return model


@pytest.fixture
def pair() -> tuple[HookedTransformer, dict[str, torch.Tensor]]:
    """A destination model and a *different* source state dict. Different
    seeds matter: with identical weights every graft is a no-op and every
    assertion below passes vacuously.
    """
    dst = _model(0)
    src = {k: v.clone() for k, v in _model(1).state_dict().items()}
    return dst, src


def _changed_keys(before: dict[str, torch.Tensor], after: dict[str, torch.Tensor]) -> set[str]:
    return {k for k in before if not torch.equal(before[k], after[k])}


def test_source_and_destination_actually_differ(pair: tuple) -> None:
    """Guard on the fixture itself -- if the two models were identical,
    every other test in this file would pass without testing anything."""
    dst, src = pair
    assert _changed_keys(dst.state_dict(), src), "fixture models are identical; tests are vacuous"


def test_base_a_changes_nothing(pair: tuple) -> None:
    dst, src = pair
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    diag.graft(dst, src, "base_a", layer=LAYER, head=HEAD)
    assert _changed_keys(before, dst.state_dict()) == set()


@pytest.mark.parametrize("cell", ["full", "base_b"])
def test_full_graft_reproduces_the_source_exactly(pair: tuple, cell: str) -> None:
    """The R == 1 oracle: the `full` cell must leave the model bit-identical
    to B, so its recovery is 1 by construction, not by approximation."""
    dst, src = pair
    diag.graft(dst, src, cell, layer=LAYER, head=HEAD)
    after = dst.state_dict()
    for k, v in src.items():
        assert torch.equal(after[k], v), f"{k} does not match the source after a full graft"


@pytest.mark.parametrize(
    ("cell", "expected_shorts"),
    [
        ("qk_q", ["W_Q", "b_Q"]),
        ("qk_qk", ["W_Q", "b_Q", "W_K", "b_K"]),
        ("ov", ["W_V", "b_V", "W_O"]),
        ("head", ["W_Q", "b_Q", "W_K", "b_K", "W_V", "b_V", "W_O"]),
    ],
)
def test_head_scoped_cell_touches_exactly_its_own_tensors(
    pair: tuple, cell: str, expected_shorts: list[str]
) -> None:
    dst, src = pair
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    diag.graft(dst, src, cell, layer=LAYER, head=HEAD)
    expected = {f"blocks.{LAYER}.attn.{s}" for s in expected_shorts}
    assert _changed_keys(before, dst.state_dict()) == expected


@pytest.mark.parametrize("cell", ["qk_q", "qk_qk", "ov", "head"])
def test_head_scoped_cell_leaves_the_other_head_alone(pair: tuple, cell: str) -> None:
    """The slice, not the tensor. A graft of the whole [n_heads, ...]
    tensor would pass every assertion above and be wrong: it would adapt
    a second head the arm never claims to touch.
    """
    dst, src = pair
    other = 1 - HEAD
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    diag.graft(dst, src, cell, layer=LAYER, head=HEAD)
    after = dst.state_dict()
    for key in _changed_keys(before, after):
        assert torch.equal(before[key][other], after[key][other]), (
            f"{key} changed for head {other}, which cell {cell!r} does not name"
        )
        assert not torch.equal(before[key][HEAD], after[key][HEAD]), (
            f"{key} did not change for head {HEAD}, which cell {cell!r} does name"
        )


def test_layer_host_touches_the_host_block_and_only_it(pair: tuple) -> None:
    dst, src = pair
    before = {k: v.clone() for k, v in dst.state_dict().items()}
    diag.graft(dst, src, "layer_host", layer=LAYER, head=HEAD)
    changed = _changed_keys(before, dst.state_dict())
    assert changed, "layer_host grafted nothing"
    assert all(k.startswith(f"blocks.{LAYER}.") for k in changed), sorted(changed)
    # and it is strictly more than the whole-head cell
    dst2, src2 = _model(0), {k: v.clone() for k, v in _model(1).state_dict().items()}
    b2 = {k: v.clone() for k, v in dst2.state_dict().items()}
    diag.graft(dst2, src2, "head", layer=LAYER, head=HEAD)
    assert _changed_keys(b2, dst2.state_dict()) < changed


def test_cells_nest_as_the_ladder_claims(pair: tuple) -> None:
    """qk_q subset qk_qk subset head, and ov subset head. The diagnostic
    reads its cells as a ladder of progressively larger grafts; if they
    did not actually nest, the ladder's ordering would be meaningless."""
    sets = {}
    for cell in ("qk_q", "qk_qk", "ov", "head"):
        dst = _model(0)
        src = {k: v.clone() for k, v in _model(1).state_dict().items()}
        before = {k: v.clone() for k, v in dst.state_dict().items()}
        diag.graft(dst, src, cell, layer=LAYER, head=HEAD)
        sets[cell] = _changed_keys(before, dst.state_dict())
    assert sets["qk_q"] < sets["qk_qk"] < sets["head"]
    assert sets["ov"] < sets["head"]
    assert sets["qk_qk"].isdisjoint(sets["ov"])


def test_unknown_cell_raises(pair: tuple) -> None:
    dst, src = pair
    with pytest.raises(ValueError, match="unknown cell"):
        diag.graft(dst, src, "not_a_cell", layer=LAYER, head=HEAD)


def test_evaluate_returns_every_component_in_range_on_a_toy_model() -> None:
    """Plumbing smoke for the eval path: the right head's attention
    pattern is what PMS reads, both NLL halves are finite, and R is
    computed from them. A randomly-initialized model has no induction, so
    the assertion is on ranges and finiteness, not on values -- the
    values are the diagnostic's output, not something to pin here.
    """
    from indbw.evalset import build_eval_tokens

    model = _model(0)
    period = 8
    tokens = build_eval_tokens(4, period, seed=0, d_vocab=50)
    obs = diag.evaluate(
        model, tokens, batch_size=2, layer=LAYER, head=HEAD, period=period, icl_a=0.0, icl_b=1.0
    )
    assert set(obs) == {"recovery", "icl", "nll_first", "nll_second", "pms"}
    assert all(np.isfinite(v) for v in obs.values())
    assert 0.0 <= obs["pms"] <= 1.0
    assert obs["nll_first"] > 0.0 and obs["nll_second"] > 0.0
    # icl_a=0, icl_b=1 makes R == ICL exactly; the identity is what pins
    # that recovery is actually being computed from the two halves here.
    assert obs["recovery"] == pytest.approx(obs["nll_first"] - obs["nll_second"], rel=1e-12)


def test_evaluate_reads_the_head_it_is_asked_for() -> None:
    """Discrimination guard on the head index: zeroing head 0's OV path
    must not change head 1's PMS, and vice versa. A hardcoded or
    off-by-one head index would sail through the range checks above.
    """
    from indbw.evalset import build_eval_tokens

    period = 8
    tokens = build_eval_tokens(4, period, seed=0, d_vocab=50)
    model = _model(0)
    pms_by_head = [
        diag.evaluate(
            model, tokens, batch_size=2, layer=LAYER, head=h, period=period, icl_a=0.0, icl_b=1.0
        )["pms"]
        for h in (0, 1)
    ]
    assert pms_by_head[0] != pms_by_head[1], "both heads report identical PMS; head index is inert"
