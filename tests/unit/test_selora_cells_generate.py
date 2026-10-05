"""Spec for selora.cells (enumeration, resumability) and selora.generate (formats, harness)."""

from __future__ import annotations

import json

import numpy as np
import pytest
import torch

from indbw.selora import cells, generate, pipeline, prereg
from indbw.selora.sae import content_mask

SCORES = {m: list(np.linspace(0, 1, 26)) for m in cells.MODELS}


def test_cell_counts_and_unique_ids() -> None:
    cs = cells.enumerate_cells(SCORES)
    per_model = 1 + 5 * len(cells.NAMED_SEEDS) + prereg.N_RANDOM_DRAWS
    assert len(cs) == 2 * per_model == 80
    assert len({c.run_id for c in cs}) == len(cs)
    late = [c for c in cs if c.arm == "late" and c.model == cells.MODELS[0]]
    assert [c.layers for c in late] == [tuple(range(13, 26))] * 3


def test_random_arms_never_collide_with_named_sets() -> None:
    cs = cells.enumerate_cells(SCORES)
    named = {c.layers for c in cs if not c.arm.startswith("random") and c.arm != "base"}
    assert not any(c.layers in named for c in cs if c.arm.startswith("random"))


def test_resume_skips_completed(tmp_path) -> None:  # type: ignore[no-untyped-def]
    cs = cells.enumerate_cells(SCORES)
    path = tmp_path / "r.jsonl"
    path.write_text("".join(json.dumps({"run_id": c.run_id}) + "\n" for c in cs[:10]))
    rest = cells.pending(cs, path)
    assert len(rest) == len(cs) - 10 and rest[0] == cs[10]
    assert cells.pending(cs, tmp_path / "missing.jsonl") == cs


def test_run_id_depends_on_every_field() -> None:
    a = cells.Cell("m", "late", (1, 2), 0)
    for b in (
        cells.Cell("m", "late", (1, 2), 1),
        cells.Cell("m", "late", (1, 3), 0),
        cells.Cell("m", "late", (1, 2), 0, lr=1e-4),
        cells.Cell("n", "late", (1, 2), 0),
    ):
        assert a.run_id != b.run_id


def test_prompt_formats() -> None:
    assert generate.gsm8k_train_text("Q?", "A #### 1").startswith(generate.gsm8k_prompt("Q?"))
    p = generate.mmlu_prompt("Q", ["w", "x", "y", "z"])
    assert p.endswith("Answer:") and "C. y" in p
    with pytest.raises(ValueError):
        generate.mmlu_prompt("Q", ["a", "b"])


def test_harnesses_run_and_logprob_is_batch_invariant(tiny_gemma2, stub_tokenizer) -> None:  # type: ignore[no-untyped-def]
    tok = stub_tokenizer
    prompts = [generate.mmlu_prompt(f"question {i}", ["aa", "bb", "cc", "dd"]) for i in range(5)]
    one = generate.mmlu_logprob_preds(tiny_gemma2, tok, prompts, batch_size=1)
    many = generate.mmlu_logprob_preds(tiny_gemma2, tok, prompts, batch_size=5)
    assert one == many and set(one) <= set("ABCD")  # left-padding must not change predictions
    gen = generate.mmlu_generation_preds(tiny_gemma2, tok, prompts, max_new_tokens=4)
    assert len(gen) == 5
    flags = generate.gsm8k_correct_flags(
        tiny_gemma2, tok, ["1+1?", "2+2?"], [2.0, 4.0], max_new_tokens=4
    )
    assert len(flags) == 2 and set(flags) <= {0, 1}


def test_fvu_by_layer_matches_manual_and_is_zero_for_base_vs_base(
    tiny_gemma2, tiny_sae_factory
) -> None:  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(3, 64, (4, 9), generator=g)
    ids[:, 0] = 2
    attn = torch.ones_like(ids)
    saes = {1: tiny_sae_factory(seed=1), 2: tiny_sae_factory(seed=2)}
    ref = pipeline.reference_stats(tiny_gemma2, ids, attn, [1, 2])
    f = pipeline.fvu_by_layer(tiny_gemma2, saes, ids, attn, ref)
    assert set(f) == {1, 2} and all(v.shape == (4,) and bool((v > 0).all()) for v in f.values())
    assert not content_mask(attn)[:, 0].any()
    again = pipeline.fvu_by_layer(tiny_gemma2, saes, ids, attn, ref)
    for li in f:
        assert torch.equal(
            f[li], again[li]
        )  # deterministic: delta between identical models is exactly 0


def test_pick_sae_path_nearest_l0_and_exact_layer_match() -> None:
    files = [
        "layer_1/width_16k/average_l0_20/params.npz",
        "layer_1/width_16k/average_l0_65/params.npz",
        "layer_1/width_16k/average_l0_120/params.npz",
        "layer_1/width_65k/average_l0_70/params.npz",  # wrong width
        "layer_11/width_16k/average_l0_70/params.npz",  # layer 11 must not match layer 1
    ]
    assert cells.pick_sae_path(files, 1) == "layer_1/width_16k/average_l0_65/params.npz"
    with pytest.raises(FileNotFoundError):
        cells.pick_sae_path(files, 2)


def test_shuffle_answers_is_a_seeded_derangement_preserving_the_multiset() -> None:
    q = [f"q{i}" for i in range(50)]
    a = [f"a{i}" for i in range(50)]
    s = generate.shuffle_answers(q, a, seed=0)
    assert [x for x, _ in s] == q
    answers = [y for _, y in s]
    assert sorted(answers) == sorted(a)
    assert all(ans != a[i] for i, ans in enumerate(answers))  # no question keeps its own answer
    assert s == generate.shuffle_answers(q, a, seed=0) and s != generate.shuffle_answers(
        q, a, seed=1
    )
    with pytest.raises(ValueError):
        generate.shuffle_answers(["q"], ["a"], seed=0)


def test_shuffled_arm_is_optional_and_off_by_default() -> None:
    assert cells.INCLUDE_SHUFFLED is False
    base = cells.enumerate_cells(SCORES)
    assert not any(c.arm == "shuffled_answers" for c in base)
    withs = cells.enumerate_cells(SCORES, include_shuffled=True)
    shuffled = [c for c in withs if c.arm == "shuffled_answers"]
    assert len(withs) == len(base) + len(shuffled) and len(shuffled) == 2 * len(cells.NAMED_SEEDS)
    assert all(c.layers == tuple(range(cells.N_LAYERS)) for c in shuffled)
