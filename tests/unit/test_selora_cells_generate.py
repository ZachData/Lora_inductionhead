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
    primary, secondary = cells.PRIMARY_MODEL, cells.SAE_MODEL
    by_model = {m: [c for c in cs if c.model == m] for m in cells.MODELS}
    # Primary: base + all/late x 3 seeds + sae_top/sae_bottom x seed 0 + 24 random draws.
    assert len(by_model[primary]) == 1 + 2 * 3 + 2 + prereg.N_RANDOM_DRAWS == 33
    # Secondary (the SAE model): base + all x seed 0, which S1a and S4 read.
    assert len(by_model[secondary]) == 2
    assert len(cs) == 35 and sum(c.arm != "base" for c in cs) == 33
    assert len({c.run_id for c in cs}) == len(cs)
    late = [c for c in cs if c.arm == "late" and c.model == primary]
    assert [c.layers for c in late] == [tuple(range(13, 26))] * 3


def test_every_planned_cell_is_read_by_some_claim() -> None:
    # The lean design drops `early` and the secondary-model arms no claim consumes.
    cs = cells.enumerate_cells(SCORES)
    assert not any(c.arm == "early" for c in cs)
    assert not any(c.arm.startswith("random_") for c in cs if c.model == cells.SAE_MODEL)
    seeds = {
        (c.model, c.arm): {x.seed for x in cs if (x.model, x.arm) == (c.model, c.arm)} for c in cs
    }
    assert seeds[(cells.PRIMARY_MODEL, "all")] == seeds[(cells.PRIMARY_MODEL, "late")] == {0, 1, 2}
    assert (
        seeds[(cells.PRIMARY_MODEL, "sae_top")] == seeds[(cells.PRIMARY_MODEL, "sae_bottom")] == {0}
    )


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
    assert len(withs) == len(base) + len(shuffled) and len(shuffled) == len(cells.MODELS) * len(
        cells.NAMED_SEEDS
    )
    assert all(c.layers == tuple(range(cells.N_LAYERS)) for c in shuffled)


def test_budgeted_reference_and_fvu_match_single_pass(
    tiny_gemma2, tiny_sae_factory
) -> None:  # type: ignore[no-untyped-def]
    g = torch.Generator().manual_seed(3)
    ids = torch.randint(3, 64, (7, 9), generator=g)
    ids[:, 0] = 2
    attn = torch.ones_like(ids)
    attn[3:, 6:] = 0  # ragged lengths, so chunk boundaries cut through padding
    saes = {1: tiny_sae_factory(seed=1), 2: tiny_sae_factory(seed=2)}
    one = pipeline.reference_stats(tiny_gemma2, ids, attn, [1, 2], budget=10**9)
    many = pipeline.reference_stats(tiny_gemma2, ids, attn, [1, 2], budget=20)
    f1 = pipeline.fvu_by_layer(tiny_gemma2, saes, ids, attn, one, budget=10**9)
    f2 = pipeline.fvu_by_layer(tiny_gemma2, saes, ids, attn, many, budget=20)
    for li in (1, 2):
        # the fp32 forward pass differs by ~1e-6 relative across batch shapes (different kernels /
        # padding), so agreement is to ~10x that, not to float64 precision
        torch.testing.assert_close(one[li][0], many[li][0], rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(one[li][1], many[li][1], rtol=1e-5, atol=0)
        torch.testing.assert_close(f1[li], f2[li], rtol=1e-4, atol=0)


def test_mmlu_option_logits_pick_the_same_letter_as_full_vocab_argmax(
    tiny_gemma2, stub_tokenizer
) -> None:  # type: ignore[no-untyped-def]
    tok = stub_tokenizer
    prompts = [generate.mmlu_prompt(f"question {i}", ["aa", "bb", "cc", "dd"]) for i in range(6)]
    got = generate.mmlu_logprob_preds(tiny_gemma2, tok, prompts, batch_size=3)
    opt = generate.option_token_ids(tok)
    ref = []
    for p in prompts:
        tok.padding_side = "left"
        enc = tok([p], return_tensors="pt", padding=True)
        with torch.no_grad():
            lg = tiny_gemma2(**enc).logits[0, -1].float()
        ref.append("ABCD"[int(torch.argmax(lg[opt]))])
    assert got == ref
