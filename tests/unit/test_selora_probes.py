"""Spec for selora.probes: capture correctness, splice discrimination, divergence null."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from indbw.selora import probes
from indbw.selora.sae import JumpReLUSAE


def _batch(vocab: int = 64, b: int = 3, t: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(0)
    ids = torch.randint(3, vocab, (b, t), generator=g)
    ids[:, 0] = 2  # BOS
    return ids, torch.ones(b, t, dtype=torch.long)


def test_capture_matches_hidden_states_except_last(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    ids, attn = _batch()
    cap = probes.capture_resid(tiny_gemma2, ids, attn, [0, 1, 3])
    hs = tiny_gemma2(input_ids=ids, attention_mask=attn, output_hidden_states=True).hidden_states
    torch.testing.assert_close(cap[0], hs[1].float())  # mid-stack: identical
    torch.testing.assert_close(cap[1], hs[2].float())
    # Final layer: HF's last hidden state is post-final-norm, a different quantity.
    assert not torch.allclose(cap[3], hs[4].float(), atol=1e-4)


def test_capture_is_layer_specific(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    ids, attn = _batch()
    cap = probes.capture_resid(tiny_gemma2, ids, attn, [0, 1])
    assert not torch.allclose(cap[0], cap[1])


def _identity_sae(d: int) -> JumpReLUSAE:
    eye = torch.eye(d, dtype=torch.float64)
    z = torch.zeros(d, dtype=torch.float64)
    return JumpReLUSAE(eye, eye.clone(), z, z.clone(), torch.full((d,), -1e30, dtype=torch.float64))


def test_splice_discriminates_perfect_from_destroyed_sae(tiny_gemma2) -> None:  # type: ignore[no-untyped-def]
    ids, attn = _batch()
    d = tiny_gemma2.config.hidden_size
    clean = probes.spliced_ce_per_seq(tiny_gemma2, ids, attn, None, None)
    perfect = probes.spliced_ce_per_seq(tiny_gemma2, ids, attn, 1, _identity_sae(d))
    torch.testing.assert_close(
        perfect, clean, atol=1e-5, rtol=1e-5
    )  # float32 round-trip via float64
    eye = torch.eye(d, dtype=torch.float64)
    z = torch.zeros(d, dtype=torch.float64)
    dead = JumpReLUSAE(eye, torch.zeros(d, d, dtype=torch.float64), z, z.clone(), z.clone())
    broken = probes.spliced_ce_per_seq(tiny_gemma2, ids, attn, 1, dead)
    assert bool((broken - clean).abs().max() > 1e-3)


def test_splice_requires_sae() -> None:
    with pytest.raises(ValueError):
        probes.spliced_ce_per_seq(
            None, torch.zeros(1, 2, dtype=torch.long), torch.ones(1, 2), 0, None
        )  # type: ignore[arg-type]


def _classes(rng: np.random.Generator, n: int = 60, d: int = 200) -> tuple[np.ndarray, np.ndarray]:
    labels = np.repeat([0, 1], n // 2)
    acts = np.maximum(rng.normal(size=(n, d)), 0.0)
    return acts, labels


def test_divergence_discriminates_planted_from_null() -> None:
    rng = np.random.default_rng(0)
    acts, y = _classes(rng)
    null_res = probes.divergence_with_null(acts, y, k=10, n_perm=300, rng=rng)
    planted = acts.copy()
    planted[y == 1, :10] += 3.0
    pl_res = probes.divergence_with_null(planted, y, k=10, n_perm=300, rng=rng)
    assert null_res["p_value"] > 0.05
    assert abs(float(null_res["excess"])) < 0.5  # selection bias is subtracted out
    assert pl_res["p_value"] == pytest.approx(1 / 301)
    assert float(pl_res["excess"]) > 5.0
    # The raw statistic alone would mislead: it is clearly positive on pure noise.
    assert float(null_res["observed"]) > 1.5


def test_divergence_strata_confounded_labels_have_trivial_null() -> None:
    # Each stratum contains one class only, so within-stratum permutation cannot change
    # the labelling: null == observed for every draw and p must be exactly 1.
    rng = np.random.default_rng(1)
    acts, y = _classes(rng)
    res = probes.divergence_with_null(acts, y, k=5, n_perm=50, rng=rng, strata=y.copy())
    assert res["p_value"] == 1.0
    assert res["excess"] == pytest.approx(0.0, abs=1e-12)


def test_divergence_guards() -> None:
    rng = np.random.default_rng(2)
    acts, y = _classes(rng)
    with pytest.raises(ValueError, match="constant"):
        probes.divergence_with_null(np.ones_like(acts), y, 5, 10, rng)
    with pytest.raises(ValueError):
        probes.divergence_with_null(acts, np.array([1] + [0] * 59), 5, 10, rng)
    bad = acts.copy()
    bad[0, 0] = np.nan
    with pytest.raises(ValueError):
        probes.divergence_with_null(bad, y, 5, 10, rng)
    with pytest.raises(ValueError):
        probes.topk_mean_abs(np.zeros(4), 5)


def test_welch_dead_feature_is_zero_not_nan() -> None:
    acts = np.zeros((10, 3))
    acts[:5, 1] = 1.0
    acts[5:, 1] = 2.0
    acts[:, 2] = np.arange(10)
    eff = probes.welch_effects(acts, np.array([0] * 5 + [1] * 5))
    assert eff[0] == 0.0 and np.isfinite(eff).all()
    assert eff[1] > 100  # zero within-class variance: huge but finite t via the guard


def test_make_strata_separates_length_and_digit_bins() -> None:
    lengths = np.array([5, 5, 50, 50, 5, 50])
    digits = np.array([0, 4, 0, 4, 0, 4])
    st = probes.make_strata(lengths, digits, n_len_bins=2)
    assert st[0] == st[4] and st[1] != st[0] and st[2] != st[0] and st[3] == st[5]
    assert len(set(st.tolist())) == 4


def test_digit_strata_kill_a_pure_digit_confound() -> None:
    # Class 1 = every prompt with digits; the SAE effect is a digit feature. Stratifying on
    # digit count makes the labelling unpermutable, so the null equals the observed (p = 1):
    # the design cannot attribute a digit effect to "math".
    rng = np.random.default_rng(4)
    n = 60
    digits = np.r_[np.zeros(30), np.full(30, 3)]
    y = (digits > 0).astype(int)
    acts = np.maximum(rng.normal(size=(n, 50)), 0)
    acts[y == 1, 0] += 5.0
    st = probes.make_strata(np.full(n, 10), digits, n_len_bins=1)
    res = probes.divergence_with_null(acts, y, 3, 50, rng, strata=st)
    assert res["p_value"] == 1.0


def _layers_with_peak(
    rng: np.random.Generator, peak: int, n_layers: int = 12
) -> tuple[list[np.ndarray], np.ndarray]:
    y = np.repeat([0, 1], 40)
    out = []
    for li in range(n_layers):
        a = np.maximum(rng.normal(size=(80, 60)), 0)
        a[y == 1, :5] += 2.5 if li == peak else 0.3
        out.append(a)
    return out, y


def test_peak_layer_bootstrap_discriminates() -> None:
    rng = np.random.default_rng(5)
    acts, y = _layers_with_peak(rng, peak=8)
    nulls = np.array([probes.divergence_with_null(a, y, 5, 50, rng)["null_mean"] for a in acts])
    inside = probes.peak_layer_bootstrap(acts, y, nulls, 5, 200, rng, 6, 10)
    outside = probes.peak_layer_bootstrap(acts, y, nulls, 5, 200, rng, 0, 4)
    assert inside >= 0.95 and outside <= 0.05


def test_peak_layer_bootstrap_guards() -> None:
    rng = np.random.default_rng(6)
    acts, y = _layers_with_peak(rng, 3, n_layers=4)
    with pytest.raises(ValueError):
        probes.peak_layer_bootstrap(acts, y, np.zeros(3), 5, 10, rng, 0, 1)
