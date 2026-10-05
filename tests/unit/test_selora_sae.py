"""Spec for selora.sae: reconstruction metrics against closed-form answers."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from indbw.selora.sae import (
    JumpReLUSAE,
    content_mask,
    fvu_per_seq,
    l0_per_seq,
    load_sae_npz,
    sq_dev_per_seq,
)

D = 8


def _identity_sae(threshold: float = -1e30, b_dec: torch.Tensor | None = None) -> JumpReLUSAE:
    eye = torch.eye(D, dtype=torch.float64)
    return JumpReLUSAE(
        eye,
        eye.clone(),
        torch.zeros(D, dtype=torch.float64),
        torch.zeros(D, dtype=torch.float64) if b_dec is None else b_dec,
        torch.full((D,), threshold, dtype=torch.float64),
    )


def _x(seed: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(3, 6, D, generator=g, dtype=torch.float64)
    mask = content_mask(torch.ones(3, 6, dtype=torch.long))
    return x, mask


def test_identity_sae_has_zero_fvu() -> None:
    x, mask = _x()
    mu = x[mask].mean(0)
    den = sq_dev_per_seq(x, mu, mask)
    np.testing.assert_allclose(
        fvu_per_seq(x, _identity_sae(), mask, mu, den).numpy(), 0, atol=1e-12
    )


def test_mean_predicting_sae_has_fvu_exactly_one() -> None:
    # W_dec = 0 and b_dec = mu  =>  reconstruction == mu  =>  FVU == 1.
    x, mask = _x(1)
    mu = x[mask].mean(0)
    sae = _identity_sae(b_dec=mu)
    sae = JumpReLUSAE(
        sae.W_enc, torch.zeros(D, D, dtype=torch.float64), sae.b_enc, mu, sae.threshold
    )
    den = sq_dev_per_seq(x, mu, mask)
    np.testing.assert_allclose(fvu_per_seq(x, sae, mask, mu, den).numpy(), 1.0, rtol=1e-12)


def test_bos_and_padding_do_not_affect_fvu() -> None:
    x, _ = _x(2)
    attn = torch.ones(3, 6, dtype=torch.long)
    attn[:, -1] = 0  # right padding
    mask = content_mask(attn)
    mu = x[mask].mean(0)
    sae = _identity_sae(threshold=0.5)  # lossy: zeroes small entries
    base = fvu_per_seq(x, sae, mask, mu, sq_dev_per_seq(x, mu, mask))
    x2 = x.clone()
    x2[:, 0] = 1e6  # BOS blow-up
    x2[:, -1] = -1e6  # pad garbage
    again = fvu_per_seq(x2, sae, mask, mu, sq_dev_per_seq(x, mu, mask))
    np.testing.assert_allclose(base.numpy(), again.numpy(), rtol=1e-12)
    assert not mask[:, 0].any() and not mask[:, -1].any()


def test_constant_activations_raise_not_return_a_number() -> None:
    x = torch.ones(2, 5, D, dtype=torch.float64)
    mask = content_mask(torch.ones(2, 5, dtype=torch.long))
    mu = x[mask].mean(0)
    den = sq_dev_per_seq(x, mu, mask)
    with pytest.raises(ValueError, match="constant"):
        fvu_per_seq(x, _identity_sae(), mask, mu, den)


def test_nan_and_empty_sequence_raise() -> None:
    x, mask = _x()
    mu = x[mask].mean(0)
    bad = x.clone()
    bad[0, 1, 0] = float("nan")
    with pytest.raises(ValueError):
        sq_dev_per_seq(bad, mu, mask)
    with pytest.raises(ValueError):
        sq_dev_per_seq(x, mu, torch.zeros_like(mask))


def test_lossy_sae_fvu_strictly_between_extremes() -> None:
    x, mask = _x(3)
    mu = x[mask].mean(0)
    f = fvu_per_seq(x, _identity_sae(threshold=0.8), mask, mu, sq_dev_per_seq(x, mu, mask))
    assert bool((f > 0).all()) and bool((f < 1).all())


def test_l0_closed_form() -> None:
    x = torch.zeros(1, 4, D, dtype=torch.float64)
    x[0, 1:, :3] = 1.0  # three active features on every content token
    mask = content_mask(torch.ones(1, 4, dtype=torch.long))
    assert l0_per_seq(x, _identity_sae(threshold=0.5), mask).item() == 3.0


def test_npz_roundtrip_and_missing_key(tmp_path, tiny_sae_factory) -> None:  # type: ignore[no-untyped-def]
    s = tiny_sae_factory()
    p = tmp_path / "params.npz"
    np.savez(
        p,
        W_enc=s.W_enc.numpy(),
        W_dec=s.W_dec.numpy(),
        b_enc=s.b_enc.numpy(),
        b_dec=s.b_dec.numpy(),
        threshold=s.threshold.numpy(),
    )
    t = load_sae_npz(p, dtype=torch.float64)
    x = torch.randn(2, 3, s.d_model, dtype=torch.float64)
    torch.testing.assert_close(t.reconstruct(x), s.reconstruct(x))
    np.savez(tmp_path / "bad.npz", W_enc=s.W_enc.numpy())
    with pytest.raises(KeyError):
        load_sae_npz(tmp_path / "bad.npz")


def test_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        JumpReLUSAE(
            torch.zeros(4, 6), torch.zeros(5, 4), torch.zeros(6), torch.zeros(4), torch.zeros(6)
        )
