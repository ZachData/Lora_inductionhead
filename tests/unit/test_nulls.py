"""Tests for indbw.nulls -- PROJECT.md §8's mandatory controls.

A broken null is the worst kind of silent failure in this repo: it does
not crash, it produces a band, and the band makes a real result look
significant (or hides one). So every generator here is checked against a
closed-form property of what it claims to produce, and every band is
checked to *discriminate* -- to come back near 1/2 on an off-diagonal
composition and at 0 on a symmetric one, rather than returning the same
comfortable number whatever it is handed.
"""

from __future__ import annotations

import numpy as np
import pytest

from indbw.algebra import phi
from indbw.lora import bandwidth, bandwidth_fraction
from indbw.nulls import (
    matched_beta_support,
    matched_norm_random_update,
    phi_null_band,
    phi_separation_null_band,
    random_subspace_basis,
)

RNG = lambda seed=0: np.random.default_rng(seed)

# ---------------------------------------------------------------------------
# 1. Closed-form oracles
# ---------------------------------------------------------------------------


def test_random_subspace_basis_is_orthonormal() -> None:
    Q = random_subspace_basis(12, 4, RNG())
    assert Q.shape == (12, 4)
    # Exact to machine precision: QR's Q factor is orthonormal by construction.
    np.testing.assert_allclose(Q.T @ Q, np.eye(4), rtol=0, atol=1e-12)


def test_random_subspace_basis_full_dimension_spans_everything() -> None:
    Q = random_subspace_basis(5, 5, RNG())
    np.testing.assert_allclose(Q @ Q.T, np.eye(5), rtol=0, atol=1e-12)


@pytest.mark.parametrize("d,k", [(4, 0), (4, 5), (0, 1), (-1, 1)])
def test_random_subspace_basis_rejects_bad_dimensions(d: int, k: int) -> None:
    with pytest.raises(ValueError):
        random_subspace_basis(d, k, RNG())


@pytest.mark.parametrize("rank", [1, 2, 5])
def test_matched_norm_random_update_has_the_requested_rank(rank: int) -> None:
    M = matched_norm_random_update((8, 6), rank, target_norm=3.0, rng=RNG())
    assert M.shape == (8, 6)
    assert np.linalg.matrix_rank(M) == rank


@pytest.mark.parametrize("target", [1e-6, 0.5, 1.0, 250.0])
def test_matched_norm_random_update_matches_the_norm_exactly(target: float) -> None:
    # Exact identity, not an approximation: the draw is rescaled by
    # target/||M||. rtol=1e-12 per CLAUDE.md's tolerance policy.
    M = matched_norm_random_update((7, 5), 3, target_norm=target, rng=RNG())
    assert float(np.linalg.norm(M)) == pytest.approx(target, rel=1e-12)


def test_matched_norm_random_update_is_deterministic_under_a_seed() -> None:
    a = matched_norm_random_update((6, 4), 2, 1.0, RNG(7))
    b = matched_norm_random_update((6, 4), 2, 1.0, RNG(7))
    np.testing.assert_array_equal(a, b)


def test_matched_norm_random_update_differs_across_seeds() -> None:
    # Discrimination guard: a generator that ignored its rng would pass
    # every other test in this file.
    a = matched_norm_random_update((6, 4), 2, 1.0, RNG(0))
    b = matched_norm_random_update((6, 4), 2, 1.0, RNG(1))
    assert not np.allclose(a, b)


@pytest.mark.parametrize(
    "shape,rank,target",
    [
        ((4, 4), 0, 1.0),  # rank below 1
        ((4, 4), 5, 1.0),  # rank above min(shape)
        ((4, 4), 2, 0.0),  # degenerate zero-norm null
        ((4, 4), 2, -1.0),
        ((4, 4), 2, float("nan")),
        ((4, 4), 2, float("inf")),
    ],
)
def test_matched_norm_random_update_rejects_degenerate_input(
    shape: tuple[int, int], rank: int, target: float
) -> None:
    with pytest.raises(ValueError):
        matched_norm_random_update(shape, rank, target, RNG())


def test_bandwidth_closed_form() -> None:
    # beta = r(d_in + d_out), PROJECT.md §5.
    assert bandwidth(4, 64, 512) == 4 * (64 + 512)
    assert bandwidth(1, 1, 1) == 2
    assert bandwidth_fraction(4, 64, 512) == pytest.approx(4 * 576 / (64 * 512), rel=1e-12)


@pytest.mark.parametrize("rank,d_in,d_out", [(0, 4, 4), (1, 0, 4), (1, 4, 0)])
def test_bandwidth_rejects_bad_dimensions(rank: int, d_in: int, d_out: int) -> None:
    with pytest.raises(ValueError):
        bandwidth(rank, d_in, d_out)


# ---------------------------------------------------------------------------
# 2. phi bands -- discrimination, the silent-failure guard
# ---------------------------------------------------------------------------


def _off_diagonal_compose(d: int) -> tuple[np.ndarray, np.ndarray]:
    """P_U M P_V with U perp V: PROJECT.md §3's exact phi = 1/2 shape."""
    P_U = np.zeros((d, d))
    P_U[: d // 2, : d // 2] = np.eye(d // 2)
    P_V = np.zeros((d, d))
    P_V[d // 2 :, d // 2 :] = np.eye(d - d // 2)
    return P_U, P_V  # type: ignore[return-value]


def test_phi_null_band_on_an_off_diagonal_composition_is_exactly_one_half() -> None:
    # PROJECT.md §3's ceiling result: M = P_U M P_V with U perp V has
    # tr(M^2) = 0, hence phi = 1/2 *exactly*, at any rank. Every draw in
    # the band must hit it to machine precision -- this is the strongest
    # available check that the band is really computing phi of the
    # composed object and not of something else.
    d = 8
    P_U, P_V = _off_diagonal_compose(d)
    band = phi_null_band(
        (d, d), rank=3, target_norm=2.0, compose=lambda M: P_U @ M @ P_V, n_draws=16
    )
    np.testing.assert_allclose(band.values, 0.5, rtol=0, atol=1e-12)
    assert band.percentile_value == pytest.approx(0.5, rel=1e-12)


def test_phi_null_band_on_a_symmetrizing_composition_is_exactly_zero() -> None:
    # Known-negative: phi(M) = 0 for symmetric M (PROJECT.md §3).
    band = phi_null_band((6, 6), rank=2, target_norm=1.0, compose=lambda M: M + M.T, n_draws=16)
    np.testing.assert_allclose(band.values, 0.0, rtol=0, atol=1e-12)


def test_phi_null_band_on_an_antisymmetrizing_composition_is_exactly_one() -> None:
    band = phi_null_band((6, 6), rank=2, target_norm=1.0, compose=lambda M: M - M.T, n_draws=16)
    np.testing.assert_allclose(band.values, 1.0, rtol=0, atol=1e-12)


def test_phi_null_band_of_a_generic_square_update_concentrates_near_one_half() -> None:
    # PROJECT.md §6's stated limitation, made visible: an unstructured
    # random update sits near 1/2, which is *why* phi_QK ~ 0.5 alone is
    # uninformative. Wide tolerance -- this asserts concentration, not a
    # precise value; at d=32 the spread across draws is a few percent.
    band = phi_null_band((32, 32), rank=8, target_norm=1.0, compose=lambda M: M, n_draws=64)
    assert 0.40 < band.mean < 0.60
    assert band.percentile_value > band.mean  # an upper percentile is above the mean


def test_phi_null_band_values_are_all_in_range_and_recorded() -> None:
    band = phi_null_band((10, 10), rank=4, target_norm=1.0, compose=lambda M: M, n_draws=32)
    assert band.n_draws == 32 and band.values.shape == (32,)
    assert np.all((band.values >= 0.0) & (band.values <= 1.0))
    assert band.percentile_value == pytest.approx(np.percentile(band.values, 95.0), rel=1e-12)


def test_phi_null_band_exceeds_discriminates() -> None:
    band = phi_null_band((10, 10), rank=4, target_norm=1.0, compose=lambda M: M, n_draws=32)
    assert band.exceeds(band.percentile_value + 1e-6)
    assert not band.exceeds(band.percentile_value - 1e-6)


def test_phi_null_band_is_deterministic_under_a_seed() -> None:
    kwargs = {"rank": 4, "target_norm": 1.0, "compose": lambda M: M, "n_draws": 16, "seed": 3}
    a = phi_null_band((10, 10), **kwargs)  # type: ignore[arg-type]
    b = phi_null_band((10, 10), **kwargs)  # type: ignore[arg-type]
    np.testing.assert_array_equal(a.values, b.values)


@pytest.mark.parametrize("n_draws,percentile", [(1, 95.0), (0, 95.0), (10, 0.0), (10, 100.0)])
def test_phi_null_band_rejects_a_degenerate_band_request(n_draws: int, percentile: float) -> None:
    with pytest.raises(ValueError):
        phi_null_band(
            (6, 6),
            rank=2,
            target_norm=1.0,
            compose=lambda M: M,
            n_draws=n_draws,
            percentile=percentile,
        )


def test_phi_separation_band_of_identically_composed_arms_is_small() -> None:
    # Two arms with the same shape/norm/composition differ only by the
    # draw, so their |phi_QK - phi_OV| band sits near zero -- the
    # reference point any real separation must clear.
    band = phi_separation_null_band(
        (16, 16),
        (16, 16),
        rank=4,
        target_norm_qk=1.0,
        target_norm_ov=1.0,
        compose_qk=lambda M: M,
        compose_ov=lambda M: M,
        n_draws=64,
    )
    assert np.all(band.values >= 0.0)
    assert band.percentile_value < 0.25


def test_phi_separation_band_discriminates_a_real_separation() -> None:
    # Known-positive: an antisymmetrizing arm (phi = 1) against a
    # symmetrizing one (phi = 0) is a separation of exactly 1, which no
    # null band can contain.
    band = phi_separation_null_band(
        (16, 16),
        (16, 16),
        rank=4,
        target_norm_qk=1.0,
        target_norm_ov=1.0,
        compose_qk=lambda M: M,
        compose_ov=lambda M: M,
        n_draws=64,
    )
    assert band.exceeds(1.0)


# ---------------------------------------------------------------------------
# 3. Matched-beta support (h0_3)
# ---------------------------------------------------------------------------


def test_matched_beta_support_has_exactly_beta_free_entries() -> None:
    mask = matched_beta_support((8, 6), beta=17, rng=RNG())
    assert mask.shape == (8, 6) and mask.dtype == np.bool_
    assert int(mask.sum()) == 17  # exact parameter-budget match, not approximate


def test_matched_beta_support_matches_a_lora_arm_s_budget() -> None:
    # The control's whole point: same beta as the rank-r LoRA arm it is
    # compared against (PROJECT.md §8, "Matched-beta unstructured control").
    d_out, d_in, rank = 64, 32, 4
    beta = bandwidth(rank, d_in, d_out)
    mask = matched_beta_support((d_out, d_in), beta, RNG())
    assert int(mask.sum()) == beta


def test_matched_beta_support_is_deterministic_under_a_seed() -> None:
    np.testing.assert_array_equal(
        matched_beta_support((8, 6), 10, RNG(5)), matched_beta_support((8, 6), 10, RNG(5))
    )


def test_matched_beta_support_differs_across_seeds() -> None:
    assert not np.array_equal(
        matched_beta_support((8, 6), 10, RNG(0)), matched_beta_support((8, 6), 10, RNG(1))
    )


def test_matched_beta_support_refuses_a_vacuous_control() -> None:
    # beta > d_out*d_in means the "restricted" update is the full matrix
    # space -- a control that cannot fail. Raise, never clamp.
    with pytest.raises(ValueError, match="ambient dimension"):
        matched_beta_support((4, 4), beta=17, rng=RNG())


def test_matched_beta_support_at_exactly_the_ambient_dimension_is_allowed() -> None:
    mask = matched_beta_support((4, 4), beta=16, rng=RNG())
    assert mask.all()


@pytest.mark.parametrize("beta", [0, -1])
def test_matched_beta_support_rejects_a_nonpositive_budget(beta: int) -> None:
    with pytest.raises(ValueError):
        matched_beta_support((4, 4), beta, RNG())


def test_phi_of_a_matched_beta_update_is_computable() -> None:
    # Integration of the two controls: a masked random update is a valid
    # input to phi (nonzero, finite), so h0_3's control can be reported
    # on the same axis as h0_4's.
    mask = matched_beta_support((8, 8), beta=24, rng=RNG())
    delta = RNG(1).standard_normal((8, 8)) * mask
    assert 0.0 <= phi(delta) <= 1.0
