"""Tests for indbw.train: LoRA hook injection, freezing, the training
loop, and adapter snapshotting.

Uses `tiny_model` (tests/conftest.py) throughout -- CLAUDE.md's fixtures
rule, never a real checkpoint outside tests/integration/. Each test is
fast (tiny dims, few steps); the real timed run this module exists to
support lives in scripts/run_g2.py.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from indbw.evalset import build_eval_tokens
from indbw.probes import icl_score, recovery
from indbw.train import (
    ARMS,
    TrainConfig,
    TrainingBudgetExceeded,
    _check_finite_loss,
    build_hooks,
    compute_recovery,
    factor_shapes,
    first_copy_nll,
    freeze_base_model,
    init_lora_factors,
    load_snapshot,
    ov_hooks,
    qk_both_hooks,
    qk_hooks,
    save_snapshot,
    second_copy_nll,
    train_lora,
)

# ---------------------------------------------------------------------------
# 1. Closed-form oracles: second_copy_nll / first_copy_nll on hand-built logits
# ---------------------------------------------------------------------------


def test_second_copy_nll_uniform_logits_equals_log_vocab() -> None:
    # All-zero logits -> uniform softmax -> NLL = log(vocab) exactly,
    # for every example, regardless of which tokens were sampled.
    vocab, T, batch = 5, 4, 3
    tokens = torch.randint(0, vocab, (batch, 2 * T))
    logits = torch.zeros(batch, 2 * T, vocab)
    nll = second_copy_nll(logits, tokens, T)
    assert nll.shape == (batch,)
    assert torch.allclose(nll, torch.full((batch,), math.log(vocab)), rtol=0, atol=1e-6)


def test_first_copy_nll_uniform_logits_equals_log_vocab() -> None:
    vocab, T, batch = 5, 4, 3
    tokens = torch.randint(0, vocab, (batch, 2 * T))
    logits = torch.zeros(batch, 2 * T, vocab)
    nll = first_copy_nll(logits, tokens, T)
    assert nll.shape == (batch,)
    assert torch.allclose(nll, torch.full((batch,), math.log(vocab)), rtol=0, atol=1e-6)


def test_second_copy_nll_confident_correct_logits_near_zero() -> None:
    # A logit spike of +50 at the true next-token class, -50 elsewhere,
    # drives softmax cross-entropy to (machine-precision-adjacent) zero.
    vocab, T = 4, 3
    tokens = torch.tensor([[0, 1, 2, 0, 1, 2]])  # 2T = 6
    logits = torch.full((1, 6, vocab), -50.0)
    for t in range(6 - 1):
        logits[0, t, tokens[0, t + 1]] = 50.0
    nll = second_copy_nll(logits, tokens, T)
    assert nll.item() < 1e-3


def test_second_copy_nll_seq_len_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        second_copy_nll(torch.zeros(1, 5, 3), torch.zeros(1, 5, dtype=torch.long), T=4)


def test_first_copy_nll_requires_t_at_least_2() -> None:
    with pytest.raises(ValueError):
        first_copy_nll(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long), T=1)


def test_second_copy_nll_requires_t_at_least_2() -> None:
    with pytest.raises(ValueError):
        second_copy_nll(torch.zeros(1, 2, 3), torch.zeros(1, 2, dtype=torch.long), T=1)


def test_second_copy_nll_excludes_the_seam_position() -> None:
    # The seam (predicting token T from logits at T-1) must not influence
    # second_copy_nll at all -- set it to something wildly wrong (which
    # would blow up the mean if included) and confirm the result is
    # unaffected, matching scripts/g0_sweep.py's nll[:, T:] windowing
    # (see second_copy_nll's docstring).
    vocab, T = 4, 3
    tokens = torch.tensor([[0, 1, 2, 0, 1, 2]])  # 2T = 6
    logits = torch.full((1, 6, vocab), -50.0)
    for t in range(6 - 1):
        logits[0, t, tokens[0, t + 1]] = 50.0  # everywhere confidently correct...
    logits[0, T - 1, :] = 0.0  # ...except the seam position, made uniform
    nll = second_copy_nll(logits, tokens, T)
    assert nll.item() < 1e-3  # seam's high NLL must not leak into the mean


def test_check_finite_loss_raises_on_nan() -> None:
    with pytest.raises(FloatingPointError):
        _check_finite_loss(torch.tensor(float("nan")), step=1)


def test_check_finite_loss_raises_on_inf() -> None:
    with pytest.raises(FloatingPointError):
        _check_finite_loss(torch.tensor(float("inf")), step=1)


def test_check_finite_loss_passes_finite() -> None:
    _check_finite_loss(torch.tensor(1.23), step=1)  # must not raise


# ---------------------------------------------------------------------------
# 2. init_lora_factors: closed-form zero delta at init, rank validation
# ---------------------------------------------------------------------------


def test_init_lora_factors_delta_is_exactly_zero_at_init() -> None:
    factors = init_lora_factors(d_out=6, d_in=4, rank=3, alpha=2.0, seed=0)
    delta = factors.delta()
    assert torch.equal(delta, torch.zeros(6, 4))


def test_init_lora_factors_rejects_invalid_rank() -> None:
    with pytest.raises(ValueError):
        init_lora_factors(d_out=6, d_in=4, rank=0, alpha=1.0, seed=0)


def test_init_lora_factors_rejects_invalid_dims() -> None:
    with pytest.raises(ValueError):
        init_lora_factors(d_out=0, d_in=4, rank=1, alpha=1.0, seed=0)


def test_factor_shapes_unknown_arm_raises(tiny_model) -> None:
    with pytest.raises(ValueError):
        factor_shapes(tiny_model, "XY")  # type: ignore[arg-type]


def test_build_hooks_unknown_arm_raises(tiny_model) -> None:
    factors = init_lora_factors(6, 4, rank=2, alpha=1.0, seed=0)
    with pytest.raises(ValueError):
        build_hooks("XY", 0, 0, factors)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# 3. Discrimination: zero delta -> untouched forward pass; nonzero -> changed
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("arm,hook_builder", [("QK", qk_hooks), ("OV", ov_hooks)])
def test_zero_delta_hooks_leave_forward_pass_unchanged(tiny_model, arm, hook_builder) -> None:
    d_out, d_in = factor_shapes(tiny_model, arm)
    factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)
    hooks = hook_builder(layer=1, head=0, factors=factors)

    tokens = torch.randint(0, tiny_model.cfg.d_vocab, (2, 12))
    with torch.no_grad():
        base_logits = tiny_model(tokens, return_type="logits")
        hooked_logits = tiny_model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")

    # B is zero at init -> delta() is exactly zero -> bit-identical forward
    # pass. This is the silent-failure guard: a hook wired to the wrong
    # tensor, or one that silently no-ops, would pass a looser tolerance
    # test just as easily as a correct one.
    assert torch.equal(base_logits, hooked_logits)


@pytest.mark.parametrize("arm,hook_builder", [("QK", qk_hooks), ("OV", ov_hooks)])
def test_nonzero_delta_hooks_change_forward_pass(tiny_model, arm, hook_builder) -> None:
    d_out, d_in = factor_shapes(tiny_model, arm)
    factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)
    with torch.no_grad():
        factors.B += 1.0  # perturb off the zero-init point
    hooks = hook_builder(layer=1, head=0, factors=factors)

    tokens = torch.randint(0, tiny_model.cfg.d_vocab, (2, 12))
    with torch.no_grad():
        base_logits = tiny_model(tokens, return_type="logits")
        hooked_logits = tiny_model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")

    assert not torch.allclose(base_logits, hooked_logits)


def test_qk_delta_at_later_layer_does_not_affect_earlier_layer_activations(tiny_model) -> None:
    # Injecting into layer 1's query must not change layer 0's output --
    # causality guard: a bug that wires the hook to the wrong layer index
    # could easily leak backwards and this test would catch it.
    d_out, d_in = factor_shapes(tiny_model, "QK")
    factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)
    with torch.no_grad():
        factors.B += 1.0
    hooks = qk_hooks(layer=1, head=0, factors=factors)

    tokens = torch.randint(0, tiny_model.cfg.d_vocab, (2, 12))
    with torch.no_grad():
        _, base_cache = tiny_model.run_with_cache(tokens, return_type="logits")
        with tiny_model.hooks(fwd_hooks=hooks):
            _, hooked_cache = tiny_model.run_with_cache(tokens, return_type="logits")

    assert torch.equal(
        base_cache["blocks.0.hook_resid_post"], hooked_cache["blocks.0.hook_resid_post"]
    )


# ---------------------------------------------------------------------------
# 3b. qk_both_hooks -- diagnostic-only two-matrix hook (REVIEW.md 2026-08-14
# W_K-bottleneck follow-up). Same discrimination/causality guards as
# qk_hooks/ov_hooks above, since this is new hook-composition logic whose
# correctness the diagnostic's conclusion depends on.
# ---------------------------------------------------------------------------


def test_qk_both_hooks_zero_delta_leaves_forward_pass_unchanged(tiny_model) -> None:
    d_out, d_in = factor_shapes(tiny_model, "QK")
    q_factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)
    k_factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=1)
    hooks = qk_both_hooks(layer=1, head=0, q_factors=q_factors, k_factors=k_factors)

    tokens = torch.randint(0, tiny_model.cfg.d_vocab, (2, 12))
    with torch.no_grad():
        base_logits = tiny_model(tokens, return_type="logits")
        hooked_logits = tiny_model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")

    assert torch.equal(base_logits, hooked_logits)


def test_qk_both_hooks_nonzero_delta_changes_forward_pass(tiny_model) -> None:
    d_out, d_in = factor_shapes(tiny_model, "QK")
    q_factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)
    k_factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=1)
    with torch.no_grad():
        q_factors.B += 1.0
        k_factors.B += 1.0
    hooks = qk_both_hooks(layer=1, head=0, q_factors=q_factors, k_factors=k_factors)

    tokens = torch.randint(0, tiny_model.cfg.d_vocab, (2, 12))
    with torch.no_grad():
        base_logits = tiny_model(tokens, return_type="logits")
        hooked_logits = tiny_model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")

    assert not torch.allclose(base_logits, hooked_logits)


def test_qk_both_hooks_q_only_and_k_only_each_move_the_forward_pass(tiny_model) -> None:
    # Guards against a copy-paste bug where add_delta_k silently reads/writes
    # the q tensor (or vice versa) -- perturbing only one factor must still
    # change the output, and perturbing neither must not.
    d_out, d_in = factor_shapes(tiny_model, "QK")
    tokens = torch.randint(0, tiny_model.cfg.d_vocab, (2, 12))
    with torch.no_grad():
        base_logits = tiny_model(tokens, return_type="logits")

    zero_factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)
    q_only = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)
    with torch.no_grad():
        q_only.B += 1.0
    hooks_q_only = qk_both_hooks(layer=1, head=0, q_factors=q_only, k_factors=zero_factors)
    with torch.no_grad():
        logits_q_only = tiny_model.run_with_hooks(
            tokens, fwd_hooks=hooks_q_only, return_type="logits"
        )
    assert not torch.allclose(base_logits, logits_q_only)

    k_only = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=1)
    with torch.no_grad():
        k_only.B += 1.0
    hooks_k_only = qk_both_hooks(layer=1, head=0, q_factors=zero_factors, k_factors=k_only)
    with torch.no_grad():
        logits_k_only = tiny_model.run_with_hooks(
            tokens, fwd_hooks=hooks_k_only, return_type="logits"
        )
    assert not torch.allclose(base_logits, logits_k_only)
    assert not torch.allclose(logits_q_only, logits_k_only)


def test_qk_both_hooks_gradient_flows_to_both_factors_only(tiny_model) -> None:
    freeze_base_model(tiny_model)
    d_out, d_in = factor_shapes(tiny_model, "QK")
    q_factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)
    k_factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=1)
    hooks = qk_both_hooks(layer=1, head=0, q_factors=q_factors, k_factors=k_factors)

    tokens = torch.randint(0, tiny_model.cfg.d_vocab, (2, 10))
    logits = tiny_model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")
    loss = second_copy_nll(logits, tokens, T=5).mean()
    loss.backward()

    assert q_factors.B.grad is not None and torch.all(torch.isfinite(q_factors.B.grad))
    assert q_factors.A.grad is not None and torch.all(torch.isfinite(q_factors.A.grad))
    assert k_factors.B.grad is not None and torch.all(torch.isfinite(k_factors.B.grad))
    assert k_factors.A.grad is not None and torch.all(torch.isfinite(k_factors.A.grad))
    assert all(p.grad is None for p in tiny_model.parameters())


def test_qk_both_hooks_at_later_layer_does_not_affect_earlier_layer_activations(tiny_model) -> None:
    d_out, d_in = factor_shapes(tiny_model, "QK")
    q_factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)
    k_factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=1)
    with torch.no_grad():
        q_factors.B += 1.0
        k_factors.B += 1.0
    hooks = qk_both_hooks(layer=1, head=0, q_factors=q_factors, k_factors=k_factors)

    tokens = torch.randint(0, tiny_model.cfg.d_vocab, (2, 12))
    with torch.no_grad():
        _, base_cache = tiny_model.run_with_cache(tokens, return_type="logits")
        with tiny_model.hooks(fwd_hooks=hooks):
            _, hooked_cache = tiny_model.run_with_cache(tokens, return_type="logits")

    assert torch.equal(
        base_cache["blocks.0.hook_resid_post"], hooked_cache["blocks.0.hook_resid_post"]
    )


# ---------------------------------------------------------------------------
# 4. Freezing and gradient isolation
# ---------------------------------------------------------------------------


def test_freeze_base_model_disables_all_grads(tiny_model) -> None:
    freeze_base_model(tiny_model)
    assert all(not p.requires_grad for p in tiny_model.parameters())


def test_gradient_flows_only_to_factors_not_base_model(tiny_model) -> None:
    freeze_base_model(tiny_model)
    d_out, d_in = factor_shapes(tiny_model, "QK")
    factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)
    hooks = qk_hooks(layer=1, head=0, factors=factors)

    tokens = torch.randint(0, tiny_model.cfg.d_vocab, (2, 10))
    logits = tiny_model.run_with_hooks(tokens, fwd_hooks=hooks, return_type="logits")
    loss = second_copy_nll(logits, tokens, T=5).mean()
    loss.backward()

    assert factors.B.grad is not None and torch.all(torch.isfinite(factors.B.grad))
    assert factors.A.grad is not None and torch.all(torch.isfinite(factors.A.grad))
    assert all(p.grad is None for p in tiny_model.parameters())


# ---------------------------------------------------------------------------
# 5. Optimization mechanics: a gradient step on a fixed batch reduces loss
# ---------------------------------------------------------------------------


def test_overfitting_a_single_batch_reduces_loss(tiny_model) -> None:
    # OV arm, not QK: on a randomly-initialized model, attention patterns
    # carry no prefix-matching structure by construction (PROJECT.md §2's
    # own definition of a pre-transition checkpoint), so *which* position
    # a QK-only delta attends to barely matters when the frozen OV
    # circuit downstream of it is random noise -- confirmed empirically,
    # QK-only stalls near a ~6% reduction even at higher rank/lr/steps.
    # OV directly reshapes what gets written to the residual stream from
    # whatever position attention already lands on, so it is the arm
    # this mechanics test (does the optimizer step actually work, not
    # whether QK alone can install induction) should exercise.
    freeze_base_model(tiny_model)
    d_out, d_in = factor_shapes(tiny_model, "OV")
    factors = init_lora_factors(d_out, d_in, rank=16, alpha=8.0, seed=0)
    hooks = ov_hooks(layer=1, head=0, factors=factors)
    optimizer = torch.optim.Adam([factors.B, factors.A], lr=0.1)

    batch = build_eval_tokens(n_eval=8, T=6, seed=0, d_vocab=tiny_model.cfg.d_vocab)

    losses = []
    for _ in range(60):
        logits = tiny_model.run_with_hooks(batch, fwd_hooks=hooks, return_type="logits")
        loss = second_copy_nll(logits, batch, T=6).mean()
        losses.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # Repeatedly training on the *same* batch is essentially guaranteed to
    # reduce loss substantially -- a much lower flakiness risk than
    # asserting monotone improvement on resampled batches.
    assert losses[-1] < losses[0] * 0.7


# ---------------------------------------------------------------------------
# 6. compute_recovery: closed-form R=0 when the eval model IS the A baseline
# ---------------------------------------------------------------------------


def test_compute_recovery_is_zero_when_icl_a_equals_the_evaluated_model(tiny_model) -> None:
    d_out, d_in = factor_shapes(tiny_model, "QK")
    factors = init_lora_factors(d_out, d_in, rank=4, alpha=8.0, seed=0)  # delta == 0
    hooks = qk_hooks(layer=1, head=0, factors=factors)
    eval_tokens = build_eval_tokens(n_eval=16, T=6, seed=1, d_vocab=tiny_model.cfg.d_vocab)

    # icl_a computed from this exact (zero-delta) model; icl_b arbitrary
    # but distinct, so recovery's denominator is nonzero.
    n = eval_tokens.shape[0]
    nll_first = np.empty(n)
    nll_second = np.empty(n)
    with torch.no_grad():
        logits = tiny_model.run_with_hooks(eval_tokens, fwd_hooks=hooks, return_type="logits")
    nll_first[:] = first_copy_nll(logits, eval_tokens, T=6).numpy()
    nll_second[:] = second_copy_nll(logits, eval_tokens, T=6).numpy()
    icl_a = icl_score(nll_first, nll_second)
    icl_b = icl_a + 1.0

    r = compute_recovery(tiny_model, hooks, eval_tokens, T=6, icl_a=icl_a, icl_b=icl_b)
    assert r == pytest.approx(0.0, abs=1e-6)
    # Cross-check against the pure closed-form oracle directly.
    assert r == pytest.approx(recovery(icl_a, icl_a, icl_b), abs=1e-12)


# ---------------------------------------------------------------------------
# 7. train_lora: stop-on-criterion, stop-on-max-steps, budget, snapshots
# ---------------------------------------------------------------------------


def _base_config(tmp_path, **overrides) -> TrainConfig:
    defaults = {
        "arm": "QK",
        "layer": 1,
        "head": 0,
        "rank": 4,
        "alpha": 8.0,
        "lr": 0.05,
        "max_steps": 6,
        "batch_size": 4,
        "T": 6,
        "d_vocab": 50,
        "train_seed": 0,
        "icl_a": -1e9,  # trivially satisfied criterion by default
        "icl_b": 1e9,
        "criterion_r": -1e9,
        "eval_every": 2,
        "eval_n": 8,
        "eval_seed": 0,
        "max_wall_clock_s": 3600.0,
        "snapshot_dir": None,
        "snapshot_every": 2,
    }
    defaults.update(overrides)
    return TrainConfig(**defaults)


def test_train_lora_stops_early_on_trivially_satisfied_criterion(tiny_model) -> None:
    config = _base_config(None, criterion_r=-1e9)  # any R satisfies immediately
    result = train_lora(tiny_model, config)
    assert result.reached_criterion is True
    assert result.steps_run == config.eval_every  # stops at the first eval checkpoint
    assert result.steps_run < config.max_steps


def test_train_lora_exhausts_max_steps_on_impossible_criterion(tiny_model) -> None:
    config = _base_config(None, criterion_r=1e9)  # unreachable
    result = train_lora(tiny_model, config)
    assert result.reached_criterion is False
    assert result.steps_run == config.max_steps
    assert math.isnan(result.final_recovery) is False  # an eval did run at max_steps


def test_train_lora_raises_on_exhausted_wall_clock_budget(tiny_model) -> None:
    config = _base_config(None, max_wall_clock_s=-1.0, criterion_r=1e9)
    with pytest.raises(TrainingBudgetExceeded):
        train_lora(tiny_model, config)


def test_train_lora_snapshots_round_trip(tiny_model, tmp_path) -> None:
    config = _base_config(
        None, criterion_r=1e9, max_steps=5, snapshot_dir=tmp_path, snapshot_every=2
    )
    result = train_lora(tiny_model, config)
    assert len(result.snapshot_paths) >= 2
    for path in result.snapshot_paths:
        assert path.exists()
    loaded_first = load_snapshot(result.snapshot_paths[0])
    assert loaded_first["arm"] == "QK"
    assert loaded_first["layer"] == 1
    assert loaded_first["head"] == 0

    # The training loop appends one final snapshot after the loop exits
    # (see train_lora) -- its factors must match the returned result
    # exactly, since both come from the same `factors` object at exit.
    loaded_last = load_snapshot(result.snapshot_paths[-1])
    assert np.array_equal(loaded_last["B"], result.B)
    assert np.array_equal(loaded_last["A"], result.A)


def test_save_and_load_snapshot_round_trip(tmp_path) -> None:
    factors = init_lora_factors(d_out=6, d_in=4, rank=3, alpha=2.0, seed=0)
    with torch.no_grad():
        factors.B += torch.randn_like(factors.B)
    path = save_snapshot(tmp_path, step=42, factors=factors, arm="QK", layer=2, head=3, alpha=2.0)
    loaded = load_snapshot(path)
    B, A = factors.numpy()
    assert np.array_equal(loaded["B"], B)
    assert np.array_equal(loaded["A"], A)
    assert loaded["step"] == 42
    assert loaded["arm"] == "QK"
    assert loaded["layer"] == 2
    assert loaded["head"] == 3
    assert loaded["alpha"] == 2.0


def test_train_lora_B_A_shapes_match_arm_convention(tiny_model) -> None:
    config = _base_config(None, arm="QK", rank=4, max_steps=2, criterion_r=1e9, eval_every=1)
    result = train_lora(tiny_model, config)
    d_out, d_in = factor_shapes(tiny_model, "QK")
    assert result.B.shape == (d_out, 4)
    assert result.A.shape == (4, d_in)


def test_arms_tuple_contains_qk_and_ov() -> None:
    assert set(ARMS) == {"QK", "OV"}
