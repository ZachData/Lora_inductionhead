"""Tier-2 smoke test (CLAUDE.md TDD contract, kind 5).

"One test runs the entire path -- load toy model -> inject LoRA -> 5
training steps -> compute every probe -> emit a results record ->
validate schema -- on a randomly-initialized 2-layer model in under 60
seconds. It asserts nothing about science. It exists so that integration
breakage surfaces in tier 2 rather than three hours into a real run."

Read that literally: **nothing here is evidence about induction.** The
model is random, the A/B ICL baselines are invented, and the recovery
number is meaningless. The only claims are structural -- every stage
runs, hands the next stage the shapes it expects, and the record that
comes out the far end validates and recomputes its own verdict.

Which is exactly the failure this catches. `train.py`'s hooks depend on
TransformerLens's within-block hook ordering (REVIEW.md, 2026-08-14),
`probes.py` depends on TL's weight-layout conventions, and neither is
exercised by any tier-1 test that does not also hand-build its inputs.
A TL version bump that broke either would otherwise first show up on a
real checkpoint, hours into a sweep.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import torch
from transformer_lens import HookedTransformer

from indbw.algebra import phi, principal_angles, truncate_svd
from indbw.evalset import build_eval_tokens
from indbw.lora import bandwidth, build_delta
from indbw.nulls import matched_norm_random_update, phi_null_band
from indbw.probes import (
    clamp_recovery,
    copying_score,
    icl_score,
    prefix_matching_score,
    prev_token_score,
    recovery,
)
from indbw.schema import METRIC_VERSION, Criterion, ResultsRecord, append_record, load_records
from indbw.train import (
    TrainConfig,
    build_hooks,
    compute_recovery,
    first_copy_nll,
    second_copy_nll,
    train_lora,
)

T = 16  # 2T = 32 <= tiny_model's n_ctx of 64
LAYER, HEAD = 1, 0  # a layer-1 head can K-compose with layer 0, as a real induction head does
SMOKE_BUDGET_S = 60.0


def test_full_path_end_to_end(tiny_model: HookedTransformer, tmp_path: Path) -> None:
    t0 = time.time()
    d_vocab = int(tiny_model.cfg.d_vocab)

    # --- 1. eval set + base-model ICL, the A/B baselines recovery needs -----
    eval_tokens = build_eval_tokens(8, T, seed=0, d_vocab=d_vocab)
    assert eval_tokens.shape == (8, 2 * T)

    with torch.no_grad():
        logits = tiny_model(eval_tokens, return_type="logits")
    nll_first = first_copy_nll(logits, eval_tokens, T).numpy()
    nll_second = second_copy_nll(logits, eval_tokens, T).numpy()
    icl_a = icl_score(nll_first, nll_second)
    # Invented, not measured: a random model has no B checkpoint. Offset
    # so that ICL(B) != ICL(A) and `recovery` is defined at all.
    icl_b = icl_a + 1.0

    # --- 2. inject LoRA + 5 training steps ---------------------------------
    config = TrainConfig(
        arm="QK",
        layer=LAYER,
        head=HEAD,
        rank=2,
        alpha=2.0,
        lr=1e-2,
        max_steps=5,
        batch_size=2,
        T=T,
        d_vocab=d_vocab,
        train_seed=0,
        icl_a=icl_a,
        icl_b=icl_b,
        criterion_r=1e9,  # unreachable: never stop early, always run all 5 steps
        eval_every=5,
        eval_n=4,
        eval_seed=0,
        max_wall_clock_s=SMOKE_BUDGET_S,
        snapshot_dir=tmp_path / "snapshots",
        snapshot_every=2,
    )
    result = train_lora(tiny_model, config)

    assert result.steps_run == 5
    assert len(result.loss_history) == 5
    assert all(np.isfinite(result.loss_history))
    assert result.B.shape == (tiny_model.cfg.d_model, config.rank)
    assert result.A.shape == (config.rank, tiny_model.cfg.d_head)
    assert result.snapshot_paths, "snapshotting produced no adapter files"
    # The update actually moved: B is zero-initialized, so a nonzero B is
    # the cheapest possible check that gradients reached the factors at
    # all rather than the loop silently training nothing.
    assert np.any(result.B != 0.0)

    # --- 3. every probe, on the adapted model ------------------------------
    factors_hooks = build_hooks(
        config.arm, config.layer, config.head, _factors_from(result, config.alpha)
    )
    with torch.no_grad(), tiny_model.hooks(fwd_hooks=factors_hooks):
        _, cache = tiny_model.run_with_cache(eval_tokens[:1])
    attn = cache[f"blocks.{LAYER}.attn.hook_pattern"][0, HEAD].numpy().astype(np.float64)

    pms = prefix_matching_score(attn, T)
    ptok = prev_token_score(attn)

    W_O = tiny_model.W_O[LAYER, HEAD].detach().numpy().astype(np.float64)
    W_V = tiny_model.W_V[LAYER, HEAD].detach().numpy().astype(np.float64)
    W_Q = tiny_model.W_Q[LAYER, HEAD].detach().numpy().astype(np.float64)
    W_K = tiny_model.W_K[LAYER, HEAD].detach().numpy().astype(np.float64)
    M_OV = W_O.T @ W_V.T  # PROJECT.md §3
    copying = copying_score(
        tiny_model.W_U.detach().numpy().astype(np.float64).T,
        M_OV,
        tiny_model.W_E.detach().numpy().astype(np.float64),
    )

    r_unclamped = compute_recovery(
        tiny_model, factors_hooks, eval_tokens, T, icl_a, icl_b, batch_size=4
    )
    r = clamp_recovery(r_unclamped)

    for name, value in [("pms", pms), ("prev_token", ptok), ("copying", copying), ("R", r)]:
        assert 0.0 <= value <= 1.0, f"{name} out of range: {value}"
    assert np.isfinite(r_unclamped)
    assert recovery(icl_a, icl_a, icl_b) == 0.0  # R(A) = 0 by construction

    # --- 4. the structural readouts (algebra + nulls) ----------------------
    delta_wq = build_delta("unconstrained", (result.B, result.A), config.alpha, config.rank)
    delta_m_qk = delta_wq @ W_K.T  # composition rule, PROJECT.md §3
    phi_qk = phi(delta_m_qk)
    assert 0.0 <= phi_qk <= 1.0

    band = phi_null_band(
        delta_wq.shape,
        config.rank,
        float(np.linalg.norm(delta_wq)),
        compose=lambda dW: dW @ W_K.T,
        n_draws=8,
    )
    assert band.n_draws == 8
    null_delta = matched_norm_random_update(
        delta_wq.shape, config.rank, float(np.linalg.norm(delta_wq)), np.random.default_rng(0)
    )
    assert float(np.linalg.norm(null_delta)) == np_approx(float(np.linalg.norm(delta_wq)))

    truncated = truncate_svd(delta_m_qk, 1)  # M5's probe, mechanically
    assert np.linalg.matrix_rank(truncated) <= 1
    angles = principal_angles(delta_wq, W_Q)
    assert np.all((angles >= -1e-12) & (angles <= np.pi / 2 + 1e-12))

    # --- 5. emit a results record and validate it --------------------------
    observed = {
        "final_recovery": r,
        "phi_qk": phi_qk,
        "pms": pms,
        "prev_token_score": ptok,
        "copying_score": copying,
        "phi_null_p95": band.percentile_value,
        "beta": float(bandwidth(config.rank, tiny_model.cfg.d_head, tiny_model.cfg.d_model)),
    }
    criteria = (Criterion(metric="final_recovery", op=">=", threshold=0.80),)
    record = ResultsRecord(
        row="SMOKE",
        null_tested="none -- smoke test, asserts nothing about science",
        criteria=criteria,
        observed=observed,
        verdict="pass" if all(c.holds(observed) for c in criteria) else "fail",
        metric_version=METRIC_VERSION,
        git_sha="0" * 40,
        run_config_hash="smoke",
        seed=config.train_seed,
        checkpoint_revision="randomly-initialized tiny_model (no checkpoint)",
        eval_set_hash="smoke",
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        transformer_lens_version="smoke",
        wall_clock_s=result.wall_clock_s,
        hardware="smoke",
    )
    assert record.is_self_consistent()

    # Round-trip through the real writer/loader, not just the dataclass:
    # tier 2.5 reads records off disk, so a serialization break has to
    # fail here too. Written under tmp_path, never into results/.
    out = tmp_path / "smoke.jsonl"
    append_record(out, record)
    (loaded,) = load_records(out)
    assert loaded.observed == record.observed
    assert loaded.recomputed_verdict() == record.verdict

    elapsed = time.time() - t0
    assert elapsed < SMOKE_BUDGET_S, f"smoke test took {elapsed:.1f}s, budget {SMOKE_BUDGET_S}s"


def _factors_from(result: object, alpha: float) -> object:
    from indbw.train import LoRAFactors

    return LoRAFactors(
        B=torch.tensor(result.B, dtype=torch.float32),  # type: ignore[attr-defined]
        A=torch.tensor(result.A, dtype=torch.float32),  # type: ignore[attr-defined]
        alpha=alpha,
    )


def np_approx(value: float) -> float:
    """pytest.approx with this repo's exact-identity tolerance."""
    import pytest

    return pytest.approx(value, rel=1e-12)  # type: ignore[return-value]
