"""Oracles and silent-failure guards for the retrain harness.

Everything here guards a failure that a 9.4-hour run would not report.
The run produces a loss curve and a PMS trajectory either way; none of
these bugs makes it crash, and several of them make it produce a
*plausible* trajectory that answers the wrong question.

  - The LR schedule is the single most likely way the transition fails to
    reproduce. Step 512 sits inside Pythia's warmup, so a schedule that
    starts at peak (the default for almost every training script) runs
    ~3x the intended learning rate through exactly the window under
    study. Closed-form, so tested exactly.
  - Gradient accumulation without the 1/N loss scaling multiplies the
    effective LR by the accumulation factor -- 256 here. The run
    completes; it is just not the run that was configured. Tested by
    equivalence against a single large batch, which is a real oracle.
  - The rolling checkpoint buffer exists to capture weights from *before*
    onset, which cannot be known to save until onset has already
    happened. If the retention logic is wrong the run finishes normally
    and the pre-onset weights are simply gone -- with 9.4 hours spent.
"""

from __future__ import annotations

import math

import pytest
import torch

from indbw.retrain import (
    PYTHIA_70M_SCHEDULE,
    RetrainConfig,
    RollingCheckpointBuffer,
    lr_at_step,
)


# --------------------------------------------------------------------------
# LR schedule: closed form, exact
# --------------------------------------------------------------------------


def test_warmup_is_linear_from_zero_to_peak() -> None:
    s = PYTHIA_70M_SCHEDULE
    assert lr_at_step(0, s) == pytest.approx(0.0, abs=1e-15)
    assert lr_at_step(s.warmup_steps, s) == pytest.approx(s.peak_lr, rel=1e-12)
    half = lr_at_step(s.warmup_steps // 2, s)
    assert half == pytest.approx(s.peak_lr * (s.warmup_steps // 2) / s.warmup_steps, rel=1e-12)


def test_checkpoint_a_sits_inside_warmup_at_about_a_third_of_peak() -> None:
    """The number that makes this schedule load-bearing. A script that
    starts at peak LR would run 2.8x too hot through the transition."""
    s = PYTHIA_70M_SCHEDULE
    assert 512 < s.warmup_steps, "step 512 must be inside warmup for this to matter"
    lr_a = lr_at_step(512, s)
    assert lr_a == pytest.approx(s.peak_lr * 512 / s.warmup_steps, rel=1e-12)
    assert lr_a == pytest.approx(3.58e-4, rel=1e-2)
    assert s.peak_lr / lr_a == pytest.approx(2.79, rel=1e-2)


def test_lr_rises_monotonically_across_the_whole_transition_window() -> None:
    """512 -> 1000 is entirely within warmup, so the LR is still climbing
    while induction forms. A constant-LR run is a different experiment."""
    s = PYTHIA_70M_SCHEDULE
    values = [lr_at_step(t, s) for t in range(512, 1001, 8)]
    assert all(b > a for a, b in zip(values[:-1], values[1:], strict=True))
    assert values[-1] == pytest.approx(s.peak_lr * 1000 / s.warmup_steps, rel=1e-12)


def test_cosine_decay_matches_its_closed_form_after_warmup() -> None:
    s = PYTHIA_70M_SCHEDULE
    min_lr = s.peak_lr * s.min_lr_ratio
    for step in (s.warmup_steps + 1, 50_000, 100_000):
        progress = (step - s.warmup_steps) / (s.total_steps - s.warmup_steps)
        expected = min_lr + (s.peak_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        assert lr_at_step(step, s) == pytest.approx(expected, rel=1e-12)


def test_lr_floors_at_min_ratio_after_warmup_but_not_during_it() -> None:
    """min_lr is the floor of the *decay*, not a global floor. Warmup
    ramps from zero and is legitimately below it -- asserting otherwise
    would force a wrong schedule that starts at 1e-4 instead of 0."""
    s = PYTHIA_70M_SCHEDULE
    min_lr = s.peak_lr * s.min_lr_ratio
    assert lr_at_step(s.total_steps, s) == pytest.approx(min_lr, rel=1e-12)
    assert lr_at_step(s.total_steps * 2, s) == pytest.approx(min_lr, rel=1e-12)
    post_warmup = range(s.warmup_steps, s.total_steps, 977)
    assert all(lr_at_step(t, s) >= min_lr - 1e-15 for t in post_warmup)
    # and the warmup side really is below it, which is what makes the
    # distinction above a real one rather than a weakened assertion
    assert lr_at_step(0, s) < min_lr
    assert lr_at_step(100, s) < min_lr


def test_schedule_constants_match_pythia_70m() -> None:
    """Pinned because they are the contract with the published run, not a
    tuning choice: peak 1e-3, warmup 1% of 143000 iters, cosine to 0.1x."""
    s = PYTHIA_70M_SCHEDULE
    assert s.peak_lr == 1e-3
    assert s.total_steps == 143_000
    assert s.warmup_steps == 1_430
    assert s.min_lr_ratio == 0.1


# --------------------------------------------------------------------------
# Rolling checkpoint buffer: the pre-onset capture
# --------------------------------------------------------------------------


def test_buffer_retains_only_the_last_n_before_any_trigger(tmp_path) -> None:
    buf = RollingCheckpointBuffer(tmp_path, keep=3, capture_steps=5)
    for step in range(10):
        buf.offer(step, {"w": torch.tensor([float(step)])})
    assert buf.retained_steps() == [7, 8, 9]


def test_trigger_keeps_the_pre_onset_window_that_was_already_buffered(tmp_path) -> None:
    """The whole reason the buffer exists. Onset is only detectable after
    it has happened, so the steps *before* it must already be on disk --
    a buffer that cleared on trigger would lose exactly the side of the
    transition the experiment needs."""
    buf = RollingCheckpointBuffer(tmp_path, keep=3, capture_steps=4)
    for step in range(10):
        buf.offer(step, {"w": torch.tensor([float(step)])})
    pre_onset = buf.retained_steps()
    buf.trigger(at_step=9)
    for step in range(10, 16):
        buf.offer(step, {"w": torch.tensor([float(step)])})
    retained = buf.retained_steps()
    assert all(s in retained for s in pre_onset), "pre-onset checkpoints were evicted"


def test_after_trigger_every_step_is_kept_for_capture_steps_then_it_stops(tmp_path) -> None:
    buf = RollingCheckpointBuffer(tmp_path, keep=2, capture_steps=3)
    for step in range(5):
        buf.offer(step, {"w": torch.tensor([float(step)])})
    buf.trigger(at_step=4)
    for step in range(5, 12):
        buf.offer(step, {"w": torch.tensor([float(step)])})
    retained = buf.retained_steps()
    assert {5, 6, 7}.issubset(retained), "capture window not fully retained"
    assert buf.capture_exhausted


def test_trigger_is_idempotent_so_a_noisy_probe_cannot_restart_capture(tmp_path) -> None:
    """PMS is estimated on a finite eval set and will wobble around any
    threshold. Re-triggering would extend the capture window indefinitely
    and fill the disk."""
    buf = RollingCheckpointBuffer(tmp_path, keep=2, capture_steps=2)
    buf.offer(0, {"w": torch.tensor([0.0])})
    buf.trigger(at_step=0)
    first = buf.triggered_at
    buf.trigger(at_step=5)
    assert buf.triggered_at == first


def test_buffer_round_trips_tensor_values(tmp_path) -> None:
    """Guard against writing empty or aliased state: a buffer that saved
    references rather than copies would silently store the *final*
    weights under every step's filename."""
    buf = RollingCheckpointBuffer(tmp_path, keep=3, capture_steps=1)
    live = torch.tensor([0.0])
    for step in range(3):
        live.fill_(float(step))
        buf.offer(step, {"w": live})
    for step in buf.retained_steps():
        loaded = buf.load(step)
        assert loaded["w"].item() == float(step), "buffer stored an alias, not a copy"


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------


def test_run_id_is_stable_and_sensitive(tmp_path) -> None:
    """Config-as-data: the hash is the run ID (CLAUDE.md). Stable across
    construction, and changed by any field that changes the run."""
    a = RetrainConfig(from_step=512, to_step=2000, out_dir=tmp_path)
    b = RetrainConfig(from_step=512, to_step=2000, out_dir=tmp_path)
    c = RetrainConfig(from_step=512, to_step=1000, out_dir=tmp_path)
    assert a.run_id() == b.run_id()
    assert a.run_id() != c.run_id()


def test_effective_batch_matches_pythia_by_default() -> None:
    cfg = RetrainConfig(from_step=512, to_step=2000)
    assert cfg.micro_bs * cfg.grad_accum == 1024
    assert cfg.tokens_per_optimizer_step() == 1024 * 2048


def test_config_rejects_a_backwards_or_empty_span() -> None:
    with pytest.raises(ValueError, match="to_step"):
        RetrainConfig(from_step=2000, to_step=512)
    with pytest.raises(ValueError, match="to_step"):
        RetrainConfig(from_step=512, to_step=512)


def test_config_rejects_an_off_grid_start(tmp_path) -> None:
    """from_step must be a real published checkpoint -- there is nothing
    to resume from otherwise, and the failure would otherwise appear as a
    download error hours into setup."""
    with pytest.raises(ValueError, match="checkpoint grid"):
        RetrainConfig(from_step=513, to_step=2000)


# --------------------------------------------------------------------------
# Gradient accumulation: equivalence against a single large batch
# --------------------------------------------------------------------------


def _tiny_lm() -> torch.nn.Module:
    """A minimal token model with a real cross-entropy head. Small enough
    to compare gradients exactly, and it exercises the same
    loss-shape/reduction path the real loop uses."""
    torch.manual_seed(0)
    return torch.nn.Sequential(torch.nn.Embedding(16, 8), torch.nn.Linear(8, 16))


def _loss(model: torch.nn.Module, batch: torch.Tensor) -> torch.Tensor:
    logits = model(batch[:, :-1])
    return torch.nn.functional.cross_entropy(
        logits.reshape(-1, logits.shape[-1]), batch[:, 1:].reshape(-1)
    )


def test_accumulated_gradient_equals_the_single_large_batch_gradient() -> None:
    """The oracle for accumulation. Omitting the 1/N loss scaling is the
    classic bug: the run completes, the loss curve looks ordinary, and
    the effective learning rate is N times what was configured -- 256x in
    the real config. Nothing downstream reports it.
    """
    from indbw.retrain import accumulate_gradients

    torch.manual_seed(1)
    full = torch.randint(0, 16, (8, 5))

    big = _tiny_lm()
    _loss(big, full).backward()
    expected = [p.grad.clone() for p in big.parameters()]

    small = _tiny_lm()
    micro = [full[i : i + 2] for i in range(0, 8, 2)]
    accumulate_gradients(small, micro, _loss)
    got = [p.grad.clone() for p in small.parameters()]

    assert len(expected) == len(got)
    for e, g in zip(expected, got, strict=True):
        assert torch.allclose(e, g, rtol=1e-5, atol=1e-7), "accumulated gradient != large-batch"


def test_missing_loss_scaling_would_be_caught_by_that_oracle() -> None:
    """Discrimination guard on the guard: confirm the test above can
    actually fail. Summing micro-batch losses without dividing by N
    inflates the gradient by exactly N, so if this assertion ever starts
    failing, the oracle has gone blind."""
    from indbw.retrain import accumulate_gradients

    torch.manual_seed(1)
    full = torch.randint(0, 16, (8, 5))
    micro = [full[i : i + 2] for i in range(0, 8, 2)]

    correct = _tiny_lm()
    accumulate_gradients(correct, micro, _loss)

    unscaled = _tiny_lm()
    for chunk in micro:
        _loss(unscaled, chunk).backward()

    ratios = [
        (u.grad / c.grad)[c.grad.abs() > 1e-6]
        for c, u in zip(correct.parameters(), unscaled.parameters(), strict=True)
    ]
    inflation = torch.cat([r.flatten() for r in ratios]).median().item()
    assert inflation == pytest.approx(len(micro), rel=1e-3)


def test_accumulate_returns_the_mean_loss_over_micro_batches() -> None:
    from indbw.retrain import accumulate_gradients

    torch.manual_seed(1)
    full = torch.randint(0, 16, (8, 5))
    model = _tiny_lm()
    micro = [full[i : i + 2] for i in range(0, 8, 2)]
    reported = accumulate_gradients(model, micro, _loss)

    ref = _tiny_lm()
    with torch.no_grad():
        expected = sum(_loss(ref, c).item() for c in micro) / len(micro)
    assert reported == pytest.approx(expected, rel=1e-6)


def test_accumulate_rejects_an_empty_micro_batch_list() -> None:
    from indbw.retrain import accumulate_gradients

    with pytest.raises(ValueError, match="at least one"):
        accumulate_gradients(_tiny_lm(), [], _loss)


# --------------------------------------------------------------------------
# Data sources
# --------------------------------------------------------------------------


def test_synthetic_source_shape_and_determinism() -> None:
    from indbw.retrain import SyntheticSource

    a = SyntheticSource(d_vocab=50, seed=0).next_batch(3, 7)
    b = SyntheticSource(d_vocab=50, seed=0).next_batch(3, 7)
    c = SyntheticSource(d_vocab=50, seed=1).next_batch(3, 7)
    assert a.shape == (3, 7) and a.dtype == torch.long
    assert torch.equal(a, b), "same seed must give the same stream"
    assert not torch.equal(a, c), "different seeds must differ"
    assert int(a.max()) < 50 and int(a.min()) >= 0


def test_synthetic_source_advances_rather_than_repeating() -> None:
    """A source that returned the same batch forever would train happily
    and overfit one batch, with a loss curve that looks like fast
    learning."""
    from indbw.retrain import SyntheticSource

    src = SyntheticSource(d_vocab=50, seed=0)
    assert not torch.equal(src.next_batch(2, 5), src.next_batch(2, 5))


def test_hf_stream_source_packs_documents_contiguously() -> None:
    """Packing logic only -- the dataset itself is faked, since the
    session that wrote this had no HuggingFace access. What is pinned is
    that documents are concatenated with an EOS between them and sliced
    into exact [micro_bs, seq_len] blocks with no gaps and no reuse."""
    from indbw.retrain import HFStreamSource

    class FakeTokenizer:
        eos_token_id = 99

        def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
            return {"input_ids": [int(c) for c in text.split(",")]}

    src = HFStreamSource.__new__(HFStreamSource)  # bypass load_dataset
    src.tokenizer = FakeTokenizer()
    src.text_field = "text"
    src._buf = []
    src._it = iter([{"text": "1,2,3"}, {"text": "4,5,6"}, {"text": "7,8,9"}])

    first = src.next_batch(2, 3)
    assert first.tolist() == [[1, 2, 3], [99, 4, 5]], "documents not packed with EOS separator"
    second = src.next_batch(1, 3)
    assert second.tolist() == [[6, 99, 7]], "leftover buffer not carried across batches"


# --------------------------------------------------------------------------
# Probe adapter
# --------------------------------------------------------------------------


def _neox(n_layers: int = 2, n_heads: int = 2, d_vocab: int = 50):
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    cfg = GPTNeoXConfig(
        hidden_size=32,
        num_hidden_layers=n_layers,
        num_attention_heads=n_heads,
        intermediate_size=64,
        vocab_size=d_vocab,
        max_position_embeddings=64,
        attn_implementation="eager",
    )
    return GPTNeoXForCausalLM(cfg).eval()


def test_probe_returns_every_component_in_range() -> None:
    from indbw.evalset import build_eval_tokens
    from indbw.retrain import probe_induction

    model = _neox()
    tokens = build_eval_tokens(4, 8, seed=0, d_vocab=50)
    out = probe_induction(model, tokens, period=8, layer=1, head=0, batch_size=2, device="cpu")
    assert set(out) == {"pms", "icl", "nll_first", "nll_second"}
    assert 0.0 <= out["pms"] <= 1.0
    assert out["nll_first"] > 0 and out["nll_second"] > 0
    assert out["icl"] == pytest.approx(out["nll_first"] - out["nll_second"], rel=1e-9)


def test_probe_reads_the_head_and_layer_it_is_asked_for() -> None:
    """Discrimination guard: a hardcoded or off-by-one index would sail
    through every range check above and silently probe the wrong head for
    the entire run."""
    from indbw.evalset import build_eval_tokens
    from indbw.retrain import probe_induction

    model = _neox()
    tokens = build_eval_tokens(4, 8, seed=0, d_vocab=50)
    pms = {
        (layer, head): probe_induction(
            model, tokens, period=8, layer=layer, head=head, batch_size=2, device="cpu"
        )["pms"]
        for layer in (0, 1)
        for head in (0, 1)
    }
    assert len(set(pms.values())) == len(pms), f"indices are inert: {pms}"


def test_probe_restores_training_mode() -> None:
    """The probe runs mid-training. Leaving the model in eval() would
    disable dropout for the rest of the run without any error."""
    from indbw.evalset import build_eval_tokens
    from indbw.retrain import probe_induction

    model = _neox()
    model.train()
    probe_induction(model, build_eval_tokens(2, 8, seed=0, d_vocab=50), 8, 1, 0, 2, device="cpu")
    assert model.training, "probe left the model in eval mode"


def test_probe_raises_rather_than_returning_zero_when_attentions_are_missing() -> None:
    """A degenerate reading must not be reportable as a number: PMS 0.0
    from a broken probe and PMS 0.0 from a pre-induction model are the
    same value, and the whole experiment is that curve over time."""
    from indbw.retrain import DegenerateProbeError, probe_induction
    from indbw.evalset import build_eval_tokens

    class NoAttentions:
        training = False

        def eval(self) -> None: ...
        def train(self, mode: bool = True) -> None: ...
        def __call__(self, *a, **k):
            class Out:
                attentions = None
                logits = torch.zeros(2, 16, 50)

            return Out()

    with pytest.raises(DegenerateProbeError, match="attention"):
        probe_induction(
            NoAttentions(), build_eval_tokens(2, 8, seed=0, d_vocab=50), 8, 1, 0, 2, device="cpu"
        )
