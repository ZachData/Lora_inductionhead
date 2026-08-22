"""The whole retrain path on a toy model, in seconds.

CLAUDE.md's smoke-test spec (kind 5), applied to the retrain harness:
load -> schedule -> accumulate -> step -> probe -> buffer -> resume, on a
randomly-initialized 2-layer model, asserting nothing about science. It
exists so that integration breakage surfaces here rather than nine hours
into a real run -- which is the specific cost this harness carries and
the reason it gets its own smoke test rather than relying on the unit
tests of its parts.

Every piece below is unit-tested in isolation. What only this catches is
the wiring: a probe that cannot read the model the loop trains, a buffer
handed a state_dict shape it cannot serialize, an optimizer whose
param_groups the schedule never actually reaches.
"""

from __future__ import annotations

import torch

from indbw.evalset import build_eval_tokens
from indbw.retrain import (
    RetrainConfig,
    RollingCheckpointBuffer,
    SyntheticSource,
    accumulate_gradients,
    lr_at_step,
    probe_induction,
)


def _toy():
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    torch.manual_seed(0)
    return GPTNeoXForCausalLM(
        GPTNeoXConfig(
            hidden_size=32,
            num_hidden_layers=2,
            num_attention_heads=2,
            intermediate_size=64,
            vocab_size=50,
            max_position_embeddings=64,
            attn_implementation="eager",
        )
    ).train()


def test_retrain_path_runs_end_to_end(tmp_path) -> None:
    cfg = RetrainConfig(
        from_step=512,
        to_step=515,
        micro_bs=2,
        grad_accum=3,
        seq_len=32,
        probe_every=1,
        probe_n_eval=4,
        probe_T=8,
        probe_layer=1,
        probe_head=0,
        buffer_steps=2,
        capture_steps=2,
        out_dir=tmp_path,
    )
    model = _toy()
    opt = torch.optim.AdamW(model.parameters(), lr=lr_at_step(cfg.from_step), betas=cfg.betas)
    buf = RollingCheckpointBuffer(cfg.run_dir() / "ckpt", cfg.buffer_steps, cfg.capture_steps)
    src = SyntheticSource(50, cfg.seed)
    ev = build_eval_tokens(cfg.probe_n_eval, cfg.probe_T, 0, 50)

    def loss_fn(m, chunk):
        return m(chunk, labels=chunk).loss

    step, lrs, losses = cfg.from_step, [], []
    while step < cfg.to_step:
        lr = lr_at_step(step)
        for g in opt.param_groups:
            g["lr"] = lr
        micro = [src.next_batch(cfg.micro_bs, cfg.seq_len) for _ in range(cfg.grad_accum)]
        losses.append(accumulate_gradients(model, micro, loss_fn))
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()
        step += 1
        lrs.append(lr)
        obs = probe_induction(model, ev, cfg.probe_T, cfg.probe_layer, cfg.probe_head, 2, "cpu")
        assert 0.0 <= obs["pms"] <= 1.0
        buf.offer(step, model.state_dict())

    assert all(torch.isfinite(torch.tensor(x)) for x in losses)
    # the schedule actually reached the optimizer, rather than the loop
    # setting a local it never applied
    assert opt.param_groups[0]["lr"] == lrs[-1]
    assert lrs[0] < lrs[-1], "step 512 is inside warmup; lr must be rising"
    assert buf.retained_steps() == [513 + 1, 514 + 1][-cfg.buffer_steps :]

    # resume round-trip
    state = {"model": model.state_dict(), "optimizer": opt.state_dict(), "step": step}
    path = cfg.run_dir() / "latest.pt"
    torch.save(state, path)
    fresh = _toy()
    fresh_opt = torch.optim.AdamW(fresh.parameters(), lr=1e-9)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    fresh.load_state_dict(loaded["model"])
    fresh_opt.load_state_dict(loaded["optimizer"])
    assert loaded["step"] == step
    for a, b in zip(model.parameters(), fresh.parameters(), strict=True):
        assert torch.equal(a, b), "resumed weights differ from saved"
