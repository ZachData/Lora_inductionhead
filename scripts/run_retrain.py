"""Re-run pythia-70m forward from a published checkpoint, densely probed.

    python scripts/run_retrain.py --from-step 512 --to-step 2000

~9.4 h on an RTX 3080 at the throughput measured in PROJECT.md §11.
Resumable: re-running the same command picks up from the last saved
state, so a crash or a reboot costs at most `--save-every` steps.

What it produces, in `data/retrain/<run-id>/`:
  probe.jsonl   PMS / ICL / NLL halves every `--probe-every` steps
  ckpt/         the rolling pre-onset buffer, then the capture window
  latest.pt     resume state (model + optimizer + step + buffer)
  config.json   the frozen config whose hash is the run id
  final/        weights at `--to-step`, for comparison against published B

Read PROJECT.md §11 before interpreting anything this writes. In
particular this is a **re-run from A, not a replay of Pythia**: Adam is
cold-started at `--from-step` because Pythia publishes weights and not
optimizer moments, so the trajectory diverges from the published one
immediately. `--to-step 2000` is the default because step 2000 is
checkpoint B, which makes that divergence measurable instead of assumed.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from indbw.evalset import build_eval_tokens
from indbw.retrain import (
    PYTHIA_70M_SCHEDULE,
    RetrainConfig,
    RollingCheckpointBuffer,
    SyntheticSource,
    accumulate_gradients,
    lr_at_step,
    probe_induction,
)

MODEL_NAME = "EleutherAI/pythia-70m"


def git_sha() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:  # noqa: BLE001 - provenance is best-effort, not a reason to abort a 9h run
        return "unknown"


def load_checkpoint_hf(step: int, device: str, dtype: torch.dtype) -> tuple[Any, Any]:
    """pythia-70m at `step` as a plain HF model, plus its tokenizer.

    Deliberately not TransformerLens: HookedTransformer exists to expose
    every intermediate through a hook, which is what this repo's probes
    need and what makes it the wrong vehicle for a multi-hour training
    run (PROJECT.md §11). Probing here reads attentions off the HF model
    directly via `probe_induction`.
    """
    from transformers import AutoTokenizer, GPTNeoXForCausalLM

    revision = f"step{step}"
    model = GPTNeoXForCausalLM.from_pretrained(
        MODEL_NAME, revision=revision, dtype=dtype, attn_implementation="eager"
    )
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, revision=revision)
    return model.to(device).train(), tokenizer


def build_source(name: str, tokenizer: Any, cfg: RetrainConfig) -> Any:
    if name == "synthetic":
        print(
            "  WARNING: --data-source synthetic cannot form induction -- there is\n"
            "  nothing in a uniform-random stream to induct on. Smoke path only.",
            flush=True,
        )
        return SyntheticSource(d_vocab=len(tokenizer), seed=cfg.seed)
    if name == "hf-stream":
        from indbw.retrain import HFStreamSource

        return HFStreamSource(tokenizer, seed=cfg.seed)
    raise ValueError(f"unknown data source {name!r}")


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--from-step", type=int, default=512, help="published checkpoint to resume")
    ap.add_argument("--to-step", type=int, default=2000, help="2000 = checkpoint B")
    ap.add_argument("--micro-bs", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=256)
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--probe-every", type=int, default=4)
    ap.add_argument("--onset-pms", type=float, default=0.05)
    ap.add_argument("--buffer-steps", type=int, default=24)
    ap.add_argument("--capture-steps", type=int, default=60)
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="data/retrain")
    ap.add_argument("--data-source", default="hf-stream", choices=["hf-stream", "synthetic"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument("--max-hours", type=float, default=24.0, help="hard wall-clock budget")
    args = ap.parse_args()

    if args.device == "cpu":
        print(
            "ERROR: refusing to run on CPU -- this is ~930 h for the shortest\n"
            "segment (PROJECT.md §11). Run scripts/bench_retrain_throughput.py"
            " --doctor.",
            file=sys.stderr,
        )
        raise SystemExit(2)

    cfg = RetrainConfig(
        from_step=args.from_step,
        to_step=args.to_step,
        micro_bs=args.micro_bs,
        grad_accum=args.grad_accum,
        seq_len=args.seq_len,
        seed=args.seed,
        probe_every=args.probe_every,
        onset_pms=args.onset_pms,
        buffer_steps=args.buffer_steps,
        capture_steps=args.capture_steps,
        save_every=args.save_every,
        out_dir=Path(args.out_dir),
        dtype=args.dtype,
        data_source=args.data_source,
    )
    run_dir = cfg.run_dir()
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(json.dumps(cfg.to_dict(), indent=2, sort_keys=True) + "\n")

    torch.manual_seed(cfg.seed)
    torch.set_float32_matmul_precision("high")
    dtype = getattr(torch, cfg.dtype)

    print(f"run id   {cfg.run_id()}   ->  {run_dir}")
    print(
        f"segment  step {cfg.from_step} -> {cfg.to_step}  "
        f"({cfg.total_optimizer_steps()} optimizer steps, "
        f"{cfg.total_optimizer_steps() * cfg.tokens_per_optimizer_step() / 1e9:.2f}B tokens)"
    )
    print(
        f"batch    micro {cfg.micro_bs} x accum {cfg.grad_accum} = "
        f"{cfg.micro_bs * cfg.grad_accum} seqs x {cfg.seq_len} tok"
    )
    print(
        f"lr       {lr_at_step(cfg.from_step):.3e} at start -> "
        f"{lr_at_step(cfg.to_step):.3e} at end  (peak {PYTHIA_70M_SCHEDULE.peak_lr:.0e}, "
        f"warmup ends step {PYTHIA_70M_SCHEDULE.warmup_steps})"
    )

    print(f"\nloading {MODEL_NAME} @ step{cfg.from_step} ...", flush=True)
    model, tokenizer = load_checkpoint_hf(cfg.from_step, args.device, torch.float32)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=lr_at_step(cfg.from_step),
        betas=cfg.betas,
        eps=cfg.eps,
        weight_decay=cfg.weight_decay,
    )
    buffer = RollingCheckpointBuffer(run_dir / "ckpt", cfg.buffer_steps, cfg.capture_steps)
    source = build_source(cfg.data_source, tokenizer, cfg)
    eval_tokens = build_eval_tokens(cfg.probe_n_eval, cfg.probe_T, 0, int(model.config.vocab_size))

    step = cfg.from_step
    latest = run_dir / "latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=args.device, weights_only=False)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        step = int(state["step"])
        buffer.triggered_at = state.get("triggered_at")
        print(f"resumed from {latest} at step {step}")

    probe_path = run_dir / "probe.jsonl"
    t0 = time.time()
    print(f"\nstarting at step {step}\n", flush=True)

    while step < cfg.to_step:
        elapsed_h = (time.time() - t0) / 3600
        if elapsed_h > args.max_hours:
            print(f"hit --max-hours ({args.max_hours}) at step {step}; state saved, resumable")
            break

        lr = lr_at_step(step)
        for group in optimizer.param_groups:
            group["lr"] = lr

        micro = [
            source.next_batch(cfg.micro_bs, cfg.seq_len).to(args.device)
            for _ in range(cfg.grad_accum)
        ]

        def loss_fn(m: Any, chunk: torch.Tensor) -> torch.Tensor:
            with torch.autocast(device_type="cuda", dtype=dtype):
                return m(chunk, labels=chunk).loss

        loss = accumulate_gradients(model, micro, loss_fn)
        if not torch.isfinite(torch.tensor(loss)):
            raise FloatingPointError(f"non-finite loss ({loss}) at step {step}")
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optimizer.step()
        step += 1

        if step % cfg.probe_every == 0 or step == cfg.to_step:
            obs = probe_induction(
                model,
                eval_tokens,
                cfg.probe_T,
                cfg.probe_layer,
                cfg.probe_head,
                batch_size=8,
                device=args.device,
            )
            rec = {"step": step, "lr": lr, "train_loss": loss, "elapsed_s": time.time() - t0, **obs}
            with probe_path.open("a") as f:
                f.write(json.dumps(rec) + "\n")
                f.flush()
            print(
                f"  step {step:>5}  loss {loss:7.4f}  lr {lr:.2e}  "
                f"PMS {obs['pms']:.4f}  ICL {obs['icl']:7.3f}  "
                f"[{(time.time() - t0) / 3600:.2f}h]",
                flush=True,
            )

            buffer.offer(step, model.state_dict())
            if obs["pms"] >= cfg.onset_pms and buffer.triggered_at is None:
                buffer.trigger(step)
                print(
                    f"  >>> ONSET: PMS {obs['pms']:.4f} >= {cfg.onset_pms} at step {step}. "
                    f"Pre-onset buffer retained: {buffer.retained_steps()}",
                    flush=True,
                )

        if step % cfg.save_every == 0:
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "step": step,
                    "triggered_at": buffer.triggered_at,
                },
                latest,
            )

    final_dir = run_dir / "final"
    final_dir.mkdir(exist_ok=True)
    model.save_pretrained(final_dir)
    (run_dir / "provenance.json").write_text(
        json.dumps(
            {
                "git_sha": git_sha(),
                "run_id": cfg.run_id(),
                "final_step": step,
                "wall_clock_s": time.time() - t0,
                "torch": torch.__version__,
                "hardware": torch.cuda.get_device_name(0)
                if args.device == "cuda"
                else platform.machine(),
                "onset_step": buffer.triggered_at,
                "retained_checkpoints": buffer.retained_steps(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    print(f"\ndone at step {step} in {(time.time() - t0) / 3600:.2f} h -> {run_dir}")
    if buffer.triggered_at is None:
        print(
            "  NOTE: onset never triggered. Either PMS stayed below --onset-pms,\n"
            "  or the segment did not reach the transition. Check probe.jsonl."
        )


if __name__ == "__main__":
    main()
