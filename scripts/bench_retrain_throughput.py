"""Measure real training throughput before committing to the retrain.

The retrain from checkpoint A is sized by one number nobody has measured:
achieved MFU on the target GPU. Across a plausible 15-35% range the
512->1100 segment is anywhere from 5.6 to 13.2 hours on a 10GB RTX 3080 --
the difference between "start it tonight" and "it will not finish". This
script replaces the guess with a measurement in a couple of minutes.

It builds a pythia-70m-shaped GPTNeoX **from config only** -- no
checkpoint download, no network -- because the throughput of a shape does
not depend on the values in it. Then it runs real forward+backward+step
iterations and reports tokens/s, achieved FLOP/s against
`analyze_training_budget.per_token_flops`, MFU against the card's bf16
peak, peak VRAM, and the extrapolated wall-clock for the segment.

Deliberately *not* TransformerLens. HookedTransformer exists to expose
every intermediate through a hook, which is what makes the probes in this
repo possible and also what makes it the wrong vehicle for a
multi-hour training run. Train with plain HF here; load the resulting
weights into HookedTransformer for probing afterwards. Measure both with
`--compare-tl` if you want the gap rather than taking this on faith.

Numbers to know for the 3080 specifically: `--peak-tflops` defaults to
59.5, which is GA102 *consumer* bf16/fp16 tensor throughput with FP32
accumulate. The widely-quoted 119 TFLOPS figure is the FP16-accumulate
rate, which torch AMP does not use; sizing against it halves every MFU
number and makes a feasible run look infeasible.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from analyze_training_budget import PYTHIA_TOKENS_PER_STEP, per_token_flops

# pythia-70m architecture (PROJECT.md §2), written out rather than
# downloaded so this runs with no network and no cached checkpoint.
PYTHIA_70M = {
    "hidden_size": 512,
    "num_hidden_layers": 6,
    "num_attention_heads": 8,
    "intermediate_size": 2048,
    "vocab_size": 50304,
    "max_position_embeddings": 2048,
    "rotary_pct": 0.25,
    "use_parallel_residual": True,
}
# RTX 3080 (GA102 consumer): bf16 tensor with FP32 accumulate.
DEFAULT_PEAK_TFLOPS = 59.5


class NoAcceleratorError(RuntimeError):
    """Raised rather than falling back to CPU. A CPU number here is not a
    conservative estimate of a GPU number, it is a different measurement
    by three orders of magnitude -- and it arrives formatted identically,
    complete with a wall-clock projection and a fits/over verdict. That is
    exactly the silent failure CLAUDE.md's TDD contract is written
    against, so this refuses instead of reporting.
    """


def cuda_diagnostics() -> str:
    """What to check when torch cannot see the GPU. Printed on refusal --
    the failure is nearly always the install, not the machine."""
    built_for = torch.version.cuda or "none (this is a CPU-only build of torch)"
    return "\n".join(
        [
            f"  torch {torch.__version__}, built against CUDA: {built_for}",
            f"  torch.cuda.is_available(): {torch.cuda.is_available()}",
            f"  torch.cuda.device_count():  {torch.cuda.device_count()}",
            "",
            "  Most likely causes, in order:",
            "   1. A CPU-only torch. conda-forge's `pytorch` resolves to the CPU",
            "      build unless the CUDA variant is requested explicitly. If the",
            "      line above says 'CPU-only build', this is it -- reinstall from",
            "      the PyTorch index (see --help epilog).",
            "   2. Driver not visible to the environment (containers, toolbox, and",
            "      atomic/immutable distros). Check `nvidia-smi` outside Python; if",
            "      that works and torch still says False, it is the container.",
            "   3. A driver too old for the CUDA the wheel was built against.",
            "",
            "  Re-run with --allow-cpu only to exercise the code path. The numbers",
            "  it prints are not usable for sizing a run.",
        ]
    )


def resolve_device(requested: str, allow_cpu: bool) -> str:
    if requested == "cuda" and not torch.cuda.is_available():
        raise NoAcceleratorError(
            "asked for --device cuda, but torch reports no CUDA device.\n\n" + cuda_diagnostics()
        )
    if requested == "cpu" and not allow_cpu:
        raise NoAcceleratorError(
            "refusing to benchmark on CPU: the result would not answer the "
            "question this script exists for.\n\n" + cuda_diagnostics()
        )
    return requested


def build_model(seq_len: int, device: str) -> Any:
    from transformers import GPTNeoXConfig, GPTNeoXForCausalLM

    cfg_kwargs = dict(PYTHIA_70M)
    cfg_kwargs["max_position_embeddings"] = max(seq_len, cfg_kwargs["max_position_embeddings"])
    config = GPTNeoXConfig(**cfg_kwargs)
    model = GPTNeoXForCausalLM(config)
    return model.to(device).train()


def bench(
    micro_bs: int,
    seq_len: int,
    device: str,
    steps: int,
    warmup: int,
    compile_model: bool,
    dtype: torch.dtype,
) -> dict[str, Any]:
    model = build_model(seq_len, device)
    compiled = False
    if compile_model:
        # torch.compile pulls in inductor, which pulls in triton, and a
        # torch/triton version mismatch raises on *import* -- long before
        # any kernel is generated. Eager is a valid measurement (it just
        # understates achievable throughput), so degrade to it and say so
        # rather than losing the run to a toolchain problem.
        try:
            model = torch.compile(model)
            compiled = True
        except Exception as exc:  # noqa: BLE001 - any import/backend failure is the same story here
            print(
                f"  torch.compile unavailable ({type(exc).__name__}: "
                f"{str(exc).splitlines()[0][:120]}) -- measuring eager.\n"
                f"  Eager understates throughput, so treat the result as a floor.",
                flush=True,
            )
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), weight_decay=0.01)

    gen = torch.Generator(device="cpu").manual_seed(0)
    batch = torch.randint(
        0, PYTHIA_70M["vocab_size"], (micro_bs, seq_len), generator=gen, dtype=torch.long
    ).to(device)

    use_amp = device == "cuda" and dtype is not torch.float32
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    def one_step() -> None:
        with torch.autocast(device_type=device, dtype=dtype, enabled=use_amp):
            loss = model(batch, labels=batch).loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

    for _ in range(warmup):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(steps):
        one_step()
    if device == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    tokens = steps * micro_bs * seq_len
    f_tok = per_token_flops(
        d_model=PYTHIA_70M["hidden_size"],
        n_layers=PYTHIA_70M["num_hidden_layers"],
        d_vocab=PYTHIA_70M["vocab_size"],
        seq_len=seq_len,
    )["total"]
    peak_mem = torch.cuda.max_memory_allocated() / 1024**3 if device == "cuda" else float("nan")
    return {
        "micro_bs": micro_bs,
        "seq_len": seq_len,
        "dtype": str(dtype).replace("torch.", ""),
        "compiled": compiled,
        "timed_steps": steps,
        "elapsed_s": elapsed,
        "s_per_step": elapsed / steps,
        "tokens_per_s": tokens / elapsed,
        "achieved_flops": f_tok * tokens / elapsed,
        "peak_vram_gb": peak_mem,
    }


def project(tokens_per_s: float, achieved_flops: float, peak_tflops: float) -> str:
    lines = [
        f"  achieved {achieved_flops / 1e12:.1f} TFLOP/s"
        f"  =  MFU {100 * achieved_flops / (peak_tflops * 1e12):.1f}%"
        f"  vs. {peak_tflops:.1f} TFLOP/s peak",
        "",
        "  projected wall-clock for the retrain segment:",
    ]
    for end in (1000, 1100, 1250, 1500):
        tok = (end - 512) * PYTHIA_TOKENS_PER_STEP
        hours = tok / tokens_per_s / 3600
        verdict = "fits 15h" if hours <= 15 else "OVER 15h"
        lines.append(
            f"    512 -> {end:<5}  {tok / 1e9:5.2f}B tokens   {hours:6.2f} h   [{verdict}]"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "If torch cannot see the GPU, the usual fix is a CUDA build of torch in a\n"
            "clean venv (conda-forge's `pytorch` is CPU-only unless asked otherwise, and\n"
            "a stray `triton` from another channel is what breaks torch.compile):\n"
            "\n"
            "    python -m venv .venv && source .venv/bin/activate\n"
            "    pip install torch --index-url https://download.pytorch.org/whl/cu124\n"
            "    pip install transformers\n"
            "\n"
            "That wheel pins its own matching triton, so install torch from the PyTorch\n"
            "index *before* anything else that might pull a different one. Verify with\n"
            "    python scripts/bench_retrain_throughput.py --doctor\n"
        ),
    )
    ap.add_argument(
        "--doctor",
        action="store_true",
        help="report what torch sees of the GPU and exit, running no benchmark",
    )
    ap.add_argument("--micro-bs", type=int, nargs="+", default=[4, 8])
    ap.add_argument("--seq-len", type=int, default=2048)
    ap.add_argument("--steps", type=int, default=12)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--allow-cpu",
        action="store_true",
        help="exercise the code path on CPU; the numbers are not usable for sizing",
    )
    ap.add_argument("--peak-tflops", type=float, default=DEFAULT_PEAK_TFLOPS)
    ap.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    ap.add_argument(
        "--compile", action="store_true", help="wrap in torch.compile (slow first step)"
    )
    ap.add_argument("--out", default="data/retrain_throughput.json")
    args = ap.parse_args()

    if args.doctor:
        print(cuda_diagnostics())
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info()
            print(
                f"\n  OK: {torch.cuda.get_device_name(0)}, "
                f"{total / 1024**3:.1f} GB total / {free / 1024**3:.1f} GB free"
            )
            print(f"  bf16 supported: {torch.cuda.is_bf16_supported()}")
        raise SystemExit(0 if torch.cuda.is_available() else 1)

    # Silences inductor's TF32 warning and matches what the real run
    # should use; with bf16 autocast it affects only the fp32 residue.
    torch.set_float32_matmul_precision("high")

    dtype = getattr(torch, args.dtype)
    try:
        device = resolve_device(args.device, args.allow_cpu)
    except NoAcceleratorError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2) from None
    args.device = device
    if device == "cuda":
        print(f"device: {torch.cuda.get_device_name(0)}")
    else:
        print("device: cpu  (--allow-cpu given; numbers are NOT usable for sizing)")
    print(f"dtype: {args.dtype}   seq_len: {args.seq_len}   compile: {args.compile}\n")

    results = []
    for mb in args.micro_bs:
        try:
            r = bench(mb, args.seq_len, args.device, args.steps, args.warmup, args.compile, dtype)
        except torch.OutOfMemoryError:
            print(f"micro_bs {mb}: OOM -- back off, or use a chunked cross-entropy\n")
            continue
        results.append(r)
        accum = PYTHIA_TOKENS_PER_STEP // (mb * args.seq_len)
        print(
            f"micro_bs {mb}:  {r['s_per_step'] * 1000:7.1f} ms/micro-step   "
            f"{r['tokens_per_s']:9,.0f} tok/s   peak VRAM {r['peak_vram_gb']:.2f} GB   "
            f"(grad-accum x{accum} for Pythia's 1024-seq batch)"
        )
        print(project(r["tokens_per_s"], r["achieved_flops"], args.peak_tflops))
        print()

    if results:
        best = max(results, key=lambda r: r["tokens_per_s"])
        print(f"best: micro_bs {best['micro_bs']} at {best['tokens_per_s']:,.0f} tok/s")
        out = REPO_ROOT / args.out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(
                {"peak_tflops": args.peak_tflops, "results": results}, indent=2, sort_keys=True
            )
            + "\n"
        )
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
