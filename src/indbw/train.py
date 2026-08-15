"""Training loop and adapter snapshotting.

PROJECT.md §3/§8. Adapts exactly one matrix per bilinear form: W_Q for the
QK arm, W_O for the OV arm (never both -- the composition rule, §3, that
keeps rank and parity structure meaningful). Injection is via forward
hooks on a `HookedTransformer`, never an in-place edit to the model's own
parameter tensor: the base weight (and everything else -- LayerNorm,
embeddings, all other heads/layers) stays frozen, and the low-rank
factors (B, A) are the *only* trainable tensors. This is what makes
"freeze LayerNorm and embeddings" (PROJECT.md §8) and "adapt exactly one
matrix" (§3) mechanically true rather than a convention that could drift.

Shape convention matches `indbw.lora.unconstrained_delta` exactly, so a
learned (B, A) pair can be handed straight to that module for downstream
symmetry/rank analysis without reshaping:
  - QK arm: base weight is `W_Q[layer, head]`, shape [d_model, d_head].
    B: [d_model, r], A: [r, d_head]. delta = (alpha/r) B @ A, added to the
    query computed from that head's normalized residual input.
  - OV arm: base weight is `W_O[layer, head]`, shape [d_head, d_model].
    B: [d_head, r], A: [r, d_model]. delta = (alpha/r) B @ A, added to
    that head's contribution to the residual stream.
Only the "unconstrained" parameterization (indbw.lora) is used here --
G2/G3's generous-rank positive control and M1/M2's rank sweeps are both
plain LoRA on one matrix. The symmetric/antisymmetric triad (M3) installs
its square delta differently (PROJECT.md §11, open question -- not
resolved by this module).

Training objective (PROJECT.md §11: "Induction objective ... Unresolved
and consequential -- settle before G3"; settled here, since G2 already
needs one to time a run against -- logged in PROJECT.md §10):
synthetic repeated-random NLL. Minimize mean per-token NLL on the second
copy of a repeated-random sequence -- exactly the quantity `icl_score`
(PROJECT.md §5) subtracts from the first-copy NLL, so a trained update
that lowers this loss raises ICL score by construction. This is the
"cleanest" option PROJECT.md §11 names; the risk it names alongside it
(a circuit that only works on synthetic input) is exactly what the
held-out natural-text check (M8) exists to catch, not something this
module needs to guard against itself.

Storage: LoRA factors only (B, A as plain numpy arrays via
`save_snapshot`), never a merged model (PROJECT.md §8, "Storage").
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import torch
import torch.nn.functional as F
from transformer_lens import HookedTransformer

from indbw.evalset import build_eval_tokens
from indbw.probes import icl_score, recovery

Arm = Literal["QK", "OV"]
ARMS: tuple[Arm, ...] = ("QK", "OV")

#: What `HookedTransformer.run_with_hooks` accepts for `fwd_hooks`. The
#: callables are typed `Callable[..., Any]` rather than a precise
#: (tensor, HookPoint) -> tensor signature because that is what
#: transformer_lens's own annotation says; narrowing it here would be a
#: fiction mypy would then have to be told to ignore at the call site.
HookList = list[tuple[str | Callable[..., Any], Callable[..., Any]]]


class TrainingBudgetExceeded(RuntimeError):
    """Hard wall-clock budget inside the training loop, independent of the
    OS-level 4h instance cap (CLAUDE.md, "Fail-fast in production paths").
    Raised, never silently swallowed -- a run that hits this needs a human
    decision (bigger budget? smaller rank first?), not a truncated result
    that looks like it finished normally.
    """


@dataclass
class LoRAFactors:
    """Trainable rank-r factors for one arm's delta. `B` starts at zero
    (see `init_lora_factors`) so the adapted model is bit-identical to
    the base model before any training step -- the zero-init discrimination
    test in test_train.py depends on this.
    """

    B: torch.Tensor
    A: torch.Tensor
    alpha: float

    def delta(self) -> torch.Tensor:
        r = self.B.shape[1]
        return (self.alpha / r) * (self.B @ self.A)

    def numpy(self) -> tuple[np.ndarray, np.ndarray]:
        return (
            self.B.detach().cpu().numpy().copy(),
            self.A.detach().cpu().numpy().copy(),
        )


def init_lora_factors(
    d_out: int,
    d_in: int,
    rank: int,
    alpha: float,
    seed: int,
    device: str = "cpu",
    dtype: torch.dtype = torch.float32,
) -> LoRAFactors:
    """B: [d_out, r] zero-initialized, A: [r, d_in] small random. Standard
    LoRA init -- guarantees delta() == 0 exactly at construction.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    if d_out < 1 or d_in < 1:
        raise ValueError(f"d_out and d_in must be >= 1, got {d_out}, {d_in}")
    gen = torch.Generator(device="cpu").manual_seed(seed)
    A = (torch.randn(rank, d_in, generator=gen, dtype=dtype) / (d_in**0.5)).to(device)
    A.requires_grad_(True)
    B = torch.zeros(d_out, rank, dtype=dtype, device=device, requires_grad=True)
    return LoRAFactors(B=B, A=A, alpha=alpha)


def freeze_base_model(model: HookedTransformer) -> None:
    """No parameter of the base model is trainable. Freezes everything,
    including LayerNorm and embeddings (PROJECT.md §8) -- the only
    trainable tensors in a training run are a `LoRAFactors`' B and A.
    """
    for p in model.parameters():
        p.requires_grad_(False)


def _qk_shapes(model: HookedTransformer) -> tuple[int, int]:
    return int(model.cfg.d_model), int(model.cfg.d_head)


def _ov_shapes(model: HookedTransformer) -> tuple[int, int]:
    return int(model.cfg.d_head), int(model.cfg.d_model)


def factor_shapes(model: HookedTransformer, arm: Arm) -> tuple[int, int]:
    """(d_out, d_in) for `init_lora_factors`, matching the base weight's
    own shape for `arm` (see module docstring's shape convention)."""
    if arm == "QK":
        return _qk_shapes(model)
    if arm == "OV":
        return _ov_shapes(model)
    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


def qk_hooks(layer: int, head: int, factors: LoRAFactors) -> HookList:
    """Forward hooks adding `factors.delta()` to head `head`'s query at
    `layer`. Captures that head's normalized residual input (what W_Q
    actually multiplies) via a hook on ln1's output, one step earlier in
    the same forward pass, and adds `x @ delta` onto `hook_q`'s slice for
    this head only -- every other head, every other layer, is untouched.
    """
    box: dict[str, torch.Tensor] = {}

    def capture(x: torch.Tensor, hook: object) -> torch.Tensor:
        box["x"] = x
        return x

    def add_delta(q: torch.Tensor, hook: object) -> torch.Tensor:
        delta_w = factors.delta()  # [d_model, d_head]
        x = box["x"]  # [batch, pos, d_model]
        delta_q = torch.einsum("bpd,dh->bph", x, delta_w)
        q = q.clone()
        q[:, :, head, :] = q[:, :, head, :] + delta_q
        return q

    return [
        (f"blocks.{layer}.ln1.hook_normalized", capture),
        (f"blocks.{layer}.attn.hook_q", add_delta),
    ]


def ov_hooks(layer: int, head: int, factors: LoRAFactors) -> HookList:
    """Forward hooks adding `factors.delta()`'s contribution of head
    `head` at `layer` directly onto that block's summed attention output
    (post W_O, pre residual-add) -- the OV analogue of `qk_hooks`.
    """
    box: dict[str, torch.Tensor] = {}

    def capture(z: torch.Tensor, hook: object) -> torch.Tensor:
        box["z"] = z
        return z

    def add_delta(attn_out: torch.Tensor, hook: object) -> torch.Tensor:
        delta_w = factors.delta()  # [d_head, d_model]
        z_head = box["z"][:, :, head, :]  # [batch, pos, d_head]
        delta_out = torch.einsum("bph,hd->bpd", z_head, delta_w)
        return attn_out + delta_out

    return [
        (f"blocks.{layer}.attn.hook_z", capture),
        (f"blocks.{layer}.hook_attn_out", add_delta),
    ]


def qk_both_hooks(
    layer: int, head: int, q_factors: LoRAFactors, k_factors: LoRAFactors
) -> HookList:
    """Diagnostic-only: adds independent deltas to *both* Q and K for one
    head. Not part of the `Arm`/`build_hooks`/`TrainConfig` system and
    never used by G2/G3 or the M1/M2 sweep -- PROJECT.md §3 requires
    adapting exactly one matrix per bilinear form, since a two-matrix
    delta destroys the rank/parity structure that analysis depends on.

    Exists only to test a diagnostic question raised in PROJECT.md §11 /
    REVIEW.md after G3's failure: does freezing $W_K$ at checkpoint A cap
    what any rank of $\\Delta W_Q$ can reach, i.e. is G3's plateau a
    $W_K$-bottleneck rather than a capacity limit? Letting $W_K$ adapt
    too is the direct way to ask that, purely as a probe -- if it is
    used for anything beyond a diagnostic run, that is a §3 protocol
    change requiring the same human sign-off this function's use already
    required (REVIEW.md, 2026-08-14 entry).

    Shares one capture of `ln1.hook_normalized` (both hook_q and hook_k
    read that same normalized residual input in this architecture) and
    adds each factor's delta independently -- same capture-then-add
    mechanics as `qk_hooks`, just duplicated onto a second hook point.
    """
    box: dict[str, torch.Tensor] = {}

    def capture(x: torch.Tensor, hook: object) -> torch.Tensor:
        box["x"] = x
        return x

    def add_delta_q(q: torch.Tensor, hook: object) -> torch.Tensor:
        delta_w = q_factors.delta()
        x = box["x"]
        delta_q = torch.einsum("bpd,dh->bph", x, delta_w)
        q = q.clone()
        q[:, :, head, :] = q[:, :, head, :] + delta_q
        return q

    def add_delta_k(k: torch.Tensor, hook: object) -> torch.Tensor:
        delta_w = k_factors.delta()
        x = box["x"]
        delta_k = torch.einsum("bpd,dh->bph", x, delta_w)
        k = k.clone()
        k[:, :, head, :] = k[:, :, head, :] + delta_k
        return k

    return [
        (f"blocks.{layer}.ln1.hook_normalized", capture),
        (f"blocks.{layer}.attn.hook_q", add_delta_q),
        (f"blocks.{layer}.attn.hook_k", add_delta_k),
    ]


def build_hooks(arm: Arm, layer: int, head: int, factors: LoRAFactors) -> HookList:
    if arm == "QK":
        return qk_hooks(layer, head, factors)
    if arm == "OV":
        return ov_hooks(layer, head, factors)
    raise ValueError(f"unknown arm {arm!r}, expected one of {ARMS}")


def _check_finite_loss(loss: torch.Tensor, step: int) -> None:
    """NaN/Inf check on loss after every training step -- raise
    immediately, never let it run to completion (CLAUDE.md)."""
    if not torch.isfinite(loss):
        raise FloatingPointError(f"non-finite loss ({loss.item()}) at step {step}")


def second_copy_nll(logits: torch.Tensor, tokens: torch.Tensor, T: int) -> torch.Tensor:
    """Per-example mean NLL on the second copy's target tokens (positions
    T+1..2T-1, predicted from logits at positions T..2T-2).

    The seam position -- predicting token T (the first token of the
    second copy) from position T-1 (the last token of the first copy) --
    is excluded from both this and `first_copy_nll`, matching
    `scripts/g0_sweep.py`'s windowing exactly (`nll[:, :T-1]` /
    `nll[:, T:]` there). That windowing already produced the committed
    G0/G1 results (A/B checkpoints, their ICL values); this module reuses
    those recorded ICL(A)/ICL(B) values as recovery baselines (PROJECT.md
    §5), so matching the window here is not a style choice -- a mismatch
    would silently compute recovery against baselines measured on a
    different quantity. (The excluded seam position also isn't a clean
    induction test in the first place: token T-1's *first* occurrence is
    typically its only occurrence within the first copy alone, so nothing
    has been seen yet to copy from.)

    logits: [batch, 2T, vocab]. tokens: [batch, 2T]. Returns [batch].
    """
    seq_len = tokens.shape[1]
    if seq_len != 2 * T:
        raise ValueError(f"expected seq_len == 2*T ({2 * T}), got {seq_len}")
    if T < 2:
        raise ValueError(f"second_copy_nll requires T >= 2, got {T}")
    pred_logits = logits[:, T : 2 * T - 1, :]  # [batch, T-1, vocab]
    targets = tokens[:, T + 1 : 2 * T]  # [batch, T-1]
    nll = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    return nll.mean(dim=1)  # [batch]


def first_copy_nll(logits: torch.Tensor, tokens: torch.Tensor, T: int) -> torch.Tensor:
    """Per-example mean NLL on the first copy's target tokens (positions
    1..T-1, predicted from logits at positions 0..T-2 -- position 0 has
    no context to predict from, so it is excluded). Returns [batch]."""
    seq_len = tokens.shape[1]
    if seq_len != 2 * T:
        raise ValueError(f"expected seq_len == 2*T ({2 * T}), got {seq_len}")
    if T < 2:
        raise ValueError(f"first_copy_nll requires T >= 2, got {T}")
    pred_logits = logits[:, 0 : T - 1, :]
    targets = tokens[:, 1:T]
    nll = F.cross_entropy(
        pred_logits.reshape(-1, pred_logits.shape[-1]), targets.reshape(-1), reduction="none"
    ).reshape(targets.shape)
    return nll.mean(dim=1)


@torch.no_grad()
def compute_recovery(
    model: HookedTransformer,
    hooks: HookList,
    eval_tokens: torch.Tensor,
    T: int,
    icl_a: float,
    icl_b: float,
    batch_size: int = 32,
) -> float:
    """Recovery R of the adapted model (base model + `hooks`) against the
    fixed A/B ICL baselines, over `eval_tokens` (PROJECT.md §5). Unclamped
    -- see `indbw.probes.recovery`.
    """
    n = eval_tokens.shape[0]
    nll_first = np.empty(n)
    nll_second = np.empty(n)
    for start in range(0, n, batch_size):
        batch = eval_tokens[start : start + batch_size]
        logits = model.run_with_hooks(batch, fwd_hooks=hooks, return_type="logits")
        nll_first[start : start + batch.shape[0]] = first_copy_nll(logits, batch, T).numpy()
        nll_second[start : start + batch.shape[0]] = second_copy_nll(logits, batch, T).numpy()
    icl = icl_score(nll_first, nll_second)
    return recovery(icl, icl_a, icl_b)


@dataclass
class TrainConfig:
    arm: Arm
    layer: int
    head: int
    rank: int
    alpha: float
    lr: float
    max_steps: int
    batch_size: int
    T: int
    d_vocab: int
    train_seed: int
    icl_a: float
    icl_b: float
    criterion_r: float = 0.80
    eval_every: int = 10
    eval_n: int = 128
    eval_seed: int = 0
    max_wall_clock_s: float = 3600.0
    snapshot_dir: Path | None = None
    snapshot_every: int = 50


@dataclass
class TrainResult:
    steps_run: int
    wall_clock_s: float
    reached_criterion: bool
    final_recovery: float
    recovery_history: list[tuple[int, float]] = field(default_factory=list)
    loss_history: list[float] = field(default_factory=list)
    B: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    A: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    snapshot_paths: list[Path] = field(default_factory=list)


def save_snapshot(
    directory: Path, step: int, factors: LoRAFactors, arm: Arm, layer: int, head: int, alpha: float
) -> Path:
    """LoRA factors only, never a merged model (PROJECT.md §8, "Storage")."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    B, A = factors.numpy()
    path = directory / f"step_{step:06d}.npz"
    np.savez(path, B=B, A=A, step=step, arm=arm, layer=layer, head=head, alpha=alpha)
    return path


def load_snapshot(path: Path) -> dict[str, object]:
    data = np.load(path, allow_pickle=False)
    return {
        "B": data["B"],
        "A": data["A"],
        "step": int(data["step"]),
        "arm": str(data["arm"]),
        "layer": int(data["layer"]),
        "head": int(data["head"]),
        "alpha": float(data["alpha"]),
    }


def train_lora(model: HookedTransformer, config: TrainConfig) -> TrainResult:
    """Train a rank-`config.rank` LoRA delta for one arm from a frozen
    base model until recovery R reaches `config.criterion_r` or
    `config.max_steps` is exhausted, whichever comes first.

    Raises `TrainingBudgetExceeded` if wall-clock exceeds
    `config.max_wall_clock_s` before either stop condition -- a hard
    budget independent of any OS-level cap (CLAUDE.md). Raises
    `FloatingPointError` immediately on a non-finite loss.
    """
    freeze_base_model(model)
    d_out, d_in = factor_shapes(model, config.arm)
    factors = init_lora_factors(d_out, d_in, config.rank, config.alpha, seed=config.train_seed)
    optimizer = torch.optim.Adam([factors.B, factors.A], lr=config.lr)
    hooks = build_hooks(config.arm, config.layer, config.head, factors)

    eval_tokens = build_eval_tokens(config.eval_n, config.T, config.eval_seed, config.d_vocab)
    train_rng = np.random.default_rng(config.train_seed + 1)  # distinct stream from eval

    t0 = time.time()
    reached = False
    recovery_history: list[tuple[int, float]] = []
    loss_history: list[float] = []
    snapshot_paths: list[Path] = []
    step = 0

    for step in range(1, config.max_steps + 1):
        elapsed = time.time() - t0
        if elapsed > config.max_wall_clock_s:
            raise TrainingBudgetExceeded(
                f"exceeded max_wall_clock_s={config.max_wall_clock_s}s at step {step} "
                f"(elapsed={elapsed:.1f}s), before reaching criterion or max_steps"
            )

        batch = build_eval_tokens(
            config.batch_size, config.T, int(train_rng.integers(0, 2**31 - 1)), config.d_vocab
        )
        logits = model.run_with_hooks(batch, fwd_hooks=hooks, return_type="logits")
        loss = second_copy_nll(logits, batch, config.T).mean()
        _check_finite_loss(loss, step)

        optimizer.zero_grad()
        # torch's `backward` is unannotated in the installed stubs; the
        # ignore is a stub gap, not a real typing problem here.
        loss.backward()  # type: ignore[no-untyped-call]
        optimizer.step()
        loss_history.append(float(loss.item()))

        if config.snapshot_dir is not None and step % config.snapshot_every == 0:
            snapshot_paths.append(
                save_snapshot(
                    config.snapshot_dir,
                    step,
                    factors,
                    config.arm,
                    config.layer,
                    config.head,
                    config.alpha,
                )
            )

        if step % config.eval_every == 0 or step == config.max_steps:
            # batch_size=8, not compute_recovery's default 32: a full-vocab
            # logits tensor at batch_size*config.T*2*d_vocab*4 bytes gets
            # large fast (32 batches at T=128/d_vocab=50304 is ~1.65GB per
            # batch), which swap-thrashed a real periodic eval during a G2
            # run to an unrecoverable halt (PROJECT.md §10, 2026-08-14). 8
            # keeps a single batch's peak footprint well bounded regardless
            # of what eval_n a caller picks.
            r = compute_recovery(
                model, hooks, eval_tokens, config.T, config.icl_a, config.icl_b, batch_size=8
            )
            recovery_history.append((step, r))
            print(
                f"  step {step}/{config.max_steps} loss={loss.item():.4f} R={r:.4f} "
                f"elapsed={time.time() - t0:.0f}s",
                flush=True,
            )
            if r >= config.criterion_r:
                reached = True
                break

    wall_clock_s = time.time() - t0
    if config.snapshot_dir is not None:
        snapshot_paths.append(
            save_snapshot(
                config.snapshot_dir,
                step,
                factors,
                config.arm,
                config.layer,
                config.head,
                config.alpha,
            )
        )
    B, A = factors.numpy()
    final_recovery = recovery_history[-1][1] if recovery_history else float("nan")

    return TrainResult(
        steps_run=step,
        wall_clock_s=wall_clock_s,
        reached_criterion=reached,
        final_recovery=final_recovery,
        recovery_history=recovery_history,
        loss_history=loss_history,
        B=B,
        A=A,
        snapshot_paths=snapshot_paths,
    )
