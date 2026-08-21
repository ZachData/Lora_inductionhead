"""Re-run pythia-70m forward from a published checkpoint, densely probed.

PROJECT.md §11 (2026-08-21). The published checkpoint grid jumps 512 ->
1000, which is 488 optimizer steps and 1.02B tokens -- so "the induction
transition is bracketed by two adjacent checkpoints" (G0) is true of the
grid and useless as a statement about the weights. §1.1 abandoned
delta-reproduction for exactly this reason. Re-running the segment with a
probe every few steps replaces that 1.02B-token bracket with one a few
optimizer steps wide, which is the object §1.1 said did not exist.

**This is a re-run from A, not a replay of Pythia.** Pythia publishes
weights but not optimizer moments, so Adam restarts cold at `from_step`
and the trajectory diverges from the original immediately. That is
acceptable for the stated purpose -- a densely-sampled trajectory that
forms induction, from A -- and it is the reason `to_step=2000` is the
default: step 2000 is checkpoint B, so the endpoint can be compared
against the published weights and the divergence measured rather than
assumed. Any run that stops earlier gives up that check.

What lives here is the part worth testing (schedule, buffer, config);
`scripts/run_retrain.py` is the loop that uses it.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import torch


#: Pythia's published grid (PROJECT.md §2): step 0, log-spaced to 512,
#: then every 1000. Only these are resumable -- there is nothing to load
#: for any other step.
def _pythia_grid() -> tuple[int, ...]:
    log_spaced = [0] + [2**i for i in range(10)]  # 0, 1, 2, ... 512
    return tuple(log_spaced + list(range(1000, 144000, 1000)))


PYTHIA_CHECKPOINT_STEPS = _pythia_grid()


@dataclass(frozen=True)
class LRSchedule:
    """GPT-NeoX's annealing schedule, which is what produced the published
    checkpoints. Linear warmup to `peak_lr`, then cosine decay to
    `peak_lr * min_lr_ratio` at `total_steps`, flat thereafter.
    """

    peak_lr: float
    warmup_steps: int
    total_steps: int
    min_lr_ratio: float


#: pythia-70m's actual schedule (Biderman et al. 2023): peak 1e-3, warmup
#: 1% of 143000 iterations, cosine to 0.1x peak. Checkpoint A (step 512)
#: is *inside* warmup, at ~3.58e-4 -- 2.8x below peak and still climbing
#: through the entire transition window. A harness that starts at peak LR,
#: which is what almost every training script does by default, runs the
#: window under study at nearly three times the intended rate.
PYTHIA_70M_SCHEDULE = LRSchedule(
    peak_lr=1e-3, warmup_steps=1_430, total_steps=143_000, min_lr_ratio=0.1
)


def lr_at_step(step: int, schedule: LRSchedule = PYTHIA_70M_SCHEDULE) -> float:
    """Learning rate at a global optimizer step, on the original schedule.

    `step` is Pythia's global step, not an offset into the re-run: the
    whole point is to continue the published schedule from where the
    checkpoint left off rather than to start a new one.
    """
    if step < 0:
        raise ValueError(f"step must be >= 0, got {step}")
    min_lr = schedule.peak_lr * schedule.min_lr_ratio
    if step < schedule.warmup_steps:
        return schedule.peak_lr * step / schedule.warmup_steps
    if step >= schedule.total_steps:
        return min_lr
    progress = (step - schedule.warmup_steps) / (schedule.total_steps - schedule.warmup_steps)
    return min_lr + (schedule.peak_lr - min_lr) * 0.5 * (1.0 + math.cos(math.pi * progress))


@dataclass(frozen=True)
class RetrainConfig:
    """One run, serialized to JSON and hashed. The hash is the run ID
    (CLAUDE.md, "Config as data"), which is also what makes resumption
    free: a run resumes only into its own directory.

    Defaults reproduce Pythia's batch (1024 sequences x 2048 tokens) via
    gradient accumulation, at the micro-batch measured to fit a 10GB card
    (micro_bs 8 OOMs -- PROJECT.md §11, 2026-08-21).
    """

    from_step: int
    to_step: int
    micro_bs: int = 4
    grad_accum: int = 256
    seq_len: int = 2048
    seed: int = 0
    # Optimizer: GPT-NeoX defaults, which is what trained the checkpoints.
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    # Probing and capture.
    probe_every: int = 4
    probe_n_eval: int = 128
    probe_T: int = 128
    probe_layer: int = 3
    probe_head: int = 6
    onset_pms: float = 0.05
    buffer_steps: int = 24
    capture_steps: int = 60
    # Durability.
    save_every: int = 25
    out_dir: Path = field(default=Path("data/retrain"))
    dtype: str = "bfloat16"
    data_source: str = "hf-stream"

    def __post_init__(self) -> None:
        if self.to_step <= self.from_step:
            raise ValueError(f"to_step ({self.to_step}) must exceed from_step ({self.from_step})")
        if self.from_step not in PYTHIA_CHECKPOINT_STEPS:
            raise ValueError(
                f"from_step {self.from_step} is not on Pythia's checkpoint grid; "
                "there is no published checkpoint to resume from"
            )
        if self.micro_bs < 1 or self.grad_accum < 1:
            raise ValueError("micro_bs and grad_accum must both be >= 1")

    def tokens_per_optimizer_step(self) -> int:
        return self.micro_bs * self.grad_accum * self.seq_len

    def total_optimizer_steps(self) -> int:
        return self.to_step - self.from_step

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["out_dir"] = str(self.out_dir)
        d["betas"] = list(self.betas)
        return d

    def run_id(self) -> str:
        """Config hash. Excludes out_dir -- where results are written does
        not change what was computed, and including it would give the same
        run two identities on two machines."""
        d = {k: v for k, v in self.to_dict().items() if k != "out_dir"}
        payload = json.dumps(d, sort_keys=True).encode()
        return hashlib.sha256(payload).hexdigest()[:16]

    def run_dir(self) -> Path:
        return Path(self.out_dir) / self.run_id()


class RollingCheckpointBuffer:
    """Keeps the last `keep` checkpoints, then everything for
    `capture_steps` once triggered.

    Onset is only detectable after it has happened, so the weights from
    *before* it have to already be on disk when the probe notices. That is
    the entire reason this class exists: a naive "start saving when PMS
    crosses" captures only the post-onset side, and the pre-onset side is
    the one the delta is measured against.

    Checkpoints are written as they are offered and deleted as they age
    out, so peak disk is bounded by (keep + capture_steps) files.
    """

    def __init__(self, directory: Path, keep: int, capture_steps: int) -> None:
        if keep < 1:
            raise ValueError(f"keep must be >= 1, got {keep}")
        if capture_steps < 1:
            raise ValueError(f"capture_steps must be >= 1, got {capture_steps}")
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.keep = keep
        self.capture_steps = capture_steps
        self.triggered_at: int | None = None
        self._retained: list[int] = []

    @property
    def capture_exhausted(self) -> bool:
        if self.triggered_at is None:
            return False
        return len(self._retained) >= self.keep + self.capture_steps

    def _path(self, step: int) -> Path:
        return self.directory / f"step_{step:06d}.pt"

    def offer(self, step: int, state: dict[str, torch.Tensor]) -> Path | None:
        """Persist `state` for `step`, evicting the oldest if untriggered.

        Returns the written path, or None once the capture window is full
        (at which point there is nothing further to record and continuing
        to write would just consume disk).
        """
        if self.capture_exhausted:
            return None
        path = self._path(step)
        # .clone() on every tensor: callers pass a live state_dict, whose
        # tensors are mutated in place by the next optimizer step. Saving
        # references would put the final weights under every filename and
        # nothing downstream would notice.
        torch.save({k: v.detach().to("cpu").clone() for k, v in state.items()}, path)
        self._retained.append(step)
        if self.triggered_at is None:
            while len(self._retained) > self.keep:
                self._path(self._retained.pop(0)).unlink(missing_ok=True)
        return path

    def trigger(self, at_step: int) -> None:
        """Freeze retention and begin the capture window. Idempotent: PMS
        is estimated on a finite eval set and wobbles around any
        threshold, so re-triggering would extend capture indefinitely."""
        if self.triggered_at is None:
            self.triggered_at = at_step

    def retained_steps(self) -> list[int]:
        return sorted(self._retained)

    def load(self, step: int) -> dict[str, torch.Tensor]:
        loaded: dict[str, torch.Tensor] = torch.load(
            self._path(step), map_location="cpu", weights_only=True
        )
        return loaded


def accumulate_gradients(
    model: torch.nn.Module,
    micro_batches: Sequence[torch.Tensor],
    loss_fn: Callable[[torch.nn.Module, torch.Tensor], torch.Tensor],
) -> float:
    """Accumulate gradients over `micro_batches` as one large batch, and
    return the mean loss. Leaves gradients on the parameters; the caller
    clips and steps.

    The 1/N is the whole content of this function and the reason it is
    not inlined into the loop. Without it the accumulated gradient is N
    times too large -- 256x under the default config -- which multiplies
    the effective learning rate by 256 while producing a loss curve that
    looks entirely ordinary. Nothing downstream of the optimizer would
    report it, so it is pinned by an equivalence oracle against a single
    large batch in tests/unit/test_retrain.py.

    Exact only when every micro-batch has the same number of loss-bearing
    positions, which holds here: fixed `micro_bs` and fixed `seq_len`,
    with no padding anywhere in the pipeline.
    """
    if len(micro_batches) == 0:
        raise ValueError("accumulate_gradients needs at least one micro-batch")
    model.zero_grad(set_to_none=True)
    total = 0.0
    n = len(micro_batches)
    for chunk in micro_batches:
        loss = loss_fn(model, chunk)
        # torch's `backward` is unannotated in the installed stubs; the
        # ignore is a stub gap, matching train.py's existing one.
        (loss / n).backward()  # type: ignore[no-untyped-call]
        total += float(loss.item())
    return total / n


class DegenerateProbeError(RuntimeError):
    """The probe could not produce a real reading. Raised rather than
    returned as a number: a PMS of 0.0 from a model whose attentions came
    back empty is indistinguishable, in a log or a plot, from a PMS of
    0.0 from a model that genuinely has no induction -- and the entire
    experiment is the shape of that curve over time.
    """


@torch.no_grad()
def probe_induction(
    model: Any,
    eval_tokens: torch.Tensor,
    period: int,
    layer: int,
    head: int,
    batch_size: int = 8,
    device: str = "cuda",
) -> dict[str, float]:
    """PMS on (`layer`, `head`) plus the ICL decomposition, read straight
    off an HF model -- no TransformerLens in the training path.

    `eval_tokens` is [n, 2*period] from `indbw.evalset.build_eval_tokens`,
    so this is measuring exactly the quantity PROJECT.md §5 defines, on
    exactly the fixed eval set the committed G0/G1/G2 results used.

    Batched because the logits tensor is the memory hazard here as
    everywhere else in this repo: n=128 at 2*period=256 and vocab 50304
    is 6.6 GB in one go.
    """
    from indbw.probes import icl_score, prefix_matching_score
    from indbw.train import first_copy_nll, second_copy_nll

    import numpy as np

    was_training = model.training
    model.eval()
    n = eval_tokens.shape[0]
    nll_first = np.empty(n)
    nll_second = np.empty(n)
    pms_vals: list[float] = []
    try:
        for start in range(0, n, batch_size):
            batch = eval_tokens[start : start + batch_size].to(device)
            out = model(batch, output_attentions=True)
            if out.attentions is None or len(out.attentions) <= layer:
                raise DegenerateProbeError(
                    "model returned no attention patterns for "
                    f"layer {layer}; the attention implementation must be one that "
                    "can emit them (set attn_implementation='eager' for the probe)"
                )
            logits = out.logits.float().cpu()
            tok = batch.cpu()
            nll_first[start : start + tok.shape[0]] = first_copy_nll(logits, tok, period).numpy()
            nll_second[start : start + tok.shape[0]] = second_copy_nll(logits, tok, period).numpy()
            patt = out.attentions[layer][:, head].double().cpu().numpy()
            pms_vals.extend(prefix_matching_score(p, period) for p in patt)
    finally:
        model.train(was_training)

    if not pms_vals:
        raise DegenerateProbeError("probe produced no readings; eval set was empty")
    icl = icl_score(nll_first, nll_second)
    return {
        "pms": float(np.mean(pms_vals)),
        "icl": icl,
        "nll_first": float(nll_first.mean()),
        "nll_second": float(nll_second.mean()),
    }


class SyntheticSource:
    """Uniform-random tokens. Not a stand-in for the Pile -- it cannot
    form induction, since there is nothing in a uniform stream to induct
    on. It exists so the loop, the schedule, the accumulation and the
    checkpoint logic can be exercised end to end with no network, which
    is the only part of this harness that can be tested offline.
    """

    def __init__(self, d_vocab: int, seed: int) -> None:
        self.d_vocab = d_vocab
        self._gen = torch.Generator(device="cpu").manual_seed(seed)

    def next_batch(self, micro_bs: int, seq_len: int) -> torch.Tensor:
        return torch.randint(
            0, self.d_vocab, (micro_bs, seq_len), generator=self._gen, dtype=torch.long
        )


class HFStreamSource:
    """Streams and packs a HuggingFace text dataset into fixed-length
    token sequences.

    **Untested against a real dataset.** The session that wrote this had
    no HuggingFace access, so the packing logic is covered only by unit
    tests over a fake iterator; the dataset name, the field name, and the
    streaming behaviour are not verified. Check the first batch decodes
    to sensible text before starting a 9-hour run.

    This does *not* reproduce Pythia's batch order. Doing so needs the
    pre-shuffled pretokenized Pile and an index into it; the deliberate
    decision (PROJECT.md §11) is that batch-order fidelity is not worth
    the setup here, because Adam is cold-started at `from_step` anyway
    and the trajectory therefore diverges from the published one on the
    first step regardless.
    """

    def __init__(
        self,
        tokenizer: Any,
        dataset_name: str = "monology/pile-uncopyrighted",
        split: str = "train",
        text_field: str = "text",
        seed: int = 0,
    ) -> None:
        from datasets import load_dataset  # type: ignore[import-untyped]

        self.tokenizer = tokenizer
        self.text_field = text_field
        stream = load_dataset(dataset_name, split=split, streaming=True)
        self._it = iter(stream.shuffle(seed=seed, buffer_size=10_000))
        self._buf: list[int] = []

    def _fill(self, n_tokens: int) -> None:
        eos = self.tokenizer.eos_token_id
        while len(self._buf) < n_tokens:
            doc = next(self._it)
            ids = self.tokenizer(doc[self.text_field], add_special_tokens=False)["input_ids"]
            self._buf.extend(ids)
            if eos is not None:
                self._buf.append(eos)

    def next_batch(self, micro_bs: int, seq_len: int) -> torch.Tensor:
        need = micro_bs * seq_len
        self._fill(need)
        flat = self._buf[:need]
        self._buf = self._buf[need:]
        return torch.tensor(flat, dtype=torch.long).view(micro_bs, seq_len)
