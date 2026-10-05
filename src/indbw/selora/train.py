"""Minimal causal-LM LoRA training loop with the fail-fast guards CLAUDE.md requires.

Matches the original recipe (AdamW, linear decay, grad accumulation, full-sequence
causal LM loss) without TRL, so the loop is testable on a tiny random model and
the NaN / step / wall-clock guards live in code we own.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass

import numpy as np
import torch


@dataclass(frozen=True)
class TrainConfig:
    lr: float = 2e-4
    batch_size: int = 4
    grad_accum: int = 4
    epochs: float = 3.0
    max_steps: int | None = None  # optimiser steps; hard cap independent of epochs
    wall_clock_budget_s: float = 6 * 3600.0
    seed: int = 0


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(examples: list[list[int]], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(e) for e in examples)
    ids = torch.full((len(examples), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(examples), n), dtype=torch.long)
    for i, e in enumerate(examples):
        ids[i, : len(e)] = torch.tensor(e)
        mask[i, : len(e)] = 1
    labels = ids.masked_fill(mask == 0, -100)
    return {"input_ids": ids, "attention_mask": mask, "labels": labels}


def batch_order(
    n_examples: int, batch_size: int, grad_accum: int, total_steps: int, seed: int
) -> list[list[int]]:
    """Example indices for every micro-batch, a function of (n, shape, seed) only.

    Shared by every arm with the same seed (common random numbers), so arm differences
    carry no data-order noise. Each epoch is a fresh permutation; a short tail is dropped.
    """
    rng = np.random.default_rng(seed)
    out: list[list[int]] = []
    order: list[int] = []
    for _ in range(total_steps * grad_accum):
        if len(order) < batch_size:
            order = [int(i) for i in rng.permutation(n_examples)]
        out.append(order[:batch_size])
        order = order[batch_size:]
    return out


def total_steps(n_examples: int, cfg: TrainConfig) -> int:
    if cfg.max_steps is not None:
        return cfg.max_steps
    per_epoch = math.ceil(n_examples / (cfg.batch_size * cfg.grad_accum))
    return max(1, int(per_epoch * cfg.epochs))


def train_lora(
    model: torch.nn.Module, examples: list[list[int]], pad_id: int, cfg: TrainConfig
) -> list[float]:
    """Train the model's trainable (LoRA) parameters; return per-optimiser-step losses.

    Raises on non-finite loss, on a model with no trainable parameters, and
    when the wall-clock budget is exceeded.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if not params:
        raise ValueError("no trainable parameters")
    seed_everything(cfg.seed)
    opt = torch.optim.AdamW(params, lr=cfg.lr)
    total = total_steps(len(examples), cfg)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1.0 - s / total))
    batches = batch_order(len(examples), cfg.batch_size, cfg.grad_accum, total, cfg.seed)
    device = next(model.parameters()).device
    model.train()
    t0 = time.monotonic()
    losses: list[float] = []
    for step in range(total):
        opt.zero_grad(set_to_none=True)
        acc = 0.0
        for micro in range(cfg.grad_accum):
            take = batches[step * cfg.grad_accum + micro]
            batch = {
                k: v.to(device) for k, v in collate([examples[i] for i in take], pad_id).items()
            }
            loss = model(**batch).loss / cfg.grad_accum
            if not torch.isfinite(loss):
                raise FloatingPointError(f"non-finite loss at optimiser step {step}")
            loss.backward()
            acc += float(loss.detach())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        sched.step()
        losses.append(acc)
        if time.monotonic() - t0 > cfg.wall_clock_budget_s:
            raise TimeoutError(f"wall-clock budget exceeded at step {step}")
    model.eval()
    return losses
