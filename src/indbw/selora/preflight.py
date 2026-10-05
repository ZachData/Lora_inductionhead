"""Preflight checks run once real checkpoints are reachable, before any sweep (STATE.md D1).

Pure functions over already-loaded objects: each returns a list of human-readable problems
(empty = pass) so `scripts/preflight_selora.py` can report everything at once.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import torch

from indbw.selora.prereg import ASSUMPTIONS
from indbw.selora.sae import JumpReLUSAE

GEMMA2_2B_D_MODEL = 2304
GEMMA_SCOPE_16K = 16384
BASE_FVU_RANGE = (0.0, 0.5)  # outside this the SAE/model pairing or the hook is wrong
ASSUMPTION_REL_TOL = (
    0.5  # a measured quantity off by >50% means the power analysis was for another study
)


def check_sae(
    sae: JumpReLUSAE, d_model: int = GEMMA2_2B_D_MODEL, d_sae: int = GEMMA_SCOPE_16K
) -> list[str]:
    out = []
    if sae.d_model != d_model:
        out.append(f"SAE d_model {sae.d_model} != model d_model {d_model}")
    if sae.d_sae != d_sae:
        out.append(f"SAE width {sae.d_sae} != expected {d_sae}")
    for name in ("W_enc", "W_dec", "b_enc", "b_dec", "threshold"):
        t = getattr(sae, name)
        if not bool(torch.isfinite(t).all()):
            out.append(f"{name} has NaN/Inf")
    if float(sae.W_dec.abs().max()) == 0.0:
        out.append("W_dec is all zeros")
    return out


def check_base_fvu(fvu_by_layer: Mapping[int, float]) -> list[str]:
    lo, hi = BASE_FVU_RANGE
    return [
        f"layer {li}: base FVU {v} outside [{lo}, {hi}]"
        for li, v in sorted(fvu_by_layer.items())
        if not (math.isfinite(v) and lo <= v <= hi)
    ]


def _flat(v: Any) -> list[Any]:
    if isinstance(v, Mapping):
        return [x for sub in v.values() for x in _flat(sub)]
    if isinstance(v, list | tuple):
        return list(v)
    return [v]


def check_populated(row: Mapping[str, Any], keys: Sequence[str]) -> list[str]:
    """First output row of a sweep: every key present, non-empty and not constant."""
    out = []
    for k in keys:
        if k not in row:
            out.append(f"{k}: missing")
            continue
        vals = _flat(row[k])
        if not vals:
            out.append(f"{k}: empty")
        elif len(vals) > 2 and len(set(map(str, vals))) == 1:
            out.append(f"{k}: constant ({vals[0]!r} x{len(vals)})")
    return out


def steps_per_cell(n_examples: int, batch_size: int, grad_accum: int, epochs: float) -> int:
    return int(math.ceil(n_examples / (batch_size * grad_accum)) * epochs)


def estimate_sweep_hours(sec_per_step: float, steps: int, n_cells: int, n_parallel: int) -> float:
    if sec_per_step <= 0 or n_parallel < 1:
        raise ValueError("sec_per_step must be > 0 and n_parallel >= 1")
    return sec_per_step * steps * n_cells / n_parallel / 3600.0


def compare_assumptions(measured: Mapping[str, float]) -> list[str]:
    out = []
    for k, v in measured.items():
        if k not in ASSUMPTIONS:
            raise KeyError(f"{k} is not a recorded assumption")
        a = ASSUMPTIONS[k]
        if abs(v - a) > ASSUMPTION_REL_TOL * abs(a):
            out.append(
                f"{k}: measured {v:.4g} vs assumed {a:.4g} (power analysis no longer applies)"
            )
    return out
