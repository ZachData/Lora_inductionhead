"""Run cells as data: one frozen dataclass per training run, hashed, resumable."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

from indbw.selora import prereg
from indbw.selora.placement import named_sets, random_sets, sae_guided_sets

MODELS = ("google/gemma-2-2b", "google/gemma-2-2b-it")
N_LAYERS = 26
N_ADAPT = 13
NAMED_SEEDS = (0, 1, 2)
PRIMARY_MODEL = MODELS[1]  # S3/S5 and the random placements (instruct, where placement mattered)
SAE_MODEL = MODELS[0]  # the model Gemma Scope was trained on: S2 and S4, S1a replication
#: Trained named arms per model and the seeds each needs. Lean design: a cell exists only if a
#: registered claim reads it. S3a pools 3 seeds of all/late; S3b and S5 rank a seed-0 score
#: against seed-0 random draws, so sae_top/sae_bottom need seed 0 only; S4 reads the SAE
#: model's all-layer seed-0 cell. `early` and the secondary model's other arms feed no claim.
ARM_SEEDS: dict[str, dict[str, tuple[int, ...]]] = {
    PRIMARY_MODEL: {"all": NAMED_SEEDS, "late": NAMED_SEEDS, "sae_top": (0,), "sae_bottom": (0,)},
    SAE_MODEL: {"all": (0,)},
}
RANDOM_MODELS = (PRIMARY_MODEL,)
RANDOM_SEED_BASE = 1000  # layer-set draws use this; training seed is separate
INCLUDE_SHUFFLED = False  # optional shuffled-answer control arm (STATE.md C5, needs sign-off)


@dataclass(frozen=True)
class Cell:
    model: str
    arm: str  # all | late | early | sae_top | sae_bottom | random_NN | base
    layers: tuple[int, ...]
    seed: int
    lr: float = 2e-4
    r: int = 8
    alpha: int = 32
    epochs: float = 3.0

    @property
    def run_id(self) -> str:
        return hashlib.sha256(json.dumps(asdict(self), sort_keys=True).encode()).hexdigest()[:16]


def enumerate_cells(
    sae_scores: dict[str, list[float]], include_shuffled: bool = INCLUDE_SHUFFLED
) -> list[Cell]:
    """All training cells. `sae_scores[model]` are per-layer divergence excesses frozen BEFORE training."""
    import numpy as np

    out: list[Cell] = []
    for model in MODELS:
        named = named_sets(N_LAYERS, N_ADAPT)
        guided = sae_guided_sets(np.asarray(sae_scores[model]), N_ADAPT)
        arms = {**named, **guided}
        out.append(Cell(model, "base", (), 0))
        for arm, seeds in ARM_SEEDS[model].items():
            for s in seeds:
                out.append(Cell(model, arm, arms[arm], s))
        if model in RANDOM_MODELS:
            rand = random_sets(
                N_LAYERS,
                N_ADAPT,
                prereg.N_RANDOM_DRAWS,
                RANDOM_SEED_BASE,
                exclude=set(arms.values()),
            )
            for i, layers in enumerate(rand):
                out.append(Cell(model, f"random_{i:02d}", layers, 0))
        if include_shuffled:
            for s in NAMED_SEEDS:
                out.append(Cell(model, "shuffled_answers", named["all"], s))
    if len({c.run_id for c in out}) != len(out):
        raise ValueError("duplicate run ids: two cells are identical")
    return out


def completed_ids(path: str | Path) -> set[str]:
    p = Path(path)
    if not p.exists():
        return set()
    return {json.loads(line)["run_id"] for line in p.read_text().splitlines() if line.strip()}


def pending(cells: Iterable[Cell], results_path: str | Path) -> list[Cell]:
    done = completed_ids(results_path)
    return [c for c in cells if c.run_id not in done]


SAE_REPO = "google/gemma-scope-2b-pt-res"
SAE_WIDTH = "width_16k"
SAE_TARGET_L0 = 70  # fixed before any run; Gemma Scope ships several sparsities per layer


def pick_sae_path(repo_files: Iterable[str], layer: int) -> str:
    """The layer's 16k residual SAE whose average L0 is nearest SAE_TARGET_L0 (ties: lower L0)."""
    import re

    pat = re.compile(rf"^layer_{layer}/{SAE_WIDTH}/average_l0_(\d+)/params\.npz$")
    found = [(int(m.group(1)), f) for f in repo_files if (m := pat.match(f))]
    if not found:
        raise FileNotFoundError(f"no {SAE_WIDTH} SAE for layer {layer} in {SAE_REPO}")
    return min(found, key=lambda t: (abs(t[0] - SAE_TARGET_L0), t[0]))[1]
