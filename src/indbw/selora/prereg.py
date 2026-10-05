"""Pre-registered claims for the selective-LoRA x SAE extension (PROJECT.md section 13).

Every criterion is a `Criterion` (observed[metric] <op> threshold), so the
verdict is recomputable by CI exactly as for the G-gates. `PREREG_STATUS`
is "draft" until a human records sign-off in PROJECT.md section 10; the
real-run entry point refuses to run while it is "draft", and no threshold
below may be edited after "registered" -- a change is a new claim id.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Final, Literal

from indbw.schema import Criterion
from indbw.selora.stats import min_attainable_p

SELORA_METRIC_VERSION: Final[str] = "0.2.0"
PREREG_STATUS: Final[str] = "draft"

ALPHA: Final[float] = 0.05
FVU_MARGIN: Final[float] = 0.05  # SAE "still valid": FVU may rise by at most 5 points
FVU_ABSOLUTE: Final[float] = 0.15  # and stay under 15% (the report's headline bound)
ACC_MARGIN: Final[float] = 0.03  # non-inferiority margin, accuracy
TOPK: Final[int] = 200
N_RANDOM_DRAWS: Final[int] = 24  # floor 1/25 = 0.04 < ALPHA, so the rank test can reject
PEAK_RANGE: Final[tuple[int, int]] = (6, 10)
PEAK_STABILITY: Final[float] = 0.8
N_PERM: Final[int] = 4000  # sign-flip permutations per cell
N_BOOT: Final[int] = 4000  # bootstrap resamples per test
N_PERM_DIVERGENCE: Final[int] = 2000  # label permutations per layer
N_LAYERS: Final[int] = 26
MMLU_GEN_N: Final[int] = 500  # generation scoring subset; log-prob scoring uses the full test set

#: What the power simulations assume (tests/unit/test_selora_design.py). Compared against the
#: first real numbers by preflight before any verdict is read (STATE.md practice 3).
ASSUMPTIONS: dict[str, float] = {
    "gsm8k_accuracy": 0.38,  # source report, Gemma-2-2B-IT arms (~38%)
    "gsm8k_n_items": 1319,
    "gsm8k_discordance_low": 0.15,  # fraction of items two arms answer differently
    "gsm8k_discordance_high": 0.25,
    "seed_item_correlation": 0.5,  # how much of that disagreement repeats across seeds
    "fvu_delta_sd": 0.02,  # per-prompt sd of FVU_ft - FVU_base
    "sae_prompts": 1024,
}

Multiplicity = Literal["iut", "holm", "gatekeep", "single"]


@dataclass(frozen=True)
class Claim:
    id: str
    statement: str
    null: str
    criteria: tuple[Criterion, ...]
    multiplicity: Multiplicity  # iut: every cell at alpha; holm: any-of; gatekeep: GATE_SEQUENCE
    controls: tuple[str, ...]  # arms/nulls that must accompany the claim


CLAIMS: Final[tuple[Claim, ...]] = (
    Claim(
        "S0",
        "Instrument validity: the residual stream is unchanged below the first adapted layer",
        "residual stream below the first adapted layer differs under LoRA",
        (Criterion("max_abs_resid_diff_below_first_adapted", "==", 0.0),),
        "single",
        ("late-layer arm",),
    ),
    Claim(
        "S1a",
        "Gemma Scope SAEs remain valid on LoRA-tuned models (non-inferiority on FVU)",
        "fine-tuning raises FVU by >= FVU_MARGIN at some adapted layer",
        (
            Criterion("n_cells_p_above_alpha", "==", 0.0),
            Criterion("max_fvu_ft_upper95", "<=", FVU_ABSOLUTE),
        ),
        "iut",
        ("spectrum-matched random LoRA", "seed-to-seed noise floor"),
    ),
    Claim(
        "S1b",
        "The SAE shift is specific to the learned update, not its spectrum",
        "trained and spectrum-matched random LoRA shift FVU equally",
        (Criterion("n_cells_p_above_alpha", "==", 0.0),),
        "iut",
        ("spectrum-matched random LoRA",),
    ),
    Claim(
        "S2a",
        "Math vs general prompts separate in SAE space beyond a label-permutation null",
        "class labels are exchangeable within length x digit-count strata",
        (Criterion("n_layers_holm_p_below_alpha", ">=", 1.0),),
        "holm",
        ("length x digit stratified permutation",),
    ),
    Claim(
        "S2b",
        "Divergence peaks at layers 6-10 (the old layer-8 result), at top-k = 200",
        "peak layer of divergence excess is not concentrated in the early-middle layers",
        (Criterion("peak_in_range_frequency", ">=", PEAK_STABILITY),),
        "single",
        ("top-100 sensitivity", "length x digit stratified permutation"),
    ),
    Claim(
        "S3a",
        "Late-13 LoRA is non-inferior to all-26 on GSM8K",
        "late-13 is worse than all-26 by >= ACC_MARGIN",
        (
            Criterion("pooled_lower95_diff", ">", -ACC_MARGIN),
            Criterion("p_adjusted", "<=", ALPHA),
        ),
        "gatekeep",
        ("no-LoRA base", "3 seeds", "full 1319-item test set"),
    ),
    Claim(
        "S3b",
        "Late-13 retains MMLU (log-prob, full test set) better than random 13-layer placements",
        "late-13 is one more exchangeable draw among 13-of-26 layer sets",
        (Criterion("p_adjusted", "<=", ALPHA),),
        "gatekeep",
        ("24 random draws", "no-LoRA base", "same-seed comparison"),
    ),
    Claim(
        "S4",
        "Below-chance generation-scored MMLU is a formatting artefact",
        "parseable-answer accuracy is also below chance",
        (
            Criterion("acc_parseable_lower95", ">=", 0.25),
            Criterion("acc_logprob_lower95", ">", 0.25),
        ),
        "single",
        ("same model scored both ways",),
    ),
    Claim(
        "S5",
        "SAE-guided top-13 placement beats random placements on GSM8K",
        "SAE-top-13 is one more exchangeable draw among 13-of-26 layer sets",
        (Criterion("p_value", "<=", ALPHA),),
        "single",
        (
            "24 random draws",
            "sae_bottom arm",
            "sae scores frozen before training",
            "same-seed comparison",
        ),
    ),
)

#: Fixed-sequence order for the gatekeep family: primary (instruct) model first, task gain
#: before retention. Testing stops at the first non-rejection.
GATE_SEQUENCE: Final[tuple[tuple[str, str], ...]] = (
    ("S3a", "google/gemma-2-2b-it"),
    ("S3b", "google/gemma-2-2b-it"),
    ("S3a", "google/gemma-2-2b"),
    ("S3b", "google/gemma-2-2b"),
)


@dataclass(frozen=True)
class TestSpec:
    """Smallest p the claim's test can return, and the multiplicity rule it must clear."""

    floor: float
    multiplicity: Multiplicity
    family_size: int = 1

    def threshold(self) -> float:
        return ALPHA / self.family_size if self.multiplicity == "holm" else ALPHA


TEST_SPECS: Final[dict[str, TestSpec]] = {
    "S1a": TestSpec(min_attainable_p(N_BOOT), "iut"),
    "S1b": TestSpec(min_attainable_p(N_PERM), "iut"),
    "S2a": TestSpec(min_attainable_p(N_PERM_DIVERGENCE), "holm", N_LAYERS),
    "S2b": TestSpec(0.0, "single"),  # frequency criterion, no p-value floor
    "S3a": TestSpec(min_attainable_p(N_BOOT), "gatekeep"),
    "S3b": TestSpec(min_attainable_p(N_RANDOM_DRAWS), "gatekeep"),
    "S4": TestSpec(0.0, "single"),  # bootstrap bounds, no p-value floor
    "S5": TestSpec(min_attainable_p(N_RANDOM_DRAWS), "single"),
}


def feasibility_problems(specs: dict[str, TestSpec] | None = None) -> list[str]:
    """Claims whose test cannot reach the threshold its multiplicity rule imposes."""
    out = []
    for cid, spec in (TEST_SPECS if specs is None else specs).items():
        if spec.floor > spec.threshold():
            out.append(
                f"{cid}: smallest attainable p {spec.floor:.4g} > threshold {spec.threshold():.4g}"
                f" ({spec.multiplicity}, family of {spec.family_size})"
            )
    return out


def claim(claim_id: str) -> Claim:
    for c in CLAIMS:
        if c.id == claim_id:
            return c
    raise KeyError(claim_id)


def prereg_hash() -> str:
    """Hash of every claim and threshold; recorded in each result so edits are visible."""
    payload = json.dumps([asdict(c) for c in CLAIMS], sort_keys=True, default=str)
    consts = json.dumps(
        [
            ALPHA,
            FVU_MARGIN,
            FVU_ABSOLUTE,
            ACC_MARGIN,
            TOPK,
            N_RANDOM_DRAWS,
            PEAK_RANGE,
            PEAK_STABILITY,
            N_PERM,
            N_BOOT,
            N_PERM_DIVERGENCE,
            MMLU_GEN_N,
            GATE_SEQUENCE,
            ASSUMPTIONS,
        ],
        sort_keys=True,
    )
    return hashlib.sha256((payload + consts).encode()).hexdigest()[:16]


def require_registered() -> None:
    if PREREG_STATUS != "registered":
        raise RuntimeError(
            "selora pre-registration is still 'draft': record the human sign-off in "
            "PROJECT.md section 10 and set PREREG_STATUS='registered' before any real run"
        )
