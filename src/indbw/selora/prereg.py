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
from typing import Final

from indbw.schema import Criterion

SELORA_METRIC_VERSION: Final[str] = "0.1.0"
PREREG_STATUS: Final[str] = "draft"

ALPHA: Final[float] = 0.05
FVU_MARGIN: Final[float] = 0.05  # SAE "still valid": FVU may rise by at most 5 points
FVU_ABSOLUTE: Final[float] = 0.15  # and stay under 15% (the report's headline bound)
ACC_MARGIN: Final[float] = 0.03  # non-inferiority margin, accuracy
TOPK: Final[int] = 200
N_RANDOM_DRAWS: Final[int] = 24  # floor 1/25 = 0.04 < ALPHA, so the rank test can reject
PEAK_RANGE: Final[tuple[int, int]] = (6, 10)
PEAK_STABILITY: Final[float] = 0.8


@dataclass(frozen=True)
class Claim:
    id: str
    statement: str
    null: str
    criteria: tuple[Criterion, ...]
    family: str  # multiplicity family; Holm is applied within it
    controls: tuple[str, ...]  # arms/nulls that must accompany the claim


CLAIMS: Final[tuple[Claim, ...]] = (
    Claim(
        "S0",
        "Instrument validity: SAE FVU / reconstruction is unchanged below the first adapted layer",
        "residual stream below the first adapted layer differs under LoRA",
        (Criterion("max_abs_resid_diff_below_first_adapted", "==", 0.0),),
        "S0",
        ("late-layer arm",),
    ),
    Claim(
        "S1a",
        "Gemma Scope SAEs remain valid on LoRA-tuned models (non-inferiority on FVU)",
        "fine-tuning raises FVU by >= FVU_MARGIN at some adapted layer",
        (
            Criterion("n_cells_holm_p_above_alpha", "==", 0.0),
            Criterion("max_fvu_ft_upper95", "<=", FVU_ABSOLUTE),
        ),
        "S1a",
        ("matched-norm random LoRA", "no-LoRA noise floor (prompt-split)"),
    ),
    Claim(
        "S1b",
        "The SAE shift is specific to the learned update, not its norm",
        "trained and matched-norm random LoRA shift FVU equally",
        (Criterion("n_cells_holm_p_above_alpha", "==", 0.0),),
        "S1b",
        ("matched-norm random LoRA",),
    ),
    Claim(
        "S2a",
        "Math vs general prompts separate in SAE space beyond a label-permutation null",
        "class labels are exchangeable within length strata",
        (Criterion("n_layers_holm_p_below_alpha", ">=", 1.0),),
        "S2a",
        ("length-stratified permutation", "numeric-prose control class"),
    ),
    Claim(
        "S2b",
        "Divergence peaks at layers 6-10 (the old layer-8 result), at top-k = 200",
        "peak layer of divergence excess is not concentrated in the early-middle layers",
        (Criterion("peak_in_range_frequency", ">=", PEAK_STABILITY),),
        "S2b",
        ("top-100 sensitivity", "numeric-prose control class"),
    ),
    Claim(
        "S3a",
        "Late-13 LoRA is non-inferior to all-26 on GSM8K (instruct model)",
        "late-13 is worse than all-26 by >= ACC_MARGIN",
        (
            Criterion("min_seed_lower95_diff", ">", -ACC_MARGIN),
            Criterion("holm_p", "<=", ALPHA),
        ),
        "S3",
        ("no-LoRA base", "3 seeds", "full 1319-item test set"),
    ),
    Claim(
        "S3b",
        "Late-13 retains MMLU (logprob) better than random 13-layer placements",
        "late-13 is one more exchangeable draw among 13-of-26 layer sets",
        (Criterion("holm_p", "<=", ALPHA),),
        "S3",
        ("24 random draws", "no-LoRA base"),
    ),
    Claim(
        "S4",
        "Below-chance generation-scored MMLU is a formatting artefact",
        "parseable-answer accuracy is also below chance",
        (
            Criterion("acc_parseable_lower95", ">=", 0.25),
            Criterion("acc_logprob_lower95", ">", 0.25),
        ),
        "S4",
        ("same model scored both ways",),
    ),
    Claim(
        "S5",
        "SAE-guided top-13 placement beats random placements on GSM8K",
        "SAE-top-13 is one more exchangeable draw among 13-of-26 layer sets",
        (Criterion("holm_p", "<=", ALPHA),),
        "S5",
        ("24 random draws", "sae_bottom arm", "sae scores frozen before training"),
    ),
)


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
        ]
    )
    return hashlib.sha256((payload + consts).encode()).hexdigest()[:16]


def require_registered() -> None:
    if PREREG_STATUS != "registered":
        raise RuntimeError(
            "selora pre-registration is still 'draft': record the human sign-off in "
            "PROJECT.md section 10 and set PREREG_STATUS='registered' before any real run"
        )
