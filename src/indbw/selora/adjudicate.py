"""Turn observed numbers for a pre-registered claim into a schema-valid results record.

The verdict is *computed* from the claim's own criteria, never passed in, and a
record is refused unless every control the claim names is attached -- the
Popper rule that a result without its nulls is about the pipeline, not the claim.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
from typing import Any, Literal

import numpy as np
import torch

from indbw.schema import ResultsRecord
from indbw.selora.prereg import SELORA_METRIC_VERSION, claim, prereg_hash


def config_hash(config: dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(config, sort_keys=True, default=str).encode()).hexdigest()[:16]


def _git_sha() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def hardware() -> str:
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"
    return f"{platform.machine()}/{gpu}"


def adjudicate(
    claim_id: str,
    observed: dict[str, Any],
    controls_attached: list[str],
    config: dict[str, Any],
    seed: int,
    checkpoint_revision: str,
    eval_set_hash: str,
    wall_clock_s: float,
) -> ResultsRecord:
    c = claim(claim_id)
    missing = [x for x in c.controls if x not in controls_attached]
    if missing:
        raise ValueError(f"{claim_id}: refusing to adjudicate without controls {missing}")
    obs = dict(observed)
    obs["controls_attached"] = sorted(controls_attached)
    obs["prereg_hash"] = prereg_hash()
    for crit in c.criteria:
        v = obs.get(crit.metric)
        if not isinstance(v, int | float) or isinstance(v, bool) or not np.isfinite(v):
            raise ValueError(f"{claim_id}: criterion metric {crit.metric!r} missing or not finite")
    verdict: Literal["pass", "fail"] = "pass" if all(x.holds(obs) for x in c.criteria) else "fail"
    try:
        import transformers

        tfv = f"hf-transformers {transformers.__version__}"
    except ImportError:  # pragma: no cover
        tfv = "n/a"
    return ResultsRecord(
        row=claim_id,
        null_tested=c.null,
        criteria=c.criteria,
        observed=obs,
        verdict=verdict,
        metric_version=SELORA_METRIC_VERSION,
        git_sha=_git_sha(),
        run_config_hash=config_hash(config),
        seed=seed,
        checkpoint_revision=checkpoint_revision,
        eval_set_hash=eval_set_hash,
        torch_version=torch.__version__,
        numpy_version=np.__version__,
        transformer_lens_version=tfv,
        wall_clock_s=wall_clock_s,
        hardware=hardware(),
    )
