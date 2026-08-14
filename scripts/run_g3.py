"""G3: positive control -- generous rank on W_Q alone reaches criterion?

PROJECT.md §8 gate. "Generous rank on $W_Q$ alone. Does $R$ exceed 0.80
at all? Fail: the single-matrix composition rule (§3) is too
restrictive; fall back to per-circuit adaptation and downgrade the H2
claim, or the objective/loop is broken."

This is *not* a second training run. scripts/run_g2.py's timing run
already is a generous-rank (r=64=d_head), $W_Q$-only, checkpoint-A
adaptation -- exactly what G3 asks for, just framed around "how long"
instead of "does it happen at all" (see run_g2.py's own module
docstring and PROJECT.md §10's 2026-08-14 decision log entries).
Retraining identically to answer a question the first run already
answered would burn the compute-hours CLAUDE.md's provenance/
resumability sections exist to conserve. This script reads G2's raw
artifact (data/g2_generous_rank_run.json) and emits G3's own,
independently schema-validated results record from it.

If that artifact doesn't exist yet (G2 hasn't run), this raises rather
than silently fabricating a result.

Usage:
    python scripts/run_g3.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_ARTIFACT_PATH = REPO_ROOT / "data" / "g2_generous_rank_run.json"
OUT_PATH = REPO_ROOT / "results" / "g3_positive_control.jsonl"

sys.path.insert(0, str(REPO_ROOT / "src"))

CRITERION_R = 0.80  # PROJECT.md §6, same threshold G2's run was already evaluated against


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def package_version(name: str) -> str:
    import importlib.metadata

    return importlib.metadata.version(name)


def main() -> None:
    from indbw.schema import METRIC_VERSION, Criterion, ResultsRecord, append_record

    if not RAW_ARTIFACT_PATH.exists():
        raise SystemExit(
            f"{RAW_ARTIFACT_PATH} not found -- run scripts/run_g2.py first. "
            "G3 reads G2's raw run artifact rather than retraining (see this "
            "script's module docstring)."
        )
    raw = json.loads(RAW_ARTIFACT_PATH.read_text())

    if raw["budget_exceeded"]:
        raise SystemExit(
            "G2's run hit its wall-clock budget before finishing -- no usable "
            "final_recovery_canonical to evaluate G3 against. Re-run G2 with a "
            "larger budget, or run a dedicated G3 training run, before closing G3."
        )

    observed = {
        "final_recovery_canonical": raw["final_recovery_canonical"],
        "steps_run": float(raw["steps_run"]),
    }
    criteria = (Criterion(metric="final_recovery_canonical", op=">=", threshold=CRITERION_R),)
    verdict = "pass" if all(c.holds(observed) for c in criteria) else "fail"

    eval_set_hash = hashlib.sha256(
        json.dumps(
            {"n_eval": 512, "T": raw["config"]["T"], "seed": raw["config"]["eval_seed"]},
            sort_keys=True,
        ).encode()
    ).hexdigest()[:16]

    record = ResultsRecord(
        row="G3",
        null_tested="generous-rank (r=64=d_head) adaptation of W_Q alone, from checkpoint A, "
        "does not reach recovery R>=0.80 -- the single-matrix composition rule (PROJECT.md §3) "
        "would be too restrictive, or the objective/loop is broken",
        criteria=criteria,
        observed=observed,
        verdict=verdict,
        metric_version=METRIC_VERSION,
        git_sha=git_sha(),
        run_config_hash=raw["run_config_hash"],  # same run as G2 -- shared by design, see docstring
        seed=raw["config"]["train_seed"],
        checkpoint_revision=f"step {raw['config']['a_step']} (A)",
        eval_set_hash=eval_set_hash,
        torch_version=package_version("torch"),
        numpy_version=package_version("numpy"),
        transformer_lens_version=package_version("transformer_lens"),
        wall_clock_s=raw["wall_clock_s"],  # G2's run wall-clock, reused -- this row did not retrain
        hardware=f"{platform.machine()}/{platform.processor() or 'unknown'}",
    )
    append_record(OUT_PATH, record)

    print(f"G3 verdict: {verdict}")
    print(
        f"  final_recovery={raw['final_recovery_canonical']:.4f} "
        f"(criterion: >= {CRITERION_R}) steps_run={raw['steps_run']}"
    )
    print(f"  self-consistent: {record.is_self_consistent()}")
    print(f"  reused G2's run (run_config_hash={raw['run_config_hash']}, no retraining)")
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
