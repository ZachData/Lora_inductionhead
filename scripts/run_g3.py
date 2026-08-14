"""G3: positive control -- generous rank on W_Q alone reaches criterion.

PROJECT.md §8 gate: "Generous rank on $W_Q$ alone. Does $R$ exceed 0.80
at all? **Fail:** the single-matrix composition rule (§3) is too
restrictive; fall back to per-circuit adaptation and downgrade the H2
claim, or the objective/loop is broken."

**No training happens here.** G2's timing run *is* this experiment --
generous-rank (r = d_head = 64) QK-arm adaptation of W_Q from checkpoint
A, asking whether R reaches 0.80 -- measured under a different framing
(how long vs. whether at all). scripts/run_g2.py was written to leave
behind `data/g2_generous_rank_run.json` precisely so this row closes
without paying for the same generous-rank run twice (PROJECT.md §10,
2026-08-14). This script reads that artifact plus G2's own schema-valid
results record, and emits G3's separately-validated record.

Two things it refuses to do rather than produce a verdict from:

  - **A missing artifact.** G2 writes no artifact when it exits on
    `TrainingBudgetExceeded`. A budget-exceeded G2 has not answered
    G3's question -- it timed out, which is not the same as the control
    failing -- so there is nothing here to close the gate with.
  - **A non-finite recovery.** `Criterion(">=", 0.80)` evaluates False
    on NaN, which would silently turn "the run produced no number" into
    "the positive control failed" -- and a failed G3 stops the phase
    (CLAUDE.md, "A gate that fails stops the phase"). That is the most
    expensive false negative available here, so it raises instead.

It also cross-checks the artifact against G2's record: two files written
by the same run disagreeing about its recovery means one of them is
wrong, and neither can then be trusted to close a gate.

Usage:
    python scripts/run_g3.py
"""

from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from indbw.schema import (
    METRIC_VERSION,
    Criterion,
    ResultsRecord,
    append_record,
    load_records,
)

G2_ARTIFACT_PATH = REPO_ROOT / "data" / "g2_generous_rank_run.json"
G2_RECORD_PATH = REPO_ROOT / "results" / "g2_cpu_timing.jsonl"
OUT_PATH = REPO_ROOT / "results" / "g3_positive_control.jsonl"

#: PROJECT.md §6's success band. Pre-registered; never edit here.
CRITERION_R = 0.80

NULL_TESTED = (
    "generous-rank (r = d_head = 64) adaptation of W_Q alone, from checkpoint A, "
    "cannot reach recovery R >= 0.80 -- i.e. the single-matrix composition rule "
    "(PROJECT.md §3) is too restrictive, or the objective/loop is broken"
)


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def load_g2_record(path: Path = G2_RECORD_PATH) -> ResultsRecord:
    """The most recent G2 record. Raises if none exists or none is for G2."""
    if not path.exists():
        raise SystemExit(
            f"no G2 results record at {path} -- G2 has not been closed, so G3 has nothing "
            "to read. Run scripts/run_g2.py first."
        )
    g2_records = [r for r in load_records(path) if r.row == "G2"]
    if not g2_records:
        raise SystemExit(f"{path} contains no row=='G2' record")
    return g2_records[-1]


def load_g2_artifact(path: Path = G2_ARTIFACT_PATH) -> dict[str, Any]:
    if not path.exists():
        raise SystemExit(
            f"no G2 run artifact at {path}. G2 writes this only on a run that completed "
            "(it is skipped on a TrainingBudgetExceeded exit), so either G2 has not run or "
            "it timed out. A timed-out G2 has not answered G3's question and must not be "
            "read as a G3 failure -- re-run G2 with a larger budget."
        )
    data: dict[str, Any] = json.loads(path.read_text())
    return data


def build_g3_record(g2_record: ResultsRecord, artifact: dict[str, Any], sha: str) -> ResultsRecord:
    """G3's results record, derived from G2's already-validated one.

    Provenance is inherited from the G2 record (same run, same eval set,
    same checkpoint, same seed, same package versions) except `git_sha`,
    which is this commit -- the record is *emitted* now even though the
    compute happened then, and conflating the two would misattribute the
    analysis code that produced this verdict.
    """
    if artifact.get("budget_exceeded"):
        raise ValueError(
            "G2's artifact is flagged budget_exceeded: the generous-rank run did not finish, "
            "so it cannot close G3 either way"
        )

    recovery_record = g2_record.observed.get("final_recovery_canonical")
    recovery_artifact = artifact.get("final_recovery_canonical")
    if recovery_record is None or recovery_artifact is None:
        raise ValueError(
            "final_recovery_canonical missing from G2's record or artifact -- "
            "G3's observed value has no source"
        )
    if not math.isfinite(float(recovery_artifact)):
        raise ValueError(
            f"G2 reported a non-finite recovery ({recovery_artifact}). A criterion comparison "
            "against NaN evaluates False, which would record 'the positive control failed' "
            "when what actually happened is 'the run produced no number'. Fix the run first."
        )
    if not math.isclose(float(recovery_record), float(recovery_artifact), rel_tol=1e-12):
        raise ValueError(
            f"G2's record ({recovery_record}) and artifact ({recovery_artifact}) disagree about "
            "final_recovery_canonical -- one of them was written wrong, and neither can close a gate"
        )
    if g2_record.run_config_hash != artifact.get("run_config_hash"):
        raise ValueError(
            f"G2's record ({g2_record.run_config_hash}) and artifact "
            f"({artifact.get('run_config_hash')}) describe different runs"
        )

    observed: dict[str, Any] = {
        "final_recovery_canonical": float(recovery_artifact),
        "rank": float(artifact["config"]["rank"]),
        "steps_run": float(artifact["steps_run"]),
        "reached_criterion_flag": 1.0 if artifact["reached_criterion"] else 0.0,
    }
    criteria = (Criterion(metric="final_recovery_canonical", op=">=", threshold=CRITERION_R),)
    verdict: Any = "pass" if all(c.holds(observed) for c in criteria) else "fail"

    return ResultsRecord(
        row="G3",
        null_tested=NULL_TESTED,
        criteria=criteria,
        observed=observed,
        verdict=verdict,
        metric_version=METRIC_VERSION,
        git_sha=sha,
        run_config_hash=g2_record.run_config_hash,
        seed=g2_record.seed,
        checkpoint_revision=g2_record.checkpoint_revision,
        eval_set_hash=g2_record.eval_set_hash,
        torch_version=g2_record.torch_version,
        numpy_version=g2_record.numpy_version,
        transformer_lens_version=g2_record.transformer_lens_version,
        wall_clock_s=float(artifact["wall_clock_s"]),
        hardware=g2_record.hardware,
    )


def main() -> int:
    g2_record = load_g2_record()
    artifact = load_g2_artifact()
    record = build_g3_record(g2_record, artifact, git_sha())
    append_record(OUT_PATH, record)

    print(f"G3 verdict: {record.verdict}")
    print(
        f"  R={record.observed['final_recovery_canonical']:.4f} "
        f"(criterion R >= {CRITERION_R}) at rank {int(record.observed['rank'])}, "
        f"{int(record.observed['steps_run'])} steps"
    )
    print(f"  reused G2's run {record.run_config_hash} -- no retraining")
    print(f"  self-consistent: {record.is_self_consistent()}")
    print(f"  wrote {OUT_PATH}")
    if record.verdict == "fail":
        print(
            "  G3 FAILED. PROJECT.md §8: the single-matrix composition rule may be too "
            "restrictive, or the objective/loop is broken. A gate that fails stops the "
            "phase -- record the decision in PROJECT.md §10 before anything else runs."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
