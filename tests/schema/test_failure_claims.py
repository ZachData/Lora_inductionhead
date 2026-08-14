"""Tier 2.5: no failure claim in results/ without its mandatory controls.

CLAUDE.md's falsification rules: "No failure claim without its controls
attached. A rank cell that failed must report the matched-lr arm and the
10x-steps arm alongside it, or the result is about the optimizer and
cannot be reported as capacity."

Runs over every record in results/, in the same spirit as the verdict
recomputation beside it: a control that was owed and never run must show
up as a red build on *every* subsequent push, not just the one that
recorded the failing cell. Until sweep records exist this is a no-op --
gate records carry no arm/rank and are ignored by the gate.
"""

from __future__ import annotations

from pathlib import Path

from indbw.schema import ResultsRecord, load_records
from indbw.sweep import check_failure_claims

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _all_records() -> list[ResultsRecord]:
    if not RESULTS_DIR.exists():
        return []
    records: list[ResultsRecord] = []
    for path in sorted(RESULTS_DIR.glob("*.jsonl")):
        records.extend(load_records(path))
    return records


def test_every_failure_claim_has_its_controls() -> None:
    problems = check_failure_claims(_all_records())
    assert not problems, "unsupported failure claim(s) in results/:\n" + "\n".join(
        f"  - {p}" for p in problems
    )
