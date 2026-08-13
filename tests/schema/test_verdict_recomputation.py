"""Tier 2.5 (CLAUDE.md): recompute every results record's verdict.

Runs over every record in results/, not just new ones -- a
METRIC_VERSION bump or a criterion edit must surface as a failure on
every affected historical record, not just the one just written.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from indbw.schema import load_records

RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"


def _record_files() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.jsonl"))


@pytest.mark.parametrize("path", _record_files(), ids=lambda p: p.name)
def test_every_record_verdict_recomputes(path: Path) -> None:
    records = load_records(path)
    assert records, f"{path.name} contains no records"
    for record in records:
        assert record.is_self_consistent(), (
            f"{path.name} row={record.row!r}: stored verdict {record.verdict!r} != "
            f"recomputed {record.recomputed_verdict()!r} for observed={record.observed}"
        )
