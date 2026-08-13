"""Results record schema and METRIC_VERSION.

A record is invalid without: null tested, pre-registered criterion,
observed value, verdict, METRIC_VERSION (PROJECT.md §9). The verdict
must be recomputable from the criterion and observed value -- CI
recomputes it in tier 2.5 (CLAUDE.md, "Computable verdicts").

A pre-registered criterion is one or more machine-evaluable comparisons
against a named entry in `observed` (CLAUDE.md: "every falsification
criterion... must be expressible as a comparison a machine can
evaluate"). Compound gates (e.g. G0: "fail if no transition, OR bracket
wider than 3") are one `Criterion` per clause, ANDed together --
`recomputed_verdict` is "pass" only if every clause holds.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

METRIC_VERSION = "0.1.0"

_Op = Literal["<=", "<", ">=", ">", "=="]

_OPS: dict[_Op, Any] = {
    "<=": lambda v, t: v <= t,
    "<": lambda v, t: v < t,
    ">=": lambda v, t: v >= t,
    ">": lambda v, t: v > t,
    "==": lambda v, t: v == t,
}


@dataclass(frozen=True)
class Criterion:
    """One machine-evaluable comparison: observed[metric] <op> threshold."""

    metric: str
    op: _Op
    threshold: float

    def holds(self, observed: dict[str, Any]) -> bool:
        if self.metric not in observed:
            raise KeyError(f"criterion references {self.metric!r}, not present in observed")
        if self.op not in _OPS:
            raise ValueError(f"unknown comparison op {self.op!r}")
        value = observed[self.metric]
        return bool(_OPS[self.op](value, self.threshold))


@dataclass(frozen=True)
class ResultsRecord:
    """A schema-valid results record (PROJECT.md §9, CLAUDE.md "Provenance").

    Every field here is mandatory -- there is no default for any of
    them, so constructing a record with a missing field raises
    TypeError immediately rather than producing a record that looks
    valid but is missing provenance.
    """

    row: str  # status-board row id, e.g. "G0"
    null_tested: str
    criteria: tuple[Criterion, ...]
    observed: dict[str, Any]
    verdict: Literal["pass", "fail"]
    metric_version: str
    git_sha: str
    run_config_hash: str
    seed: int
    checkpoint_revision: str
    eval_set_hash: str
    torch_version: str
    numpy_version: str
    transformer_lens_version: str
    wall_clock_s: float
    hardware: str

    def recomputed_verdict(self) -> Literal["pass", "fail"]:
        return "pass" if all(c.holds(self.observed) for c in self.criteria) else "fail"

    def is_self_consistent(self) -> bool:
        """True iff the stored verdict matches what recomputing it now yields."""
        return self.verdict == self.recomputed_verdict()

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def to_json_line(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def record_from_dict(d: dict[str, Any]) -> ResultsRecord:
    """Reconstruct a ResultsRecord from a plain dict (e.g. parsed JSON).

    Raises TypeError if any mandatory field is missing -- dataclass
    construction with no defaults is the validator (CLAUDE.md: "CI
    rejects records missing any field").
    """
    d = dict(d)
    d["criteria"] = tuple(Criterion(**c) for c in d["criteria"])
    return ResultsRecord(**d)


def load_records(path: str | Path) -> list[ResultsRecord]:
    records = []
    for line in Path(path).read_text().splitlines():
        line = line.strip()
        if not line:
            continue
        records.append(record_from_dict(json.loads(line)))
    return records


def append_record(path: str | Path, record: ResultsRecord) -> None:
    """Append one record to a results JSONL file, durably (CLAUDE.md resumability)."""
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a") as f:
        f.write(record.to_json_line() + "\n")
        f.flush()
        os.fsync(f.fileno())
