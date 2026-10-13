"""Hash lock for the selora metric surface (same mechanism as indbw.metric_hash).

Changing a definition in sae.py / probes.py / evals.py / stats.py without bumping
`SELORA_METRIC_VERSION` fails `tests/unit/test_selora_metric_hash.py`. Regenerate the
lock deliberately with `python scripts/check_selora_metric_hash.py --update` *after*
bumping the version.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Final

from indbw.metric_hash import normalized_definitions
from indbw.selora.prereg import SELORA_METRIC_VERSION

SELORA_METRIC_MODULES: Final[tuple[str, ...]] = ("sae.py", "probes.py", "evals.py", "stats.py")
SRC: Final[Path] = Path(__file__).resolve().parent
LOCK: Final[Path] = Path(__file__).resolve().parents[3] / "selora_metric_hash.json"


def compute() -> dict[str, object]:
    defs = {
        m: {
            n: hashlib.sha256(d.encode()).hexdigest()[:16]
            for n, d in sorted(normalized_definitions((SRC / m).read_text()).items())
        }
        for m in SELORA_METRIC_MODULES
    }
    return {
        "metric_version": SELORA_METRIC_VERSION,
        "aggregate": hashlib.sha256(json.dumps(defs, sort_keys=True).encode()).hexdigest(),
        "definitions": defs,
    }


def check() -> list[str]:
    """Problems vs the committed lock; empty means the surface matches its version."""
    lock = json.loads(LOCK.read_text())
    cur = compute()
    if cur["aggregate"] == lock["aggregate"]:
        return []
    changed = sorted(
        f"{m}:{n}"
        for m, d in cur["definitions"].items()  # type: ignore[attr-defined]
        for n, h in d.items()
        if lock["definitions"].get(m, {}).get(n) != h
    )
    if cur["metric_version"] == lock["metric_version"]:
        return [
            f"selora metric definitions changed ({changed}) without bumping SELORA_METRIC_VERSION"
        ]
    return [f"lock is stale for version {cur['metric_version']}: run --update ({changed})"]


def update() -> None:
    LOCK.write_text(json.dumps(compute(), indent=2, sort_keys=True) + "\n")
