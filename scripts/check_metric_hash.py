"""Tier-0 metric-hash gate (CLAUDE.md, "Metric-version enforcement").

    python scripts/check_metric_hash.py            # check; exit 1 on a problem
    python scripts/check_metric_hash.py --update   # regenerate metric_hash.json

The check fails if a metric definition in algebra.py or probes.py moved
without a METRIC_VERSION bump. `--update` refuses to regenerate the
lockfile in exactly that situation, so the only way past the gate is the
deliberate one: bump METRIC_VERSION, regenerate, commit both.

Mechanism and what counts as a change: src/indbw/metric_hash.py.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from indbw.metric_hash import check, update


def main(argv: list[str]) -> int:
    if "--update" in argv:
        try:
            mh = update()
        except ValueError as exc:
            print(f"metric-hash update refused:\n{exc}", file=sys.stderr)
            return 1
        print(f"metric_hash.json written: METRIC_VERSION={mh.metric_version} {mh.aggregate[:16]}")
        return 0

    problems = check()
    if problems:
        print("metric-hash check FAILED:", file=sys.stderr)
        for problem in problems:
            print(f"- {problem}", file=sys.stderr)
        return 1
    print("metric-hash check passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
