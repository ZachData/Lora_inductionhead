"""Tier-0 check for the selora metric surface. `--update` rewrites the lock."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from indbw.selora.metric_hash import check, update

if __name__ == "__main__":
    if "--update" in sys.argv:
        update()
        print("selora_metric_hash.json written")
        sys.exit(0)
    problems = check()
    for p in problems:
        print(p)
    sys.exit(1 if problems else 0)
