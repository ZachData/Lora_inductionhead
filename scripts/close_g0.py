"""Close out the G0 gate: merge sweep telemetry, locate the transition,
emit a schema-valid results record (PROJECT.md §8/§9).

data/g0_sweep*.jsonl (the shared file plus one per worker, see
scripts/g0_sweep.py) is raw per-checkpoint telemetry -- not itself a
results record. This script is the one-time step that turns that
telemetry into the actual G0 verdict: merge every shard, run
indbw.gates.locate_transition over the full grid, and append the
result to results/g0_transition.jsonl via indbw.schema.

Idempotent to run again (e.g. after more telemetry lands): it always
reads the full current merge and appends a fresh record rather than
mutating an old one, matching the append-only results/ convention.

Usage:
    python scripts/close_g0.py
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
OUT_PATH = REPO_ROOT / "results" / "g0_transition.jsonl"

sys.path.insert(0, str(REPO_ROOT / "src"))

# Must match scripts/g0_sweep.py's defaults -- every worker ran with no
# --n-eval/--T/--seed override (infra/worker-bootstrap.sh), so the whole
# grid was swept under PROJECT.md §5's pre-registered eval set.
N_EVAL = 512
T = 128
SEED = 0


def merge_sweep_records() -> dict[int, dict]:
    merged: dict[int, dict] = {}
    for path in sorted(DATA_DIR.glob("g0_sweep*.jsonl")):
        for line in path.read_text().splitlines():
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            merged[rec["step"]] = rec
    return merged


def git_sha() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()


def package_version(name: str) -> str:
    import importlib.metadata

    return importlib.metadata.version(name)


def main() -> None:
    from indbw.gates import locate_transition
    from indbw.models import checkpoint_steps
    from indbw.schema import METRIC_VERSION, Criterion, ResultsRecord, append_record

    merged = merge_sweep_records()
    all_steps = checkpoint_steps()
    missing = [s for s in all_steps if s not in merged]
    if missing:
        raise SystemExit(
            f"G0 sweep incomplete: {len(missing)}/{len(all_steps)} checkpoints missing "
            f"from data/g0_sweep*.jsonl (e.g. {missing[:5]}) -- refusing to close the gate "
            f"on a partial grid."
        )

    steps = sorted(merged)
    max_pms = [merged[s]["max_pms"] for s in steps]
    icl = [merged[s]["icl"] for s in steps]
    total_wall_clock = sum(merged[s]["wall_clock_s"] for s in steps)

    result = locate_transition(steps, max_pms, icl)

    observed = {
        "found": int(result.found),
        "bracket_width": result.bracket_width if result.found else float("inf"),
        "a_step": result.a_step,
        "b_step": result.b_step,
    }
    criteria = (
        Criterion(metric="found", op="==", threshold=1),
        Criterion(metric="bracket_width", op="<=", threshold=3),
    )
    verdict = "pass" if all(c.holds(observed) for c in criteria) else "fail"

    config = {
        "n_eval": N_EVAL,
        "T": T,
        "seed": SEED,
        "n_checkpoints": len(steps),
        "locate_transition_defaults": True,
    }
    run_config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:16]
    eval_set_hash = hashlib.sha256(
        json.dumps({"n_eval": N_EVAL, "T": T, "seed": SEED}, sort_keys=True).encode()
    ).hexdigest()[:16]

    record = ResultsRecord(
        row="G0",
        null_tested="no induction transition exists on the pythia-70m checkpoint grid, "
        "or the transition bracket spans more than 3 consecutive checkpoints",
        criteria=criteria,
        observed=observed,
        verdict=verdict,
        metric_version=METRIC_VERSION,
        git_sha=git_sha(),
        run_config_hash=run_config_hash,
        seed=SEED,
        checkpoint_revision=f"all {len(steps)} grid steps (full sweep, step {steps[0]}-{steps[-1]})",
        eval_set_hash=eval_set_hash,
        torch_version=package_version("torch"),
        numpy_version=package_version("numpy"),
        transformer_lens_version=package_version("transformer_lens"),
        wall_clock_s=total_wall_clock,
        hardware=f"{platform.machine()}/{platform.processor() or 'unknown'}",
    )

    append_record(OUT_PATH, record)
    print(f"G0 verdict: {verdict}")
    print(f"  a_step={result.a_step} b_step={result.b_step} bracket_width={result.bracket_width}")
    print(f"  self-consistent: {record.is_self_consistent()}")
    print(f"  wrote {OUT_PATH}")


if __name__ == "__main__":
    main()
