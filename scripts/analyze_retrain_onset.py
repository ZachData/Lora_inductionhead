"""What the 512->2000 retrain's probe trace says about induction onset.

Reads `data/retrain/<run_id>/probe.jsonl` -- the 381-row, stride-4 trace
the retrain wrote on 2026-09-01 and that PROJECT.md did not record until
2026-09-07 (10). Writes a JSON summary and a PNG.

Analysis of already-collected data. No new measurement, no new metric --
`recovery` is PROJECT.md 5's hashed function, applied to the ICL the run
already logged, against the ICL_A/ICL_B constants the committed G0 record
fixed. No verdict: what any of this means is a human call (CLAUDE.md 9).

WHAT THIS IS FOR
----------------
Two things the trace can settle that nothing else in the repo can.

1. **The onset has a shape, and the public grid cannot show it.** Pythia
   publishes step 512 then step 1000, so G0's bracket is coarse because
   the grid is coarse. This trace has 82 points inside that gap.

2. **It separates real induction from prior-flattening.** PROJECT.md 11
   (2026-08-19) flags an unexplained anomaly: in the full-fine-tune
   diagnostic, training loss fell substantially while R declined, and the
   candidate mechanism offered was that second-copy NLL has a large
   non-induction descent direction -- flattening the output distribution
   toward the ln(50304) = 10.83 uniform floor, which lowers *both* copies
   equally and moves ICL by zero. That entry ends: "Untested, and cheap to
   test: log first_copy_nll alongside the training loss -- currently only
   the second half is recorded, so the two cannot be told apart in any
   existing run's loss_history."

   This run logs both halves. It cannot resolve the anomaly *in the G3
   runs* -- different objective, different optimizer state, and the two
   are not comparable run-to-run -- but it does establish what the two
   halves do when induction genuinely forms, which is the missing
   reference the entry was asking for.

Usage:
    python scripts/analyze_retrain_onset.py [--run-id ID] [--out-dir DIR]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

DEFAULT_RUN_ID = "cb25e3f6c2185c1e"

# Fixed by the committed G0 record and reused verbatim by
# diagnose_g3_reachability.py. Re-deriving them here would make this
# trace's R incomparable with every other R in the project.
ICL_A = -0.024705959483981133
ICL_B = 11.512047719210386

# PROJECT.md 5's induction-bearing bar, 6's decision bands, and the
# retrain config's own onset trigger. No new threshold is introduced.
PMS_THRESHOLDS = (0.05, 0.3, 0.5, 0.8, 0.9)
R_THRESHOLDS = (0.5, 0.8)


def load_probe_series(path: Path) -> list[dict[str, Any]]:
    """Probe rows, deduplicated by step and sorted.

    The run was resumed twice and `probe.jsonl` therefore replays a few
    steps: it contains backward step transitions at 868->852 and
    1540->1528. `onset_bracket.json` records that the replays are *not*
    the same trajectory (different train_loss at identical steps, because
    the streaming-data position was not checkpointed), and that session 1
    is the clean pass over the onset bracket.

    First occurrence therefore wins, which keeps the 528-852 window on
    session 1's trajectory as that file recommends. Documented rather
    than silent: taking the last occurrence instead would mix two
    trajectories inside the very window the analysis is about.
    """
    rows: dict[int, dict[str, Any]] = {}
    with path.open() as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.setdefault(int(row["step"]), row)
    if not rows:
        raise ValueError(f"{path} contained no probe rows")
    return [rows[s] for s in sorted(rows)]


def first_crossing(steps: list[int], values: list[float], threshold: float) -> int | None:
    """First step whose value is >= threshold, or None if never reached.

    Matches `gates.locate_transition`'s convention ("first grid step where
    ... crosses") and the retrain config's own `onset_pms` trigger, so the
    onset step this reports is the same number `onset_bracket.json`
    already records. Deliberately *not* interpolated: the trace is a
    noisy finite-eval estimate, and an interpolated crossing would invent
    resolution the stride-4 grid does not have.
    """
    if len(steps) != len(values):
        raise ValueError(f"steps and values disagree: {len(steps)} vs {len(values)}")
    if not steps:
        raise ValueError("empty series")
    for step, value in zip(steps, values, strict=True):
        if value >= threshold:
            return step
    return None


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    from indbw.probes import recovery

    steps = [int(r["step"]) for r in rows]
    pms = [float(r["pms"]) for r in rows]
    icl = [float(r["icl"]) for r in rows]
    nll_1 = [float(r["nll_first"]) for r in rows]
    nll_2 = [float(r["nll_second"]) for r in rows]
    rec = [recovery(v, ICL_A, ICL_B) for v in icl]

    pms_cross = {str(t): first_crossing(steps, pms, t) for t in PMS_THRESHOLDS}
    r_cross = {str(t): first_crossing(steps, rec, t) for t in R_THRESHOLDS}

    # The lag between prefix matching appearing and recovery following.
    # 4 (prefix matching) warns that a lumped metric hides which
    # component is the bottleneck; the localization bisection showed the
    # two dissociate under grafting. This is the same pair measured along
    # a real trajectory instead.
    lag = None
    if pms_cross["0.3"] is not None and r_cross["0.5"] is not None:
        lag = r_cross["0.5"] - pms_cross["0.3"]

    first_idx, last_idx = 0, len(rows) - 1
    return {
        "n_probe_rows": len(rows),
        "step_first": steps[first_idx],
        "step_last": steps[last_idx],
        "pms_crossing_steps": pms_cross,
        "recovery_crossing_steps": r_cross,
        "pms_0p3_to_R_0p5_lag_steps": lag,
        "icl_a_baseline": ICL_A,
        "icl_b_baseline": ICL_B,
        "at_first_step": {
            "step": steps[first_idx],
            "pms": pms[first_idx],
            "icl": icl[first_idx],
            "recovery": rec[first_idx],
            "nll_first": nll_1[first_idx],
            "nll_second": nll_2[first_idx],
        },
        "at_last_step": {
            "step": steps[last_idx],
            "pms": pms[last_idx],
            "icl": icl[last_idx],
            "recovery": rec[last_idx],
            "nll_first": nll_1[last_idx],
            "nll_second": nll_2[last_idx],
        },
        # The flattening discriminator (PROJECT.md 11, 2026-08-19).
        # Flattening lowers both halves together and moves ICL by zero;
        # induction collapses the second half alone. Reported as the two
        # deltas so the reader can see which happened rather than being
        # told.
        "nll_first_delta": nll_1[last_idx] - nll_1[first_idx],
        "nll_second_delta": nll_2[last_idx] - nll_2[first_idx],
        "uniform_floor_nats": 10.825754,  # ln(50304), the chance level
        "max_pms": max(pms),
        "max_recovery": max(rec),
        "series": {
            "step": steps,
            "pms": pms,
            "icl": icl,
            "recovery": rec,
            "nll_first": nll_1,
            "nll_second": nll_2,
            "train_loss": [float(r["train_loss"]) for r in rows],
        },
    }


def write_plot(summary: dict[str, Any], path: Path) -> bool:
    """Best-effort figure. Returns False if matplotlib is unavailable --
    the JSON summary is the artifact of record, the PNG is a convenience."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping the figure", flush=True)
        return False

    s = summary["series"]
    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)

    axes[0].plot(s["step"], s["pms"], lw=1.2)
    axes[0].axhline(0.3, ls="--", lw=0.8, color="grey")
    axes[0].set_ylabel("PMS (head 3.6)")
    axes[0].set_title("pythia-70m retrain 512->2000: induction onset at stride 4")

    axes[1].plot(s["step"], s["recovery"], lw=1.2, color="tab:orange")
    for t in R_THRESHOLDS:
        axes[1].axhline(t, ls="--", lw=0.8, color="grey")
    axes[1].set_ylabel("recovery R")

    axes[2].plot(s["step"], s["nll_first"], lw=1.2, label="first copy")
    axes[2].plot(s["step"], s["nll_second"], lw=1.2, label="second copy")
    axes[2].axhline(summary["uniform_floor_nats"], ls=":", lw=1.0, color="k", label="uniform floor")
    axes[2].set_ylabel("NLL (nats)")
    axes[2].set_xlabel("pretraining step")
    axes[2].legend(fontsize=8)

    onset = summary["pms_crossing_steps"]["0.05"]
    if onset is not None:
        for ax in axes:
            ax.axvline(onset, lw=0.8, color="tab:red", alpha=0.6)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", default=DEFAULT_RUN_ID)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "data")
    args = parser.parse_args()

    probe_path = REPO_ROOT / "data" / "retrain" / args.run_id / "probe.jsonl"
    rows = load_probe_series(probe_path)
    summary = summarize(rows)
    summary["run_id"] = args.run_id
    summary["source"] = str(probe_path.relative_to(REPO_ROOT))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    json_path = args.out_dir / f"retrain_onset_{args.run_id}.json"
    json_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    wrote_plot = write_plot(summary, args.out_dir / f"retrain_onset_{args.run_id}.png")

    print(f"rows: {summary['n_probe_rows']}  steps {summary['step_first']}-{summary['step_last']}")
    print(f"PMS crossings:      {summary['pms_crossing_steps']}")
    print(f"recovery crossings: {summary['recovery_crossing_steps']}")
    print(f"PMS>=0.3 -> R>=0.5 lag: {summary['pms_0p3_to_R_0p5_lag_steps']} steps")
    print(
        f"first-copy NLL delta {summary['nll_first_delta']:+.3f}, "
        f"second-copy {summary['nll_second_delta']:+.3f} "
        f"(uniform floor {summary['uniform_floor_nats']:.3f})"
    )
    print(f"wrote {json_path}" + (" and the PNG" if wrote_plot else ""))


if __name__ == "__main__":
    main()
