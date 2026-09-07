"""Merge the re-probe shards into one onset curve, all heads.

Reads `data/reprobe_shard*.jsonl`, checks coverage against the bracket
the shards were cut from, and writes a merged JSON plus a figure.

Analysis of authorized measurements. No new metric, no verdict. In
particular this reports the per-head PMS curves *descriptively*: PROJECT.md
11 (2026-09-03) asks whether induction at B is carried by one head or by
three, and flags rescoping M1-M8 as a human call. Drawing the curves is
not making that call, and nothing here concludes anything about it.

Coverage is checked rather than assumed. A spot reclaim kills a shard
mid-run, and a merged file that is quietly missing 12 of 82 checkpoints
still plots as a smooth curve -- the gaps are interleaved by design, so
they close up visually. `--require-complete` refuses to write in that
case; without it the merge proceeds and records exactly which steps are
missing.

Usage:
    python scripts/merge_reprobe.py [--require-complete]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from reprobe_retrain_checkpoints import bracket_steps  # noqa: E402

ICL_A = -0.024705959483981133
ICL_B = 11.512047719210386


def merge_shards(paths: list[Path]) -> list[dict[str, Any]]:
    """Rows from every shard, deduplicated by step and sorted.

    Duplicates are possible: a shard that is reclaimed after writing a
    row and then relaunched re-does nothing (it resumes), but a shard
    relaunched with a *different* --n-shards would overlap. First
    occurrence wins and the count is reported by the caller.
    """
    rows: dict[int, dict[str, Any]] = {}
    for p in sorted(paths):
        with p.open() as fh:
            for line in fh:
                if line.strip():
                    r = json.loads(line)
                    rows.setdefault(int(r["step"]), r)
    return [rows[s] for s in sorted(rows)]


def coverage(rows: list[dict[str, Any]]) -> tuple[list[int], list[int]]:
    """(present, missing) against the bracket the shards were cut from."""
    have = {int(r["step"]) for r in rows}
    want = bracket_steps()
    return sorted(have), [s for s in want if s not in have]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--require-complete", action="store_true")
    ap.add_argument("--data-dir", type=Path, default=REPO_ROOT / "data")
    args = ap.parse_args()

    from indbw.probes import recovery

    paths = sorted(args.data_dir.glob("reprobe_shard*.jsonl"))
    if not paths:
        raise SystemExit("no reprobe_shard*.jsonl found")
    rows = merge_shards(paths)
    present, missing = coverage(rows)
    print(f"{len(paths)} shards, {len(rows)} unique steps, {len(missing)} missing")
    if missing:
        print(f"  MISSING: {missing}")
        if args.require_complete:
            raise SystemExit("refusing to write an incomplete merge (--require-complete)")

    steps = [int(r["step"]) for r in rows]
    icl = [float(r["icl"]) for r in rows]
    rec = [recovery(v, ICL_A, ICL_B) for v in icl]
    grids = [r["pms_grid"] for r in rows]
    n_layers, n_heads = len(grids[0]), len(grids[0][0])

    out = {
        "n_steps": len(rows),
        "steps": steps,
        "missing_steps": missing,
        "complete": not missing,
        "n_eval": rows[0]["n_eval"],
        "T": rows[0]["T"],
        "eval_seed": rows[0]["eval_seed"],
        "icl": icl,
        "recovery": rec,
        "nll_first": [float(r["nll_first"]) for r in rows],
        "nll_second": [float(r["nll_second"]) for r in rows],
        "pms_by_head": {
            f"{ell}.{h}": [g[ell][h] for g in grids]
            for ell in range(n_layers)
            for h in range(n_heads)
        },
        "prev_token_by_head": {
            f"{ell}.{h}": [r["prev_token_grid"][ell][h] for r in rows]
            for ell in range(n_layers)
            for h in range(n_heads)
        },
    }
    dest = args.data_dir / "reprobe_merged.json"
    dest.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(f"wrote {dest}")

    final = {k: v[-1] for k, v in out["pms_by_head"].items()}
    top = sorted(final.items(), key=lambda kv: -kv[1])[:6]
    print(f"PMS at step {steps[-1]}, top heads: " + ", ".join(f"({k})={v:.4f}" for k, v in top))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib unavailable; skipping the figure")
        return

    fig, axes = plt.subplots(2, 1, figsize=(9, 8), sharex=True)
    for key, series in out["pms_by_head"].items():
        if max(series) < 0.05:
            axes[0].plot(steps, series, lw=0.6, color="lightgrey", zorder=1)
    for key, _ in top:
        axes[0].plot(steps, out["pms_by_head"][key], lw=1.5, label=f"head {key}", zorder=2)
    axes[0].set_ylabel("PMS")
    axes[0].legend(fontsize=8)
    axes[0].set_title(
        f"retrain onset re-probed at N_eval={out['n_eval']}, all {n_layers * n_heads} heads"
    )
    axes[1].plot(steps, rec, lw=1.4, color="tab:orange")
    axes[1].axhline(0.5, ls="--", lw=0.8, color="grey")
    axes[1].axhline(0.8, ls="--", lw=0.8, color="grey")
    axes[1].set_ylabel("recovery R")
    axes[1].set_xlabel("pretraining step")
    fig.tight_layout()
    fig.savefig(args.data_dir / "reprobe_merged.png", dpi=130)
    print(f"wrote {args.data_dir / 'reprobe_merged.png'}")


if __name__ == "__main__":
    main()
