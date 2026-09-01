# Handoff

_Updated by Claude at natural checkpoints — see `~/.claude/CLAUDE.md` for the convention. Read this first when resuming in this repo._

## Current goal
Unblock the M1–M8 rank sweeps, which are stalled on G3's failed positive control (`PROJECT.md` status board). Six optimization-axis diagnostics have already come back negative sharing an unexplained plateau (`R ≈ 0.01`). The remaining, undone diagnostic is the reachability graft.

## What's been done (most recent last)
- Confirmed (2026-09-01, prior session) that six G3 diagnostics — lr, rank, frozen-W_K, head choice, full fine-tune, step-budget accounting — all disfavor their respective hypotheses; none explain the plateau.
- Built `scripts/diagnose_g3_reachability.py` (7th diagnostic): grafts checkpoint B's real weights into A component-by-component to test whether the QK arm's success criterion is reachable *at all*, independent of optimization. 17 unit tests, never executed — prior session's network policy blocked HuggingFace access.
- Verified checkpoint contents for the separate `data/retrain/` onset-bracket work (spot check of 4/82 checkpoints, all differ, delta norms scale sanely with step gap) — see REVIEW.md 2026-09-01 entries. That thread has two still-open human-call items (buffer eviction design after resume; whether a 58-checkpoint post-onset window is adequate for D1) but is lower priority since D1 is itself blocked upstream on M1/M2.
- **This session (2026-09-01):** confirmed HuggingFace access now works from this box (`curl huggingface.co` → 200), removing the blocker on the reachability graft. Confirmed `/home/ubuntu/venv` has the project deps installed (Python 3.12 — fine for this script since it writes no results record and doesn't touch `metric_hash.json`/mypy, the two things pinned to 3.11). User explicitly approved running the graft next.

## Next step (in progress / about to run)
Run `scripts/diagnose_g3_reachability.py` (default cells) from `/home/ubuntu/Lora_inductionhead` with `source /home/ubuntu/venv/bin/activate` first. Per the script's own docstring: diagnostic only, no protocol/status-board change — reading the result and deciding what it means is a human call (CLAUDE.md falsification discipline). REVIEW.md's 2026-08-19/08-22 entries are the context for why this is the discriminating experiment.

## Open questions / blockers
- §11's first open question ("induction objective ... settle before G3") looks stale — G3 has already run. Flagged to the user, not yet resolved; worth checking whether PROJECT.md needs a §10 correction.
- The two retrain-harness human calls in REVIEW.md's 2026-09-01 entries (buffer eviction design; bracket adequacy for D1) — not addressed this session.

## Useful paths
- Spec / status board: `PROJECT.md` (§11 = open questions, §10 = decisions log)
- Review queue (things needing human judgment): `REVIEW.md`
- Diagnostic script: `scripts/diagnose_g3_reachability.py`
- venv: `/home/ubuntu/venv` (Python 3.12; CI/metric-hash-sensitive work needs 3.11 instead, see `infra/rvm.env`)
- Branch: `main`, clean, up to date with `origin/main` as of commit `1ca6854`
