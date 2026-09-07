# Handoff

_Session/environment mechanics only. Findings, decisions, and the next-step plan live in `PROJECT.md` (§10 decisions log, §11 open questions) — that is the single source of truth for the science. This file exists so a fresh session doesn't have to re-derive what box it's on and where things are; do not let it re-accumulate a parallel narrative._

## Where things stand
Read `PROJECT.md` §10's 2026-09-03, 2026-09-05 and **2026-09-07** entries and §11 first. Short version: eight diagnostics agree G3's plateau is not an optimization failure — the reachability graft shows $B$'s own $W_Q$ for the target head doesn't move $R$, which upper-bounds any trained update. The prereq A/B contrast supplies a candidate mechanism (a 2.6–15× weaker prerequisite at $A$ than at $B$, same head). The 2026-09-05 localization bisection sharpens this further: grafting blocks 0-2 (on top of block 3, always-grafted) restores prefix matching almost fully (PMS 0.895 vs full's 0.964) while R stays at 0.10 — matching and recovery dissociate. None of this carries a verdict. `layer_host_plus_ln_final` is **closed for good** (8/8 spot reclaims across `t4g.medium` and `t4g.large`, §10 2026-09-05) — do not retry it on spot; it would need on-demand, an explicit AZ, or another region, none of which this project may authorize.

**Corrected 2026-09-07, and it changes the framing the previous handoff gave:** the earlier claim that "the blocker is a human decision, not a missing measurement" was true of what `PROJECT.md` recorded and false of what the repo held. The **512→2000 retrain ran on 2026-09-01** (6.9 h, RTX 3080) and was undocumented — §11 still called it "unrun." It reproduces the induction transition (onset step 620; endpoint PMS 0.971 / ICL 11.72 vs published $B$'s 0.9652 / 11.512) and leaves an 82-checkpoint bracket at stride 4 over steps 528–852 in S3, 48 GB. See §10 2026-09-07 and the board row.

**Current plan, user-authorized 2026-09-07 (§10):** (b) measure the candidate head's copying score and OV value spread at $A$ — two forward passes, the Cor 17.2/17.3 precondition for M1/M2 meaning anything, never measured for any head; then (a) exploit the retrain bracket. G3 stays failed and M1–M4 stay blocked; neither is a route around the gate.

## Environment notes
- Orchestrator: `t4g.small`, 1.8 GB RAM, no swap by default. Its venv may be Python 3.12 and may not have `pytest` installed — check before assuming tier 1–2 can run locally; `pip install -e ".[dev]"` fixes both. CI runs 3.11 and is the authoritative green/red signal.
- Do not bump `METRIC_VERSION` to silence `test_committed_lockfile_matches_current_metric_source` under a 3.12 venv — that's a known `ast.dump` false positive (PROJECT.md §11), not a real metric change.
- No `ec2:GetConsoleOutput`, so a worker cannot be debugged by console. A pushed result commit (`git log`) is the only completion signal.
- Spot works in this account as of 2026-09-03 (verified by real launch, `InstanceLifecycle: spot`). The spot-*only* enforcement (a `Deny` on non-spot `RunInstances`) is not yet in place — see REVIEW.md 2026-09-03 for the guardrail gap. Prefer spot explicitly rather than relying on it being enforced.
- Spot reclaims on `t4g.medium` were frequent on 2026-09-05 (6 in a row for one specific job, ~5-45 min runtimes each) — capacity for this type is volatile at times, not a launch problem. `diagnose_g3_reachability.py` only pushes/syncs once at the very end of a run, so a reclaim mid-run loses everything computed so far; splitting a multi-cell job into one-worker-per-cell (separate `--out` shards, merged by hand afterward) worked far better and is the pattern to reuse rather than a single big job, until the script itself is fixed to sync incrementally (REVIEW.md 2026-09-05).
- `diagnose_g3_reachability.py` now syncs each cell to S3 as it finishes (fixed 2026-09-05, §10) — the monolithic-push gap that cost a 45-minute job is closed for *that* script. Check any other long-running script for the same pattern before launching it on spot.
- The retrain's payload lives in `s3://research-vm-shared-176048535722/retrain/cb25e3f6c2185c1e/` (176 objects, 48.5 GB: `ckpt/` 168 checkpoints ×282 MB, plus `latest.pt` at 845 MB). This box has ~47 GB free, so the bracket cannot be pulled whole — pull the specific steps you need. Locally only the JSON/JSONL metadata is checked in.
- Terminated instances age out of `aws ec2 describe-instances` after a few hours in this account — don't rely on instance state to diagnose an old worker's outcome; git log and the S3 backstop (`s3://research-vm-shared-176048535722/worker-results/<id>/`) are the only durable signals.

## Useful paths
- Plain-language narrative (what/why/what we've found, no section-number refs): `OVERVIEW.md`
- Spec / status board / findings / plan: `PROJECT.md` (§10 decisions log, §11 open questions)
- Review queue (things needing human judgment, more granular than §10/§11): `REVIEW.md`
- Reachability graft script (candidate to extend for localization + truncation): `scripts/diagnose_g3_reachability.py`
- Worker mechanism: `infra/wrapper.sh` + `infra/worker-bootstrap.sh`; see `CLAUDE.md` "Workers"
- Branch: `main`
