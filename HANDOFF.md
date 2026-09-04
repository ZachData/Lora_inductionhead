# Handoff

_Session/environment mechanics only. Findings, decisions, and the next-step plan live in `PROJECT.md` (§10 decisions log, §11 open questions) — that is the single source of truth for the science. This file exists so a fresh session doesn't have to re-derive what box it's on and where things are; do not let it re-accumulate a parallel narrative._

## Where things stand
Read `PROJECT.md` §10's 2026-09-03 entries and §11 first. Short version: eight diagnostics now agree G3's plateau is not an optimization failure — the reachability graft shows $B$'s own $W_Q$ for the target head doesn't move $R$, which upper-bounds any trained update. The prereq A/B contrast supplies a candidate mechanism (a 2.6–15× weaker prerequisite at $A$ than at $B$, same head). Neither carries a verdict; §11 has a proposed training-free next phase (localize the reachability gap, then rank-truncate the real $B-A$ delta instead of training) that needs human sign-off before anything runs under it. **The blocker is a human decision, not a missing measurement.**

## Environment notes
- Orchestrator: `t4g.small`, 1.8 GB RAM, no swap by default. Its venv may be Python 3.12 and may not have `pytest` installed — check before assuming tier 1–2 can run locally; `pip install -e ".[dev]"` fixes both. CI runs 3.11 and is the authoritative green/red signal.
- Do not bump `METRIC_VERSION` to silence `test_committed_lockfile_matches_current_metric_source` under a 3.12 venv — that's a known `ast.dump` false positive (PROJECT.md §11), not a real metric change.
- No `ec2:GetConsoleOutput`, so a worker cannot be debugged by console. A pushed result commit (`git log`) is the only completion signal.
- Spot works in this account as of 2026-09-03 (verified by real launch, `InstanceLifecycle: spot`). The spot-*only* enforcement (a `Deny` on non-spot `RunInstances`) is not yet in place — see REVIEW.md 2026-09-03 for the guardrail gap. Prefer spot explicitly rather than relying on it being enforced.

## Useful paths
- Spec / status board / findings / plan: `PROJECT.md` (§10 decisions log, §11 open questions)
- Review queue (things needing human judgment, more granular than §10/§11): `REVIEW.md`
- Reachability graft script (candidate to extend for localization + truncation): `scripts/diagnose_g3_reachability.py`
- Worker mechanism: `infra/wrapper.sh` + `infra/worker-bootstrap.sh`; see `CLAUDE.md` "Workers"
- Branch: `main`
