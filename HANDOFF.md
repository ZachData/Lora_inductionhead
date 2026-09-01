# Handoff

_Updated by Claude at natural checkpoints — see `~/.claude/CLAUDE.md` for the convention. Read this first when resuming in this repo._

## Current goal
Unblock the M1–M8 rank sweeps, which are stalled on G3's failed positive control (`PROJECT.md` status board). Six optimization-axis diagnostics have already come back negative sharing an unexplained plateau (`R ≈ 0.01`). The remaining, undone diagnostic is the reachability graft.

## What's been done (most recent last)
- Confirmed (2026-09-01, prior session) that six G3 diagnostics — lr, rank, frozen-W_K, head choice, full fine-tune, step-budget accounting — all disfavor their respective hypotheses; none explain the plateau.
- Built `scripts/diagnose_g3_reachability.py` (7th diagnostic): grafts checkpoint B's real weights into A component-by-component to test whether the QK arm's success criterion is reachable *at all*, independent of optimization. 17 unit tests, never executed — prior session's network policy blocked HuggingFace access.
- Verified checkpoint contents for the separate `data/retrain/` onset-bracket work (spot check of 4/82 checkpoints, all differ, delta norms scale sanely with step gap) — see REVIEW.md 2026-09-01 entries. That thread has two still-open human-call items (buffer eviction design after resume; whether a 58-checkpoint post-onset window is adequate for D1) but is lower priority since D1 is itself blocked upstream on M1/M2.
- **This session (2026-09-01):** confirmed HuggingFace access now works from this box (`curl huggingface.co` → 200), removing the blocker on the reachability graft. Confirmed `/home/ubuntu/venv` has the project deps installed (Python 3.12 — fine for this script since it writes no results record and doesn't touch `metric_hash.json`/mypy, the two things pinned to 3.11). User explicitly approved running the graft next.

## Next step
**Attempted this session; failed on memory, not logic.** `scripts/diagnose_g3_reachability.py` was run — HuggingFace access works fine now (checkpoint A's 76 tensors downloaded and loaded), but the process was OOM-killed loading checkpoint B alongside it (confirmed via `journalctl -k`: anon-rss 1.13–1.26 GB vs. this box's 1.8 GB total — the same box and the same magnitude as the earlier mypy OOM in §11). See the new REVIEW.md 2026-09-01 entry for the three untried fixes (bigger instance / load-then-free instead of holding both models / lower-precision dtype) and why none was picked without sign-off. **This is now the actual next step**: get a human call on which fix, then re-run. `data/g3_reachability_diag.jsonl` does not exist — nothing was produced.

## Side thread this session: instance sizing + spot workers
- User asked to upgrade this orchestrator box by one step (t4g.small → t4g.medium) and to prefer spot for sweep workers.
- **This box cannot resize itself.** Confirmed via dry-run: its role (`research-vm-ssm-role`) allows `ec2:StopInstances` but denies `ec2:ModifyInstanceAttribute`, `ec2:StartInstances`, `ec2:RequestSpotInstances`. Stopping it would also kill this session with no way to restart it from in here. Instance is `i-017b99c1cdafa4a92`, us-east-2. Commands for the user to run from wherever holds fuller credentials:
  ```
  aws ec2 stop-instances --instance-ids i-017b99c1cdafa4a92 --region us-east-2
  aws ec2 wait instance-stopped --instance-ids i-017b99c1cdafa4a92 --region us-east-2
  aws ec2 modify-instance-attribute --instance-id i-017b99c1cdafa4a92 --instance-type t4g.medium --region us-east-2
  aws ec2 start-instances --instance-ids i-017b99c1cdafa4a92 --region us-east-2
  ```
  Not yet run — waiting on the user (or whoever has those creds) to do it.
- **Sweep workers now request spot** — implemented and pushed. `infra/wrapper.sh`'s `run-instances` call now passes one-time spot market options; the wait-loop also verifies each worker's `git log`-matched result commit before treating the sweep as done, since a spot reclaim terminates the instance the same way a successful push does (see REVIEW.md 2026-09-01 entry for the full reasoning, including why persistent+stop-on-interruption was considered and rejected). Untested against a real sweep/reclaim.
- Noted but not touched: `infra/rvm.env`'s `RVM_INSTANCE_TYPE` is dead config — neither script reads it; the worker instance type actually comes from the `research-vm-worker-template` launch template (t4g.small). Human call on whether to wire it up or delete it.

## Open questions / blockers
- Which fix for the reachability-graft OOM (REVIEW.md, this date) — bigger instance, partial-tensor loading, or lower precision. No `rvm` CLI is available from this box to self-serve a bigger instance.
- §11's first open question ("induction objective ... settle before G3") looks stale — G3 has already run. Flagged to the user, not yet resolved; worth checking whether PROJECT.md needs a §10 correction.
- The two retrain-harness human calls in REVIEW.md's 2026-09-01 entries (buffer eviction design; bracket adequacy for D1) — not addressed this session.

## Useful paths
- Spec / status board: `PROJECT.md` (§11 = open questions, §10 = decisions log)
- Review queue (things needing human judgment): `REVIEW.md`
- Diagnostic script: `scripts/diagnose_g3_reachability.py`
- venv: `/home/ubuntu/venv` (Python 3.12; CI/metric-hash-sensitive work needs 3.11 instead, see `infra/rvm.env`)
- Branch: `main`, clean, up to date with `origin/main` as of commit `1ca6854`
