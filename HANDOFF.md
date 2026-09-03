# Handoff

_Updated by Claude at natural checkpoints — see `~/.claude/CLAUDE.md` for the convention. Read this first when resuming in this repo._

## Current goal
Unblock the M1–M8 rank sweeps, which are stalled on G3's failed positive control (`PROJECT.md` status board). Six optimization-axis diagnostics have already come back negative sharing an unexplained plateau (`R ≈ 0.01`). The remaining, undone diagnostic is the reachability graft.

## What's been done (most recent last)
- Confirmed (2026-09-01, prior session) that six G3 diagnostics — lr, rank, frozen-W_K, head choice, full fine-tune, step-budget accounting — all disfavor their respective hypotheses; none explain the plateau.
- Built `scripts/diagnose_g3_reachability.py` (7th diagnostic): grafts checkpoint B's real weights into A component-by-component to test whether the QK arm's success criterion is reachable *at all*, independent of optimization. 17 unit tests.
- Verified checkpoint contents for the separate `data/retrain/` onset-bracket work — see REVIEW.md 2026-09-01 entries. Two still-open human-call items there (buffer eviction design after resume; whether a 58-checkpoint post-onset window is adequate for D1), lower priority since D1 is blocked upstream on M1/M2.
- 2026-09-01: HuggingFace access confirmed working; the graft was attempted on the orchestrator and **OOM-killed** loading checkpoint B alongside A (1.8 GB box).
- **This session (2026-09-03): stopped treating the OOM as a blocker and sent the job to a worker, which is what workers are for.** Details below.

## This session

### The reachability graft is running on a worker
The OOM was a statement about the orchestrator (`t4g.small`, 1.8 GB), not about the code. Launched it on a **`t4g.medium` (4 GB) on-demand worker**, `i-0a399eeb8ed09de7a`, worker id `g3reach`, running `python scripts/diagnose_g3_reachability.py` off `main` at `e6acb6d`. It writes `data/g3_reachability_diag.jsonl` and pushes a `Worker g3reach result` commit. Non-numeric worker id on purpose: `wrapper.sh` greps *all* history for `^Worker <id> result$`, so a stray `Worker 0 result` would make a future sweep believe worker 0 had already finished.

**If that commit is not on `main`, the run did not land** — check the instance, don't assume.

### Fixed: the per-worker `cmd` override was documented but never implemented
`infra/prompt.md` has told the agent for several commits that a manifest entry "may also carry `cmd` to override the default worker command." Nothing implemented it — `worker-bootstrap.sh` hardcoded `g0_sweep.py`, so a worker could only ever run a checkpoint-grid sweep. That is a large part of why memory-bound one-off jobs kept being treated as blocked-on-a-human instead of being sent to a worker. Now implemented (`e6acb6d`), along with a per-worker `instance_type` override, since the launch template is `t4g.small` in every version — i.e. a default worker would OOM exactly as the orchestrator did. `worker-bootstrap.sh` also now allocates a 3 GB swapfile, which `PROJECT.md` §11 recommends in three separate places.

`CLAUDE.md` gained a positive **"Workers"** section. The permission already existed but was phrased as an exception buried in the "Do not" list, which reads as a prohibition at a glance. It now also carries cost discipline: prefer cheap `t4g`, size up one step at a time, avoid GPU instances absent human sign-off (the probe path is forward-only on a 70M model and has no use for one).

### Found: spot launches cannot succeed in this account
Commit `ca3e20d` moved workers to spot in *two* places — `wrapper.sh`'s flag and `research-vm-worker-template` **version 4** (the default), which bakes the same options in. Both fail with `AuthFailure.ServiceLinkedRoleCreationNotPermitted`: `AWSServiceRoleForEC2Spot` does not exist in the account and `research-vm-ssm-role` cannot create it. **Every sweep launched through `wrapper.sh` currently fails at the first `run-instances`.** Worked around here by pinning `--launch-template ...,Version=3` (last version without market options) and going on-demand; the template default and `wrapper.sh` were left untouched rather than silently reverting the user's explicit spot request.

Also worth knowing: **`run-instances --dry-run` reports success for the spot call that then fails.** Dry-run does not check service-linked-role creation. A launch pre-flight built on `--dry-run` alone will greenlight a path that cannot work.

## Next step
1. Confirm the `Worker g3reach result` commit landed and read `data/g3_reachability_diag.jsonl`. The discriminating cell is `qk_q`: it is an upper bound on what *any* ΔW_Q can reach from A, so if it does not move R, no rank and no step budget can, and G3's plateau is an existence problem rather than an optimization one. `full` must give R == 1 by construction — if it does not, distrust the whole run. Interpreting the result is a human call (CLAUDE.md falsification discipline); do not write a status-board verdict off it unassisted.
2. Human call needed on spot: either `aws iam create-service-linked-role --aws-service-name spot.amazonaws.com` (one IAM call, forbidden to me), or revert `ca3e20d`. Leaving `main` with a sweep path that cannot launch is the one option that should not persist.

## Open questions / blockers
- The spot SLR decision above (REVIEW.md 2026-09-03).
- §11's first open question ("induction objective ... settle before G3") looks stale — G3 has already run. Flagged previously, still unresolved; worth checking whether `PROJECT.md` needs a §10 correction.
- The two retrain-harness human calls in REVIEW.md's 2026-09-01 entries (buffer eviction design; bracket adequacy for D1).
- `infra/rvm.env`'s `RVM_INSTANCE_TYPE` is dead config — neither script reads it; the worker type comes from the launch template. Now partly superseded by the manifest's `instance_type` key. Wire it up or delete it.

## Environment notes
- This orchestrator is `i-00ba65488bfbbcf1b` (**not** the `i-017b99c1cdafa4a92` an earlier handoff named — it is a fresh box). `t4g.small`, 1.8 GB, no swap.
- Its venv is **Python 3.12** and shipped without `pytest`; `pip install -e ".[dev]"` had never been run here. Installing `pytest`+`hypothesis` is enough for tiers 1–2: **460 passed, 1 failed**. The one failure is `test_committed_lockfile_matches_current_metric_source`, the known 3.12 `ast.dump`/`type_params` false positive (REVIEW.md 2026-09-01), which fails identically on a clean tree. **Do not bump `METRIC_VERSION` to make it green, no matter what its failure message says** — that invalidates every historical record. CI runs 3.11 and is green.
- No `ec2:GetConsoleOutput` permission, so a worker cannot be debugged from here by console. The pushed result commit is the only completion signal — which is why `wrapper.sh` checks `git log` rather than instance state.

## Useful paths
- Spec / status board: `PROJECT.md` (§10 = decisions log, §11 = open questions)
- Review queue (things needing human judgment): `REVIEW.md`
- Diagnostic script: `scripts/diagnose_g3_reachability.py`
- Worker mechanism: `infra/wrapper.sh` + `infra/worker-bootstrap.sh`; see CLAUDE.md "Workers"
- venv: `/home/ubuntu/venv` (3.12; CI/metric-hash-sensitive work needs 3.11)
- Branch: `main`
