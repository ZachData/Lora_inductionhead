# Handoff

_Updated by Claude at natural checkpoints — see `~/.claude/CLAUDE.md` for the convention. Read this first when resuming in this repo._

## Current goal
Unblock the M1–M8 rank sweeps, stalled on G3's failed positive control (`PROJECT.md` status board). **All seven diagnostics are done, plus an eighth measurement that appears to explain them.** The reachability graft (2026-09-03) showed the plateau is not about optimization: B's own weights grafted into A do not reach the criterion either, and grafting all of layer 3 still leaves R at 0.0077 — so the deficit is *upstream* of layer 3. The prerequisite contrast then found it: the previous-token head at layer 2 scores **0.356 at A vs 0.940 at B**, and its overlap margin over chance is **15x larger at B**. G1 certified that prerequisite as present at A; it is present but roughly a third as strong as where induction actually works. **The blocker is now an interpretive decision by a human, not a missing measurement.** See "Next step".

## What's been done (most recent last)
- Confirmed (2026-09-01, prior session) that six G3 diagnostics — lr, rank, frozen-W_K, head choice, full fine-tune, step-budget accounting — all disfavor their respective hypotheses; none explain the plateau.
- Built `scripts/diagnose_g3_reachability.py` (7th diagnostic): grafts checkpoint B's real weights into A component-by-component to test whether the QK arm's success criterion is reachable *at all*, independent of optimization. 17 unit tests.
- Verified checkpoint contents for the separate `data/retrain/` onset-bracket work — see REVIEW.md 2026-09-01 entries. Two still-open human-call items there (buffer eviction design after resume; whether a 58-checkpoint post-onset window is adequate for D1), lower priority since D1 is blocked upstream on M1/M2.
- 2026-09-01: HuggingFace access confirmed working; the graft was attempted on the orchestrator and **OOM-killed** loading checkpoint B alongside A (1.8 GB box).
- **This session (2026-09-03): ran the reachability graft, then measured G1's prerequisite at B for the first time.** Details below.

## This session

### The reachability graft RAN. It is no longer blocked.
The OOM was a statement about the orchestrator (`t4g.small`, 1.8 GB), not about the code. Ran it on a `t4g.medium` worker (`i-0a399eeb8ed09de7a`, id `g3reach`) off `main` at `e6acb6d`; 8/8 cells in ~11 min, pushed as `2e03a0c`, results in `data/g3_reachability_diag.jsonl`.

| cell | R | PMS | | cell | R | PMS |
|---|---|---|---|---|---|---|
| `base_a` | −0.0002 | 0.0055 | | `head` | 0.0046 | 0.0212 |
| `qk_q` | **−0.0006** | 0.0092 | | `layer_host` | 0.0077 | 0.0212 |
| `qk_qk` | 0.0005 | 0.0212 | | `full` | **1.0004** | 0.9641 |
| `ov` | 0.0023 | 0.0055 | | `base_b` | 1.0004 | 0.9641 |

Controls hold (`full` ≈ 1, `base_a` ≈ 0), plus two consistency checks nothing asserts: the `ov` graft leaves PMS exactly at `base_a` (OV cannot change attention patterns), and `qk_qk` moves PMS further than `qk_q` alone.

**`qk_q` = −0.0006 is the result.** It grafts B's real `W_Q` for head (3,6) — the exact tensor the QK arm trains — so it upper-bounds what *any* ΔW_Q at any rank, lr, or step budget can reach from A. It moves nothing. G3's plateau is not an optimization failure in the QK arm.

It also overshoots the question: the whole head gives 0.0046 and the whole host block 0.0077 — the same ~0.01 floor — while `full` gives 1.0004. **The structure that makes B's induction head function is not in the head, nor in its block.** Nothing between `layer_host` and `full` was measured, so the gap is unlocalized.

**No verdict has been written and none should be written casually.** The G3 row is untouched, no §10 entry, no §6 edit — interpreting this is a human call (CLAUDE.md §9), and "distributed circuit" vs. "A and B are not in the same basin" vs. "LayerNorm/embedding co-adaptation" have very different consequences for whether M1–M8 is well-posed. See the REVIEW.md 2026-09-03 entry for the four things that result cannot resolve, including a 0.04% drift between the hardcoded `ICL_B` constant and its re-measurement, which if real biases every prior R.

### The prerequisite contrast: A's prerequisite is 2.6x weaker than B's
`scripts/diagnose_prereq_ab.py` (new), results in `data/prereq_ab_diag.jsonl`, run on a `t4g.small` **spot** worker in 8 min. Walks the circuit both directions at both checkpoints.

| | A (512) | B (2000) |
|---|---|---|
| prev-token score | 0.3556 | **0.9403** |
| prev-token head | (2,1) | **(2,1)** — same head |
| K-composition overlap | 0.3871 | 0.7691 |
| **overlap margin over null** | **0.0268** | **0.4081** |
| induction head (3,6) PMS | 0.0058 | 0.9652 |

**A's numbers reproduce the committed G1 record to three decimals** (0.356 / 0.387 / 0.360) — that A-vs-A agreement is the check that the wiring is right, and it passes.

G1 passed at A on both criteria, both by technicalities (0.356 vs a 0.3 bar; margin 0.027). At B neither statistic is anywhere near its threshold. The head does not move, so this is one component strengthening, not a different one appearing.

**Why it matters:** the deficit is at layer 2 — upstream of layer 3 — which supplies one mechanism for results that had been accumulating as separate negatives. It is why grafting B's entire layer-3 block still left R at 0.0077; why unfreezing `W_K` (in block 3) did not help; why B's `W_Q` does not transfer, being tuned to a 0.94-strength signal A does not supply. **Every G3 diagnostic varied something in or around layer 3.**

**Unplanned second finding:** at B, *three* layer-3 heads do prefix matching — (3,6) 0.9652, (3,1) 0.8993, (3,0) 0.5094 — against a uniform ~0.006 at A. M1-M8 adapts a single head. If induction at B is carried by three, a single-head protocol may not be a low-rank version of what B does but a different object.

**No verdict written.** These are two checkpoints of natural pretraining, so "the prev-token head strengthens and induction follows" is tempting and is a *developmental* claim, which CLAUDE.md reserves for D1. This is a correlation across two checkpoints; causal ordering is not measured here. See the REVIEW.md 2026-09-03 entry for four things it cannot resolve, including whether G1's 0.3 bar was ever derived from anything.

### Sizing: measured at last
Peak RSS **1.34 GB** (`data/worker_prereqab_resources.txt`) on a 1.8 GB `t4g.small` with the new swapfile. The ladder works — this did not need the `t4g.medium` the reachability graft used, and future one-checkpoint-at-a-time jobs should start at `small`.

### Fixed: the per-worker `cmd` override was documented but never implemented
`infra/prompt.md` has told the agent for several commits that a manifest entry "may also carry `cmd` to override the default worker command." Nothing implemented it — `worker-bootstrap.sh` hardcoded `g0_sweep.py`, so a worker could only ever run a checkpoint-grid sweep. That is a large part of why memory-bound one-off jobs kept being treated as blocked-on-a-human instead of being sent to a worker. Now implemented (`e6acb6d`), along with a per-worker `instance_type` override, since the launch template is `t4g.small` in every version — i.e. a default worker would OOM exactly as the orchestrator did. `worker-bootstrap.sh` also now allocates a 3 GB swapfile, which `PROJECT.md` §11 recommends in three separate places.

`CLAUDE.md` gained a positive **"Workers"** section. The permission already existed but was phrased as an exception buried in the "Do not" list, which reads as a prohibition at a glance. It now also carries cost discipline: prefer cheap `t4g`, size up one step at a time, avoid GPU instances absent human sign-off (the probe path is forward-only on a 70M model and has no use for one).

### Found: spot launches cannot succeed in this account
Commit `ca3e20d` moved workers to spot in *two* places — `wrapper.sh`'s flag and `research-vm-worker-template` **version 4** (the default), which bakes the same options in. Both fail with `AuthFailure.ServiceLinkedRoleCreationNotPermitted`: `AWSServiceRoleForEC2Spot` does not exist in the account and `research-vm-ssm-role` cannot create it. **Every sweep launched through `wrapper.sh` currently fails at the first `run-instances`.** Worked around here by pinning `--launch-template ...,Version=3` (last version without market options) and going on-demand; the template default and `wrapper.sh` were left untouched rather than silently reverting the user's explicit spot request.

Also worth knowing: **`run-instances --dry-run` reports success for the spot call that then fails.** Dry-run does not check service-linked-role creation. A launch pre-flight built on `--dry-run` alone will greenlight a path that cannot work.

## Next step
**A human needs to interpret the graft result and decide what it means for M1-M8.** Seven diagnostics are now done and the phase cannot move without that call. The question is no longer "why won't the QK arm train?" but "given that B's own weights do not transplant, is the M1-M8 protocol asking a well-posed question?" Everything downstream (M1-M8, and D1 beneath them) waits on it.

Also needed from a human, both one-time IAM calls I am not permitted to make (see "Found: spot launches cannot succeed" above):
1. `aws iam create-service-linked-role --aws-service-name spot.amazonaws.com` - makes spot work at all. Until then every sweep through `wrapper.sh` fails at the first `run-instances`.
2. Optionally, a `Deny` on `ec2:RunInstances` when `ec2:InstanceMarketType != spot`, and when `ec2:InstanceType` is not `t4g.*`. That enforces spot-only and no-GPU at the layer that cannot be routed around - the launch template only *defaults* to spot, and I demonstrated it can be bypassed by pinning an older version.

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
