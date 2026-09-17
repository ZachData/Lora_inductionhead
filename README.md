# Induction Bandwidth (`Lora_inductionhead`)

**What is the minimum-dimension weight update that installs a working induction circuit into a pre-induction `pythia-70m` checkpoint, and does the update's symmetry structure differ between the circuit's prefix-matching (QK) and copying (OV) halves?**

Research codebase. `PROJECT.md` is the source of truth for the science (hypotheses, pre-registered thresholds, metric definitions, status board, decisions log); `CLAUDE.md` is the operational rulebook; `OVERVIEW.md` is the plain-language narrative; `HANDOFF.md` is session/environment mechanics; `REVIEW.md` is the queue of things awaiting human judgment.

## Status (2026-09-17)

**The training programme is stopped at gate G3.** Gates G0–G2 passed (checkpoint $A$ = step 512, $B$ = step 2000, bracket width 2; induction head L3H6, previous-token head L2H1; CPU affordable). G3 — the positive control, a generous-rank QK-only LoRA update reaching criterion — **failed**, and nine diagnostics rule out optimisation as the cause: $B$'s own $W_Q$ grafted into $A$ does not move recovery $R$, which upper-bounds any trained update. Grafting blocks 0–2 restores prefix matching (PMS 0.895) while $R$ stays at 0.10 — matching and recovery dissociate. The gradient is not gated. M1–M8 (the rank sweeps that answer the headline question) do not start until a human reads §10/§11 and decides. **There is no rank-$r$ induction subspace yet.**

What the repo *has* produced:

- **A dense induction-onset bracket.** A full-parameter continuation of `pythia-70m` from step 512 to 2000 on Pythia's own LR schedule (RTX 3080, 6.9 h), with a weight snapshot every 4 optimiser steps. Clean bracket: steps 528–852, 82 checkpoints, onset at step 620. Weights in S3 (`research-vm-shared-176048535722/retrain/cb25e3f6c2185c1e/`, 48 GB); metadata under `data/retrain/`. This fills the gap between the published step-512 and step-1000 checkpoints. It is *reachability, not development* — Adam is cold-started at 512, so it diverges from the published trajectory from step one.
- **An ordered six-head cascade.** The bracket re-probed at `n_eval=512` across all 48 heads (`data/reprobe_merged.{json,png}`): 3.6 → 3.1 → 4.6 → 4.7 → 3.0 → 3.5 by first PMS ≥ 0.10, with the L2H1 previous-token head going 0.389 → 0.947 inside the window.
- **A known-broken readout.** §5's copying score (full-vocab argmax hit rate) reads ≤ 2.8e-4 for every head at both checkpoints, including $B$'s L3H6, which demonstrably does induction. A known positive at the floor is a readout failure; it is parked in `REVIEW.md` because fixing it touches `METRIC_VERSION`.

## Relationship to Mets

As of 2026-09-10 this project's **concepts, analysis, and artifacts were absorbed into [MetastableStateAnalysis](https://github.com/ZachData/MetastableStateAnalysis) ("Mets"), Phase 8 (`p8_scale_ladder/`)**, as the pythia-70m exploration rung of a scale ladder. See Mets `PROJECT.md` §3.9 / §3.9-A and `p8_scale_ladder/design-8.md` "Relation to the sister project" for the record of what was taken and why.

This repo **stays the independent upstream**. The split is one-directional in each direction:

| Direction | What moves |
|---|---|
| Here → Mets | The dense bracket, the six-head cascade, the $\varphi$ (antisymmetric-fraction) question, the shared `S + Λ` operator decomposition, and the recorded dead ends so they are not rediscovered. |
| Mets → here | Instrument fixes. The outstanding one is the **copying score**: Mets' `tools/run/copying_score_sweep.py` (Elhage's $\sum\lambda / \sum|\lambda|$ over the token-to-token circuit, continuous and transpose-invariant) is the candidate replacement for the argmax readout above. Also a prerequisite check for M1–M8: Mets found heads whose SVD basis is *anti-ordered* (bottom-$r$ beats top-$r$), which would make a minimum-rank result in the SVD basis measure the opposite of what it intends. |
| Stays here | The training half — EC2 spot workers, S3, `METRIC_VERSION` CI, the LoRA fitting, and the G0–G3 gate structure. Mets explicitly does not import G3's failed gate or the M1–M8 block. |

## Layout

```
src/indbw/
  algebra.py    sym/antisym split, φ, principal angles, SVD truncation
  probes.py     PMS, prev-token score, copying score, ICL score, recovery R
  models.py     Pythia checkpoint loader + cache
  lora.py       injection; unconstrained / symmetric / antisymmetric parameterizations
  train.py      training loop, adapter snapshotting
  nulls.py      matched-norm random updates, random subspaces
  sweep.py      cell config-as-data, run-ID hash, resumability, control gate
  gates.py      G0–G3 pass/fail criteria as code
  schema.py     results record, METRIC_VERSION
  metric_hash.py  hash of the metric-defining source (tier-0 gate)
  evalset.py    seeded random-token eval set + hash
  natural_text.py held-out natural-text loss (M8)
  existence.py  existence derivation for the rank-r update
  retrain.py    full-parameter 512→2000 continuation (the dense bracket)
scripts/        sweep runners, G0–G3 diagnostics, re-probe, analysis (plots to disk)
infra/          EC2 spot worker pattern: wrapper.sh, worker-bootstrap.sh, teardown.sh
data/           per-worker JSONL shards, merged re-probe, retrain metadata, resource logs
results/        results records (one per cell; verdict recomputed in CI)
tests/
  unit/         closed-form oracles, no model loading
  property/     hypothesis invariants
  schema/       results-record validation + verdict recomputation
  smoke/        full path on a tiny random model, ~4 s
  integration/  real pythia-70m checkpoints, on demand
```

## Commands

```
pip install -e ".[dev]"
pytest tests/unit tests/property -q      # tiers 1+2, must pass before any commit
pytest tests/integration -q               # tier 3, slow, downloads checkpoints
python scripts/check_metric_hash.py       # tier 0: metric-defining source hash vs METRIC_VERSION
```

CI (GitHub Actions) runs ruff/format/mypy/metric-hash, unit, property + smoke, and schema validation with **verdict recomputation** over every results record on every push. Integration runs on the `run-integration` label or nightly.

## Working rules, in brief

Full text in `CLAUDE.md`. The ones that matter most:

- The test file is the spec. Every probe has a discrimination test (known-positive vs known-negative) so a readout that silently returns a constant is caught at the source.
- Metric definitions are hashed; changing one without bumping `METRIC_VERSION` fails the build.
- Every results record carries the null tested, the pre-registered criterion, the observed value, the verdict, and full provenance. The verdict must be recomputable from the other fields, and CI recomputes it.
- Hypotheses and thresholds are fixed by humans in `PROJECT.md` §6. Agents do not propose new tests, change thresholds, or route around a failed gate.
- Everything in M1–M8 is a claim about the loss landscape near $A$, never about how induction forms during pretraining.
- Workers are spot-only, smallest instance that works, results written incrementally.
