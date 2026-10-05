# STATE — Extension S (selective LoRA × SAE validity)

Working checklist for `PROJECT.md` §13. Read this first in any session touching `src/indbw/selora/`.
`PROJECT.md` stays the source of truth for claims and decisions; this file tracks *work*.

**Last updated:** 2026-10-05 · **Branch:** `claude/selora-sae-placement` · **Prereg:** DRAFT (unsigned)

## Constraints right now

- No GPU, small CPU box, no route to `huggingface.co` / `neuronpedia.org`. Nothing real can run.
- So: everything is built and verified against random tiny Gemma2 models, synthetic SAEs and simulated data.
- Code that touches the network lives only in `scripts/` and is labelled **unverified**.

## Practices (Extension S)

1. **Simulate every claim before sign-off**: planted-true and planted-false data; type-I ≤ α at the null boundary, power ≥ 0.8 at the assumed effect. Kept as permanent tests (`tests/unit/test_selora_design.py`).
2. **Multiplicity is part of the claim** (`Claim.multiplicity`), and a feasibility test proves the smallest attainable p can clear the strictest threshold that rule applies.
3. **Design assumptions are written down** (`prereg.ASSUMPTIONS`) and compared with the first real data before any verdict is read.
4. **Sign once, then freeze**: after `PREREG_STATUS="registered"` a threshold change is a new claim id.
5. **Every readout has a discrimination test** (planted positive, planted negative, constant input raises).
6. **Loaders thin and in `scripts/`; measurement in `src/`.**
7. **Reproducible from the record**: seed before every random draw; same cell → identical row.
8. **Preflight before any sweep** (`scripts/preflight_selora.py`): downloads, shapes, base-FVU sanity, timing probe, first output populated.
9. **Stage compute**: S0/S1/S2 (forward passes, few adapters) before the 80-cell sweep.
10. **No verdicts on partial data**: adjudication refuses until every planned cell has a row.

## Checklist

Legend: [x] done (tested offline) · [ ] open · [~] provisioned (written, cannot be exercised without network/GPU)

### A. Defects found in review (fix test-first)
- [x] A1 S3b infeasible under Holm(4) with 24 draws → fixed-sequence gatekeeping + feasibility test
- [x] A2 LoRA init unseeded (`get_peft_model` before seeding) → seed first; determinism test
- [x] A3 rank test compared 3-seed mean vs 1-seed draws → same-seed scores only
- [x] A4 `holm_p` held raw p → only the family-adjust step produces the criterion field

### B. Statistical power
- [x] B1 S3a: pooled seeds with crossed (seed × item) bootstrap, replacing worst-seed rule; type-I/power simulation tests
- [x] B2 S1a/S1b: intersection-union (each cell at α), Holm removed; S2a keeps Holm
- [x] B3 MMLU log-prob on the full test set; generation on the 500-item subset
- [x] B4 margin tests via studentized bootstrap (bootstrap-t), sign-flip kept for exact symmetric nulls
- [x] B5 S2b peak stability via prompt-level bootstrap (replaces Gaussian approximation)

### C. Nulls and controls
- [x] C1 spectrum-matched random LoRA (trained singular values, random singular vectors)
- [x] C2 seed-to-seed ΔFVU noise floor (replaces prompt-split)
- [x] C3 length × digit-count stratified permutation (replaces numeric-prose dataset)
- [x] C4 common data order across arms for a given seed (test)
- [x] C5 shuffled-answer training arm — built and tested, off by default (`cells.INCLUDE_SHUFFLED`); inclusion is E4

### D. Provisioning
- [x] D1a `selora/preflight.py` checks (tested)
- [~] D1b `scripts/preflight_selora.py` (needs network + GPU)
- [x] D2 `prereg.ASSUMPTIONS` + power simulations against them
- [x] D3 completeness guard before adjudication
- [~] D4 `scripts/run_selora.py` loaders + `adjudicate` stage (dataset names, SAE layout unverified)
- [x] D5 `report.build_records`: complete rows → one record per claim / gate step; controls attached only from data

### E. Needs a human
- [ ] E1 Sign off §13.4 claims, thresholds, multiplicity rules, assumptions (→ §10, flip `PREREG_STATUS`)
- [ ] E2 Allow `huggingface.co` (and optionally `neuronpedia.org`); HF token with Gemma licence
- [ ] E3 GPU sign-off in §10 (first use: preflight + 20-step timing probe)
- [ ] E4 Decide whether C5 (shuffled-answer arm) is in the registered design

### F. First real session (after E1–E3)
- [ ] F1 `preflight_selora.py` green
- [ ] F2 compare `ASSUMPTIONS` with preflight numbers; if off, stop and revise before sign-off is used
- [ ] F3 `divergence` per model → freeze SAE scores
- [ ] F4 S0/S1 on the late arm only (cheapest kill)
- [ ] F5 full sweep, then adjudicate once

## Simulation results (tests/design, 2026-10-05)

| Procedure | Condition | Result |
|---|---|---|
| S3a pooled, crossed bootstrap | true diff = −0.03 (null boundary) | type-I 0.013 (discordance 0.15 and 0.25) |
| S3a pooled, crossed bootstrap | true diff = 0 | power 0.99 (disc 0.15), 0.80 (disc 0.25) |
| S3a worst-seed rule (replaced) | true diff = 0 | power 0.67 (disc 0.15), 0.39 (disc 0.25) |
| Fixed sequence, 4 steps | global null | FWER ≤ α (exact: P(p₁ ≤ α)) |
| S1a bootstrap-t | skewed ΔFVU at the margin | type-I within α + 0.045 over 200 reps |

## Open offline work (no network needed)

- [ ] `check_selora_metric_hash.py --update` should refuse without a `SELORA_METRIC_VERSION` bump
- [ ] If discordance ≥ 0.25 is plausible, draft a 5-seed variant of S3a with its simulation, for E1
- [ ] Measure nothing, decide: S2 on base only vs both models; S4/S5 configurations (REVIEW.md 2026-10-05)
