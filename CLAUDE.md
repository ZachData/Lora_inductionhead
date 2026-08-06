# CLAUDE.md

Operational rules. `PROJECT.md` is the source of truth for what the experiment *is* — hypotheses, thresholds, metric definitions, status board, decisions log. Read it before doing anything non-trivial. This file only covers how to work.

## What this repo is

A research codebase measuring the minimum-dimension weight update that installs an induction circuit into a pre-induction `pythia-70m` checkpoint, and whether the update's symmetry structure differs between the prefix-matching (QK) and copying (OV) halves.

## Stack

- Python 3.11, PyTorch, TransformerLens (`HookedTransformer`), pytest
- Models: `EleutherAI/pythia-70m` by checkpoint revision, cached locally
- All new code in `src/indbw/`

```
src/indbw/
  algebra.py    sym/antisym split, φ, principal angles, SVD truncation
  probes.py     PMS, prev-token score, copying score, ICL score, recovery R
  models.py     Pythia checkpoint loader + cache
  lora.py       injection; unconstrained / symmetric / antisymmetric parameterizations
  train.py      training loop, adapter snapshotting
  nulls.py      matched-norm random updates, random subspaces
  schema.py     results record, METRIC_VERSION
tests/
  unit/         pure functions vs closed-form oracles; no model loading
  property/     invariants under random input
  integration/  real checkpoints, slow, not on every push
```

## Commands

```
pytest tests/unit tests/property -q      # tier 1+2, must pass before any commit
pytest tests/integration -q               # tier 3, slow
pytest tests/unit/test_algebra.py -x -v   # single file
pip install -e ".[dev]"
```

---

## TDD contract

**The test file is the spec. Write or read it first. No implementation ships without one.**

The dominant risk in this codebase is *silent failure*: a readout that returns a constant, or empty, or the same number for every input looks completely fine in logs and poisons every downstream claim. Tests exist to catch that at the source, not to check that code runs.

Four kinds of test, and every module needs the ones that apply:

**1. Closed-form oracles.** The mathematics in `PROJECT.md` §3 gives exact answers. Test to machine precision, not to a tolerance you tuned until it passed.

- $\varphi(ba^\top) = \frac12(1 - c^2)$ where $c = \hat a \cdot \hat b$
- $\varphi(M) = 1/2$ exactly for any $M = P_U M P_V$ with $U \perp V$
- $\varphi(M) = 0$ for symmetric $M$; $\varphi(M) = 1$ for antisymmetric $M$
- A nonzero real antisymmetric matrix has even rank
- The antisymmetric parameterization produces a matrix satisfying $M^\top = -M$ to float tolerance

**2. Invariants (property tests).** Over random inputs:

- $\|M\|_F^2 = \|S\|_F^2 + \|\Lambda\|_F^2$
- $\langle S, \Lambda \rangle = 0$
- $\varphi \in [0, 1]$
- $\varphi$ invariant under orthogonal conjugation $M \mapsto QMQ^\top$
- $\operatorname{rank}(\Delta W) \le r$ for every parameterization
- Principal angles are in $[0, \pi/2]$ and equal 0 for identical subspaces

**3. Discrimination tests — the silent-failure guards.** Every probe must be shown to return *different* answers on a known-positive and a known-negative. A probe that only ever gets tested on real data can be broken in a way no assertion catches.

- PMS on a hand-constructed toy induction head: high. On a randomly-initialized model: near chance.
- Copying score on an identity OV circuit: near 1. On random OV: near 0.
- Recovery $R$: exactly 0 at $A$, exactly 1 at $B$, by construction.
- Every readout, given an all-zeros or all-constant input, must raise or return a sentinel — never a plausible-looking number.

**4. Schema tests.** A results record is invalid unless it carries the null tested, the pre-registered criterion, the observed value, the verdict, and `METRIC_VERSION`. CI rejects records missing any field.

**Metric-version enforcement.** CI hashes the metric-defining functions in `probes.py` and `algebra.py`. If the hash changes and `METRIC_VERSION` did not, the build fails. Changing a metric definition silently invalidates every prior result; this makes that impossible rather than merely discouraged.

**5. Smoke test.** One test runs the entire path — load toy model → inject LoRA → 5 training steps → compute every probe → emit a results record → validate schema — on a randomly-initialized 2-layer model in under 60 seconds. It asserts nothing about science. It exists so that integration breakage surfaces in tier 2 rather than three hours into a real run.

**Fixtures.** Unit tests use tiny models from `tests/conftest.py`. Never instantiate a model inline in a test. Never load a real checkpoint outside `tests/integration/`.

### When to write a test

Not all code here deserves one, and testing exploratory code wastes time and ossifies things that should stay fluid.

| Write a test if… | Skip it if… |
|---|---|
| The output is a number that lands in a results record | It only produces a plot or a printout a human will look at |
| Failure is silent — wrong number, not a crash | Failure is loud and immediate |
| There is a closed-form answer to check against | The "expected" value would just be whatever the code currently returns |
| More than one caller depends on it | It is a one-off analysis script |
| It defines something (a metric, a schema, an invariant) | It orchestrates things that are themselves tested |

The short version: **test definitions and silent failures; don't test exploration.** A regression test whose expected value was copied from the current output tests nothing except that the code has not changed — if that is all a test would do, don't write it.

### Numerical tolerance policy

`algebra.py` runs in float64. The §3 identities are exact, so their tests assert to `rtol=1e-12`. Everywhere else, state the tolerance in the test with a one-line comment explaining where it comes from. **Never widen a tolerance to make a test pass** — that converts a real disagreement into a silent one. If a test needs a looser bound, say why in the commit body.

---

---

## Beyond tests

### Computable verdicts

A results record stores the observed value, the pre-registered criterion, and the verdict. **The verdict must be recomputable from the other two**, and CI recomputes it. If the stored verdict and the recomputed verdict disagree, the build fails.

This is the single most important check in the repo. Tests catch broken code; this catches a correctly-functioning pipeline whose conclusion does not follow from its own numbers — which is the characteristic failure of unattended agentic research, and the one no unit test detects.

Corollary: **every falsification criterion in `PROJECT.md` §6 must be expressible as a comparison a machine can evaluate.** If a criterion cannot be written that way, it is too vague to be pre-registered and needs rewriting before any row depending on it is run.

### Provenance

Every results record carries, without exception: git SHA, `METRIC_VERSION`, run config hash, RNG seed, checkpoint revision, eval-set hash, torch/numpy/transformer-lens versions, wall-clock, hardware. A record without provenance is unusable three months later and cannot be defended in review.

Seed everything at run start — Python, numpy, torch, CUDA if present — and record the seed. Enable `torch.use_deterministic_algorithms(True)`; if a kernel forces it off, log which one rather than silently proceeding.

### Config as data

One frozen dataclass per run, serialized to JSON, hashed. The hash *is* the run ID. This kills the "I changed a default and no longer know which runs used it" failure, and it makes resumability free.

### Resumability

An idle-CPU alarm or the 4h hard cap can stop the instance mid-sweep. Write results incrementally, one record per cell, appended. On start, skip cells whose config hash already has a record. Losing twenty hours of CPU to a stop is a self-inflicted wound.

### Fail-fast in production paths, not only in tests

Assertions inside `src/`, not only in `tests/`:

- NaN/Inf check on loss after every training step — raise immediately, do not let it run to completion
- Shape assertions at every function boundary that takes a tensor
- Range guards on metrics: `0 <= phi <= 1`, `0 <= pms <= 1`, principal angles in $[0, \pi/2]$
- Raise, don't clamp. A clamped metric looks plausible and is a lie. If a value is out of range, something upstream is wrong and you want to know now.
- Hard step and wall-clock budget inside the training loop that raises, independent of the OS-level cap

### Adversarial self-review

Any commit touching `algebra.py`, `probes.py`, or `schema.py` must include in its body a short **"how this could be wrong"** note — the most plausible way the change produces a confident wrong number. Not a summary of what changed; a statement of the failure mode not covered by the tests.

Anything the note cannot resolve goes in `REVIEW.md` as a queue for human eyes. There is no reviewer in the CD loop; this is the substitute, and it is a weak one, so keep the queue short and actually read it.

### Hygiene

- Delete dead code rather than commenting it out. The superseded `sltdiff` modules already caused one round of confusion about what the project was.
- Type hints on everything in `src/`; mypy in tier 0.
- Pre-commit hooks running ruff, format, and the metric-hash check, so a wasted CI cycle costs seconds instead of minutes.
- No notebooks. Analysis scripts that write plots to disk, so every figure has a reproducible command behind it.
- `hypothesis` for the property tests in `tests/property/`.

### Pre-flight

`wrapper.sh` runs a canary before opening Claude Code: checkpoint cache reachable, disk space sufficient, `gh` authenticated, tier 1–2 green on the current tree, the smoke test passes. Fail here and stop — the cost of a three-hour run that dies on a missing checkpoint is the whole session.

---

## CI

Five tiers. Tiers 0–2 gate every push; tier 3 runs on demand.

| Tier | Contents | Runs on | Budget |
|---|---|---|---|
| 0 | ruff, format, mypy, metric-hash check | every push | seconds |
| 1 | `tests/unit` — pure math vs closed-form oracles, no model loading | every push | < 30 s |
| 2 | `tests/property` + fixture-model tests + smoke test | every push | < 2 min |
| 2.5 | Schema validation + **verdict recomputation** over all records | every push | seconds |
| 3 | `tests/integration` — real `pythia-70m` checkpoints | label `run-integration`, or nightly | minutes |

Tier 2.5 runs over every results record in the repo, not just new ones. A metric-version bump or a criterion edit therefore surfaces as a failure on all affected historical records rather than leaving them quietly stale.

Tier 3 is separated because it downloads checkpoints and is the only tier that can fail for reasons unrelated to the change. Keeping it off the push path means a red build always means a real regression.

CI runs on GitHub Actions, on GitHub's infrastructure. No AWS involvement.

---

## CD

The EC2 work loop, gated on CI. `infra/wrapper.sh` does, in order:

1. `cd` into the repo
2. Open Claude Code seeded with a prompt to read `PROJECT.md`, take the first ○ row on the status board, follow this file's rules, update the board, and push
3. **Refuse to proceed on a dirty tree or unpushed commits** — terminate destroys the volume
4. Wait on the GitHub Actions result for the pushed commit via `gh run list`; proceed only if green
5. Terminate the instance

Red CI blocks progress. It is a gate, not a parallel signal logged and ignored.

No auto-relaunch. Whether to start the next instance is a manual decision made outside the script.

---

## Working on a status-board row

1. Read the row and the `PROJECT.md` section it references.
2. Write the test file first. If the row is a gate (G0–G3), the test *is* the pass/fail criterion — encode it, don't eyeball it.
3. Implement until tiers 1–2 are green.
4. Run the row. Emit a results record.
5. Update the status board. If a decision was made, append to §10. If a new uncertainty surfaced, add it to §11.
6. Commit, push, wait for CI.

**A gate that fails stops the phase.** Record the failure in §10 with the reason and the decision taken. Do not route around it, do not weaken its criterion, and do not proceed to the next row on the assumption it will pass later.

---

## Falsification rules

These are not style preferences. See `PROJECT.md` §9.

- **Do not propose new hypotheses or new falsification tests.** Those are fixed by humans in `PROJECT.md` §6. Agent-designed tests are where the error rate concentrates.
- **Do not change a pre-registered threshold.** If one looks wrong, say so and append to §10. Never edit it silently.
- **No failure claim without its controls attached.** A rank cell that failed must report the matched-lr arm and the 10×-steps arm alongside it, or the result is about the optimizer and cannot be reported as capacity.
- **No developmental language from reachability results.** Everything in M1–M8 is a claim about the loss landscape near $A$. Only D1 licenses talk about how induction forms during pretraining, and only if it passes.
- **Report $\varphi_{QK}$ only as a contrast against $\varphi_{OV}$.** Random matrices concentrate near $\varphi = 1/2$, so $\varphi_{QK} \approx 0.5$ against a random null is uninformative. This limitation goes in the writeup, not just the code.
- **Ambiguous results stay ambiguous.** $0.50 \le R < 0.80$ is reported as ambiguous, never rounded into a verdict.

---

## Do not

- Do not adapt both matrices of a bilinear form. $W_Q$ only for QK, $W_O$ only for OV (`PROJECT.md` §3).
- Do not train LayerNorm or embeddings. Freezing them is what makes the symmetry analysis valid.
- Do not store merged models. LoRA factors only.
- Do not sweep rank without sweeping lr independently per rank, or holding $\alpha/r$ fixed.
- Do not use 410m for training. Forward-only probes at most.
- Do not run AWS commands or launch instances.
- Do not reintroduce LLC, SGLD, crosscoders, sparsity readouts, or the Markov substrate. All dropped deliberately (`PROJECT.md` §10).
- Do not treat `sltdiff-readme.md` or any pre-2026-08 spec as authoritative. Superseded.

## When finishing

Update the status board. Append decisions to §10, uncertainties to §11. Leave the tree clean and pushed.
