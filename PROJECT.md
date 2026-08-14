# Induction Bandwidth

**What is the minimum-dimension weight update that installs a working induction circuit into a pre-induction Pythia checkpoint, and does the update's symmetry structure differ between the circuit's prefix-matching and copying halves?**

This is the living specification and the source of truth. It is updated at the end of every working session. If a claim here contradicts the code, one of them is wrong — resolve it explicitly in §10 and never let both stand.

---

## Status board

| Phase | Item | Status | Notes |
|---|---|---|---|
| Setup | `pyproject.toml`, package skeleton, CI green on empty suite | ● | CI run 31238848784 green on commit 1af4d12: tier0/1/2/2.5 all pass, tier3 correctly skipped (workflow_dispatch only) |
| Setup | `algebra.py` + unit tests against closed-form oracles | ● | §3 identities are exact; test to machine precision. 23 tests (unit oracles + property invariants), all pass at rtol=1e-12 where exact |
| Setup | `probes.py` + silent-failure guards | ● | PMS, prev-token score, copying score, ICL score, recovery R implemented as pure functions over tensors (no model loading — see §10). 29 tests (unit oracles/discrimination + property invariants), all pass |
| Setup | `models.py` Pythia loader + local checkpoint cache | ● | `load_checkpoint(step)` validated against a real download (pythia-70m, step 0): architecture matches §2 exactly (6 layers, d_model=512, 8 heads, d_head=64), forward pass finite. 4 unit tests (mocked, no network) + 2 integration tests, all pass |
| Setup | `lora.py` three parameterizations + rank/parity tests | ● | Pure functions over plain arrays (same pattern as algebra.py/probes.py), no model coupling. unconstrained: standard `(alpha/r) B A` on any rectangular shape. symmetric/antisymmetric: only defined on a square delta, built from `rank // 2` vector pairs — `rank` must be even (parity theorem). 27 tests (unit oracles/discrimination + property invariants), all pass |
| G0 | Induction transition located at 70m, bracket width measured | ● | **Gate passed.** 7-way worker sweep completed 154/154 checkpoints (data/g0_sweep_worker{0-6}.jsonl); scripts/close_g0.py merged the shards, ran `locate_transition`, and emitted results/g0_transition.jsonl. **A = step 512, B = step 2000, bracket_width = 2** (≤ 3 threshold). N_eval=512, T=128, seed=0 per §5 |
| G1 | Prerequisite check at A: prev-token head, $W_K$ overlap | ● | **Gate passed.** At A (step 512): previous-token head found at layer 2, head 1 (score 0.356 ≥ 0.3). Candidate induction head (layer 3, head 6, identified from B's PMS) has $\|P_{\text{prev-tok}}W_K\|_F/\|W_K\|_F = 0.387$ at A, vs. a random-subspace-null 95th percentile of 0.360 — significant, but the margin (0.027) is narrow. results/g1_prerequisite.jsonl |
| G2 | CPU wall-clock for one run to criterion at 70m | ◐ | **Gate. Decides CPU vs GPU.** `train.py` built (LoRA hook injection for QK/OV, freeze/train loop, snapshotting); `scripts/run_g2.py` running the real timed generous-rank run against checkpoint A now |
| G3 | Positive control: generous rank on $W_Q$ alone reaches criterion | ○ | **Gate. Fails → single-matrix rule too strict** |
| M1 | QK rank sweep (adapt $W_Q$ only) | ○ | H1 |
| M2 | OV rank sweep (adapt $W_O$ only) | ○ | H1, H3 |
| M3 | Matched-β parameterization triad (unconstrained / sym / antisym) | ○ | H2 |
| M4 | Antisymmetric fraction φ for both arms | ○ | H2 |
| M5 | SVD truncation probe on the generous-rank solution | ○ | §7 minimality |
| M6 | Matched-norm random-update nulls | ○ | h0₃, h0₄ |
| M7 | 10×-steps arm on every failing rank cell | ○ | h0₂ — mandatory before any failure claim |
| M8 | Held-out natural-text loss, before/after | ○ | collateral damage |
| D1 | Principal angles vs pretraining delta, winning arm only | ○ | **Deferred scope — single instance, not a sweep** |

Legend: ○ not started · ◐ in progress · ● done · ✗ failed/abandoned

---

## 1. Framing

### 1.1 What we are not doing

We are not reproducing the weight delta $W(B) - W(A)$. Pythia's public grid is 154 checkpoints — step 0, log-spaced to 512, then every 1000 steps. Induction onset sits near the seam. Any available A/B pair spans billions of tokens of general LM improvement, of which the induction circuit is a small fraction. A rank measured on that delta measures the wrong object.

We are also not measuring information flow. LoRA leaves activations full-rank; every token still passes through the full $d_{\text{model}}$. What rank constrains is the dimension of the *change to the operator*. The correct name for the measured quantity is the **intrinsic dimension of the update**, not bandwidth or channel capacity. The project title is a legacy label; the claims must use the precise term.

### 1.2 What we are doing

Define the target behaviorally. Train a rank-limited update from checkpoint $A$ whose objective is to install induction behavior. $B$ is used only as the reference ceiling for the recovery metric and — in deferred scope only — as the comparison object for subspace identity.

### 1.3 Stated up front

An update trained from $A$ installs *an* induction circuit, not necessarily $B$'s. Everything here is a claim about the loss landscape near $A$, not about the pretraining trajectory. Only D1 bridges to a developmental claim, and only if it comes back positive. **Until D1 runs and passes, every result is phrased as reachability.** Developmental language in an abstract backed only by M1–M8 is the failure mode this paragraph exists to prevent.

---

## 2. Substrate

**Model: `EleutherAI/pythia-70m`.** 6 layers, $d_{\text{model}} = 512$, 8 heads, $d_{\text{head}} = 64$. Chosen over 14m because induction formation at 6 layers / 4 heads / $d=128$ is not reliably documented and the substrate risk is not worth the speed. Seed variants exist at 70m if a cross-seed band is ever needed.

410m is out of scope for training on CPU: ~30× the per-step cost and ~1.6 GB of fp32 weights before optimizer state. Available for forward-only probe sweeps if a scale check is wanted; not for any sweep.

**Checkpoint definitions.**
- $A$ — last checkpoint before the transition. No head above PMS 0.1; ICL score at baseline.
- $B$ — first checkpoint after the transition stabilizes: ICL within 10% of its value two checkpoints later.

Both selected by behavioral probe. **LLC / SGLD is out of scope entirely** — we know which transition we are targeting, so the detection machinery is unnecessary. This is a deliberate narrowing, logged in §10.

---

## 3. The algebra

Full derivation in `docs/mathematics.md`. The load-bearing results:

**The two square forms.** $M_{QK} = W_Q W_K^\top$ and $M_{OV} = W_O^\top W_V^\top$, both $d \times d$, both rank $\le d_{\text{head}}$. Individual $W_Q, W_K, W_V, W_O$ are rectangular; symmetry is undefined for them, so all symmetry claims are about the composed forms only.

**Decomposition.** $M = S + \Lambda$ with $S = \frac12(M + M^\top)$, $\Lambda = \frac12(M - M^\top)$, Frobenius-orthogonal. The antisymmetric fraction:

$$\varphi(M) = \frac{\|\Lambda\|_F^2}{\|M\|_F^2} = \frac12\left(1 - \frac{\operatorname{tr}(M^2)}{\|M\|_F^2}\right)$$

**Interpretation.** $\text{score}(i,j) - \text{score}(j,i) = \frac{2}{\sqrt{d_h}} x_i^\top \Lambda x_j$. The symmetric part is order-blind — it implements similarity. Only the antisymmetric part carries directional structure.

**The ceiling result.** If $M$ is supported on an off-diagonal block ($M = P_U M P_V$, $U \perp V$) then $\operatorname{tr}(M^2) = 0$ and $\varphi = 1/2$ **exactly, at any rank**. Prefix matching is exactly this shape. So the prediction is $\varphi \approx 1/2$, not $\varphi \to 1$.

**Rank-1 is not barred.** $\varphi(ba^\top) = \frac12(1 - c^2)$ with $c = \hat a \cdot \hat b$, maximized at $1/2$ when $a \perp b$. A single relational term $uv^\top$ is cleanly rank-1. It is the *constrained* parameterizations that force companion terms: pure-antisym gives $uv^\top - vu^\top$, pure-sym gives $uv^\top + vu^\top$.

**What sets required rank.** The target operator is $M_{QK}^\star \approx \lambda \sum_t e_t p_t^\top = \lambda E P^\top$, so required rank is $\operatorname{rank}(EP^\top)$ — the dimension of the token code the match must resolve. Default expectation is a smooth $R(r)$ tracking that spectrum. A sharp low-rank threshold would be the surprising result.

**Composition rule.** Adapting both $W_Q$ and $W_K$ gives $\Delta M_{QK}$ of rank up to $3r$ with no parity structure. **Adapt exactly one matrix per form**: $W_Q$ for QK, $W_O$ for OV.

---

## 4. The circuit, split

Three components across two layers. A lumped metric cannot tell you which was the bottleneck.

| # | Component | Circuit | Role here | Metric |
|---|---|---|---|---|
| 1 | Previous-token head | QK, positional | **Prerequisite, not a target** | prev-token score: mean attn $t \to t-1$ |
| 2 | Prefix matching | QK | Arm 1: adapt $W_Q$ only | PMS: mean attn $t \to$ successor of prior occurrence |
| 3 | Copying | OV | Arm 2: adapt $W_O$ only | copying score: fraction of $t$ with $\arg\max W_U M_{OV} e_t = t$ |

Report $r^\ast_{QK}$ and $r^\ast_{OV}$ separately. They have no reason to coincide.

---

## 5. Metrics

Fixed evaluation set, fixed seed. Any change to a definition invalidates prior results and **must** bump `METRIC_VERSION` in `schema.py`. CI enforces this (§9).

**Repeated-random sequences.** $T = 128$ tokens sampled uniformly from vocab, concatenated to length $2T$. $N_{\text{eval}} = 512$.

**Prefix-matching score (PMS), per head.** Mean attention from position $t$ to $t - T + 1$, averaged over $t$ in the second copy. Induction-bearing at PMS $\ge 0.3$.

**Copying score, per head.** Fraction of vocabulary tokens $t$ for which $\arg\max_s (W_U M_{OV} e_t)_s = t$.

**ICL score.** $\text{NLL}_{\text{first copy}} - \text{NLL}_{\text{second copy}}$. Higher is stronger.

**Recovery fraction.** $R(X) = \dfrac{\text{ICL}(X) - \text{ICL}(A)}{\text{ICL}(B) - \text{ICL}(A)}$. $R(A) = 0$, $R(B) = 1$ by construction. Clamped for reporting, unclamped logged.

**Bandwidth.** $\beta = r(d_{\text{in}} + d_{\text{out}})$, reported absolutely and as a fraction of the adapted matrix's parameters.

**Antisymmetric fraction.** $\varphi$ as in §3, computed on $\Delta M_{QK}$ and $\Delta M_{OV}$.

**Prerequisite overlap.** $\|P_{\text{prev-tok}} W_K\|_F / \|W_K\|_F$ at checkpoint $A$.

---

## 6. Pre-registered hypotheses

Thresholds fixed now, before data. Decision bands on $R$: **success** $R \ge 0.80$ · **failure** $R < 0.50$ · **ambiguous** in between, reported as ambiguous and never rounded into a verdict.

### H1 — Intrinsic dimension

*There is a rank $r^\ast$ at which the arm reaches criterion, and $R(r)$ tracks the singular-value spectrum of the fitted update.*

| Null | Statement | Falsified by |
|---|---|---|
| h0₁ | $R$ is flat in $r$ | Monotone rise in $R(r)$ |
| h0₂ | At $r < r^\ast$, 10× steps closes the gap | Plateau: $R$ stops improving while $< 0.50$ |
| h0₃ | Matched-$\beta$ unstructured update performs identically | Gap at matched $\beta$ |

h0₂ carries the scientific content: **reaches criterion slower → optimization limit; plateaus below → capacity limit.** No rank-level failure may be reported without both the matched-lr arm and the 10×-steps arm attached.

### H2 — Structural asymmetry

*$\varphi(\Delta M_{QK}) \approx 0.5$ and $\varphi(\Delta M_{OV})$ is materially below it.*

| Null | Statement | Falsified by |
|---|---|---|
| h0₄ | $\varphi_{QK}$ and $\varphi_{OV}$ are indistinguishable | Separation exceeding the matched-norm random-null band |

**Stated limitation, not to be omitted from any writeup:** random matrices also concentrate near $\varphi = 1/2$, so $\varphi_{QK} \approx 0.5$ against a random null is *uninformative*. The OV side carries the signal. Report the contrast, never $\varphi_{QK}$ alone as a positive finding.

### H3 — Separability

*$r^\ast_{QK} \ne r^\ast_{OV}$.*

| Null | Statement | Falsified by |
|---|---|---|
| h0₅ | The two sub-circuits share a minimum rank | Non-overlapping thresholds across seeds |

### H4 — Identity *(deferred scope, single instance)*

*The learned QK subspace aligns with the pretraining delta restricted to the same head.*

| Null | Statement | Falsified by |
|---|---|---|
| h0₆ | Principal angles indistinguishable from a random $r$-dim subspace | Mean $\cos$ above the 95th percentile of 100 random draws |

This is the only bridge to a developmental claim. Run it once, on the winning arm, on one matrix. Do not build the sweep.

---

## 7. What can and cannot be concluded

**Sufficiency is sound.** "Rank $r$ suffices to install prefix matching from $A$" is constructive — you exhibit the update. Upper bound on intrinsic dimension. A real result.

**Necessity is not obtainable.** Failure at rank $r'$ is failure of a tuple: this optimizer, objective, initialization, budget, parameterization, checkpoint. The problem is non-convex; there is no certificate that no rank-$r'$ solution exists. **"Rank $r'$ is insufficient" must never appear without its qualifiers.**

**The strongest available necessity evidence is M5, the truncation probe:** train at generous rank to get $\Delta W^\star$, truncate its SVD to rank $r'$, evaluate the truncation directly. This removes training dynamics from the inference entirely — it is evidence about the object rather than about the optimizer, and it is strictly better than a failed training run.

**Out of reach entirely:** "the minimum structure an induction head can take." That quantifies over all implementations in all models. Reachable claim: *the minimum-dimension update installing prefix matching from $A$, under this parameterization, has rank $\approx r^\ast$, and its QK component has antisymmetric fraction $\approx \varphi$.*

---

## 8. Gates and protocol

Ordered by kill-power per compute-hour. Each has unambiguous pass/fail. Do not proceed past a failure without a §10 entry.

**G0 — Transition location.** *(hours, forward passes only)* Sweep PMS per head and ICL over the checkpoint grid at 70m. Locate $A$ and $B$. Measure bracket width. **Fail:** no transition, or bracket wider than 3 consecutive checkpoints.

**G1 — Prerequisite.** *(minutes)* At $A$: does a previous-token head exist (prev-token score $\ge 0.3$)? Does $\|P_{\text{prev-tok}} W_K\|_F / \|W_K\|_F$ show non-negligible overlap on the candidate induction head? **Fail:** with $W_K$ frozen, no $\Delta W_Q$ at any rank can build the match — a null result here is a missing prerequisite, not a capacity limit, and reading it as the latter is the most expensive available mistake.

**G2 — CPU timing.** *(one run)* Wall-clock one generous-rank run to criterion at 70m. **This has never been measured; every schedule downstream assumes it.** Output decides CPU-only vs. GPU and the affordable sweep size.

**G3 — Positive control.** *(a day)* Generous rank on $W_Q$ alone. Does $R$ exceed 0.80 at all? **Fail:** the single-matrix composition rule (§3) is too restrictive; fall back to per-circuit adaptation and downgrade the H2 claim, or the objective/loop is broken.

**Main sweep, after all gates pass.** Per arm: rank $r \in \{1,2,4,8,16,32\}$ × lr (≥3 points, swept independently per rank) × 3 seeds. Parameterization triad at matched $\beta$. Plus M5–M8.

**Mandatory controls.** Matched-norm random rank-$r$ null. Matched-$\beta$ unstructured control. Held-out natural-text loss before/after. **Freeze LayerNorm and embeddings throughout** — otherwise $\Delta M_{QK}$ is not the only thing that changed and the symmetry analysis does not hold.

**The $\alpha/r$ confound.** Rank changes the effective learning rate on $\Delta W$. Either hold $\alpha/r$ fixed or sweep lr independently per rank. This is the single most likely way to produce a clean-looking curve that means nothing.

**Storage.** LoRA factors only, never merged models. Snapshot adapters every $N$ steps — they are tiny, keep all of them; they give the formation dynamics for free.

---

## 9. Falsification discipline

Method borrowed from the POPPER framework for automated hypothesis validation (Huang, Jin, Li et al. 2025, arXiv 2502.09858), with its own error analysis as the guide to what to avoid. Popper's underlying point is the operative one: **corroboration counts only in proportion to how severe the test was.** A hypothesis that survives a test which could easily have killed it earns more than one that survives a token test.

Taken from POPPER:
- Pre-registration of nulls and thresholds before data exists.
- Explicit implication chains: for each null, if the main hypothesis is false, is this null necessarily true? h0₃ exists as a relevance guard; h0₄'s stated limitation exists because the QK arm's implication chain is weak.
- Humans fix hypotheses and falsification tests. **Agents execute the protocol, produce plots, and flag threshold crossings.** POPPER's own error analysis puts the largest failure concentration in agent-designed tests.

Left behind deliberately: the e-value / sequential-testing aggregation. It controls Type-I error over a large unbounded space of generated tests against a static database. We have four hypotheses with one or two canonical tests each and we control data generation. The aggregation buys nothing and the α-ritual would manufacture false confidence.

POPPER's failure taxonomy, mapped:
- *Misinterpreted statistics (35.9%)* → reading "rank 2 got $R = 0.31$" as capacity when it is optimization. Guarded by h0₂ + M7.
- *Tests that break the implication (17.2%)* → guarded by h0₃ and by the h0₄ limitation note.
- *Ineffective test design (28.1%)* → guarded by G0–G3 running before compute is spent.

**Every status-board row that reaches ● must emit a results record containing:** the null tested, the pre-registered falsification criterion, the observed value, the verdict, and `METRIC_VERSION`. A row cannot be closed without it. This is schema-enforced in CI, not a convention.

---

## 10. Decisions log

Append-only. Never edit an entry — supersede it.

| Date | Decision | Reason |
|---|---|---|
| (init) | Target is capability-install, not delta-reproduction | Checkpoint grid too coarse; delta contaminated by general LM improvement (§1.1) |
| (init) | Success τ=0.80, failure τ=0.50, ambiguous between | Pre-registration; prevents post-hoc threshold fitting |
| 2026-08 | Headline model is pythia-70m, not 14m | Induction formation unreliable at 6L/4H/d=128; substrate risk not worth the speed |
| 2026-08 | LLC / SGLD dropped entirely | Target transition is known; detection machinery unnecessary |
| 2026-08 | Markov substrate, crosscoders, sparsity, agreement readouts dropped | Different question; superseded design |
| 2026-08 | Package renamed `sltdiff` → `indbw` | SLT is no longer in scope; the old name misdescribes the work |
| 2026-08 | Parity prediction ($r^\ast = 2$ from antisymmetry) **withdrawn** | Off-diagonal blocks have $\varphi = 1/2$ exactly at any rank; rank-1 achieves the maximum. Corrected prediction: rank set by $\operatorname{rank}(EP^\top)$, §3 |
| 2026-08 | Circuit split into prev-token / QK / OV with separate arms and metrics | Opposite predicted symmetry; a lumped metric conflates them |
| 2026-08 | Adapt one matrix per bilinear form | Cross terms destroy the rank and parity structure (§3) |
| 2026-08 | H4 (subspace identity) reduced to a single deferred instance | Full sweep is the largest time sink; one instance answers the first question anyone asks |
| 2026-08 | M5 truncation probe added | Best available necessity evidence; removes optimizer from the inference (§7) |
| 2026-08-08 | Fixed invalid YAML in `.github/workflows/ci.yml` (unquoted `run: echo "...: ..."` parsed as an illegal nested mapping) | Every push since the workflow was added had produced 0 jobs and an immediate failure — CI was never actually green despite the Setup-row-1 code existing since 57f2dc2. Caught while closing the "CI green on empty suite" row; wrapped the offending step in a block scalar (`run: \|`) |
| 2026-08-11 | `probes.py` takes tensors (attention patterns, OV/embedding/unembedding matrices, per-token NLL), never a `HookedTransformer` | Keeps probes unit-testable against hand-constructed toy inputs per CLAUDE.md's fixture rule, and decouples this row from `models.py` (still ○). Orchestration — running a model to extract these tensors — belongs to whatever calls the probes, not to probes.py itself |
| 2026-08-11 | `recovery()` returns the unclamped value; clamping for reporting is a separate `clamp_recovery()` call, not a flag | Matches §5 ("clamped for reporting, unclamped logged") as two composable functions instead of a boolean that silently changes return semantics |
| 2026-08-11 | Pinned `transformers==5.9.0` exactly in `pyproject.toml` | Unpinned installs pulled `transformers` 5.14.1, whose `GPTNeoXForCausalLM` renamed the LM head from `embed_out` to `lm_head`; `transformer_lens` 3.7.0's neox weight conversion hardcodes `embed_out` and crashes on load. 5.9.0 is `transformer_lens`'s own declared floor and confirmed to still have `embed_out` |
| 2026-08-11 | `models.py`'s checkpoint grid is read from `transformer_lens.loading_from_pretrained.get_checkpoint_labels`, not hand-maintained | Same rationale as the probes.py tensor-only decision above: avoids a second copy of the 154-step Pythia grid that could silently drift from what `HookedTransformer.from_pretrained` actually accepts |
| 2026-08-11 | `wrapper.sh` and `worker-bootstrap.sh` now run `pip install -e ".[dev]"` after checkout, before doing any work | The transformers pin fix above only affects a fresh install — `worker-bootstrap.sh` and `wrapper.sh` were only ever `source`-ing a venv baked into the AMI at image-build time, never reinstalling from `pyproject.toml`. A dependency fix merged to the repo would silently never reach an instance launched from an older AMI. Reinstalling on every launch makes the venv self-healing from git state; costs boot time, not correctness |
| 2026-08-12 | `lora.py`'s symmetric/antisymmetric parameterizations operate on the composed square delta directly (`rank // 2` vector-pairs, `U V^T ± V U^T`), not on `W_Q`/`W_O`'s native rectangular shape | Symmetry is undefined on a rectangular matrix (§3), so the constrained parameterizations can only apply to the square composed form. `rank` is forced even by the parity theorem (docs/mathematics.md §6), not a convenience choice — odd rank raises. This keeps lora.py's tested surface a pure array-in/array-out module, same decoupling pattern as algebra.py/probes.py; wiring a given parameterization's output into a real `HookedTransformer`'s `W_Q`/`W_O` under the "one matrix per form" rule (§3) is left to `train.py`, not yet implemented |
| 2026-08-12 | Added `src/indbw/gates.py` (not in CLAUDE.md's original module list) for G0's transition-location logic; `scripts/g0_sweep.py` orchestrates the real checkpoint sweep | Gate-detection (locate A/B on a checkpoint grid from a PMS/ICL series) is a one-time analysis of swept metrics, not a per-example metric itself, so it doesn't belong under probes.py's METRIC_VERSION hash. `locate_transition`'s operationalization of §2's prose ("baseline" = ICL at step 0; "transition" = first grid step where max-over-heads PMS crosses 0.1) is fixed here, before any checkpoint was swept, per §11's own requirement |
| 2026-08-12 | Fixed a real upstream bug blocking every checkpointed model load: `transformer_lens`'s pythia-checkpoint branch (`loading_from_pretrained.py`, still present at 3.7.1, the latest release) passes `token=os.environ.get("HF_TOKEN","")` unconditionally, unlike its sibling branches which guard empty tokens. `huggingface_hub`'s current httpx backend rejects the resulting `Authorization: Bearer ` header outright (`LocalProtocolError`), where the older requests-based backend sent it silently. Worked around with a monkeypatch in `models.py` (`get_token_to_send` treats `""` as `None`) plus a `transformer_lens>=3.7.1` floor pin | tier-3's own integration test (`test_load_checkpoint_step_zero_...`) was silently broken by this — it isn't on the push path (CLAUDE.md tier 3), so nothing caught the regression until G0 needed real checkpoint loads across the full grid. The fix is intentionally narrow (normalizes only the empty-token case) so it can't mask a real auth failure |
| 2026-08-12 | G0's sweep processes each layer's attention pattern via a forward hook (compute PMS increment, discard) rather than `model.run_with_cache` (retain all layers' patterns until the batch loop deletes the cache dict) | `run_with_cache` across pythia-70m's 6 layers was enough peak memory, on this box's ~1.8GB RAM baseline (already tight per §11's "torch+transformers imports alone use ~700MB"), to force sustained swap thrashing — one checkpoint at the pre-registered N_eval=512/T=128 didn't finish in 500s. The hook-based rewrite produces bit-identical PMS/ICL numbers (verified against the `run_with_cache` version on a tiny pilot) at ~2.4s/batch steady state (batch_size=4), ≈5.5 min/checkpoint, ≈14h for the full 154-step grid |
| 2026-08-12 | `CLAUDE.md`'s blanket "do not run AWS commands or launch instances" narrowed to explicitly permit worker-instance lifecycle (launch/terminate via the `wrapper.sh`/`worker-bootstrap.sh` pattern, self-stop via `teardown.sh`), scoped to this project's tagged resources; no IAM changes | Requested by the user to parallelize G0's ~14h single-box sweep across workers after a mid-sweep disconnect lost unsaved progress (only step 0 had been committed). The single-box sweep is memory-bound (§11, ~1.8GB RAM), not embarrassingly-parallelizable in-process, but is embarrassingly parallel across the 154 independent checkpoints given a fixed eval set — a good fit for the existing worker pattern |
| 2026-08-13 | Built out `schema.py` (previously a 9-line stub: just `METRIC_VERSION`) into a real `Criterion`/`ResultsRecord` schema, and used it to close G0 | G0's row couldn't legitimately reach ● without a schema-validated results record (CLAUDE.md "every row that reaches ● must emit a results record"), and no prior row had ever exercised that path. `Criterion` is a single `observed[metric] <op> threshold` comparison; a record's `criteria` is a tuple of these ANDed together, which handles compound gate criteria (G0: "fail if not found, OR bracket > 3") without inventing per-row verdict logic. `ResultsRecord.recomputed_verdict()` is what tier 2.5's new `tests/schema/test_verdict_recomputation.py` calls against every record in `results/` |
| 2026-08-13 | **G0 gate closed: PASS.** A = step 512, B = step 2000, bracket_width = 2 (≤ 3) | `scripts/close_g0.py` merged all 7 worker shards (154/154 checkpoints, no gaps) and ran `locate_transition`. Result committed to `results/g0_transition.jsonl`. The 7-way worker sweep itself had actually finished ~17.5h before this closing step ran — the workers pushed their shards and self-terminated per `infra/worker-bootstrap.sh`, but nothing then re-invoked `locate_transition` over the merged data or updated the status board, so the row sat at stale ⏳ with fully-collected data behind it. Flagged in §11 |
| 2026-08-13 | Operationalized $P_{\text{prev-tok}}$ (docs/mathematics.md §9) as the orthogonal projector onto the previous-token head's $W_O$ row space, built via `indbw.algebra.subspace_projector` (SVD-based numerical rank, not raw QR) | Neither PROJECT.md nor docs/mathematics.md gave $P_{\text{prev-tok}}$ a concrete construction — it's the residual-stream write-subspace of the head that writes $p_t$ (docs/mathematics.md §8), i.e. $\operatorname{col}(W_O^\top)$. Fixed here, before checkpoint A was inspected, matching `gates.py`'s existing G0 precedent for operationalizing prose definitions before seeing data |
| 2026-08-13 | G1's "non-negligible overlap" (PROJECT.md §8, no numeric cutoff given) operationalized as: observed ratio exceeds the 95th percentile of the same ratio over 100 random rank-matched subspaces, seed 0 | Reuses the random-subspace-null convention PROJECT.md §6 already pre-registers for H4's $h0_6$, rather than inventing a new bare threshold. Implemented in `indbw.gates.k_composition_overlap` |
| 2026-08-13 | G1's candidate induction head is identified from checkpoint B's argmax-PMS head (layer 3, head 6, from `data/g0_sweep*.jsonl`), not from A's own PMS | A is pre-transition by definition (§2: "no head above PMS 0.1"), so A's own argmax-PMS head is noise, not a real circuit. The head that matters for the frozen-$W_K$ prerequisite check is whichever one *becomes* the induction head — i.e. B's |
| 2026-08-13 | **G1 gate closed: PASS.** Previous-token head at layer 2/head 1 (score 0.356 ≥ 0.3); candidate induction head's (layer 3/head 6) $W_K$ overlap with that head's output subspace = 0.387 vs. null 95th percentile 0.360 | `scripts/run_g1.py`, single forward pass over checkpoint A (step 512) under the same fixed eval set as G0. Result in `results/g1_prerequisite.jsonl`. **Margin is narrow (0.027)** — flagged as a caveat in §11 rather than treated as a comfortable pass |
| 2026-08-14 | Built `src/indbw/train.py`: LoRA injection via forward hooks (not an in-place weight edit) on `HookedTransformer`, freezing every base-model parameter so B/A are the only trainable tensors; `unconstrained` parameterization only (matches `indbw.lora`'s shape convention exactly, so a learned (B,A) hands straight to it for later symmetry/rank analysis) | G2 needed a real training loop to time; §3's "adapt exactly one matrix" and §8's "freeze LayerNorm and embeddings" needed to be mechanically enforced (no trainable path to anything but B/A) rather than left as a convention a future row could violate by accident |
| 2026-08-14 | §11's open question ("Induction objective... unresolved and consequential — settle before G3") settled: **synthetic repeated-random second-copy NLL**, minimized directly. This is the "cleanest" option §11 itself named | G2 already needs an objective to time a run against, and G2 precedes G3 — deferring further wasn't available. The named risk (a circuit that only works on synthetic input) is exactly M8's job to catch, not this decision's to prevent |
| 2026-08-14 | `train.py`'s `second_copy_nll`/`first_copy_nll` windowing matched exactly to `scripts/g0_sweep.py`'s existing (already-committed, G0/G1-closing) convention — both exclude the copy-boundary position (index T-1, predicting token T) from either half, rather than including it in the second copy as a naive reading of §5 might suggest | G2/G3 reuse G0's already-recorded ICL(A)/ICL(B) as recovery baselines (PROJECT.md §5's `recovery()`) rather than recomputing them from scratch — a caught-before-running mismatch: my first draft of `second_copy_nll` included the seam position, which would have silently computed `R` against baselines measured on a different windowing than the one being evaluated. Fixed before any real checkpoint was touched, same discipline as G0/G1's own pre-registered operationalizations |
| 2026-08-14 | G2's CPU-vs-GPU decision rule operationalized *before* this run's timing is known (PROJECT.md §8 names no numeric threshold): project the 108-cell main sweep's wall-clock using a 7-way worker fleet (G0's precedent fleet size); ≤24h → CPU stays affordable, else escalate to GPU. Recorded as a planning estimate alongside G2's results record, not folded into its pass/fail verdict | G2's stated purpose is exactly this decision ("Decides CPU vs GPU") but PROJECT.md gives no number to compare against — same category of gap as G0/G1's ambiguous prose, closed the same way: fix the rule before the data that would tempt post-hoc fitting exists |
| 2026-08-14 | `scripts/run_g2.py`'s real training run (generous rank r=64=d_head on $W_Q$ at the layer-3/head-6 induction head, checkpoint A) is written to also produce a raw JSON artifact (`data/g2_generous_rank_run.json`) that `scripts/run_g3.py` reads to close G3 without retraining | G2's timing run and G3's positive control ask for the literal same experiment (generous-rank $W_Q$-only adaptation from A, does R reach 0.80) under two different framings (how long vs. does it happen at all) — retraining a second time to answer a question the first run already answered burns exactly the compute-hours CLAUDE.md's provenance/resumability sections exist to conserve. Each row still gets its own commit, board update, and schema-validated results record |

---

## 11. Open questions

- Induction objective: synthetic repeated-random, natural-text induction positions, or KL-to-$B$? Synthetic is cleanest but risks a circuit that only works on synthetic input; KL-to-$B$ makes $B$ a training signal and partly undercuts §1.2. **Unresolved and consequential — settle before G3.** Whichever is chosen, evaluate on the other as a held-out check.
- Head selection rule at $A$ — must be stated and applied before seeing results, not chosen afterward.
- Local dev venv (`/home/ubuntu/venv`) runs Python 3.12 with numpy 2.5.1, whose stubs use 3.12-only syntax; `mypy` (pinned to `python_version = "3.11"` per pyproject) fails locally on that stub file alone, before it ever reaches repo code. No 3.11 interpreter is installed locally, so `mypy src` cannot be run locally at all right now — every future commit touching `src/` needs its mypy gate verified via CI, not pre-flight, until this venv gets a matching Python version. (Separately, real CI *did* catch two genuine `no-any-return` errors in the first `algebra.py` push — mypy strict itself is fine; only the local reproduction path is broken.)
- Does the matched-$\beta$ unstructured control fix its random subspace before training or resample it? Fixed is the right analogue to LoRA.
- Is the antisym parameterization's companion term $-vu^\top$ actually benign? The prediction assumes the $+$ version interferes more. Untested.
- Cross-seed *pretraining* variance band: needed for any "X misses the mechanism" claim, expensive, currently deferred. Do not make claims requiring it.
- Identity-matrix oracles are transpose-blind: `copying_score(I, I, I) == 1.0` passes even under a `W_U`/`M_OV` argument-order bug, since `I.T == I`. Closed here with an independent per-token recomputation as a second oracle. `lora.py` and `models.py` will hit the same trap testing composed forms against identity/permutation weights — plan for a non-self-transpose oracle alongside any identity-based one.
- `lora.py`'s symmetric/antisymmetric deltas are square (`d x d`, matching `M_QK`/`M_OV`), but the "one matrix per form" rule (§3) says the *trained* object is `Delta W_Q` (rectangular, `d x d_head`) with `W_K` frozen — so `Delta M_QK = Delta W_Q W_K^T` is generically neither symmetric nor antisymmetric no matter how `Delta W_Q` is parameterized, since its row space is constrained to lie in `W_K`'s column space. train.py (○, not yet built) will need to decide how M3's triad actually gets *installed* into the model under this constraint: either (a) solve for the `Delta W_Q` whose composition with frozen `W_K` best matches the target square delta (a projection, not exact for arbitrary targets), or (b) inject the square delta directly via a forward hook on the attention-score computation itself, which respects "touch only one logical piece" in spirit (no `Delta W_Q Delta W_K^T` cross term) without being a literal weight edit to `W_Q`. Undecided — flagging now so M3 doesn't silently pick one without it being a visible choice.
- **`wrapper.sh`'s "close out the row" step is not actually triggered by worker completion.** Per CLAUDE.md's CD section, `wrapper.sh` is supposed to wait for all workers to terminate and then re-invoke Claude Code to close the row — but in practice the 7 G0 workers finished, pushed their shard commits, and self-terminated, and the row then sat at stale ⏳ (fully-collected data, gate never actually evaluated) for ~17.5 hours until a manual session noticed and ran `scripts/close_g0.py`. Worth checking whether `wrapper.sh`'s polling loop is actually being kept alive on a real EC2 orchestrator instance across the whole worker run, or whether it only works when someone is present to launch it and then the box gets stopped/disconnected before the wait-and-close step executes. Until that's confirmed, don't assume a sweep row is done just because its workers finished — check for the results record.
- **G1's overlap-significance margin is narrow: 0.387 observed vs. 0.360 null 95th percentile (margin 0.027, ~7% relative).** The gate passed under the pre-registered criterion, but this is not a comfortable pass — a different null seed or a slightly different percentile choice (90th vs. 95th) could plausibly flip it. Since G1 is a one-time gate (not something to re-run/re-threshold post-hoc per CLAUDE.md's falsification discipline), this is not being re-litigated, but M1 (QK rank sweep) should be read with this in mind: if the QK arm struggles to reach criterion at low rank, this narrow prerequisite margin is a real alternative explanation to a capacity limit, not just h0₂'s optimization-limit story.
- This session's coding sandbox has ~1.8GB RAM with no swap — insufficient to load `pythia-70m` end to end (torch+transformers imports alone use ~700MB before the model; the HF→TL weight-conversion step OOM-killed the process every time with no swap). Worked around with a temporary 3GB swapfile (`fallocate`/`mkswap`/`swapon`, removed after); `tests/integration/test_models.py` then passed (2/2). No `gh workflow run` access from this session's PAT (403, insufficient scope) so `tier3-integration` couldn't be triggered as a fallback — the swapfile workaround was the only path to a definitive local read. Future sessions hitting the same OOM on an integration test should reach for temporary swap rather than assuming the code is broken. Recurred for G1's single-checkpoint forward pass (`scripts/run_g1.py`): even with `return_type=None` (skipping the unembedding matmul, since G1 only needs attention patterns), loading + one forward pass over checkpoint A still needed the same temporary-swapfile workaround on this box. A checkpoint load alone appears to be right at this box's ceiling regardless of what the forward pass computes.

---

## 12. Reading

- Elhage et al. 2021 — QK/OV circuits, K-composition; the framework §3 is written in
- Olsson et al. 2022 — induction heads, prefix-matching and copying scores
- Li et al. 2018; Aghajanyan et al. 2020 — intrinsic dimension of task updates; the correct framing for §1.1
- Hu et al. 2021 — LoRA
- Kalajdzievski 2023 — $\sqrt r$ scaling; bears on the §8 $\alpha/r$ confound
- Biderman et al. 2023 — Pythia suite, checkpoint structure, seed variants
- Huang, Jin, Li et al. 2025 (arXiv 2502.09858) — automated falsification; see §9 for what is taken and what is left
