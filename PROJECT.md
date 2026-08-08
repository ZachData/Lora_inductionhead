# Induction Bandwidth

**What is the minimum-dimension weight update that installs a working induction circuit into a pre-induction Pythia checkpoint, and does the update's symmetry structure differ between the circuit's prefix-matching and copying halves?**

This is the living specification and the source of truth. It is updated at the end of every working session. If a claim here contradicts the code, one of them is wrong — resolve it explicitly in §10 and never let both stand.

---

## Status board

| Phase | Item | Status | Notes |
|---|---|---|---|
| Setup | `pyproject.toml`, package skeleton, CI green on empty suite | ● | CI run 31238848784 green on commit 1af4d12: tier0/1/2/2.5 all pass, tier3 correctly skipped (workflow_dispatch only) |
| Setup | `algebra.py` + unit tests against closed-form oracles | ○ | §3 identities are exact; test to machine precision |
| Setup | `probes.py` + silent-failure guards | ○ | Every probe ships with a negative control test |
| Setup | `models.py` Pythia loader + local checkpoint cache | ○ | |
| Setup | `lora.py` three parameterizations + rank/parity tests | ○ | |
| G0 | Induction transition located at 70m, bracket width measured | ○ | **Gate. Fails → §8** |
| G1 | Prerequisite check at A: prev-token head, $W_K$ overlap | ○ | **Gate. Cheapest kill. Run before G2** |
| G2 | CPU wall-clock for one run to criterion at 70m | ○ | **Gate. Decides CPU vs GPU** |
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

---

## 11. Open questions

- Induction objective: synthetic repeated-random, natural-text induction positions, or KL-to-$B$? Synthetic is cleanest but risks a circuit that only works on synthetic input; KL-to-$B$ makes $B$ a training signal and partly undercuts §1.2. **Unresolved and consequential — settle before G3.** Whichever is chosen, evaluate on the other as a held-out check.
- Head selection rule at $A$ — must be stated and applied before seeing results, not chosen afterward.
- Does the matched-$\beta$ unstructured control fix its random subspace before training or resample it? Fixed is the right analogue to LoRA.
- Is the antisym parameterization's companion term $-vu^\top$ actually benign? The prediction assumes the $+$ version interferes more. Untested.
- Cross-seed *pretraining* variance band: needed for any "X misses the mechanism" claim, expensive, currently deferred. Do not make claims requiring it.

---

## 12. Reading

- Elhage et al. 2021 — QK/OV circuits, K-composition; the framework §3 is written in
- Olsson et al. 2022 — induction heads, prefix-matching and copying scores
- Li et al. 2018; Aghajanyan et al. 2020 — intrinsic dimension of task updates; the correct framing for §1.1
- Hu et al. 2021 — LoRA
- Kalajdzievski 2023 — $\sqrt r$ scaling; bears on the §8 $\alpha/r$ confound
- Biderman et al. 2023 — Pythia suite, checkpoint structure, seed variants
- Huang, Jin, Li et al. 2025 (arXiv 2502.09858) — automated falsification; see §9 for what is taken and what is left
