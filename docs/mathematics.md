# The mathematics of the LoRA-rank induction experiment

Everything from LoRA's definition to what the symmetric/antisymmetric split can and cannot establish. Read §7–§9 carefully: the algebra does not support the prediction stated in earlier conversation, and the corrected version is a better experiment.

§1–§11 are the original derivation. **§12–§18 are a later extension** asking a different question: whether an induction circuit *exists* as a low-rank update at all, separately from whether training finds one. Their one-line answer is §18 — existence here is decidable rather than merely assertable, so this is not the "theoretically possible but empirically uncertain" situation of a universal-approximation theorem; the theoretical half is decidable and simply has not been decided.

---

## 1. Notation

| Symbol | Meaning |
|---|---|
| $d$ | `d_model` (512 for pythia-70m) |
| $d_h$ | `d_head` (64 for pythia-70m, 8 heads) |
| $x_i \in \mathbb{R}^{d}$ | residual stream at position $i$ |
| $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_h}$ | per-head projections |
| $W_O \in \mathbb{R}^{d_h \times d}$ | per-head output projection |
| $A$ | pre-induction checkpoint |
| $B$ | post-induction checkpoint |
| $\|M\|_F$ | Frobenius norm, $\|M\|_F^2 = \operatorname{tr}(M^\top M)$ |

Frobenius inner product: $\langle X, Y\rangle = \operatorname{tr}(X^\top Y)$.

---

## 2. LoRA

Freeze $W$, add a trained low-rank term:

$$W' = W + \Delta W, \qquad \Delta W = \frac{\alpha}{r} B A, \quad B \in \mathbb{R}^{d_{\text{out}} \times r},\ A \in \mathbb{R}^{r \times d_{\text{in}}}$$

so $\operatorname{rank}(\Delta W) \le r$, and the trainable parameter count is $\beta = r(d_{\text{in}} + d_{\text{out}})$.

**Two things this does not do.**

It does not bottleneck activations. Every token still passes through the full $d$-dimensional residual stream; $W'$ is generically full-rank. The constraint is on the *change to the operator*, not on information flow. The right name for what we are measuring is the **intrinsic dimension of the update** (Li et al. 2018; Aghajanyan et al. 2020), not channel capacity.

It does not reduce forward/backward FLOPs. The frozen base still runs in full at every step. CPU feasibility is a question about the base model's size, not about $r$.

**The $\alpha/r$ trap.** Because the scale factor depends on $r$, changing rank changes the effective learning rate on $\Delta W$. Any "lower rank converges slower" curve is an optimizer artifact unless $\alpha/r$ is held fixed or lr is swept independently per rank. This is the single most likely way to produce a clean-looking result that means nothing.

---

## 3. The two square matrices

Attention has two bilinear forms, and both are $d \times d$ — which is what makes the whole symmetry story possible.

**QK (where to attend).** With $q_i = W_Q^\top x_i$ and $k_j = W_K^\top x_j$:

$$\text{score}(i,j) \;=\; \frac{q_i \cdot k_j}{\sqrt{d_h}} \;=\; \frac{1}{\sqrt{d_h}}\, x_i^\top \underbrace{W_Q W_K^\top}_{=:\,M_{QK}} x_j$$

**OV (what to move).** The head's write-back at position $i$:

$$\text{out}_i \;=\; \sum_j \alpha_{ij}\, \underbrace{W_O^\top W_V^\top}_{=:\,M_{OV}} x_j$$

Both $M_{QK}, M_{OV} \in \mathbb{R}^{d \times d}$, each of rank $\le d_h$. These are Elhage et al.'s QK and OV circuits. Individual $W_Q, W_K$ are $d \times d_h$ — rectangular, so "symmetric" is undefined for them. The decomposition only applies to the composed forms.

---

## 4. Symmetric / antisymmetric decomposition

Any square $M$ splits uniquely:

$$M = S + \Lambda, \qquad S = \tfrac{1}{2}(M + M^\top),\quad \Lambda = \tfrac{1}{2}(M - M^\top)$$

with $S^\top = S$, $\Lambda^\top = -\Lambda$. The two parts are Frobenius-orthogonal, $\langle S, \Lambda\rangle = 0$, so

$$\|M\|_F^2 = \|S\|_F^2 + \|\Lambda\|_F^2 .$$

Closed forms worth having, since they make this a two-line computation on any $\Delta W$:

$$\|S\|_F^2 = \tfrac{1}{2}\big(\|M\|_F^2 + \operatorname{tr}(M^2)\big), \qquad \|\Lambda\|_F^2 = \tfrac{1}{2}\big(\|M\|_F^2 - \operatorname{tr}(M^2)\big)$$

Define the **antisymmetric fraction**

$$\boxed{\;\varphi(M) \;=\; \frac{\|\Lambda\|_F^2}{\|M\|_F^2} \;=\; \frac{1}{2}\left(1 - \frac{\operatorname{tr}(M^2)}{\|M\|_F^2}\right) \;\in\; [0,1]\;}$$

This is the primary readout of the whole experiment and it costs one matrix multiply.

---

## 5. What the split means for attention

$$x_i^\top M x_j = \underbrace{x_i^\top S x_j}_{\text{order-blind}} + \underbrace{x_i^\top \Lambda x_j}_{\text{order-sensitive}}$$

because $x_j^\top S x_i = x_i^\top S x_j$ while $x_j^\top \Lambda x_i = -\,x_i^\top \Lambda x_j$. Equivalently:

$$\text{score}(i,j) - \text{score}(j,i) \;=\; \tfrac{2}{\sqrt{d_h}}\, x_i^\top \Lambda x_j$$

**The symmetric part cannot distinguish "$i$ attends to $j$" from "$j$ attends to $i$."** It implements *similarity* — attend to things that look like me. The antisymmetric part is the only source of asymmetric, directional, relational structure.

Induction is relational: attend from the current token to *the successor of* a previous occurrence of that token. Two different roles. That is the intuition for expecting $\varphi$ to be large. §7 makes it exact.

---

## 6. The parity theorem

**Theorem.** A real antisymmetric matrix has even rank. In particular, a nonzero real antisymmetric matrix has rank $\ge 2$.

*Proof sketch.* Real antisymmetric matrices are orthogonally similar to a block-diagonal form with $2\times2$ blocks $\begin{psmallmatrix}0 & \theta_k\\ -\theta_k & 0\end{psmallmatrix}$ and a zero block. Rank $= 2 \times \#\{\theta_k \ne 0\}$. Equivalently: eigenvalues are purely imaginary and come in conjugate pairs $\pm i\theta_k$. $\square$

**Corollary.** No rank-1 update can be purely antisymmetric. At $r = 2$ pure antisymmetry is constructible: $\Lambda = a_2 a_1^\top - a_1 a_2^\top$.

This corollary is what motivated the earlier "$r^\ast = 2$" prediction. §7 shows the prediction does not follow.

---

## 7. What rank-1 actually buys — the correction

Take $M = b a^\top$ with $\|a\| = \|b\| = 1$ and $c = a\cdot b$. Then $\operatorname{tr}(M^2) = (a^\top b)^2 = c^2$ and $\|M\|_F^2 = 1$, so

$$\boxed{\;\varphi(ba^\top) = \tfrac{1}{2}\,(1 - c^2)\;}$$

Two limits:

- $a \parallel b$ ($c = \pm1$): $\varphi = 0$. Pure similarity — the "attend to things like me" operator.
- $a \perp b$ ($c = 0$): $\varphi = 1/2$. **Maximally relational, at rank 1.**

More generally, if $M$ is supported on an off-diagonal block — $M = P_U M P_V$ with $U \perp V$ — then $P_U P_V = 0$ forces $(M^\top)^2 = 0$, hence $\operatorname{tr}(M^2) = 0$ and

$$\varphi(M) = \tfrac{1}{2} \quad \text{exactly, at any rank.}$$

**This is the correction.** A purely relational operator — query reading subspace $U$, key reading a disjoint subspace $V$ — has antisymmetric fraction exactly $1/2$, never more. $\varphi \to 1$ requires a *skew* map within a single subspace, which is not what prefix matching is.

Consequences:

1. **The prediction is $\varphi \approx 1/2$, not $\varphi \approx 1$.** "Induction is antisymmetric" is false as stated; "induction is off-diagonal, hence half-antisymmetric" is the true version.
2. **Rank-1 is not structurally barred.** $uv^\top$ with $u \perp v$ *is* a clean single-direction relational term with no companion.
3. **The constrained parameterizations are the ones that force companions.** Pure-antisymmetric $uv^\top - vu^\top$ gives the wanted term *and* its reverse with a minus sign; pure-symmetric $uv^\top + vu^\top$ gives it with a plus sign. Unconstrained rank-1 gives the wanted term alone.

So the sym-vs-antisym contrast at matched $\beta$ is not "expressible vs. inexpressible." It is a test of **whether the obligate reverse term helps or hurts**: $-vu^\top$ suppresses attention from $i$ to tokens matching $i$'s predecessor; $+vu^\top$ adds it, competing with the induction pattern. A real prediction, but a weaker one than parity suggested, and it should be labelled as such.

---

## 8. What sets the required rank

Prefix matching must fire when $\text{token}(j-1) = \text{token}(i)$. Let $e_t$ be the residual direction carrying "current token is $t$" on the query side, and $p_t$ the direction carrying "previous token was $t$" on the key side — the latter written into the residual stream by the previous-token head. The target operator is

$$M_{QK}^{\star} \;\approx\; \lambda \sum_{t \in \mathcal{V}} e_t\, p_t^\top \;=\; \lambda\, E P^\top$$

$$\Rightarrow\quad \operatorname{rank}(M_{QK}^\star) \;=\; \operatorname{rank}(E P^\top) \;\le\; \min(d_h,\, \dim \operatorname{span}\{e_t\},\, \dim \operatorname{span}\{p_t\})$$

**The required rank is the dimension of the token code the match must resolve** — not a parity constant. With $|\mathcal{V}| \approx 50{,}000 \gg d$, token codes are heavily superposed, and a rank-$r$ truncation of $EP^\top$ resolves the top-$r$ directions of the token-code covariance, giving partial credit.

Predicted shape of $R(r)$: a **smooth rise tracking the singular-value spectrum of $EP^\top$**, not a step at $r = 2$. A sharp low-rank threshold would be the surprising outcome — it would mean the model matches on a compressed token code of low effective dimension, which is a genuinely interesting finding. Either way, the spectrum of the fitted $\Delta W_{QK}$ is directly comparable to the observed $R(r)$ curve, which is a strong internal consistency check.

---

## 9. Composition: adapt exactly one matrix per form

Adapting both $W_Q$ and $W_K$ at rank $r$:

$$\Delta M_{QK} = \Delta W_Q W_K^\top + W_Q \Delta W_K^\top + \Delta W_Q \Delta W_K^\top$$

rank up to $3r$, no parity structure, $\varphi$ uninterpretable. Adapting $W_Q$ alone:

$$\Delta M_{QK} = (b a^\top) W_K^\top = b\,(W_K a)^\top$$

rank exactly 1 per LoRA rank, square, $\varphi$ well-defined. **Adapt $W_Q$ only for the QK arm, $W_O$ only for the OV arm.**

**Prerequisite this creates.** With $W_K$ frozen, the key-side directions are whatever $W_K$ already reads at checkpoint $A$. If $W_K$ at $A$ has no overlap with the previous-token head's output subspace, *no* $\Delta W_Q$ at any rank can build the match, and the result would read as a capacity limit when it is a missing-prerequisite. Measure $\|P_{\text{prev-tok}} W_K\|_F / \|W_K\|_F$ at $A$ before running anything. Minutes of compute, and it is the cheapest way to avoid a wasted fortnight.

**Making $P_{\text{prev-tok}}$ concrete (G1).** Neither this section nor PROJECT.md §5/§8 gives $P_{\text{prev-tok}}$ an actual construction — it is fixed here, once, before any checkpoint is inspected (same discipline as `gates.py`'s G0 operationalization). The previous-token head writes $p_t$ into the residual stream through its own OV circuit; the set of directions it can possibly write is the row space of that head's $W_O \in \mathbb{R}^{d_h \times d}$ (equivalently, the column space of $W_O^\top$). So:

$$P_{\text{prev-tok}} \;=\; U U^\top, \qquad U = \text{orthonormal basis for } \operatorname{col}(W_O^\top)$$

computed via SVD (numerical rank, not a naive QR, so a rank-deficient $W_O$ at an early checkpoint can't inflate the projector). $W_K \in \mathbb{R}^{d \times d_h}$ for the candidate induction head is then projected directly: $P_{\text{prev-tok}} W_K \in \mathbb{R}^{d \times d_h}$, and the ratio is its Frobenius norm over $\|W_K\|_F$ — in $[0,1]$ by construction, since $P_{\text{prev-tok}}$ is an orthogonal projector.

PROJECT.md §8 asks whether this ratio shows "non-negligible overlap" without a numeric cutoff. Rather than inventing one, G1 reuses the random-subspace-null convention §6 already pre-registers for H4's $h0_6$: draw $N$ random rank-matched subspaces of $\mathbb{R}^d$, compute the same ratio against each, and call the observed ratio non-negligible iff it exceeds the 95th percentile of that null. Implementation: `indbw.gates.k_composition_overlap` (built on `indbw.algebra.subspace_projector` / `projection_energy_fraction`).

The candidate induction head itself is identified independently of $A$: since $A$ is defined as pre-transition (no head above PMS 0.1), $A$'s own PMS values are uninformative for picking "the" induction head. G1 instead uses whichever (layer, head) has the highest PMS at $B$ — the same head, evaluated at $A$'s frozen weights — since that is the head whose $W_K$ any QK-arm training run would actually be adapting around.

---

## 10. Splitting the circuit — yes, and here is the split

Induction is three components across two layers, not two:

| # | Component | Layer | Circuit | What it does |
|---|---|---|---|---|
| 1 | Previous-token head | earlier | QK (positional) | attends $t \to t-1$, writes token $t{-}1$'s identity into the residual |
| 2 | Prefix matching | later | QK | attends from $t$ to the successor of a previous occurrence of token $t$ — requires K-composition with (1) |
| 3 | Copying | later | OV | moves the attended token's identity to the output logits |

Your "look back and replace forward" is (2) and (3). Component (1) is a prerequisite, not a target — it exists at $A$ or the experiment is dead.

**Separate them, with separate metrics and separate arms.** A lumped recovery metric $R$ cannot tell you which sub-circuit was the bottleneck, and the two have *opposite* predicted symmetry:

| Arm | Adapt | Metric | Predicted $\varphi$ | Why |
|---|---|---|---|---|
| Prerequisite | nothing | prev-token score at $A$: mean attn $t \to t-1$ | — | gate, not a result |
| QK | $W_Q$ only | prefix-matching score: mean attn $t \to$ successor position | $\approx 0.5$ | off-diagonal, $U \perp V$ (§7) |
| OV | $W_O$ only | copying score: fraction of $t$ with $\arg\max W_U M_{OV} e_t = t$ | $< 0.5$ | $M_{OV} \approx \lambda\sum_t u_t e_t^\top$ is identity-like in token space |
| Joint | both | recovery $R$ | — | upper bound; ceiling check |

**$\varphi(\Delta M_{QK}) \approx 0.5$ while $\varphi(\Delta M_{OV})$ is materially below it** is the cleanest falsifiable contrast available. It is nearly free to compute, it distinguishes two mechanisms that a single metric conflates, and both directions of the result mean something. The null: matched-norm random rank-$r$ update, whose $\varphi$ concentrates near $1/2$ by symmetry — so the *OV* prediction ($\varphi$ significantly below $1/2$) carries the signal, and the QK prediction ($\varphi \approx 1/2$) is only informative when contrasted against OV, not against the random null. State that asymmetry explicitly rather than reporting $\varphi_{QK} \approx 0.5$ as a positive finding on its own.

Minimum rank should be reported per sub-circuit: $r^\ast_{QK}$ and $r^\ast_{OV}$ separately. They have no reason to coincide.

---

## 11. What this can and cannot establish about minimality

**Sufficiency — sound.** "Rank $r$ suffices to install prefix matching from checkpoint $A$" is constructive. You exhibit the update. This is an upper bound on the intrinsic dimension and it is a real result.

**Necessity — not obtainable.** Failure at rank $r'$ is failure of a *tuple*: (this optimizer, this objective, this initialization, this step budget, this parameterization, this checkpoint). The optimization is non-convex; there is no certificate that no rank-$r'$ solution exists. "Rank $r'$ is insufficient" is not provable by this experiment and should never be written without its qualifiers attached.

**Three things that strengthen the lower-bound side, in increasing order of value:**

1. Multiple seeds, multiple lrs, $10\times$ step budget on every failing cell. Standard, necessary, weak.
2. **SVD truncation of a known-good solution.** Train at generous rank to get $\Delta W^\star$; compute its best rank-$r'$ approximation $\Delta W^\star_{r'}$ by truncating the SVD; evaluate directly. If truncation destroys the behavior, that is evidence about the *object* rather than about the optimizer — it removes the training dynamics from the inference entirely. Cheap, and a strictly better minimality probe than a failed training run. Do this.
3. Compare the trained $\Delta W$ spectrum to the pretraining delta $\big(W_{QK}(B) - W_{QK}(A)\big)$ restricted to the same head, via principal angles against a random-subspace null. This is the only bridge from a landscape result to a developmental claim.

**What is out of reach entirely.** "The minimum structure the induction head can take" is a universal claim over all implementations of induction in all models. This experiment studies updates from one checkpoint of one model under one objective. The reachable claim is: *the minimum-dimension update that installs prefix matching from $A$, under this parameterization, has rank $\approx r^\ast$, and its QK component has antisymmetric fraction $\approx \varphi$.* That is a real, defensible, modest result. Write it that way in the abstract and the paper survives review.

---

## 12. Three questions, not one

"Can a rank-$r$ update install induction?" is three separate claims that get conflated, and they have different epistemic status. Naming them is most of the work.

| | Claim | Form | Status here |
|---|---|---|---|
| **(E)** | **Existence** | $\exists\, \Delta W$ of rank $\le r$ achieving the criterion | Decidable in closed form / by LP (§16). Not yet decided. |
| **(A)** | **Rate** | how the achievable margin grows with $r$ | Derived below under stated code-geometry assumptions (§15) |
| **(L)** | **Reachability** | does *this* optimizer from *this* init find such a $\Delta W$ | Genuinely uncertain, but with named obstructions (§17) |

§11 already says necessity is not obtainable. That is a statement about (L), and it is correct. What §11 does not say — and what the sections below supply — is that **(E) is not in the same boat.** (E) is a finite-dimensional feasibility question about frozen matrices at checkpoint $A$; it does not involve the optimizer at all.

The universal-approximation analogy the framing question invokes is apt for (L) and misleading for (E). §18 scores it line by line. Read §13–§17 first.

**Scope note.** Everything in §12–§18 is derivation, not protocol. No hypothesis in `PROJECT.md` §6 is added, altered, or re-thresholded by any of it, and nothing here is a falsification test. Where a derived quantity would be worth measuring, that is flagged as a human call in `PROJECT.md` §11 and `REVIEW.md`, per CLAUDE.md's falsification rules.

---

## 13. The reachable set of a one-matrix update

The composition rule (§9) adapts $W_Q$ only, so the reachable updates are not "all rank-$r$ matrices."

**Lemma 13.1 (reachable set).** With $W_K$ frozen and $\Delta W_Q$ of rank $\le r$,

$$\Delta M_{QK} \;=\; \Delta W_Q W_K^\top \;\in\; \mathcal{M}_r \;:=\; \big\{\, M \in \mathbb{R}^{d\times d} \;:\; \operatorname{rank} M \le r,\ \ \operatorname{row}(M) \subseteq \operatorname{col}(W_K) \,\big\}$$

and every element of $\mathcal{M}_r$ is attained by some $\Delta W_Q$ of rank $\le r$.

*Proof.* Writing $\Delta W_Q = \sum_{s\le r} u_s a_s^\top$ gives $\Delta M_{QK} = \sum_s u_s (W_K a_s)^\top$, whose row space lies in $\operatorname{col}(W_K)$. Conversely any $M = \sum_s u_s v_s^\top$ with $v_s = W_K a_s$ is realized by $\Delta W_Q = \sum_s u_s a_s^\top$. $\square$

Two constraints, then, not one: a **rank** constraint (from $r$) and a **row-space** constraint (from frozen $W_K$). They bind differently, and the second one does not go away as $r$ grows.

**Lemma 13.2 (rank saturation — the constraint is vacuous at $r \ge d_h$).** $\Delta W_Q \in \mathbb{R}^{d \times d_h}$, so $\operatorname{rank}(\Delta W_Q) \le d_h$ unconditionally. Hence for $r \ge d_h$ the LoRA factorization imposes **no constraint whatsoever**: $\{(\alpha/r) BA : B \in \mathbb{R}^{d\times r}, A \in \mathbb{R}^{r \times d_h}\} = \mathbb{R}^{d \times d_h}$ exactly.

At `pythia-70m`, $d_h = 64$. So the QK arm's rank axis has a hard ceiling at 64, the meaningful sweep range is $1 \le r < 64$, and a run at $r = 64$ is the **unconstrained** $\Delta W_Q$ problem wearing a LoRA costume. The same holds for the OV arm, where $\Delta W_O \in \mathbb{R}^{d_h \times d}$. G3's own record already labels its cell $r = d_{\text{head}} = 64$; Lemma 13.2 is the consequence — whatever G3 measured, it was not a rank limit, because at that $r$ there was no rank limit in force.

**Lemma 13.3 (the key-side shadow).** For any $\Delta W_Q$, the update's contribution to the score depends on the key position only through

$$k_j \;=\; W_K^\top x_j \;\in\; \mathbb{R}^{d_h},$$

since $x_i^\top \Delta W_Q W_K^\top x_j = \langle \Delta W_Q^\top x_i,\, k_j\rangle$. The prefix-match decision must therefore be computable from a **$d_h$-dimensional view of the key**, whatever $r$ is and whatever $d$ is.

This is the structural fact that makes the rest of the analysis finite. The key side of the QK circuit is a 64-dimensional bottleneck by construction, and the vocabulary is $|\mathcal{V}| = 50{,}304$. §15 is about what can be resolved through that bottleneck.

---

## 14. How much score margin the behaviour needs

Before asking what rank buys, fix what has to be bought. Attention is a softmax over $m$ unmasked key positions; the target is one position $j^\star$ (the successor of the earlier occurrence).

**Proposition 14.1 (margin $\Rightarrow$ attention).** If $s_{ij^\star} - s_{ij} \ge \Delta$ for all $m-1$ non-target positions $j$, then

$$\alpha_{ij^\star} \;\ge\; \frac{1}{1 + (m-1)e^{-\Delta}} .$$

*Proof.* $\alpha_{ij^\star} = e^{s_{ij^\star}}/\sum_j e^{s_{ij}} \ge e^{s_{ij^\star}}/\big(e^{s_{ij^\star}} + (m-1)e^{s_{ij^\star}-\Delta}\big)$. $\square$

**Corollary 14.2 (the margin budget).** $\alpha_{ij^\star} \ge \tau$ is *guaranteed* once

$$\boxed{\;\Delta \;\ge\; \Delta_\tau(m) \;=\; \log\frac{(m-1)\,\tau}{1-\tau}\;}$$

At the pre-registered PMS threshold $\tau = 0.3$ (`PROJECT.md` §5) with $m = 2T = 256$: $\Delta_\tau = \log(255 \cdot 0.3/0.7) = \mathbf{4.69}$ score units.

Three things to note about that number.

1. **It is small.** A logit gap of 4.7 is an ordinary magnitude for an attention score. Nothing about the target behaviour is extreme.
2. **It is sufficient, not necessary.** PMS averages attention over query positions, so it can clear $\tau$ with per-position attention below $\tau$ on some positions. Corollary 14.2 is therefore an upper bound on what existence requires — which is the direction we want for an existence argument.
3. **It must be paid on top of the base model.** Let $\rho_i = \max_j s^A_{ij} - \min_j s^A_{ij}$ be the base model's score range over candidate positions at query $i$. The update must deliver margin $\Delta_\tau + \rho_i$ in the worst case, since the frozen scores can already favour a distractor by up to $\rho_i$. $\rho_i$ is measurable at $A$ with one forward pass.

Everything downstream is: *can an update in $\mathcal{M}_r$ deliver $\Delta_\tau + \rho$, and at what norm?*

---

## 15. A constructive low-rank update, and what it costs

Now the existence construction. It is explicit — this is not an approximation-theoretic $\exists$.

### 15.1 Assumptions, all measurable at $A$

**A1 (query code).** $x_i = q_{t_i} + \eta_i$, where $q_t \in \mathbb{R}^{d}$ is the residual direction carrying "current token is $t$" and $\eta_i$ is orthogonal-ish context.

**A2 (key code).** $x_j = p_{t_{j-1}} + \xi_j$ with $p_t$ written by the previous-token head. Since that head's write is its own OV circuit applied to the previous position, $p_t = M_{OV}^{\text{prev}} q_t$, so

$$P \;=\; M_{OV}^{\text{prev}} Q, \qquad \operatorname{rank}(P) \;\le\; \operatorname{rank}\!\big(M_{OV}^{\text{prev}}\big) \;\le\; d_h .$$

**This retires the $|\mathcal{V}|$-sized rank worry in §8.** $M_{QK}^\star \approx \lambda \sum_t q_t p_t^\top = \lambda\, Q Q^\top (M_{OV}^{\text{prev}})^\top$, whose rank is at most $d_h = 64$ no matter that $|\mathcal{V}| = 50{,}304$: the key-side code is already a 64-dimensional object before any update is applied. §8's "required rank is the dimension of the token code" is right, and the dimension of the token code *as the key side can see it* is $\le d_h$.

**A3 (key read).** $\kappa_t := W_K^\top p_t \in \mathbb{R}^{d_h}$, the matched key code. $\|\kappa_t\|$ is what G1's overlap ratio measures the availability of: $\|P_{\text{prev-tok}} W_K\|_F / \|W_K\|_F = 0.387$ at $A$.

**A4 (near-orthogonality).** $\{\hat q_t\}$ and $\{\hat\kappa_t\}$ are each families of near-unit vectors with bounded coherence. For $\hat\kappa$ this is *forced to be loose*: the Welch bound for $N$ unit vectors in $\mathbb{R}^n$ gives $\mu \ge \sqrt{(N-n)/(n(N-1))}$, which at $N = 50{,}304$, $n = 64$ is $\mu \ge 0.125$. **Key-side interference of order $1/8$ is unavoidable, at any rank, by counting alone.**

### 15.2 The construction

Let $\Pi_r$ be a rank-$r$ orthogonal projector on $\mathbb{R}^{d_h}$ and set

$$\boxed{\;\Delta W_Q^{(r)} \;:=\; \lambda \sum_{t\in\mathcal{V}} \hat q_t\, \big(\Pi_r \hat\kappa_t\big)^{\!\top} \;=\; \lambda\, \hat Q\, \hat K^\top \Pi_r \;}$$

a matched filter, sketched to $r$ dimensions. $\operatorname{rank}(\Delta W_Q^{(r)}) \le r$ by construction, and it factors as $BA$ with $r$ inner dimensions, so it is a legal LoRA parameterization. Its induced score contribution at query $i$, key $j$, under A1–A2, is

$$\Delta s(i,j) \;=\; \frac{\lambda \|q\|\|\kappa\|}{\sqrt{d_h}}\Big[\hat\kappa_{t_i}^\top \Pi_r \hat\kappa_{t_{j-1}}\Big] \;+\; O(\mu_E),$$

so the behaviour reduces to one quantity: how well a rank-$r$ sketch preserves the ability of $\hat\kappa_{t_i}$ to pick itself out of the codes present in the sequence.

### 15.3 The rate: rank versus number of competitors

**Lemma 15.1 (sketch statistics).** For $\hat\kappa$ isotropic on the unit sphere of $\mathbb{R}^n$ and $\Pi_r$ a uniformly random rank-$r$ projector,

$$\mathbb{E}\big[\hat\kappa^\top \Pi_r \hat\kappa\big] = \frac{r}{n}, \qquad \mathbb{E}\big[\hat\kappa^\top\Pi_r\hat\kappa'\big] = 0,\quad \operatorname{sd}\big[\hat\kappa^\top\Pi_r\hat\kappa'\big] \approx \frac{\sqrt r}{n}\ \ (\hat\kappa' \perp\!\!\!\perp \hat\kappa).$$

Signal grows like $r/n$; interference grows only like $\sqrt r / n$. The signal-to-interference ratio is $\sqrt r$ — independent of $n$, independent of $d$, independent of $|\mathcal{V}|$.

**Theorem 15.2 (rank threshold).** With $m$ competitors, $\max_{j} |\hat\kappa_{t_i}^\top\Pi_r\hat\kappa_{t_j}| \lesssim (\sqrt r/n)\sqrt{2\log m}$ with high probability, so the sketched matched filter separates the target from all $m-1$ distractors as soon as

$$\boxed{\;r \;\gtrsim\; 2\log m\;}$$

and, to retain a fraction $(1-\epsilon)$ of the unsketched margin, $r \gtrsim 2\log m/\epsilon^2$ — Johnson–Lindenstrauss, in its usual shape.

**This is the central rate result, and it is logarithmic in the number of competitors.** Instantiated:

| Arm | Competitors $m$ | $2\log m$ | Reading |
|---|---|---|---|
| QK (prefix match) | context positions, $2T = 256$ | $\approx 11$ | must beat the other positions in *one sequence* |
| OV (copying) | vocabulary, $|\mathcal{V}| = 50{,}304$ | $\approx 22$ | must beat *every other token* at the unembedding |

Simulation at $n = 64$ with random codes puts the 50% separation crossing slightly *below* $2\log m$ ($r \approx 8$ for $m = 256$, $r \approx 11$ for $m = 2048$) and near-certain separation at roughly $1.5\times 2\log m$ — i.e. the bound is the right order and mildly conservative. (Reproduced as a seeded discrimination test in `tests/unit/test_existence.py`.)

**Corollary 15.3 (a derived direction for H3).** Under isotropic codes, the two arms' sufficient ranks stand in ratio

$$\frac{r^\ast_{OV}}{r^\ast_{QK}} \;\approx\; \frac{\log|\mathcal{V}|}{\log 2T} \;=\; \frac{10.83}{5.55} \;\approx\; 1.95 .$$

H3 (`PROJECT.md` §6) pre-registers only $r^\ast_{QK} \ne r^\ast_{OV}$, with no direction. This derivation supplies one — OV should need roughly twice the rank — **as a consequence of stated assumptions, not as an amendment to H3.** H3's null and criterion are unchanged. If the sweep ever runs and the ordering comes out reversed, that is informative about A1–A4, not a violation of anything pre-registered.

### 15.4 The norm cost, and what G1's narrow margin actually means

**Proposition 15.4 (norm cost).** Combining Corollary 14.2 with §15.2, the scale $\lambda$ must satisfy

$$\lambda \;\ge\; \frac{\sqrt{d_h}\,\big(\Delta_\tau(m) + \rho\big)}{\|q\|\,\|\kappa\|\,\dfrac{r}{n}\Big(1 - \sqrt{2\log m / r}\Big)}$$

Read the $\|\kappa\|$ in the denominator. $\kappa_t = W_K^\top p_t$, and G1 measured exactly how much of $W_K$'s read capacity lands in the previous-token head's write subspace: $0.387$ against a null 95th percentile of $0.360$.

**So the frozen-$W_K$ overlap enters as a multiplicative norm cost, not as a feasibility barrier.** A $W_K$ perfectly aligned with the prev-token subspace would need $\lambda$; the observed alignment needs roughly $1/0.387 \approx 2.6\times$ that. Unless the overlap is *exactly* zero — in which case $\kappa_t \equiv 0$ and no $\Delta W_Q$ at any rank can work, which is precisely the failure branch G1 exists to rule out — the constraint is on how large the update must be, not on whether one exists.

This is worth stating plainly because it settles, on paper, a question the record chased empirically: `PROJECT.md` §11's "$W_K$-bottleneck hypothesis" cannot be a *hard* ceiling at overlap $0.387$. The diagnostic that unfroze $W_K$ found the same plateau, and Prop 15.4's algebra agrees with that finding rather than merely coexisting with it — a frozen $W_K$ with nonzero overlap scales the required norm and nothing else.

---

## 16. Existence is decidable, not merely assertable

§15 is an existence argument conditional on A1–A4. Those assumptions are the weak point: they are stated, not measured. This section removes them.

**Observation 16.1 (no feedback into own input).** A head's attention pattern does not influence the residual stream entering its own layer. So for the candidate head at layer $\ell$, adapting only its $W_Q$, the inputs $\{x_i^{(\ell)}\}$ are **exactly unchanged** from checkpoint $A$'s forward pass. Record them once; the map $\Delta W_Q \mapsto$ (that head's attention pattern) is then an exact, self-contained function with no model in the loop.

Note the scope: this is exact for PMS, an attention-pattern statistic of one head. It is *not* exact for ICL or $R$, which read the logits and therefore see everything the head writes downstream.

**Theorem 16.2 (convex feasibility).** Fix the eval set and the recorded activations. The set of updates achieving margin $\Delta$ everywhere,

$$\mathcal{F}_\Delta \;=\; \Big\{\Delta W_Q \in \mathbb{R}^{d\times d_h} \;:\; x_i^\top \Delta W_Q\big(k_{j^\star(i)} - k_j\big) \;\ge\; \sqrt{d_h}\,\Delta - \big(s^A_{ij^\star} - s^A_{ij}\big) \ \ \forall i,\, j \ne j^\star(i)\Big\}$$

is an intersection of half-spaces in $\Delta W_Q$ — a convex polyhedron. Each constraint is *linear* in $\Delta W_Q$ because the score is bilinear in $(x_i, k_j)$ and both are frozen.

**Corollary 16.3 (the unconstrained arm is an LP).** By Lemma 13.2 the rank constraint is vacuous at $r = d_h$, so deciding whether *any* $\Delta W_Q$ installs prefix matching at margin $\Delta$ from $A$ is exactly a linear program: $N_{\text{eval}} \times T \times (m-1)$ constraints in $d \cdot d_h = 32{,}768$ variables, solvable on CPU, no training, no optimizer, no learning rate.

The two outcomes are both worth having:

- **Feasible** $\Rightarrow$ a certificate, and the LP hands you the witness $\Delta W_Q$ itself. Sufficiency is constructive (§11); this is the construction, obtained without gradient descent.
- **Infeasible** $\Rightarrow$ a Farkas certificate that no $\Delta W_Q$ whatsoever achieves margin $\Delta$ from $A$. That is a statement about the frozen geometry of $W_K$ and the residual stream, **valid independently of optimizer, budget, initialization and parameterization** — i.e. exactly the kind of necessity statement §11 says a failed training run cannot deliver.

**Corollary 16.4 (what stays hard).** Adding $\operatorname{rank}(\Delta W_Q) \le r$ for $r < d_h$ makes it a rank-constrained LP, NP-hard in general. Two usable relaxations, neither requiring training: minimize the nuclear norm over $\mathcal{F}_\Delta$ (any feasible point it returns is a constructive existence proof at whatever rank it lands on), or truncate the SVD of a full-rank feasible point — the M5 probe (`PROJECT.md` §7) applied to an LP witness instead of a trained solution.

**This is where the universal-approximation analogy breaks in our favour.** A universal-approximation theorem asserts existence non-constructively over an unbounded architecture family; here existence is a finite linear feasibility question over frozen matrices, and it is decidable exactly.

---

## 17. Reachability: the obstructions have names

Suppose (E) is settled affirmatively. Nothing follows about whether training finds the witness. The gap is generic — but here it is not *only* generic, and the specific structure is derivable.

**Theorem 17.1 (QK gradient gating).** Let the head's output be $o_i = \sum_j \alpha_{ij} z_j$ with $z_j = M_{OV}x_j$, and let $L$ be any loss depending on the QK parameters only through the scores $s_{ij}$. Write $g_i = \partial L/\partial o_i$ and $\bar z_i = \sum_k \alpha_{ik} z_k$. Then

$$\boxed{\;\frac{\partial L}{\partial s_{ij}} \;=\; \alpha_{ij}\,\big\langle g_i,\; z_j - \bar z_i \big\rangle\;}$$

*Proof.* $\partial L/\partial \alpha_{ik} = \langle g_i, z_k\rangle$ and $\partial \alpha_{ik}/\partial s_{ij} = \alpha_{ik}(\delta_{jk} - \alpha_{ij})$; summing over $k$ gives $\alpha_{ij}(\langle g_i,z_j\rangle - \sum_k \alpha_{ik}\langle g_i,z_k\rangle)$. $\square$

**Corollary 17.2 (the gate).** $\nabla_{\Delta W_Q} L = 0$ **exactly** whenever $z_j$ is constant in $j$ — when the head writes the same vector regardless of which position it attends to, attention has nothing to change. Quantitatively,

$$\Big\|\frac{\partial L}{\partial s_{i\cdot}}\Big\|_\infty \;\le\; \|g_i\|\cdot \underbrace{\max_j \|z_j - \bar z_i\|}_{=:\ \sigma_{OV}(i)\ \text{(the value spread)}} .$$

**The QK arm's gradient is bounded by the head's existing OV value spread.** This is a property of the head at $A$, forward-measurable, and completely independent of $r$.

**Corollary 17.3 (spread is necessary, not sufficient).** $\sigma_{OV} > 0$ makes the gradient nonzero but not *useful*. Descent moves attention toward positions with low $\langle g_i, z_j\rangle$; for that to build induction, attending to the successor position must lower next-token loss, which requires the head's OV to carry the attended token's identity to the logits — a nonzero copying score. **Spread without copying yields a loss that moves and a recovery that does not.**

**Theorem 17.4 (bilinear saddle at LoRA init).** To first order, a head's contribution to ICL is a product: (prefix-match strength) $\times$ (copy strength). Standard LoRA init sets $B = 0$, $A \ne 0$, hence $\Delta W = 0$ exactly (`train.py`'s `init_lora_factors` does exactly this). At the origin of a bilinear objective both partials vanish, so the init sits at a **strict saddle**: escape is possible, but its timescale is set by curvature and gradient noise, and by nothing about $r$. LoRA's factorization adds a second degeneracy on top: with $B = 0$, $\partial L/\partial A = B^\top(\partial L/\partial \Delta W) = 0$ identically at step 0, so only $B$ moves on the first step.

**Proposition 17.5 (rank cannot repair a vanishing gradient).** The rank-$r$ parameterization's gradient is a linear image of the full one: $\partial L/\partial B = (\partial L/\partial \Delta W)A^\top$ and $\partial L/\partial A = B^\top(\partial L/\partial \Delta W)$. So if $\partial L/\partial \Delta W = 0$, every rank's gradient is zero. **A rank sweep run in a gradient-gated regime returns $R(r)$ flat in $r$ at every $r$** — which is the literal statement of $h0_1$, arrived at by a mechanism that has nothing to do with intrinsic dimension.

That last point is the one with teeth for the protocol: M1/M2's interpretability as a *capacity* measurement presupposes that the arm has non-vanishing, correctly-directed gradient at $A$. Corollaries 17.2–17.3 make that presupposition into two forward-only measurements on the candidate head at $A$ — its value spread $\sigma_{OV}$, and its copying score, which `probes.copying_score` already computes and which no record in `results/` currently reports. Whether to measure them, and what to do about the answer, is a human call (`PROJECT.md` §11, `REVIEW.md`); this section derives the dependency, it does not propose a test.

---

## 18. So: "theoretically possible but empirically uncertain"?

Partly — but the phrase flattens a distinction worth keeping, and the universal-approximator analogy fits only one of the three claims in §12.

**Where the analogy holds.** (L), reachability. Just as universal approximation says nothing about whether SGD finds the approximating network, §16's LP certificate would say nothing about whether the training loop finds the witness. Both gaps are real and neither closes by exhibiting the target.

**Where it does not.**

| | Universal approximation | This problem |
|---|---|---|
| Quantifies over | an unbounded architecture family | one head, one checkpoint, fixed shapes |
| Existence proof | non-constructive | explicit (§15.2) *and* decidable by LP (§16) |
| Constants | typically vacuous (width exponential in $1/\epsilon$) | measurable at $A$ in minutes: $\Delta_\tau$, $\rho$, $\|\kappa_t\|$, code coherence |
| Rate in the budget parameter | usually none, or a bad one | $r \gtrsim 2\log m$, logarithmic (Thm 15.2) |
| Failure of the existence half | not observable | a Farkas certificate, optimizer-independent |
| Learnability | outside the theorem | outside the theorem, **but with named obstructions** (§17) |

The honest summary is therefore **not** "theoretically possible but empirically uncertain." It is:

> **Theoretically decidable and not yet decided; empirically uncertain for a separate, identifiable, and separately measurable reason.**

Three concrete things follow.

1. **The existence question does not need the training loop.** §16's LP and §15.2's closed-form construction both run on frozen activations recorded once at $A$. Whatever is happening in the training runs, it is not evidence about (E) — and (E) has never been evaluated.
2. **The rank axis is smaller than it looks.** Lemma 13.2: $r \ge d_h = 64$ is no constraint at all, so any cell at $r = 64$ measures the unconstrained arm. The informative range is $1 \le r < 64$, and §15.2 predicts the action sits near $r \sim 10$–$25$.
3. **The uncertainty in (L) is not the generic non-convexity handwave.** Theorem 17.1 gives an exact vanishing condition, Corollary 17.2 a bound in a forward-measurable quantity, and Proposition 17.5 the consequence that rank cannot rescue it. That is a sharper position than "optimization is hard," and it is falsifiable in the ordinary sense — measure $\sigma_{OV}$ and the copying score at $A$ and the bound either binds or it does not.

**What would make §15 wrong.** A1–A4 are the load-bearing assumptions and none is verified. If the query-side token direction $q_t$ is not cleanly separable from context ($\eta_i$ large), if the previous-token head's write is weak enough that $\|\kappa_t\|$ is near the noise floor, or if the codes are anisotropic in a way that puts the discriminative directions in the *tail* of the key-code spectrum rather than the head, then §15.2's construction still exists but its required $\lambda$ blows up and the effective threshold rank rises above $2\log m$. §16 is immune to all of this, which is the argument for preferring it: the LP asks the question of the actual matrices instead of a model of them.

---

## 19. Computable quantity summary

| Quantity | Formula | Cost |
|---|---|---|
| Antisymmetric fraction | $\varphi(M) = \frac{1}{2}\!\left(1 - \operatorname{tr}(M^2)/\|M\|_F^2\right)$ | one matmul |
| QK form | $M_{QK} = W_Q W_K^\top$ | one matmul |
| OV form | $M_{OV} = W_O^\top W_V^\top$ | one matmul |
| Update rank-1 relationality | $\varphi = \frac12(1-c^2)$, $c = \hat a\cdot\hat b$ | one dot product |
| Prev-token overlap at $A$ | $\|P_{\text{prev}} W_K\|_F / \|W_K\|_F$ | projection |
| Bandwidth | $\beta = r(d_{\text{in}} + d_{\text{out}})$ | — |
| Truncation probe | $R(\Delta W^\star_{r'})$ from SVD of $\Delta W^\star$ | one SVD |

Added by §12–§18 (all implemented in `indbw.existence`, all forward-only or training-free):

| Quantity | Formula | Cost | §|
|---|---|---|---|
| Required logit margin | $\Delta_\tau(m) = \log\frac{(m-1)\tau}{1-\tau}$ | closed form | 14.2 |
| Attention floor from a margin | $\alpha \ge 1/(1 + (m-1)e^{-\Delta})$ | closed form | 14.1 |
| Base-score range at $A$ | $\rho_i = \max_j s^A_{ij} - \min_j s^A_{ij}$ | one forward pass | 14 |
| Arm rank ceiling | $\min(d_{\text{in}}, d_{\text{out}}) = d_h$; $r \ge d_h$ is vacuous | — | 13.2 |
| Matched key code | $\kappa_t = W_K^\top p_t \in \mathbb{R}^{d_h}$ | one matmul | 15.1 |
| Sketch rank threshold | $r \gtrsim 2\log m / \epsilon^2$ | closed form | 15.2 |
| Constructive update | $\Delta W_Q^{(r)} = \lambda \hat Q \hat K^\top \Pi_r$ | one projection | 15.2 |
| Welch coherence floor | $\mu \ge \sqrt{(N-n)/(n(N-1))}$ | closed form | 15.1 |
| Existence certificate | LP feasibility of $\mathcal{F}_\Delta$ | one LP, no training | 16.3 |
| QK score gradient | $\partial L/\partial s_{ij} = \alpha_{ij}\langle g_i, z_j - \bar z_i\rangle$ | one matmul | 17.1 |
| Value spread | $\sigma_{OV}(i) = \max_j \|z_j - \bar z_i\|$ | one forward pass | 17.2 |

---

## 20. Reading

- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* — QK/OV circuits, K-composition, the $W_QW_K^\top$ and $W_OW_V$ forms used throughout
- Olsson et al. 2022 — induction heads, prefix-matching and copying scores
- Li et al. 2018; Aghajanyan et al. 2020 — intrinsic dimension of task-specific updates, the correct framing for §2
- Hu et al. 2021 — LoRA
- Kalajdzievski 2023 — $\sqrt{r}$ scaling; relevant to the §2 $\alpha/r$ confound
