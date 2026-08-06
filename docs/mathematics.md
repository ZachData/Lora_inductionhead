# The mathematics of the LoRA-rank induction experiment

Everything from LoRA's definition to what the symmetric/antisymmetric split can and cannot establish. Read §7–§9 carefully: the algebra does not support the prediction stated in earlier conversation, and the corrected version is a better experiment.

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

## 12. Computable quantity summary

| Quantity | Formula | Cost |
|---|---|---|
| Antisymmetric fraction | $\varphi(M) = \frac{1}{2}\!\left(1 - \operatorname{tr}(M^2)/\|M\|_F^2\right)$ | one matmul |
| QK form | $M_{QK} = W_Q W_K^\top$ | one matmul |
| OV form | $M_{OV} = W_O^\top W_V^\top$ | one matmul |
| Update rank-1 relationality | $\varphi = \frac12(1-c^2)$, $c = \hat a\cdot\hat b$ | one dot product |
| Prev-token overlap at $A$ | $\|P_{\text{prev}} W_K\|_F / \|W_K\|_F$ | projection |
| Bandwidth | $\beta = r(d_{\text{in}} + d_{\text{out}})$ | — |
| Truncation probe | $R(\Delta W^\star_{r'})$ from SVD of $\Delta W^\star$ | one SVD |

---

## 13. Reading

- Elhage et al. 2021, *A Mathematical Framework for Transformer Circuits* — QK/OV circuits, K-composition, the $W_QW_K^\top$ and $W_OW_V$ forms used throughout
- Olsson et al. 2022 — induction heads, prefix-matching and copying scores
- Li et al. 2018; Aghajanyan et al. 2020 — intrinsic dimension of task-specific updates, the correct framing for §2
- Hu et al. 2021 — LoRA
- Kalajdzievski 2023 — $\sqrt{r}$ scaling; relevant to the §2 $\alpha/r$ confound
