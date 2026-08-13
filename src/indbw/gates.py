"""Gate-detection logic for PROJECT.md §8's G0 and G1 protocols.

Pure function over a per-checkpoint probe series -- not a probes.py metric
itself (CI's METRIC_VERSION hash covers only algebra.py/probes.py, the
per-example metric *definitions*). This module locates a one-time constant
(the A/B checkpoint pair) from those metrics swept over the checkpoint
grid; loading checkpoints and running the sweep is scripts/g0_sweep.py's
job, matching the "never load a model in a pure module" pattern used by
probes.py.

PROJECT.md §2 defines A and B in prose:
  A -- "last checkpoint before the transition. No head above PMS 0.1;
        ICL score at baseline."
  B -- "first checkpoint after the transition stabilizes: ICL within 10%
        of its value two checkpoints later."
Both conditions must be made computable to satisfy CLAUDE.md's "Computable
verdicts" rule. The operationalization below (baseline = ICL at step 0;
"transition" = the first grid step where max-over-heads PMS crosses the
0.1 threshold) is stated here, before any checkpoint has been swept, per
PROJECT.md §11's own requirement that such rules be fixed in advance.

G1 (`locate_prev_token_head`, `k_composition_overlap`) follows the same
discipline for PROJECT.md §8's prerequisite check; see
`k_composition_overlap`'s own docstring for how "non-negligible overlap"
is made computable.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from indbw.algebra import projection_energy_fraction, subspace_projector


@dataclass(frozen=True)
class TransitionResult:
    """Outcome of locating the G0 induction transition on a checkpoint grid."""

    found: bool
    a_step: int | None
    b_step: int | None
    bracket_width: int | None  # grid steps from A to B; None if not found
    passed: bool  # PROJECT.md §8: fail if not found, or bracket_width > max_bracket_width


def locate_transition(
    steps: list[int],
    max_pms: np.ndarray,
    icl: np.ndarray,
    *,
    pms_pretransition_threshold: float = 0.1,
    icl_baseline_tol: float = 0.05,
    icl_stabilize_rtol: float = 0.10,
    lookahead: int = 2,
    max_bracket_width: int = 3,
) -> TransitionResult:
    """Locate A and B on the checkpoint grid (PROJECT.md §2/§8).

    steps: ascending checkpoint step numbers, length n.
    max_pms: [n] max-over-heads prefix-matching score at each step.
    icl: [n] ICL score at each step.

    A is the checkpoint immediately before max_pms first reaches
    pms_pretransition_threshold, provided its ICL is within icl_baseline_tol
    (absolute) of icl[0] -- both conditions of PROJECT.md §2's A definition
    must hold simultaneously, or no valid A exists on this grid.

    B is the first checkpoint at or after the crossing whose ICL is within
    icl_stabilize_rtol (relative) of its value `lookahead` grid steps later.

    Raises ValueError on malformed input (shape mismatch, too-short grid,
    non-finite values) -- never returns a plausible-looking but meaningless
    result for degenerate input (CLAUDE.md TDD contract).
    """
    steps_arr = np.asarray(steps)
    max_pms = np.asarray(max_pms, dtype=float)
    icl = np.asarray(icl, dtype=float)
    n = len(steps)
    if max_pms.shape != (n,) or icl.shape != (n,):
        raise ValueError(
            f"steps (len {n}), max_pms {max_pms.shape}, icl {icl.shape} must all have length n"
        )
    if n < lookahead + 2:
        raise ValueError(f"grid too short ({n} steps) for lookahead={lookahead}")
    if not np.all(np.isfinite(max_pms)) or not np.all(np.isfinite(icl)):
        raise ValueError("locate_transition received a non-finite max_pms or icl value")
    if not np.all(np.diff(steps_arr) > 0):
        raise ValueError("steps must be strictly ascending")
    if np.any((max_pms < -1e-8) | (max_pms > 1 + 1e-8)):
        raise ValueError(f"max_pms out of [0, 1] range: {max_pms}")

    not_found = TransitionResult(
        found=False, a_step=None, b_step=None, bracket_width=None, passed=False
    )

    crossings = np.flatnonzero(max_pms >= pms_pretransition_threshold)
    if crossings.size == 0:
        return not_found  # no head ever reaches the pre-transition threshold
    first_cross = int(crossings[0])
    if first_cross == 0:
        return not_found  # no pre-transition checkpoint exists on this grid

    a_idx = first_cross - 1
    baseline_icl = float(icl[0])
    if abs(icl[a_idx] - baseline_icl) > icl_baseline_tol:
        return not_found  # PMS says pre-transition, but ICL already moved -- A ill-defined

    b_idx = None
    for j in range(first_cross, n - lookahead):
        later = icl[j + lookahead]
        denom = abs(later) if later != 0 else 1.0
        if abs(icl[j] - later) <= icl_stabilize_rtol * denom:
            b_idx = j
            break
    if b_idx is None:
        return not_found  # transition started but ICL never stabilized on this grid

    bracket_width = b_idx - a_idx
    return TransitionResult(
        found=True,
        a_step=int(steps[a_idx]),
        b_step=int(steps[b_idx]),
        bracket_width=int(bracket_width),
        passed=bracket_width <= max_bracket_width,
    )


@dataclass(frozen=True)
class PrevTokenHeadResult:
    """Outcome of locating the previous-token head at checkpoint A (PROJECT.md §8 G1)."""

    layer: int
    head: int
    score: float
    found: bool  # score >= threshold


def locate_prev_token_head(
    prev_token_scores: np.ndarray, *, threshold: float = 0.3
) -> PrevTokenHeadResult:
    """Argmax-scoring head over prev_token_scores[layer, head] at checkpoint A.

    PROJECT.md §8: "does a previous-token head exist (prev-token score
    >= 0.3)?" -- the candidate head is whichever one scores highest;
    `found` is that score crossing the pre-registered 0.3 threshold.

    Raises ValueError on malformed input (wrong ndim, non-finite, out
    of [0,1]) rather than returning a plausible-looking wrong head.
    """
    if prev_token_scores.ndim != 2:
        raise ValueError(
            f"prev_token_scores must be [n_layers, n_heads], got shape {prev_token_scores.shape}"
        )
    if not np.all(np.isfinite(prev_token_scores)):
        raise ValueError("prev_token_scores contains non-finite values")
    if np.any((prev_token_scores < -1e-8) | (prev_token_scores > 1 + 1e-8)):
        raise ValueError(f"prev_token_scores out of [0, 1] range: {prev_token_scores}")
    layer, head = np.unravel_index(int(np.argmax(prev_token_scores)), prev_token_scores.shape)
    score = float(prev_token_scores[layer, head])
    return PrevTokenHeadResult(
        layer=int(layer), head=int(head), score=score, found=score >= threshold
    )


@dataclass(frozen=True)
class OverlapResult:
    """K-composition overlap between a previous-token head and a candidate
    induction head's frozen $W_K$ (PROJECT.md §8/§9, docs/mathematics.md §9)."""

    ratio: float
    null_percentile_value: float
    significant: bool  # ratio > null_percentile_value


def k_composition_overlap(
    W_O_prev: np.ndarray,
    W_K_target: np.ndarray,
    *,
    n_null_draws: int = 100,
    percentile: float = 95.0,
    seed: int = 0,
) -> OverlapResult:
    r"""$\|P_{\text{prev-tok}} W_K\|_F / \|W_K\|_F$ vs. a random-subspace null.

    W_O_prev: [d_head, d_model], the previous-token head's output
    projection. W_K_target: [d_model, d_head], the candidate induction
    head's frozen key projection at checkpoint A.

    $P_{\text{prev-tok}}$ is the orthogonal projector onto the previous-
    token head's output subspace -- the column space of `W_O_prev.T`,
    i.e. the residual-stream directions that head's OV circuit can write
    into (docs/mathematics.md §8's $p_t$, written by component 1 of
    §10's three-part split). This is the operationalization referenced
    by docs/mathematics.md §9's "measure the overlap at A before running
    anything" and PROJECT.md §5's "prerequisite overlap" row -- neither
    source gives $P_{\text{prev-tok}}$ a concrete construction, so it is
    fixed here, before any checkpoint was inspected, matching gates.py's
    own precedent for `locate_transition`.

    PROJECT.md §8 asks for "non-negligible overlap" without a numeric
    cutoff. Rather than inventing an arbitrary one, this reuses the
    random-subspace-null convention PROJECT.md §6 already pre-registers
    for H4's h0_6 (observed value vs. the 95th percentile of the same
    statistic over random draws): `significant` is True iff the observed
    ratio exceeds the `percentile`-th percentile of the same ratio
    computed against `n_null_draws` random subspaces of matching rank.

    Raises ValueError on a d_model mismatch between the two matrices.
    """
    if W_O_prev.ndim != 2 or W_K_target.ndim != 2:
        raise ValueError("k_composition_overlap requires 2D W_O_prev and W_K_target")
    _, d_model_o = W_O_prev.shape
    d_model_k, _ = W_K_target.shape
    if d_model_o != d_model_k:
        raise ValueError(
            f"W_O_prev's d_model ({d_model_o}) must match W_K_target's d_model ({d_model_k})"
        )

    P = subspace_projector(W_O_prev.T)
    ratio = projection_energy_fraction(P, W_K_target)

    rank = round(float(np.trace(P)))
    rng = np.random.default_rng(seed)
    null_ratios = np.empty(n_null_draws)
    for i in range(n_null_draws):
        random_basis = rng.standard_normal((d_model_o, rank))
        null_ratios[i] = projection_energy_fraction(subspace_projector(random_basis), W_K_target)
    null_percentile_value = float(np.percentile(null_ratios, percentile))

    return OverlapResult(
        ratio=ratio,
        null_percentile_value=null_percentile_value,
        significant=ratio > null_percentile_value,
    )
