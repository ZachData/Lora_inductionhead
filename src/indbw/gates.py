"""Gate-detection logic for PROJECT.md §8's G0 protocol.

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
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


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
