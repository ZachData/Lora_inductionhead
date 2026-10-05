"""Complete sweep rows -> one results record per claim (or gatekeeping step).

Controls are attached only when the rows contain the data that computes them, so a claim
whose null was never run is refused by `adjudicate` rather than labelled as controlled.
Verdicts are produced once, on the complete sweep (`analysis.require_complete`).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from indbw.schema import ResultsRecord
from indbw.selora import analysis, cells, prereg
from indbw.selora.adjudicate import adjudicate

NAMED_ARMS = ("all", "late", "early", "sae_top", "sae_bottom")
PRIMARY_MODEL = "google/gemma-2-2b-it"  # S5's model (instruct, where placement mattered)
SAE_MODEL = "google/gemma-2-2b"  # S2/S4: the model the SAEs were trained on / the 9.6% result


@dataclass(frozen=True)
class Provenance:
    config: dict[str, Any]
    seed: int
    checkpoint_revision: str
    eval_set_hash: str
    wall_clock_s: float


Key = tuple[str, str, int]  # (model, arm, seed)


def _index(rows: Sequence[Mapping[str, Any]]) -> dict[Key, Mapping[str, Any]]:
    return {(r["cell"]["model"], r["cell"]["arm"], int(r["cell"]["seed"])): r for r in rows}


def _acc(preds: Sequence[Any], gold: Sequence[str]) -> float:
    return float(np.mean([p == g for p, g in zip(preds, gold, strict=True)]))


def _controls(
    idx: Mapping[Key, Mapping[str, Any]], divergence: Mapping[str, Any], gsm_items_expected: int
) -> tuple[list[str], dict[str, str]]:
    have: list[str] = []
    why_not: dict[str, str] = {}

    def mark(name: str, ok: bool, reason: str) -> None:
        (have.append(name) if ok else why_not.__setitem__(name, reason))

    trained = [r for (_, arm, _), r in idx.items() if arm != "base"]
    arms = {(m, a) for m, a, _ in idx}
    mark("late-layer arm", any(a == "late" for _, a in arms), "no late arm")
    mark(
        "spectrum-matched random LoRA",
        bool(trained) and all("fvu_rand" in r for r in trained),
        "fvu_rand missing from trained rows",
    )
    seeds = {s for _, a, s in idx if a == "late"}
    mark("seed-to-seed noise floor", len(seeds) >= 2, "fewer than 2 seeds")
    mark("3 seeds", len(seeds) >= 3, "fewer than 3 seeds")
    mark("no-LoRA base", all((m, "base") in arms for m in cells.MODELS), "base rows missing")
    n_items = {len(r["gsm_correct"]) for r in idx.values()}
    mark(
        "full 1319-item test set",
        n_items == {gsm_items_expected},
        f"GSM8K items {sorted(n_items)} != {gsm_items_expected} (full 1319-item set)",
    )
    n_rand = {m: sum(a.startswith("random_") for mm, a in arms if mm == m) for m in cells.MODELS}
    mark(
        "24 random draws",
        all(v == prereg.N_RANDOM_DRAWS for v in n_rand.values()),
        f"draws {n_rand}",
    )
    mark(
        "same-seed comparison",
        all((m, "random_00", 0) in idx for m in cells.MODELS),
        "no seed-0 draws",
    )
    mark("sae_bottom arm", any(a == "sae_bottom" for _, a in arms), "no sae_bottom arm")
    mark(
        "sae scores frozen before training",
        all(m in divergence for m in cells.MODELS),
        "divergence missing",
    )
    mark(
        "same model scored both ways",
        all("mmlu_logprob" in r and "mmlu_gen" in r for r in idx.values()),
        "an MMLU protocol missing",
    )
    mark(
        "length x digit stratified permutation",
        all(divergence.get(m, {}).get("strata") == "length x digit" for m in cells.MODELS),
        "divergence not stratified by length x digit",
    )
    mark(
        "top-100 sensitivity",
        all(divergence.get(m, {}).get("topk_sensitivity_done") for m in cells.MODELS),
        "top-100 sensitivity not run",
    )
    return have, why_not


def build_records(
    rows: Sequence[Mapping[str, Any]],
    planned: Sequence[cells.Cell],
    divergence: Mapping[str, Mapping[str, Any]],
    prov: Provenance,
    gsm_items_expected: int = 1319,
) -> list[ResultsRecord]:
    analysis.require_complete((r["run_id"] for r in rows), [c.run_id for c in planned])
    idx = _index(rows)
    have, why_not = _controls(idx, divergence, gsm_items_expected)

    def record(cid: str, obs: dict[str, Any]) -> ResultsRecord:
        missing = [c for c in prereg.claim(cid).controls if c not in have]
        if missing:
            raise ValueError(
                f"{cid}: controls not computed: "
                + "; ".join(f"{m} ({why_not[m]})" for m in missing)
            )
        return adjudicate(
            cid,
            obs,
            have,
            prov.config,
            prov.seed,
            prov.checkpoint_revision,
            prov.eval_set_hash,
            prov.wall_clock_s,
        )

    out: list[ResultsRecord] = []

    # S0: every trained cell, every layer below its first adapted layer.
    below = [
        v
        for (_, arm, _), r in idx.items()
        if arm != "base"
        for li, v in r["resid_max_abs_diff"].items()
        if int(li) < min(r["cell"]["layers"])
    ]
    out.append(
        record("S0", {"max_abs_resid_diff_below_first_adapted": float(max(below, default=0.0))})
    )

    # S1a / S1b / noise floor: named arms, seed 0, adapted layers, both models.
    s1a, s1b, floor = {}, {}, {}
    for m in cells.MODELS:
        for arm in NAMED_ARMS:
            r0 = idx[(m, arm, 0)]
            r1 = idx.get((m, arm, 1))
            for li in r0["cell"]["layers"]:
                k = f"{m}|{arm}|L{li}"
                base = np.array(r0["fvu_base"][str(li)])
                ft = np.array(r0["fvu_ft"][str(li)])
                s1a[k] = (base, ft)
                if "fvu_rand" in r0:
                    s1b[k] = (ft - base, np.array(r0["fvu_rand"][str(li)]) - base)
                if r1 is not None:
                    floor[k] = (ft, np.array(r1["fvu_ft"][str(li)]))
    obs1a = analysis.s1a_observed(s1a)
    if floor:
        obs1a.update(analysis.seed_noise_floor(floor))
    out.append(record("S1a", obs1a))
    if s1b:
        out.append(record("S1b", analysis.s1b_observed(s1b)))
    else:
        record("S1b", {})  # raises: the control is missing

    # S2a / S2b on the SAEs' own model.
    d = divergence[SAE_MODEL]
    out.append(record("S2a", {**analysis.s2a_observed(d["p_by_layer"]), "model": SAE_MODEL}))
    out.append(
        record(
            "S2b",
            {"peak_in_range_frequency": float(d["peak_in_range_frequency"]), "model": SAE_MODEL},
        )
    )

    # S3 gatekeeping family.
    steps: dict[tuple[str, str], dict[str, float]] = {}
    for m in cells.MODELS:
        late = np.array([idx[(m, "late", s)]["gsm_correct"] for s in cells.NAMED_SEEDS], float)
        allc = np.array([idx[(m, "all", s)]["gsm_correct"] for s in cells.NAMED_SEEDS], float)
        steps[("S3a", m)] = analysis.s3a_step(late, allc)
        named = {
            s: _acc(idx[(m, "late", s)]["mmlu_logprob"], idx[(m, "late", s)]["mmlu_logprob_gold"])
            for s in cells.NAMED_SEEDS
        }
        rand = np.array(
            [
                _acc(r["mmlu_logprob"], r["mmlu_logprob_gold"])
                for (mm, a, _), r in idx.items()
                if mm == m and a.startswith("random_")
            ]
        )
        steps[("S3b", m)] = analysis.rank_step(named, rand, random_seed=0)
    for (cid, m), obs in analysis.gatekeep(steps).items():
        out.append(record(cid, {**obs, "model": m}))

    # S4: base model, all-layer arm, seed 0 (the source's 9.6% configuration).
    r = idx[(SAE_MODEL, "all", 0)]
    parse = [
        float(p == g)
        for p, g in zip(r["mmlu_gen"], r["mmlu_gen_gold"], strict=True)
        if p is not None
    ]
    lp = [float(p == g) for p, g in zip(r["mmlu_logprob"], r["mmlu_logprob_gold"], strict=True)]
    out.append(
        record("S4", {**analysis.s4_observed(np.array(parse), np.array(lp)), "model": SAE_MODEL})
    )

    # S5: SAE-guided top-13 vs random placements, GSM8K, primary model.
    named = {
        s: float(np.mean(idx[(PRIMARY_MODEL, "sae_top", s)]["gsm_correct"]))
        for s in cells.NAMED_SEEDS
    }
    rand = np.array(
        [
            float(np.mean(r["gsm_correct"]))
            for (mm, a, _), r in idx.items()
            if mm == PRIMARY_MODEL and a.startswith("random_")
        ]
    )
    out.append(record("S5", {**analysis.s5_observed(named, rand, 0), "model": PRIMARY_MODEL}))
    return out
