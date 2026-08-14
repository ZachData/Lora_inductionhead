# Review queue

Append-only. Anything an "how this could be wrong" commit note couldn't
resolve lands here for human eyes (CLAUDE.md, "Adversarial self-review").

There is no reviewer in the CD loop. This is the substitute, and it is
weak — keep this queue short and actually read it.

| Date | Commit | What the note couldn't resolve |
|---|---|---|
| 2026-08-14 | 5d88030 | `train.py`'s LoRA hook injection (`qk_hooks`/`ov_hooks`) assumes TransformerLens's within-block hook execution order is stable (`ln1.hook_normalized` before `attn.hook_q`; `attn.hook_z` before `hook_attn_out`) so a "capture, then add" closure sees the right tensor. If a future TL version changed that order, the capture box would read a stale/mismatched tensor. The zero-init discrimination test (bit-identical forward pass at B=0) would not catch this, since it never depends on capture timing; only the nonzero-delta test would, and only if the reordering also broke shape compatibility rather than just changing values. No test currently pins the hook *order* itself — worth a direct assertion on TransformerLens's hook-firing sequence if this becomes a live risk (e.g. after any TL version bump). |
| 2026-08-14 | 7d87384 | G3 failed (R=0.0117 vs. 0.80) with a loss curve that never moved (flat 12.8–13.0 across all 400 steps) — logged as an *inconclusive* gate failure rather than a finding about PROJECT.md §3's composition rule, because flat-from-step-1 loss looks like an optimization/tuning problem (bad lr, or a subtler bug in the QK path) rather than genuine capacity exhaustion. This commit's note could not resolve which. `tests/unit/test_train.py`'s only convergence-mechanics test (`test_overfitting_a_single_batch_reduces_loss`) deliberately exercises the OV arm, not QK — its own docstring already flags QK-only as known-hard on a toy model, but there is no equivalent closed-loop test proving QK-only *can* converge under any hyperparameters, toy or real. Needs a human call: run an lr sweep on the same generous-rank QK config before trusting this G3 result, or add the missing QK convergence test to rule out a real bug first (PROJECT.md §11 has the same flag). |
