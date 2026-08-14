# Review queue

Append-only. Anything an "how this could be wrong" commit note couldn't
resolve lands here for human eyes (CLAUDE.md, "Adversarial self-review").

There is no reviewer in the CD loop. This is the substitute, and it is
weak — keep this queue short and actually read it.

| Date | Commit | What the note couldn't resolve |
|---|---|---|
| 2026-08-14 | 5d88030 | `train.py`'s LoRA hook injection (`qk_hooks`/`ov_hooks`) assumes TransformerLens's within-block hook execution order is stable (`ln1.hook_normalized` before `attn.hook_q`; `attn.hook_z` before `hook_attn_out`) so a "capture, then add" closure sees the right tensor. If a future TL version changed that order, the capture box would read a stale/mismatched tensor. The zero-init discrimination test (bit-identical forward pass at B=0) would not catch this, since it never depends on capture timing; only the nonzero-delta test would, and only if the reordering also broke shape compatibility rather than just changing values. No test currently pins the hook *order* itself — worth a direct assertion on TransformerLens's hook-firing sequence if this becomes a live risk (e.g. after any TL version bump). |
