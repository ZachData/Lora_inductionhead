"""Training loop and adapter snapshotting.

Snapshot every N steps — LoRA factors only, never merged models
(PROJECT.md §8, "Storage").
"""
