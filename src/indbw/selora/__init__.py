"""Selective-LoRA placement x SAE validity (PROJECT.md section 13).

Separate from the induction-bandwidth metric surface: its metrics carry
`SELORA_METRIC_VERSION` (prereg.py) and their own hash lock, so editing
them never invalidates the Pythia records under `METRIC_VERSION`.
"""
