"""Results record schema and METRIC_VERSION.

A record is invalid without: null tested, pre-registered criterion,
observed value, verdict, METRIC_VERSION (PROJECT.md §9). The verdict
must be recomputable from the criterion and observed value — CI
recomputes it in tier 2.5 (CLAUDE.md, "Computable verdicts").
"""

METRIC_VERSION = "0.1.0"
