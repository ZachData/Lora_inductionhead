Read PROJECT.md. Take the first unfinished (○) row in phase order from the
status board. Follow CLAUDE.md's development rules: write the test first,
implement, run `pytest tests/unit tests/property -q`, and only proceed if it
is green.

If this row is a sweep (multiple independent cells — rank/lr/seed/arm
combinations), do not run the cells yourself. Decide the split, write
sweep_manifest.json with the schema {"workers": [{"worker_id": 0, "arm": ...,
"rank": ..., "lr": ..., "seed": ...}]} — a worker entry may also carry "cmd"
to override the default worker command — create an empty file NEEDS_WORKERS at
the repo root, mark the row ⏳, commit, and stop there.

If this row is not a sweep, do the work directly: mark ⏳ on start, ● or ✗ on
finish. A gate that fails stops the phase; record it and do not route around
it. Append any decision to §10 (Decisions log) and any new uncertainty to §11
(Open questions). Commit with a real message and push.
