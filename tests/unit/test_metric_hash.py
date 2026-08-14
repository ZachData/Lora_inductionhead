"""Tests for indbw.metric_hash -- the tier-0 metric-version gate.

This file is the spec for what "the metric definition changed" means.
The gate's whole value is that it fires on a semantic change and stays
quiet on a cosmetic one; a gate that fires on reformatting gets routed
around, and a gate that misses a changed constant is worse than none.
Both directions are tested here against synthetic sources, plus the
live check against the committed lockfile.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from indbw.metric_hash import (
    METRIC_MODULES,
    check,
    compute_metric_hash,
    load_lock,
    normalized_definitions,
    update,
    write_lock,
)
from indbw.schema import METRIC_VERSION

REPO_ROOT = Path(__file__).resolve().parents[2]

# ---------------------------------------------------------------------------
# 1. The live gate: the committed lockfile describes the current source.
# ---------------------------------------------------------------------------


def test_committed_lockfile_matches_current_metric_source() -> None:
    # The same assertion CI tier 0 makes, run locally so a metric edit
    # surfaces in `pytest tests/unit` and not three minutes into CI.
    assert check() == []


def test_lockfile_records_the_current_metric_version() -> None:
    assert load_lock().metric_version == METRIC_VERSION


def test_every_metric_module_is_covered() -> None:
    current = compute_metric_hash()
    assert set(current.definitions) == set(METRIC_MODULES)
    # Discrimination: a module that parsed to nothing would produce an
    # empty dict and a stable-looking hash that tracks no code at all.
    for module, defs in current.definitions.items():
        assert defs, f"{module} contributed no definitions to the metric hash"


def test_known_metric_functions_are_in_the_hashed_surface() -> None:
    # Named explicitly so that deleting or moving one of PROJECT.md §5's
    # metrics out of the hashed modules fails here rather than silently
    # shrinking the surface the gate protects.
    current = compute_metric_hash()
    assert {"phi", "sym_antisym_split", "principal_angles", "truncate_svd"} <= set(
        current.definitions["algebra.py"]
    )
    assert {
        "prefix_matching_score",
        "prev_token_score",
        "copying_score",
        "icl_score",
        "recovery",
    } <= set(current.definitions["probes.py"])


# ---------------------------------------------------------------------------
# 2. Discrimination: cosmetic vs. semantic edits
# ---------------------------------------------------------------------------

_BASE = '''
"""Module docstring."""

from __future__ import annotations

import numpy as np

THRESHOLD = 0.3


def metric(x: float) -> float:
    """Docstring."""
    return x * THRESHOLD
'''


def test_docstring_edit_does_not_change_the_hash() -> None:
    edited = _BASE.replace('"""Docstring."""', '"""A much longer docstring, rewritten."""').replace(
        '"""Module docstring."""', '"""Rewritten module docstring."""'
    )
    assert normalized_definitions(edited) == normalized_definitions(_BASE)


def test_comment_and_whitespace_edits_do_not_change_the_hash() -> None:
    edited = _BASE.replace(
        "    return x * THRESHOLD",
        "    # explain the scaling\n\n    return (\n        x * THRESHOLD\n    )",
    )
    assert normalized_definitions(edited) == normalized_definitions(_BASE)


def test_import_reorder_does_not_change_the_hash() -> None:
    edited = _BASE.replace("import numpy as np", "import json\nimport numpy as np")
    assert normalized_definitions(edited) == normalized_definitions(_BASE)


def test_changed_constant_changes_the_hash() -> None:
    edited = _BASE.replace("THRESHOLD = 0.3", "THRESHOLD = 0.5")
    before, after = normalized_definitions(_BASE), normalized_definitions(edited)
    assert before["THRESHOLD"] != after["THRESHOLD"]
    assert before["metric"] == after["metric"]  # only the constant moved


def test_changed_function_body_changes_the_hash() -> None:
    edited = _BASE.replace("return x * THRESHOLD", "return x * THRESHOLD + 1")
    assert normalized_definitions(edited)["metric"] != normalized_definitions(_BASE)["metric"]


def test_added_function_changes_the_hashed_surface() -> None:
    edited = _BASE + "\n\ndef other(y: float) -> float:\n    return y\n"
    assert set(normalized_definitions(edited)) - set(normalized_definitions(_BASE)) == {"other"}


def test_unhandled_top_level_construct_raises() -> None:
    # A metric must not be able to hide inside a statement form the
    # hasher silently skips.
    with pytest.raises(ValueError, match="unhandled top-level construct"):
        normalized_definitions("if True:\n    THRESHOLD = 0.3\n")


def test_scope_whose_body_is_only_a_docstring_still_parses() -> None:
    assert "stub" in normalized_definitions('def stub() -> None:\n    """Nothing yet."""\n')


# ---------------------------------------------------------------------------
# 3. The enforcement itself: update() refuses without a version bump
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_surface(tmp_path: Path) -> Path:
    src = tmp_path / "src"
    src.mkdir()
    for module in METRIC_MODULES:
        (src / module).write_text(_BASE)
    return src


def test_update_then_check_is_clean(fake_surface: Path, tmp_path: Path) -> None:
    lock = tmp_path / "metric_hash.json"
    update(src_dir=fake_surface, lock_path=lock)
    assert check(src_dir=fake_surface, lock_path=lock) == []


def test_check_fails_when_a_metric_changes_without_a_version_bump(
    fake_surface: Path, tmp_path: Path
) -> None:
    lock = tmp_path / "metric_hash.json"
    update(src_dir=fake_surface, lock_path=lock)
    (fake_surface / "probes.py").write_text(_BASE.replace("THRESHOLD = 0.3", "THRESHOLD = 0.5"))
    problems = check(src_dir=fake_surface, lock_path=lock)
    assert len(problems) == 1
    assert "probes.py::THRESHOLD changed" in problems[0]


def test_update_refuses_when_a_metric_changed_without_a_version_bump(
    fake_surface: Path, tmp_path: Path
) -> None:
    lock = tmp_path / "metric_hash.json"
    update(src_dir=fake_surface, lock_path=lock)
    (fake_surface / "probes.py").write_text(_BASE.replace("THRESHOLD = 0.3", "THRESHOLD = 0.5"))
    with pytest.raises(ValueError, match="refusing to update"):
        update(src_dir=fake_surface, lock_path=lock)


def test_update_succeeds_once_the_version_is_bumped(fake_surface: Path, tmp_path: Path) -> None:
    lock = tmp_path / "metric_hash.json"
    update(src_dir=fake_surface, lock_path=lock)
    (fake_surface / "probes.py").write_text(_BASE.replace("THRESHOLD = 0.3", "THRESHOLD = 0.5"))
    # Simulate the bump: rewrite the lockfile's recorded version, as a
    # real METRIC_VERSION bump in schema.py would.
    stale = json.loads(lock.read_text())
    stale["metric_version"] = "0.0.0-previous"
    lock.write_text(json.dumps(stale))
    mh = update(src_dir=fake_surface, lock_path=lock)
    assert mh.metric_version == METRIC_VERSION
    assert check(src_dir=fake_surface, lock_path=lock) == []


def test_check_flags_a_stale_recorded_version(fake_surface: Path, tmp_path: Path) -> None:
    lock = tmp_path / "metric_hash.json"
    mh = update(src_dir=fake_surface, lock_path=lock)
    write_lock(
        type(mh)(
            metric_version="9.9.9",
            aggregate=mh.aggregate,
            definitions=mh.definitions,
        ),
        lock,
    )
    problems = check(src_dir=fake_surface, lock_path=lock)
    assert len(problems) == 1
    assert "9.9.9" in problems[0]


def test_missing_metric_module_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compute_metric_hash(src_dir=tmp_path)
