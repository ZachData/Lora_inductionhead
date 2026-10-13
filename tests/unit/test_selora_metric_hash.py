"""The selora metric surface may not change silently under a fixed SELORA_METRIC_VERSION."""

from __future__ import annotations

from indbw.selora import metric_hash


def test_committed_lock_matches_current_selora_metric_source() -> None:
    assert metric_hash.check() == []


def test_hash_detects_a_semantic_edit_and_ignores_cosmetics(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    for m in metric_hash.SELORA_METRIC_MODULES:
        (tmp_path / m).write_text((metric_hash.SRC / m).read_text())
    monkeypatch.setattr(metric_hash, "SRC", tmp_path)
    base = metric_hash.compute()["aggregate"]
    stats = tmp_path / "stats.py"
    stats.write_text(stats.read_text() + "\n\n# a comment\n")
    assert metric_hash.compute()["aggregate"] == base  # cosmetic: unchanged
    stats.write_text(stats.read_text().replace("1 + k) / (n + 1)", "k) / (n + 1)", 1))
    assert metric_hash.compute()["aggregate"] != base  # semantic: changed


def _lock_with(tmp_path, monkeypatch, **override):  # type: ignore[no-untyped-def]
    import json

    lock = {**metric_hash.compute(), **override}
    p = tmp_path / "lock.json"
    p.write_text(json.dumps(lock))
    monkeypatch.setattr(metric_hash, "LOCK", p)
    return p


def test_lock_records_the_interpreter_that_made_it() -> None:
    import platform

    assert metric_hash.compute()["python_version"] == ".".join(platform.python_version_tuple()[:2])


def test_other_interpreter_mismatch_is_skew_not_a_metric_change(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    # ast.dump differs across Pythons: a different aggregate from a different interpreter must not
    # read as "bump SELORA_METRIC_VERSION and regenerate", which would bake the skew in as baseline
    _lock_with(tmp_path, monkeypatch, aggregate="0" * 64, python_version="2.7")
    (msg,) = metric_hash.check()
    assert "Python 2.7" in msg and "Do not" in msg and "without bumping" not in msg


def test_same_interpreter_mismatch_is_a_metric_change(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    _lock_with(tmp_path, monkeypatch, aggregate="0" * 64)
    (msg,) = metric_hash.check()
    assert "without bumping" in msg


def test_update_refuses_to_rebaseline_from_another_interpreter(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import pytest

    p = _lock_with(tmp_path, monkeypatch, aggregate="0" * 64, python_version="2.7")
    before = p.read_text()
    with pytest.raises(ValueError, match="2.7"):
        metric_hash.update()
    assert p.read_text() == before


def test_update_may_stamp_a_lock_with_no_interpreter_record(tmp_path, monkeypatch) -> None:  # type: ignore[no-untyped-def]
    import json

    p = _lock_with(tmp_path, monkeypatch, aggregate="0" * 64, python_version="unknown")
    metric_hash.update()
    assert json.loads(p.read_text())["aggregate"] == metric_hash.compute()["aggregate"]
