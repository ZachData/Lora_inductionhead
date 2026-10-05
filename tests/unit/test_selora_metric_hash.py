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
