"""Tests for scripts/g0_sweep.py's S3 durability sync.

Not under src/indbw -- g0_sweep.py is sweep orchestration, not a metric
(see its own module docstring), so it isn't part of the METRIC_VERSION
surface. sync_to_s3 still gets tests because it's exactly the kind of
thing CLAUDE.md's TDD contract flags: a failure mode (a raised exception
here) that would silently take down hours of real sweep progress if it
ever propagated, rather than just failing to back up a file.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import g0_sweep


def test_empty_bucket_is_a_no_op(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # No bucket configured (the default, CLAUDE.md-style: existing behavior
    # unchanged unless explicitly opted in) must not touch the network at all.
    def _fail_if_called(*args: Any, **kwargs: Any) -> None:
        raise AssertionError("subprocess.run must not be called with an empty bucket")

    monkeypatch.setattr(subprocess, "run", _fail_if_called)
    target = tmp_path / "g0_sweep_worker0.jsonl"
    target.write_text('{"step": 0}\n')
    g0_sweep.sync_to_s3(target, "")


def test_configured_bucket_invokes_aws_cp_with_expected_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[list[str]] = []

    def _record(cmd: list[str], **kwargs: Any) -> None:
        calls.append(cmd)

    monkeypatch.setattr(subprocess, "run", _record)
    target = tmp_path / "g0_sweep_worker3.jsonl"
    target.write_text('{"step": 1}\n')

    g0_sweep.sync_to_s3(target, "my-research-bucket")

    assert len(calls) == 1
    cmd = calls[0]
    assert cmd[:3] == ["aws", "s3", "cp"]
    assert cmd[3] == str(target)
    assert cmd[4] == "s3://my-research-bucket/g0_sweep/g0_sweep_worker3.jsonl"


def test_sync_failure_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # The silent-failure guard this test exists for: a dead network, an
    # expired credential, or a missing bucket must degrade to a printed
    # warning, never propagate and abort the sweep mid-checkpoint.
    def _boom(*args: Any, **kwargs: Any) -> None:
        raise subprocess.CalledProcessError(1, ["aws", "s3", "cp"])

    monkeypatch.setattr(subprocess, "run", _boom)
    target = tmp_path / "g0_sweep_worker5.jsonl"
    target.write_text('{"step": 2}\n')

    g0_sweep.sync_to_s3(target, "my-research-bucket")  # must not raise


def test_sync_timeout_does_not_raise(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def _hang(*args: Any, **kwargs: Any) -> None:
        raise subprocess.TimeoutExpired(cmd=["aws", "s3", "cp"], timeout=30)

    monkeypatch.setattr(subprocess, "run", _hang)
    target = tmp_path / "g0_sweep_worker6.jsonl"
    target.write_text('{"step": 3}\n')

    g0_sweep.sync_to_s3(target, "my-research-bucket")  # must not raise
