"""Closed-form checks and validation guards for src/indbw/models.py.

PROJECT.md §2. No test here loads a real checkpoint or touches the
network — that's tests/integration/test_models.py's job (CLAUDE.md,
"Fixtures": never load a real checkpoint outside tests/integration/).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from indbw.models import MODEL_NAME, checkpoint_steps, load_checkpoint


def test_checkpoint_steps_matches_pythia_grid() -> None:
    # PROJECT.md §1.1/§2: step 0, log-spaced to 512, then every 1000
    # steps to 143000 -- 11 + 143 = 154 checkpoints, exactly.
    steps = checkpoint_steps()
    expected = [0, 1, 2, 4, 8, 16, 32, 64, 128, 256, 512] + list(range(1000, 143001, 1000))
    assert steps == expected
    assert len(steps) == 154


def test_load_checkpoint_rejects_off_grid_step_without_network_call() -> None:
    # 256 and 512 are on the grid; 500 is not (log-spacing then jumps to
    # 1000). Validation must reject this before any network/HF call.
    with patch("indbw.models.HookedTransformer.from_pretrained") as mock_load:
        with pytest.raises(ValueError):
            load_checkpoint(500)
        mock_load.assert_not_called()


def test_load_checkpoint_calls_from_pretrained_with_validated_step() -> None:
    with patch("indbw.models.HookedTransformer.from_pretrained") as mock_load:
        mock_load.return_value = "sentinel-model"
        result = load_checkpoint(1000)
        assert result == "sentinel-model"
        mock_load.assert_called_once()
        args, kwargs = mock_load.call_args
        assert args[0] == MODEL_NAME
        assert kwargs.get("checkpoint_value") == 1000


def test_load_checkpoint_forwards_extra_kwargs() -> None:
    with patch("indbw.models.HookedTransformer.from_pretrained") as mock_load:
        mock_load.return_value = "sentinel-model"
        load_checkpoint(0, device="cpu")
        _, kwargs = mock_load.call_args
        assert kwargs.get("device") == "cpu"
