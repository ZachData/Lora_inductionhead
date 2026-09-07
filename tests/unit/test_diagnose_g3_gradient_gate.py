"""Tests for the gradient-gate diagnostic's chunked copying score.

`probes.copying_score` is the metric of record and is not touched here --
it is inside the METRIC_VERSION surface (metric_hash.METRIC_MODULES), so
changing its implementation would stale every committed record.

But it cannot be called on a real checkpoint. It materializes the full
[vocab, vocab] logit matrix in one allocation, and at pythia-70m's
vocab of 50,304 that is 50304^2 * 4 bytes = 10.1 GB -- larger than any
instance this project is permitted to launch, and 5x the orchestrator's
entire RAM. Every prior memory incident in PROJECT.md 11 has this exact
shape (a full-vocab logits tensor sized by a constant nobody multiplied
out), so the arithmetic is done here rather than discovered on a worker.

`copying_score_chunked` is therefore the same definition evaluated over
token blocks. The fraction of vocab tokens whose argmax is themselves
decomposes exactly over a partition of those tokens, so chunking is
arithmetic-preserving rather than an approximation -- and that claim is
what these tests check, against the hashed function itself as the oracle,
at sizes where both fit in memory.

The silent failure this guards is specific and cheap to write: a chunked
argmax that indexes into the chunk rather than the vocabulary returns
`0.0` for every input that is not the first chunk. Zero is exactly the
value the diagnostic is looking for -- a gated OV circuit -- so a wrong
implementation and the finding under investigation are the same number.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT / "scripts"))

from diagnose_g3_gradient_gate import copying_score_chunked  # noqa: E402

from indbw.probes import copying_score  # noqa: E402


def _rng(seed: int) -> np.random.Generator:
    return np.random.default_rng(seed)


# --- the oracle: agreement with the hashed metric ----------------------


@pytest.mark.parametrize("chunk", [1, 2, 3, 7, 16, 64])
def test_matches_probes_copying_score_at_every_chunk_size(chunk: int) -> None:
    """Exact agreement with the metric of record, including chunk sizes that
    do not divide the vocab evenly (7 into 16) and the degenerate chunk=1."""
    rng = _rng(0)
    vocab, d = 16, 5
    W_U = rng.standard_normal((vocab, d))
    M_OV = rng.standard_normal((d, d))
    W_E = rng.standard_normal((vocab, d))
    assert copying_score_chunked(W_U, M_OV, W_E, chunk=chunk) == copying_score(W_U, M_OV, W_E)


def test_chunk_larger_than_vocab_is_the_unchunked_case() -> None:
    rng = _rng(1)
    vocab, d = 12, 4
    W_U, M_OV, W_E = (
        rng.standard_normal((vocab, d)),
        rng.standard_normal((d, d)),
        rng.standard_normal((vocab, d)),
    )
    assert copying_score_chunked(W_U, M_OV, W_E, chunk=10_000) == copying_score(W_U, M_OV, W_E)


# --- discrimination: the readout must separate copy from no-copy -------


def test_identity_ov_copies_and_a_permutation_does_not() -> None:
    """A known-positive and a known-negative that are NOT each other's transpose.

    PROJECT.md 11 flags identity oracles as transpose-blind: copying_score(I,
    I, I) == 1.0 passes even under a W_U/M_OV argument-order bug because
    I.T == I. A cyclic permutation is the fix -- P.T != P, it maps every
    token to a different token, so a correct implementation scores exactly
    0.0 and an argument-order bug scores something else.
    """
    vocab = 24
    eye = np.eye(vocab)
    assert copying_score_chunked(eye, eye, eye, chunk=5) == 1.0

    shift = np.roll(eye, 1, axis=0)
    assert not np.array_equal(shift, shift.T)
    assert copying_score_chunked(eye, shift, eye, chunk=5) == 0.0


def _half_copying_fixture() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """W_U = W_E = I and an M_OV under which exactly half the vocab copies.

    With identity embeddings logits[t, s] = M_OV[s, t], so column t decides
    token t. Tokens t < 6 get M_OV[t, t] = 2 and copy; tokens t >= 6 get
    M_OV[0, t] = 1 and map to token 0 instead. Score is exactly 0.5.

    A partial score is the point. An all-copy (identity) or no-copy fixture
    lets several distinct bugs return the right answer by coincidence --
    which is what an earlier version of this test did, and why the
    accompanying discrimination test below exists.
    """
    vocab = 12
    eye = np.eye(vocab)
    M_OV = np.zeros((vocab, vocab))
    for t in range(6):
        M_OV[t, t] = 2.0
    for t in range(6, vocab):
        M_OV[0, t] = 1.0
    return eye, M_OV, eye


@pytest.mark.parametrize("chunk", [1, 4, 5, 6, 7, 12, 13])
def test_partial_copying_is_exact_at_chunk_sizes_that_straddle_the_boundary(
    chunk: int,
) -> None:
    """The load-bearing guard: a known score of exactly 0.5, at chunk sizes
    that split the copying and non-copying halves in different places."""
    W_U, M_OV, W_E = _half_copying_fixture()
    got = copying_score_chunked(W_U, M_OV, W_E, chunk=chunk)
    assert got == pytest.approx(0.5)
    assert got == copying_score(W_U, M_OV, W_E)


def test_the_chunk_offset_bug_would_change_the_answer() -> None:
    """Confirms the guard above is not vacuous.

    The realistic bug is comparing each block's argmax against chunk-local
    ids (`arange(len(block))`) rather than global ones (`arange(start,
    stop)`). Simulated here to show it yields a different number on the
    same fixture -- so the test above would actually catch it.
    """
    W_U, M_OV, W_E = _half_copying_fixture()
    vocab, chunk = W_U.shape[0], 4
    logits = (W_E @ M_OV.T) @ W_U.T
    n_buggy = 0
    for start in range(0, vocab, chunk):
        block = logits[start : start + chunk]
        n_buggy += int(np.count_nonzero(np.argmax(block, axis=1) == np.arange(block.shape[0])))
    assert n_buggy / vocab != pytest.approx(0.5)
    assert copying_score_chunked(W_U, M_OV, W_E, chunk=chunk) == pytest.approx(0.5)


# --- input validation is inherited, not re-implemented -----------------


def test_rejects_all_zero_ov_like_the_hashed_metric() -> None:
    """CLAUDE.md: a readout given a degenerate input must raise, never return
    a plausible number. An all-zero M_OV makes every logit row identical, so
    argmax returns 0 for every token and the score is 1/vocab -- a small,
    entirely believable copying score for a circuit that does not exist."""
    vocab, d = 8, 3
    with pytest.raises(ValueError):
        copying_score_chunked(np.ones((vocab, d)), np.zeros((d, d)), np.ones((vocab, d)), chunk=2)


@pytest.mark.parametrize("chunk", [0, -1])
def test_rejects_nonpositive_chunk(chunk: int) -> None:
    rng = _rng(2)
    vocab, d = 8, 3
    with pytest.raises(ValueError):
        copying_score_chunked(
            rng.standard_normal((vocab, d)),
            rng.standard_normal((d, d)),
            rng.standard_normal((vocab, d)),
            chunk=chunk,
        )


def test_rejects_shape_mismatch() -> None:
    rng = _rng(3)
    with pytest.raises(ValueError):
        copying_score_chunked(
            rng.standard_normal((8, 3)),
            rng.standard_normal((3, 3)),
            rng.standard_normal((9, 3)),  # vocab disagrees with W_U
            chunk=2,
        )


# --- the OV composition from TransformerLens's weight layout -----------


def test_ov_matrix_matches_the_section_3_definition() -> None:
    """M_OV = W_O^T W_V^T (PROJECT.md 3), built from TL's [d_model, d_head]
    W_V and [d_head, d_model] W_O. Asserted against an explicit
    vector-by-vector application so a transpose error cannot survive: the
    head's action on a residual x is (x @ W_V) @ W_O.
    """
    from diagnose_g3_gradient_gate import ov_matrix

    rng = _rng(4)
    d_model, d_head = 7, 3
    W_V = rng.standard_normal((d_model, d_head))
    W_O = rng.standard_normal((d_head, d_model))

    M_OV = ov_matrix(W_V, W_O)
    assert M_OV.shape == (d_model, d_model)

    x = rng.standard_normal(d_model)
    np.testing.assert_allclose(M_OV @ x, (x @ W_V) @ W_O, rtol=1e-12, atol=1e-12)


def test_ov_matrix_is_not_symmetric_for_generic_weights() -> None:
    """Guards the same transpose-blindness the identity oracle has: if
    ov_matrix returned its own transpose the test above would still pass for
    symmetric inputs, so pin that generic weights give an asymmetric form."""
    from diagnose_g3_gradient_gate import ov_matrix

    rng = _rng(5)
    M = ov_matrix(rng.standard_normal((6, 2)), rng.standard_normal((2, 6)))
    assert not np.allclose(M, M.T)
