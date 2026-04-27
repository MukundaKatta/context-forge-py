"""Tests for ``context_forge.diversity`` (MMR)."""

from context_forge import diversify


def test_diversify_empty_returns_empty():
    assert diversify([]) == []


def test_diversify_lambda_one_pure_relevance_order():
    chunks = [
        {"id": "low", "text": "cats", "score": 0.1},
        {"id": "high", "text": "pluto pluto pluto", "score": 1.0},
        {"id": "mid", "text": "dogs", "score": 0.5},
    ]
    out = diversify(chunks, lambda_=1.0)
    assert [c["id"] for c in out] == ["high", "mid", "low"]


def test_diversify_lambda_zero_picks_diverse_after_first():
    chunks = [
        {"id": "a", "text": "pluto pluto pluto", "score": 1.0},
        {"id": "a-dup", "text": "pluto pluto pluto", "score": 0.99},
        {"id": "diverse", "text": "totally different words", "score": 0.5},
    ]
    out = diversify(chunks, lambda_=0.0)
    # First pick is the highest-relevance after which lambda=0 prefers
    # whichever is most dissimilar.
    assert out[1]["id"] == "diverse"


def test_diversify_returns_all_chunks():
    chunks = [{"id": str(i), "text": "x", "score": 0} for i in range(4)]
    out = diversify(chunks)
    assert len(out) == 4


def test_diversify_does_not_mutate_input():
    chunks = [{"id": "a", "text": "hello", "score": 1.0}]
    diversify(chunks)
    assert chunks == [{"id": "a", "text": "hello", "score": 1.0}]
