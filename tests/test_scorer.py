"""Tests for ``context_forge.scorer`` (BM25)."""

from context_forge import score_chunks, tokenize


def test_tokenize_lowercases_and_splits_words():
    assert tokenize("Hello, World! 42") == ["hello", "world", "42"]


def test_score_chunks_ranks_relevant_higher():
    chunks = [
        {"id": "a", "text": "Pluto is a dwarf planet."},
        {"id": "b", "text": "Cats are cute."},
        {"id": "c", "text": "Pluto Pluto Pluto."},
    ]
    out = score_chunks("Pluto", chunks)
    assert out[0]["id"] in ("a", "c")
    assert out[-1]["id"] == "b"


def test_score_chunks_empty_input_returns_empty():
    assert score_chunks("anything", []) == []


def test_score_chunks_assigns_score_field():
    out = score_chunks("foo", [{"id": "x", "text": "foo bar"}])
    assert "score" in out[0]
    assert out[0]["score"] >= 0


def test_score_chunks_does_not_mutate_input():
    chunks = [{"id": "a", "text": "hello"}]
    score_chunks("hello", chunks)
    assert "score" not in chunks[0]
