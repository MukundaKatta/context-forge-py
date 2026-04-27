"""Tests for ``context_forge.packer``."""

from context_forge import pack_to_budget


def test_pack_keeps_chunks_under_budget():
    chunks = [
        {"id": "a", "text": "x" * 100, "tokens": 25},
        {"id": "b", "text": "x" * 100, "tokens": 25},
    ]
    out = pack_to_budget(chunks, budget_tokens=100, per_chunk_min=10)
    assert {c["id"] for c in out["kept"]} == {"a", "b"}
    assert out["used_tokens"] == 50


def test_pack_drops_when_budget_exceeded():
    chunks = [
        {"id": "a", "text": "x" * 100, "tokens": 80},
        {"id": "b", "text": "x" * 100, "tokens": 80},
    ]
    out = pack_to_budget(chunks, budget_tokens=100, per_chunk_min=10)
    assert len(out["kept"]) == 1
    assert out["dropped"][0]["reason"] == "budget_exceeded"


def test_pack_drops_below_min_tokens():
    chunks = [{"id": "tiny", "text": "x", "tokens": 5}]
    out = pack_to_budget(chunks, budget_tokens=1000, per_chunk_min=20)
    assert out["kept"] == []
    assert out["dropped"][0]["reason"] == "below_min_tokens"


def test_pack_estimates_tokens_when_missing():
    chunks = [{"id": "a", "text": "x" * 200}]
    out = pack_to_budget(chunks, budget_tokens=1000, per_chunk_min=10)
    assert out["used_tokens"] == 50  # ceil(200/4)


def test_pack_empty_returns_zero():
    out = pack_to_budget([], budget_tokens=1000)
    assert out == {"kept": [], "dropped": [], "used_tokens": 0}
