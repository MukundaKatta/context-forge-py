"""Tests for ``context_forge.chunker``.

Focus on the ``start``/``end`` offset contract: ``text[start:end]`` of the
source document must reconstruct the chunk's content (modulo the ``\\n\\n``
joins the chunker inserts between units), even when units repeat or are
reused as overlap.
"""

from context_forge import chunk_document


def _norm(s: str) -> str:
    return " ".join(s.split())


def _assert_spans_recover_text(text, chunks):
    for c in chunks:
        span = text[c["start"] : c["end"]]
        assert _norm(span) == _norm(c["text"]), (
            f"span mismatch for {c['id']}: span={span!r} text={c['text']!r}"
        )


def test_empty_or_missing_text_returns_empty():
    assert chunk_document({"id": "d", "text": ""}) == []
    assert chunk_document({"id": "d"}) == []
    assert chunk_document("not a mapping") == []


def test_basic_fields_present():
    doc = {"id": "doc1", "text": "Para one. Some more.\n\nPara two.", "source": "s"}
    chunks = chunk_document(doc, max_tokens=100, overlap_tokens=0)
    assert chunks
    for c in chunks:
        assert c["doc_id"] == "doc1"
        assert c["source"] == "s"
        assert set(c) >= {"id", "doc_id", "source", "text", "start", "end", "tokens"}
        assert c["tokens"] > 0


def test_spans_recover_text_unique_no_overlap():
    text = (
        "Sentence one here. Sentence two here. Sentence three here. "
        "Sentence four here. Sentence five here. Sentence six here."
    )
    chunks = chunk_document({"id": "d1", "text": text}, max_tokens=10, overlap_tokens=0)
    assert len(chunks) > 1
    _assert_spans_recover_text(text, chunks)


def test_spans_recover_text_with_repeated_units():
    # Regression: repeated units used to collapse end == start + len(one unit),
    # so citation spans only covered the first unit of a multi-unit chunk.
    text = "Alpha beta gamma. Delta epsilon zeta. " * 6
    chunks = chunk_document({"id": "d1", "text": text}, max_tokens=40, overlap_tokens=0)
    assert chunks
    _assert_spans_recover_text(text, chunks)
    # The first chunk packs several repeated units; its span must cover them all.
    assert chunks[0]["end"] - chunks[0]["start"] > len(
        "Alpha beta gamma. Delta epsilon zeta."
    )


def test_spans_recover_text_with_overlap():
    text = " ".join(
        f"Unique sentence number {w} appears once."
        for w in ["alpha", "bravo", "charlie", "delta", "echo", "foxtrot"]
    )
    chunks = chunk_document({"id": "d1", "text": text}, max_tokens=12, overlap_tokens=6)
    assert len(chunks) > 1
    _assert_spans_recover_text(text, chunks)


def test_doc_id_and_source_fallbacks():
    # No id -> falls back to path; source falls back to path.
    chunks = chunk_document({"path": "a/b.md", "text": "Hello world here today."})
    assert chunks
    assert chunks[0]["doc_id"] == "a/b.md"
    assert chunks[0]["source"] == "a/b.md"


def test_respects_max_tokens_per_chunk():
    text = ("word " * 400).strip()
    chunks = chunk_document({"id": "d1", "text": text}, max_tokens=50, overlap_tokens=0)
    assert chunks
    # No chunk should blow far past the budget (units are word-bounded).
    assert all(c["tokens"] <= 60 for c in chunks)
