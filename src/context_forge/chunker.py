"""Document chunker.

Split first on blank-line paragraphs, then on sentence boundaries inside
paragraphs that exceed a chunk on their own. Falls back to word-level
splitting for sentences that are still too large. Pack units sequentially
under ``max_tokens`` with optional token-level overlap.
"""

from __future__ import annotations

import re
from typing import List, Mapping

from .estimate import estimate_tokens

_PARA = re.compile(r"\n\s*\n+")
# Split on terminal punctuation followed by whitespace; preserve the punctuation
# so the chunk text reads naturally when reassembled.
_SENT = re.compile(r"[^.!?]+[.!?]+(?:\s+|$)|[^.!?]+$")


def _split_paragraphs(text: str) -> List[str]:
    return [p.strip() for p in _PARA.split(text) if p.strip()]


def _split_sentences(paragraph: str) -> List[str]:
    parts = _SENT.findall(paragraph)
    parts = [p.strip() for p in parts if p.strip()]
    if len(parts) > 1:
        return parts
    return [paragraph]


def chunk_document(
    doc: Mapping, *, max_tokens: int = 200, overlap_tokens: int = 20
) -> List[dict]:
    """Split ``doc`` into chunk dicts.

    ``doc`` is a mapping with ``text`` and an optional ``id`` / ``path`` /
    ``source``. Returns a list of dicts with ``id``, ``doc_id``, ``source``,
    ``text``, ``start``, ``end``, ``tokens``.
    """
    overlap = max(0, overlap_tokens)
    text = "" if not isinstance(doc, Mapping) else str(doc.get("text") or "")
    doc_id = (
        str(doc.get("id"))
        if isinstance(doc, Mapping) and doc.get("id")
        else (
            str(doc.get("path"))
            if isinstance(doc, Mapping) and doc.get("path")
            else "doc"
        )
    )
    if isinstance(doc, Mapping) and doc.get("source") is not None:
        source = doc.get("source")
    elif isinstance(doc, Mapping) and doc.get("path") is not None:
        source = doc.get("path")
    elif isinstance(doc, Mapping) and doc.get("id") is not None:
        source = doc.get("id")
    else:
        source = doc_id
    if not text.strip():
        return []

    # Collect units together with their true character offsets in ``text`` so
    # citation spans stay correct even when units repeat or are reused as
    # overlap. ``locate`` advances a monotonic search cursor so identical units
    # map to successive (not the first) occurrences.
    units: List[tuple] = []  # (text, start, end)
    locate_cursor = 0

    def locate(unit: str) -> tuple:
        nonlocal locate_cursor
        idx = text.find(unit, locate_cursor)
        if idx == -1:
            idx = locate_cursor
            end = idx + len(unit)
        else:
            end = idx + len(unit)
            locate_cursor = end
        return unit, idx, end

    for paragraph in _split_paragraphs(text):
        if estimate_tokens(paragraph) <= max_tokens:
            units.append(locate(paragraph))
            continue
        for sentence in _split_sentences(paragraph):
            if estimate_tokens(sentence) <= max_tokens:
                units.append(locate(sentence))
                continue
            # Word-level fallback for runaway sentences.
            words = re.split(r"\s+", sentence)
            buffer: List[str] = []
            for word in words:
                candidate = (" ".join(buffer) + " " + word).strip() if buffer else word
                if buffer and estimate_tokens(candidate) > max_tokens:
                    units.append(locate(" ".join(buffer)))
                    buffer = [word]
                else:
                    buffer.append(word)
            if buffer:
                units.append(locate(" ".join(buffer)))

    chunks: List[dict] = []
    buffer: List[tuple] = []
    buffer_tokens = 0
    counter = [0]

    def flush():
        if not buffer:
            return
        chunk_text = "\n\n".join(u[0] for u in buffer)
        start = buffer[0][1]
        end = buffer[-1][2]
        cid = doc_id + "#" + str(counter[0])
        counter[0] += 1
        chunks.append(
            {
                "id": cid,
                "doc_id": doc_id,
                "source": source,
                "text": chunk_text,
                "start": start,
                "end": end,
                "tokens": estimate_tokens(chunk_text),
            }
        )

    for unit in units:
        unit_tokens = estimate_tokens(unit[0])
        if buffer and buffer_tokens + unit_tokens > max_tokens:
            flush()
            if overlap > 0:
                tail: List[tuple] = []
                tail_tokens = 0
                for prev in reversed(buffer):
                    t = estimate_tokens(prev[0])
                    if tail_tokens + t > overlap:
                        break
                    tail.insert(0, prev)
                    tail_tokens += t
                buffer = tail
                buffer_tokens = tail_tokens
            else:
                buffer = []
                buffer_tokens = 0
        buffer.append(unit)
        buffer_tokens += unit_tokens
    flush()
    return chunks
