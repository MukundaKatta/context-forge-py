"""High-level pipeline: ``forge`` and ``pack_context``."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping, Optional

from .chunker import chunk_document
from .diversity import diversify
from .inject import scan_injection
from .packer import pack_to_budget
from .scorer import score_chunks


@dataclass
class ForgedContext:
    """Structured result from :func:`forge` / :func:`pack_context`.

    Attributes:
        blocks: Kept context chunks (ranked, diversified, packed).
        used_tokens: Total tokens consumed by ``blocks``.
        dropped: List of ``{"id", "reason"}`` for chunks the packer cut.
        risks: Injection findings from scanning kept blocks.
        citations: ``{block_id: {"source", "span": [start, end]}}``.
    """

    blocks: List[dict]
    used_tokens: int
    dropped: List[dict]
    risks: List[dict]
    citations: dict


def _run(
    chunks: List[Mapping],
    query: str,
    *,
    budget: int,
    lambda_: float,
    per_chunk_min: int,
) -> ForgedContext:
    scored = score_chunks(query, chunks)
    diversified = diversify(scored, lambda_=lambda_)
    pack = pack_to_budget(
        diversified, budget_tokens=budget, per_chunk_min=per_chunk_min
    )

    risks: List[dict] = []
    blocks: List[dict] = []
    for chunk in pack["kept"]:
        text = chunk.get("text", "")
        for finding in scan_injection(text):
            entry = {"id": chunk.get("id")}
            entry.update(finding)
            risks.append(entry)
        blocks.append(
            {
                "id": chunk.get("id"),
                "text": text,
                "source": chunk.get("source"),
                "score": chunk.get("score") or 0,
                "tokens": chunk.get("tokens"),
            }
        )

    citations: dict = {}
    for chunk in pack["kept"]:
        cid = chunk.get("id")
        start = chunk.get("start") or 0
        text = chunk.get("text", "")
        end = chunk.get("end")
        if end is None:
            end = start + len(text)
        citations[cid] = {"source": chunk.get("source"), "span": [start, end]}

    return ForgedContext(
        blocks=blocks,
        used_tokens=pack["used_tokens"],
        dropped=pack["dropped"],
        risks=risks,
        citations=citations,
    )


def forge(
    chunks: List[Mapping],
    query: str,
    budget: int = 1200,
    *,
    lambda_: float = 0.7,
    per_chunk_min: int = 20,
) -> ForgedContext:
    """Compose ranking + diversification + scan + budget-packing on ``chunks``.

    Use this when you already have chunked context (e.g. from a vector
    store). For raw documents, see :func:`pack_context`.
    """
    if not isinstance(chunks, list):
        raise TypeError("forge: chunks must be a list of mappings")
    if isinstance(budget, bool) or not isinstance(budget, (int, float)) or budget < 0:
        raise TypeError("forge: budget must be a non-negative number")
    return _run(
        chunks,
        query if isinstance(query, str) else "",
        budget=int(budget),
        lambda_=float(lambda_),
        per_chunk_min=int(per_chunk_min),
    )


def pack_context(
    *,
    query: str = "",
    documents: Optional[Iterable[Mapping]] = None,
    budget_tokens: int = 1200,
    max_tokens: int = 200,
    overlap_tokens: int = 20,
    lambda_: float = 0.7,
    per_chunk_min: int = 20,
) -> ForgedContext:
    """Full pipeline: chunk -> score -> diversify -> risk-scan -> pack.

    ``documents`` is an iterable of ``{"id", "text", "source"?}`` mappings.
    """
    docs = list(documents or [])
    chunks: List[dict] = []
    for doc in docs:
        chunks.extend(
            chunk_document(doc, max_tokens=max_tokens, overlap_tokens=overlap_tokens)
        )
    return _run(
        chunks,
        query if isinstance(query, str) else "",
        budget=int(budget_tokens),
        lambda_=float(lambda_),
        per_chunk_min=int(per_chunk_min),
    )


def render_context_block(blocks: Iterable[Mapping]) -> str:
    """Format kept blocks as XML-like ``<context>`` tags for prompting."""
    out = []
    for index, block in enumerate(blocks):
        bid = (
            block.get("id")
            if isinstance(block, Mapping) and block.get("id")
            else "block-" + str(index)
        )
        if isinstance(block, Mapping) and block.get("source"):
            source = block.get("source")
        elif isinstance(block, Mapping) and block.get("sourceId"):
            source = block.get("sourceId")
        else:
            source = "unknown"
        text = block.get("text", "") if isinstance(block, Mapping) else ""
        out.append(
            '<context index="'
            + str(index + 1)
            + '" id="'
            + str(bid)
            + '" source="'
            + str(source)
            + '">\n'
            + str(text)
            + "\n</context>"
        )
    return "\n\n".join(out)
