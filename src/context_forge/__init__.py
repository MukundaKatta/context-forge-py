"""context_forge -- context engineering toolkit for RAG and agent prompts.

Public surface (mirrors the JS sibling):

* ``forge(chunks, query, budget=...)`` -- compose ranking + diversify + scan + pack.
* ``pack_context(query, documents, ...)`` -- full pipeline starting from raw docs.
* ``chunk_document(doc, ...)`` -- paragraph + sentence splitter with overlap.
* ``score_chunks(query, chunks)`` -- BM25 relevance scorer.
* ``diversify(scored, lambda_=...)`` -- Maximal Marginal Relevance re-rank.
* ``pack_to_budget(chunks, budget_tokens=..., per_chunk_min=...)`` -- greedy packer.
* ``scan_injection(text)`` -- prompt-injection / exfiltration risk scan.
* ``estimate_tokens(text)`` -- ceil(len(text) / 4) heuristic.
* ``render_context_block(blocks)`` -- format kept blocks as ``<context>`` XML.
* ``ForgedContext`` -- dataclass returned by ``forge`` and ``pack_context``.
"""

from .chunker import chunk_document
from .diversity import diversify
from .estimate import estimate_tokens
from .forge import ForgedContext, forge, pack_context, render_context_block
from .inject import scan_injection
from .packer import pack_to_budget
from .scorer import score_chunks, tokenize

# Backwards-friendly aliases that mirror the JS sibling exports.
risk_scan = scan_injection


def rank_chunks(query, chunks):
    """Alias for ``score_chunks(query, chunks)`` to mirror the JS sibling."""
    return score_chunks(query, chunks)


__version__ = "0.1.0"
VERSION = __version__

__all__ = [
    "VERSION",
    "ForgedContext",
    "chunk_document",
    "diversify",
    "estimate_tokens",
    "forge",
    "pack_context",
    "pack_to_budget",
    "rank_chunks",
    "render_context_block",
    "risk_scan",
    "scan_injection",
    "score_chunks",
    "tokenize",
]
