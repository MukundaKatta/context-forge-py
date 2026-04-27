"""Greedy budget packer.

Walk chunks in the order provided (already sorted by the caller -- typically
by relevance or MMR), accept each one if it fits, and record dropped chunks
with a reason so the caller can show what was cut.
"""

from __future__ import annotations

from typing import List, Mapping

from .estimate import estimate_tokens


def pack_to_budget(
    chunks: List[Mapping],
    *,
    budget_tokens: int = 1200,
    per_chunk_min: int = 20,
) -> dict:
    """Greedy pack ``chunks`` under ``budget_tokens``.

    Each chunk is a mapping with ``text`` (and optionally ``tokens`` to skip
    re-estimation). Chunks below ``per_chunk_min`` are dropped first.

    Returns ``{"kept": [...], "dropped": [...], "used_tokens": int}``. Each
    dropped entry is ``{"id": ..., "reason": "below_min_tokens" | "budget_exceeded"}``.
    """
    kept: List[dict] = []
    dropped: List[dict] = []
    used = 0
    for chunk in chunks:
        cdict = dict(chunk) if isinstance(chunk, Mapping) else {"text": ""}
        tokens = cdict.get("tokens")
        if tokens is None:
            tokens = estimate_tokens(cdict.get("text", ""))
        if tokens < per_chunk_min:
            dropped.append({"id": cdict.get("id"), "reason": "below_min_tokens"})
            continue
        if used + tokens > budget_tokens:
            dropped.append({"id": cdict.get("id"), "reason": "budget_exceeded"})
            continue
        cdict["tokens"] = tokens
        kept.append(cdict)
        used += tokens
    return {"kept": kept, "dropped": dropped, "used_tokens": used}
