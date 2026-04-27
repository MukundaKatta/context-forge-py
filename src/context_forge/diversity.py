"""Maximal Marginal Relevance (MMR) re-ranker.

Pick the chunk that maximizes
    lambda * relevance - (1 - lambda) * max_jaccard_to_already_picked.

lambda close to 1 favors pure relevance; closer to 0 favors diversity.
"""

from __future__ import annotations

from typing import List, Mapping

from .scorer import tokenize


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 0.0
    intersection = len(a & b)
    union = len(a) + len(b) - intersection
    return (intersection / union) if union else 0.0


def diversify(scored_chunks: List[Mapping], lambda_: float = 0.7) -> List[dict]:
    """Re-rank ``scored_chunks`` by MMR.

    ``lambda_`` defaults to 0.7 -- mostly relevance, some diversity.
    Returns a new list of dict copies in the picked order.
    """
    if not scored_chunks:
        return []
    chunks = [dict(c) if isinstance(c, Mapping) else {"text": ""} for c in scored_chunks]
    token_sets = [set(tokenize(c.get("text", ""))) for c in chunks]
    remaining = list(range(len(chunks)))
    selected: List[int] = []

    while remaining:
        best_pos = 0
        best_score = float("-inf")
        for pos, idx in enumerate(remaining):
            relevance = chunks[idx].get("score") or 0
            max_sim = 0.0
            for picked_idx in selected:
                sim = _jaccard(token_sets[idx], token_sets[picked_idx])
                if sim > max_sim:
                    max_sim = sim
            mmr = lambda_ * relevance - (1 - lambda_) * max_sim
            if mmr > best_score:
                best_score = mmr
                best_pos = pos
        chosen = remaining.pop(best_pos)
        selected.append(chosen)

    return [chunks[i] for i in selected]
