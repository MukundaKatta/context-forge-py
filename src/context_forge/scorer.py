"""BM25 relevance scorer (k1=1.5, b=0.75 -- canonical Robertson defaults).

Tokenization: lowercase, ``\\w+`` -- keeps it dependency-free and consistent
with the diversity module so similarity and relevance share a vocabulary.
"""

from __future__ import annotations

import math
import re
from typing import Iterable, List, Mapping

K1 = 1.5
B = 0.75

_WORD = re.compile(r"\w+")


def tokenize(text) -> List[str]:
    """Lowercase + extract ``\\w+`` words. ``None`` -> ``[]``."""
    s = "" if text is None else str(text)
    return _WORD.findall(s.lower())


def _term_frequencies(tokens: Iterable[str]) -> dict:
    tf: dict = {}
    for t in tokens:
        tf[t] = tf.get(t, 0) + 1
    return tf


def score_chunks(query, chunks: List[Mapping]) -> List[dict]:
    """Score ``chunks`` against ``query`` with BM25.

    Each input chunk should be a mapping with a ``text`` key (other keys are
    preserved). Returns a new list of dict copies, each with an extra
    ``score`` key, sorted by score descending (stable for ties).
    """
    query_tokens = list(dict.fromkeys(tokenize(query)))
    docs = []
    for chunk in chunks:
        text = chunk.get("text", "") if isinstance(chunk, Mapping) else ""
        docs.append({"chunk": chunk, "tokens": tokenize(text)})
    if not docs:
        return []

    doc_lengths = [len(d["tokens"]) for d in docs]
    avg_doc_length = (sum(doc_lengths) / len(docs)) if docs else 1
    if avg_doc_length == 0:
        avg_doc_length = 1

    df: dict = {}
    for term in query_tokens:
        cnt = 0
        for d in docs:
            if term in d["tokens"]:
                cnt += 1
        df[term] = cnt

    n = len(docs)
    idf: dict = {}
    for term in query_tokens:
        n_t = df.get(term, 0)
        # Robertson IDF with +1 smoothing to avoid negative IDF.
        idf[term] = math.log(1 + (n - n_t + 0.5) / (n_t + 0.5))

    scored: List[tuple] = []
    for index, d in enumerate(docs):
        tokens = d["tokens"]
        tf = _term_frequencies(tokens)
        doc_len = doc_lengths[index]
        score = 0.0
        for term in query_tokens:
            term_freq = tf.get(term, 0)
            if not term_freq:
                continue
            numerator = term_freq * (K1 + 1)
            denominator = term_freq + K1 * (1 - B + B * (doc_len / avg_doc_length))
            score += idf.get(term, 0.0) * (numerator / denominator)
        out = dict(d["chunk"]) if isinstance(d["chunk"], Mapping) else {"text": ""}
        out["score"] = score
        scored.append((score, index, out))

    scored.sort(key=lambda t: (-t[0], t[1]))
    return [t[2] for t in scored]
