"""Heuristic token estimator: ``ceil(len(text) / 4)``.

Picked the chars/4 heuristic because it matches OpenAI's rough tokenizer
guidance for English text and produces stable estimates across markdown,
code, and prose without a runtime dependency. The whitespace-aware variant
(words * 1.3) tends to underestimate code and dense punctuation, so we
stick with chars/4 throughout.
"""

from __future__ import annotations

import math


def estimate_tokens(text) -> int:
    """Estimate token count for a string. ``None`` and non-strings coerce to ``""``."""
    s = "" if text is None else str(text)
    if not s:
        return 0
    return math.ceil(len(s) / 4)
