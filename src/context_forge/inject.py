"""Prompt-injection and exfiltration risk scanner.

Each rule documents its own severity. We scan the raw text once per rule and
surface findings with index + snippet so callers can highlight the offending
region. Patterns intentionally favor high-precision matches; expand carefully
to avoid false positives on legitimate documentation.
"""

from __future__ import annotations

import re
from typing import List

# Zero-width chars: zero-width space, zero-width non-joiner, zero-width joiner, BOM.
ZERO_WIDTH = re.compile("[​‌‍﻿]")

_RULES = [
    (
        "ignore_instructions",
        "high",
        re.compile(
            r"ignore\s+(all|previous|prior|above)\s+instructions", re.IGNORECASE
        ),
    ),
    ("system_prefix", "high", re.compile(r"^system:\s", re.IGNORECASE | re.MULTILINE)),
    ("you_are_now", "med", re.compile(r"you\s+are\s+now\b", re.IGNORECASE)),
    (
        "role_tag",
        "high",
        re.compile(
            r"<\|system\|>|<\|assistant\|>|<\|user\|>|\[INST\]|###\s*system\b",
            re.IGNORECASE,
        ),
    ),
    ("exfil_curl", "high", re.compile(r"\bcurl\s+[^\s|]*https?://", re.IGNORECASE)),
    ("exfil_wget", "high", re.compile(r"\bwget\s+[^\s|]*https?://", re.IGNORECASE)),
    ("exfil_base64", "med", re.compile(r"\bbase64\s+-d\b", re.IGNORECASE)),
]

_URL = re.compile(r"https?://([^\s/?#]+)", re.IGNORECASE)


def _snippet_around(text: str, index: int, length: int, padding: int = 24) -> str:
    start = max(0, index - padding)
    end = min(len(text), index + length + padding)
    return re.sub(r"\s+", " ", text[start:end]).strip()


def _detect_suspicious_urls(text: str, findings: list) -> None:
    # Match http(s) URLs and flag long randomized subdomains (e.g. 16+ alphanumeric with at least one digit).
    for m in _URL.finditer(text):
        host = m.group(1)
        labels = host.split(".")
        if len(labels) < 3:
            continue
        sub = labels[0]
        long_random = (
            len(sub) >= 16
            and re.fullmatch(r"[a-z0-9]+", sub, re.IGNORECASE) is not None
            and re.search(r"\d", sub) is not None
        )
        if long_random:
            findings.append(
                {
                    "kind": "suspicious_url",
                    "severity": "med",
                    "snippet": _snippet_around(text, m.start(), m.end() - m.start()),
                    "index": m.start(),
                }
            )


def _detect_zero_width(text: str, findings: list) -> None:
    for m in ZERO_WIDTH.finditer(text):
        findings.append(
            {
                "kind": "zero_width_char",
                "severity": "low",
                "snippet": _snippet_around(text, m.start(), 1),
                "index": m.start(),
            }
        )


def scan_injection(text) -> List[dict]:
    """Scan ``text`` for prompt-injection / exfiltration patterns.

    Returns a list of finding dicts, sorted by ``index`` (document order).
    Each finding has ``kind``, ``severity``, ``snippet``, ``index``.
    """
    s = "" if text is None else str(text)
    if not s:
        return []
    findings: List[dict] = []
    for kind, severity, pattern in _RULES:
        for m in pattern.finditer(s):
            findings.append(
                {
                    "kind": kind,
                    "severity": severity,
                    "snippet": _snippet_around(s, m.start(), m.end() - m.start()),
                    "index": m.start(),
                }
            )
    _detect_zero_width(s, findings)
    _detect_suspicious_urls(s, findings)
    findings.sort(key=lambda f: f["index"])
    return findings
