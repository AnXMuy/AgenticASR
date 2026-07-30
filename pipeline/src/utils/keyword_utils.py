# -*- coding: utf-8 -*-
"""Normalize and validate keyword-list review targets."""
from __future__ import annotations

import re
from typing import Any


COMMON_KEYWORDS = {
    "\u6211\u4eec", "\u4f60\u4eec", "\u4ed6\u4eec", "\u5979\u4eec",
    "\u5b83\u4eec", "\u8fd9\u4e2a", "\u90a3\u4e2a", "\u4ec0\u4e48",
    "\u600e\u4e48", "\u600e\u6837", "\u54ea\u91cc", "\u90a3\u91cc",
    "\u8fd9\u91cc", "\u7684", "\u4e86", "\u662f", "\u5728", "\u6709",
    "\u548c", "\u4e0e", "\u6216", "\u5c31", "\u90fd", "\u4e5f", "\u8fd8",
    "\u53c8", "\u518d", "\u4e0d", "\u6ca1", "\u53ef\u4ee5", "\u80fd\u591f",
    "\u9700\u8981", "\u5e94\u8be5", "\u5fc5\u987b", "\u53ef\u80fd",
    "\u4e5f\u8bb8", "\u96c6\u5408", "\u4f1a\u8bae", "\u4e1c\u897f",
    "\u4e8b\u60c5", "\u4e00\u4e0b", "\u4eca\u5929", "\u660e\u5929",
    "we", "you", "they", "this", "that", "the", "a", "an", "and", "or", "to", "of", "in",
    "on", "for", "is", "are", "was", "were", "need", "should", "maybe",
}

REVIEW_TARGET_PATTERNS = [
    re.compile(r"[A-Za-z][A-Za-z0-9_+#.\-]*\+\+"),
    re.compile(r"\b[A-Za-z]#\b"),
    re.compile(r"\bv\d+(?:\.\d+)*\b", re.I),
    re.compile(r"\d+(?:\.\d+)?%"),
    re.compile(r"\d+(?:\.\d+)?(?:\u5143|\u5757|\u7f8e\u5143|\u516c\u91cc|\u7c73|\u5206\u949f|\u5c0f\u65f6|\u53f7|\u697c|\u5c42)"),
    re.compile(r"\d{1,2}:\d{2}"),
    re.compile(r"(?:>=|<=|!=|==|->|::|\+=|-=|\*=|/=|%d)"),
]


def dedupe_keywords(keywords: list[Any] | None) -> list[str]:
    """Deduplicate non-empty keywords while preserving order."""
    seen = set()
    result: list[str] = []
    for kw in keywords or []:
        if not isinstance(kw, str):
            continue
        kw = kw.strip()
        if not kw or kw in seen:
            continue
        seen.add(kw)
        result.append(kw)
    return result


def is_common_keyword(keyword: str) -> bool:
    """Return whether a keyword is clearly too common to review."""
    kw = keyword.strip()
    if not kw:
        return True
    if kw in COMMON_KEYWORDS:
        return True
    if len(kw) <= 1 and not re.search(r"[\u4e00-\u9fffA-Za-z0-9+#%]", kw):
        return True
    return False


def _looks_like_review_target(keyword: str) -> bool:
    if any(pattern.search(keyword) for pattern in REVIEW_TARGET_PATTERNS):
        return True
    if re.search(r"[A-Za-z][A-Za-z0-9_+#.\-]{1,}", keyword):
        return True
    if re.search(r"\d", keyword):
        return True
    if len(keyword) >= 2 and re.search(r"[\u4e00-\u9fff]", keyword):
        return True
    return False


def extract_keywords(item: dict) -> list[str]:
    """Extract review targets from current and legacy fields."""
    output = item.get("output") if isinstance(item.get("output"), dict) else {}
    for candidate in (
        output.get("keyword_list"),
        item.get("keyword_list"),
        item.get("review_targets"),
        item.get("domain_keywords"),
        item.get("error_keywords"),
    ):
        if candidate:
            return dedupe_keywords(candidate)
    return []


def normalize_review_targets(
    keywords: list[Any] | None,
    input_text: str = "",
    refined_text: str = "",
    include_inferred: bool = True,
) -> list[str]:
    """Normalize candidate values into traceable review targets."""
    normalized: list[str] = []
    for kw in dedupe_keywords(keywords):
        if is_common_keyword(kw):
            continue
        if keyword_has_evidence(kw, input_text, refined_text, allow_inferred=include_inferred):
            normalized.append(kw)
        elif _looks_like_review_target(kw):
            normalized.append(kw)
    return dedupe_keywords(normalized)


def keyword_has_evidence(
    keyword: str,
    input_text: str = "",
    refined_text: str = "",
    allow_inferred: bool = True,
) -> bool:
    """Return whether a keyword is traceable to input or refined output."""
    kw = keyword.strip()
    if not kw:
        return False
    if kw in refined_text or kw in input_text:
        return True
    if not allow_inferred:
        return False
    if len(kw) == 1 and kw in refined_text:
        return True
    if len(kw) == 1 and re.search(rf"{re.escape(kw)}[\u662f\u53eb\u4e3a]", input_text):
        return True
    if len(kw) > 1:
        chars = [c for c in kw if re.search(r"[\u4e00-\u9fffA-Za-z0-9+#%]", c)]
        if chars and sum(c in refined_text for c in chars) >= max(1, len(chars) - 1):
            return True
        if chars and sum(c in input_text for c in chars) >= max(1, len(chars) - 1):
            return True
    return False
