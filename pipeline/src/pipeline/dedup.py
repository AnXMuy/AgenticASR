# -*- coding: utf-8 -*-
"""Character n-gram deduplication and dataset-distribution statistics."""
from collections import Counter, defaultdict

from tqdm import tqdm


def char_ngrams(text: str, n: int = 3) -> set[str]:
    """Extract a set of character n-grams."""
    text = text.strip()
    if len(text) < n:
        return {text}
    return {text[i:i + n] for i in range(len(text) - n + 1)}


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute set Jaccard similarity."""
    if not set_a or not set_b:
        return 0.0
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def is_duplicate(text: str, existing_ngrams: list[set[str]], threshold: float = 0.75) -> bool:
    """Return whether text matches any retained n-gram set."""
    new_ngrams = char_ngrams(text)
    for existing in existing_ngrams:
        if jaccard_similarity(new_ngrams, existing) >= threshold:
            return True
    return False


def deduplicate(items: list[dict], text_key: str = "input", threshold: float = 0.75) -> list[dict]:
    """Retain the first record from each near-duplicate group."""
    seen_ngrams: list[set[str]] = []
    postings: dict[str, set[int]] = defaultdict(set)
    unique_items = []

    for item in tqdm(items, desc="Dedup", unit="item"):
        text = item.get(text_key, "")
        if not text:
            unique_items.append(item)
            continue

        ngrams = char_ngrams(text)
        candidate_ids: set[int] = set()
        for ngram in ngrams:
            candidate_ids.update(postings.get(ngram, ()))
        is_dup = any(
            jaccard_similarity(ngrams, seen_ngrams[index]) >= threshold
            for index in candidate_ids
        )

        if not is_dup:
            retained_index = len(seen_ngrams)
            seen_ngrams.append(ngrams)
            for ngram in ngrams:
                postings[ngram].add(retained_index)
            unique_items.append(item)

    return unique_items


def compute_distribution(items: list[dict]) -> dict:
    """Compute scene, language, and fragment distributions."""
    stats = {
        "total": len(items),
        "scene": Counter(),
        "language": Counter(),
        "is_fragment": Counter(),
    }

    for item in items:
        meta = item.get("meta", {})
        stats["scene"][meta.get("scene", "unknown")] += 1
        stats["language"][meta.get("language", "unknown")] += 1
        stats["is_fragment"][str(meta.get("is_fragment", False))] += 1

    return {k: dict(v) if isinstance(v, Counter) else v for k, v in stats.items()}
