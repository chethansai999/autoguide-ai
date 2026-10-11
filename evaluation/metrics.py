"""Evaluation metrics (proposal section 7). Pure functions, unit-tested.

Relevance is judged at DOCUMENT level (source URL) by default: chunk IDs
differ between fixed and structure-aware chunking, so URL relevance is the
only label that compares B1/B2/B3 fairly with one golden set.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence


def dedupe(items: Sequence[str]) -> list[str]:
    return list(dict.fromkeys(i for i in items if i))


def recall_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int) -> float | None:
    rel = set(relevant)
    if not rel:
        return None
    return len(rel & set(dedupe(ranked)[:k])) / len(rel)


def mrr(ranked: Sequence[str], relevant: Sequence[str]) -> float | None:
    rel = set(relevant)
    if not rel:
        return None
    for i, item in enumerate(dedupe(ranked), start=1):
        if item in rel:
            return 1.0 / i
    return 0.0


def ndcg_at_k(ranked: Sequence[str], relevant: Sequence[str], k: int = 10) -> float | None:
    rel = set(relevant)
    if not rel:
        return None
    ranked = dedupe(ranked)[:k]
    dcg = sum(1.0 / math.log2(i + 2) for i, item in enumerate(ranked) if item in rel)
    ideal = sum(1.0 / math.log2(i + 2) for i in range(min(k, len(rel))))
    return dcg / ideal if ideal else None


def percentile(values: Sequence[float], q: float) -> float | None:
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    idx = (len(values) - 1) * q
    lo, hi = math.floor(idx), math.ceil(idx)
    return values[lo] + (values[hi] - values[lo]) * (idx - lo)


def mean(values: Sequence[float | None]) -> float | None:
    values = [v for v in values if v is not None]
    return sum(values) / len(values) if values else None


def bootstrap_ci(values: Sequence[float], n: int = 2000, alpha: float = 0.05, seed: int = 13) -> tuple[float, float] | None:
    values = [v for v in values if v is not None]
    if len(values) < 2:
        return None
    rng = random.Random(seed)
    means = sorted(sum(rng.choices(values, k=len(values))) / len(values) for _ in range(n))
    return means[int(alpha / 2 * n)], means[int((1 - alpha / 2) * n) - 1]


def mcnemar_exact(b: int, c: int) -> float:
    """Two-sided exact McNemar p-value. b = A right/B wrong, c = A wrong/B right."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def refusal_rates(rows: Sequence[dict]) -> dict:
    """rows: {"answerable": bool, "refused": bool}"""
    unanswerable = [r for r in rows if not r["answerable"]]
    answerable = [r for r in rows if r["answerable"]]
    return {
        "correct_refusal_rate": (sum(r["refused"] for r in unanswerable) / len(unanswerable)) if unanswerable else None,
        "over_refusal_rate": (sum(r["refused"] for r in answerable) / len(answerable)) if answerable else None,
    }
