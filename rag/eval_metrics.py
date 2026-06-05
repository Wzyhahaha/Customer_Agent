"""RAG evaluation metrics."""

from __future__ import annotations

import math
from typing import Sequence


def recall_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    if not relevant:
        return 1.0
    top_k = retrieved[:k]
    return len(set(top_k) & relevant) / len(relevant)


def precision_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    top_k = retrieved[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant) / len(top_k)


def mrr_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    for i, item in enumerate(retrieved[:k], start=1):
        if item in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(relevant: set[str], retrieved: list[str], k: int) -> float:
    top_k = retrieved[:k]
    gains = [1.0 if item in relevant else 0.0 for item in top_k]
    dcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(gains))
    ideal_gains = sorted([1.0] * min(len(relevant), k) + [0.0] * max(0, k - len(relevant)), reverse=True)
    idcg = sum(gain / math.log2(i + 2) for i, gain in enumerate(ideal_gains))
    return dcg / idcg if idcg > 0 else 0.0


def hit_at_k(relevant: set[str], retrieved: list[str], k: int) -> bool:
    return bool(set(retrieved[:k]) & relevant)


def joint_hit_at_k(
    question_relevant: set[str],
    question_retrieved: list[str],
    domain_relevant: set[str],
    domain_retrieved: list[str],
    k: int,
) -> bool:
    return hit_at_k(question_relevant, question_retrieved, k) and hit_at_k(
        domain_relevant, domain_retrieved, k
    )


def route_accuracy(predictions: Sequence[str], labels: Sequence[str]) -> float:
    if not predictions:
        return 0.0
    return sum(p == l for p, l in zip(predictions, labels)) / len(predictions)
