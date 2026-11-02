"""Ranking metrics for top-K recommendation with binary relevance.

For one user, `recs` is the ranked list of recommended item ids (best first) and `truth` is
the set of items the user really interacted with in the evaluation period.

    Recall@K    hits in top K / |truth|
    HitRate@K   1 if at least one hit in top K, else 0
    NDCG@K      DCG@K / ideal DCG@K, where DCG = sum(rel_i / log2(i + 1)), i = 1-based rank
    MAP@K       AP@K = sum(precision@i * rel_i) / min(|truth|, K), averaged over users

Every function returns a value in [0, 1]. Scores are averaged over users (macro average).
"""

import math
from collections.abc import Mapping, Sequence

import numpy as np


def _top_k(recs: Sequence[int], k: int) -> list[int]:
    """First k unique items, keeping rank order (a repeated item must not count twice)."""
    return list(dict.fromkeys(recs))[:k]


def _relevance(recs: Sequence[int], truth: set[int], k: int) -> list[int]:
    return [1 if item in truth else 0 for item in _top_k(recs, k)]


def recall_at_k(recs: Sequence[int], truth: set[int], k: int) -> float:
    return sum(_relevance(recs, truth, k)) / len(truth)


def hit_rate_at_k(recs: Sequence[int], truth: set[int], k: int) -> float:
    return float(sum(_relevance(recs, truth, k)) > 0)


def ndcg_at_k(recs: Sequence[int], truth: set[int], k: int) -> float:
    rel = _relevance(recs, truth, k)
    dcg = sum(r / math.log2(rank + 2) for rank, r in enumerate(rel))  # rank is 0-based here
    ideal = sum(1 / math.log2(rank + 2) for rank in range(min(len(truth), k)))
    return dcg / ideal


def average_precision_at_k(recs: Sequence[int], truth: set[int], k: int) -> float:
    hits, score = 0, 0.0
    for rank, r in enumerate(_relevance(recs, truth, k), start=1):
        if r:
            hits += 1
            score += hits / rank  # precision at this rank
    return score / min(len(truth), k)


METRICS = {
    "recall": recall_at_k,
    "ndcg": ndcg_at_k,
    "map": average_precision_at_k,
    "hit": hit_rate_at_k,
}


def per_user_metrics(
    recs: Mapping[int, Sequence[int]],
    truth: Mapping[int, set[int]],
    users: Sequence[int],
    ks: Sequence[int] = (10, 200),
) -> dict[str, np.ndarray]:
    """Metric value of every user in `users` (same order), keyed like 'ndcg@10'."""
    out = {}
    for k in ks:
        for name, fn in METRICS.items():
            out[f"{name}@{k}"] = np.array([fn(recs[u], truth[u], k) for u in users])
    return out


def coverage_at_k(recs: Mapping[int, Sequence[int]], users: Sequence[int], k: int, n_items: int):
    """Share of the catalog that appears in at least one user's top K."""
    seen = {item for u in users for item in _top_k(recs[u], k)}
    return len(seen) / n_items


def summarize(values: np.ndarray, n_boot: int = 1000, seed: int = 0) -> dict[str, float]:
    """Mean with a 95% bootstrap confidence interval (resample users with replacement).

    The evaluation sets only have ~2K users, so a single number hides a lot of noise.
    """
    if len(values) == 0:
        return {"mean": float("nan"), "lo": float("nan"), "hi": float("nan")}
    rng = np.random.default_rng(seed)
    boot_means = values[rng.integers(0, len(values), size=(n_boot, len(values)))].mean(axis=1)
    lo, hi = np.percentile(boot_means, [2.5, 97.5])
    return {"mean": float(values.mean()), "lo": float(lo), "hi": float(hi)}
