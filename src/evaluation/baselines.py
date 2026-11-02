"""Non-personalised baselines. Any model added later has to beat these."""

from datetime import timedelta

import numpy as np
import polars as pl

from src.config import EVENT_WEIGHTS


def fit_popularity(
    history: pl.DataFrame,
    n: int = 200,
    window_days: int | None = None,
    weights: dict[str, int] = EVENT_WEIGHTS,
) -> list[int]:
    """Top-n items by weighted event count, most popular first.

    window_days=None counts the whole history; window_days=7 counts only the last 7 days
    before the end of `history`. If the window has fewer than n items the list is padded
    with all-time popular items, so the list always has n entries.
    """

    def rank(df: pl.DataFrame) -> list[int]:
        score = pl.col("event").replace_strict(weights, return_dtype=pl.Int64).sum().alias("score")
        # sort by itemid as a tie-breaker so the result is deterministic
        ranked = (
            df.group_by("itemid").agg(score).sort(["score", "itemid"], descending=[True, False])
        )
        return ranked["itemid"].to_list()

    all_time = rank(history)
    if window_days is None:
        return all_time[:n]

    cutoff = history["ts"].max() - timedelta(days=window_days)
    recent = rank(history.filter(pl.col("ts") > cutoff))
    seen = set(recent)
    padded = recent + [i for i in all_time if i not in seen]
    return padded[:n]


def random_recs(users: list[int], catalog: list[int], n: int = 200, seed: int = 0):
    """Sanity floor: n random items per user. A real model must be far above this."""
    rng = np.random.default_rng(seed)
    items = np.array(catalog)
    return {u: rng.choice(items, size=n, replace=False).tolist() for u in users}
