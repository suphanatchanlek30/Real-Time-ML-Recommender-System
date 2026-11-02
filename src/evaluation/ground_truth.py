"""Build the evaluation targets for a period."""

import polars as pl


def build_ground_truth(period: pl.DataFrame) -> dict[int, set[int]]:
    """user -> items they added to cart or bought during `period`.

    Plain views are not ground truth (they are weak intent and 97% of all events).
    Users without any addtocart/transaction in the period are not evaluated.
    """
    actions = period.filter(pl.col("event") != "view")
    grouped = actions.group_by("visitorid").agg(pl.col("itemid").unique())
    return {u: set(items) for u, items in zip(grouped["visitorid"], grouped["itemid"])}


def split_warm_cold(users: list[int], history: pl.DataFrame) -> tuple[list[int], list[int]]:
    """Warm = seen in `history` (before the period), cold = never seen."""
    known = set(history["visitorid"].unique().to_list())
    warm = [u for u in users if u in known]
    cold = [u for u in users if u not in known]
    return warm, cold
