"""Item and category features (batch only: they change slowly, so serving reads them from a
batch-loaded table instead of updating them per event)."""

from datetime import datetime, timedelta

import polars as pl

from src.config import EVENT_WEIGHTS
from src.features.spec import PRIOR_STRENGTH, WINDOWS_DAYS
from src.features.user import DAY_MS


def item_attributes(
    props: pl.DataFrame, category_tree: pl.DataFrame, as_of: datetime
) -> pl.DataFrame:
    """Category, parent category and availability of each item as they were *at* `as_of`.

    Taking the latest value overall would use changes made after `as_of` (leakage), so only
    property rows with ts < as_of are used and the last one per item wins. Unknown = -1.
    """
    latest = (
        props.filter(pl.col("ts") < as_of)
        .sort("ts")
        .group_by("itemid", "property")
        .agg(pl.col("value").last())
    )

    def one(prop: str, name: str) -> pl.DataFrame:
        return latest.filter(pl.col("property") == prop).select(
            "itemid", pl.col("value").alias(name)
        )

    parents = category_tree.select(
        pl.col("categoryid").cast(pl.Int64).alias("i_categoryid"),
        pl.col("parentid").cast(pl.Int64).alias("i_parent_categoryid"),
    )
    return (
        one("categoryid", "i_categoryid")
        .join(one("available", "i_available"), on="itemid", how="full", coalesce=True)
        .join(parents, on="i_categoryid", how="left")
        .with_columns(pl.col("i_categoryid", "i_parent_categoryid", "i_available").fill_null(-1))
    )


def item_features(events: pl.DataFrame, attributes: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """One row per item seen before `as_of`: popularity windows, trend, smoothed conversion."""
    h = events.filter(pl.col("ts") < as_of).with_columns(
        pl.col("event").replace_strict(EVENT_WEIGHTS, return_dtype=pl.Int64).alias("w")
    )

    def window(days: int) -> pl.Expr:
        recent = pl.col("ts") >= as_of - timedelta(days=days)
        return pl.col("w").filter(recent).sum().alias(f"i_w_{days}d")

    # global addtocart / transaction per view: the prior each item's own rate shrinks towards
    n_view = h.filter(pl.col("event") == "view").height
    prior_cart = h.filter(pl.col("event") == "addtocart").height / n_view
    prior_buy = h.filter(pl.col("event") == "transaction").height / n_view
    a = PRIOR_STRENGTH

    per_item = h.group_by("itemid").agg(
        pl.col("w").sum().alias("i_w_all"),
        *[window(d) for d in WINDOWS_DAYS],
        (pl.col("event") == "view").sum().alias("i_n_view"),
        (pl.col("event") == "addtocart").sum().alias("i_n_cart"),
        (pl.col("event") == "transaction").sum().alias("i_n_buy"),
        pl.col("visitorid").n_unique().alias("i_n_users"),
        pl.col("ts").max().alias("last_ts"),
        pl.col("ts").min().alias("first_ts"),
    )
    return (
        per_item.with_columns(
            # last day vs the average day of the last week, +1 so empty items stay finite
            ((pl.col("i_w_1d") + 1) / (pl.col("i_w_7d") / 7 + 1)).alias("i_trend"),
            ((pl.lit(as_of) - pl.col("last_ts")).dt.total_milliseconds() / DAY_MS).alias(
                "i_days_since_last"
            ),
            ((pl.lit(as_of) - pl.col("first_ts")).dt.total_milliseconds() / DAY_MS).alias(
                "i_age_days"
            ),
            ((pl.col("i_n_cart") + a * prior_cart) / (pl.col("i_n_view") + a)).alias(
                "i_cart_rate_s"
            ),
            ((pl.col("i_n_buy") + a * prior_buy) / (pl.col("i_n_view") + a)).alias("i_buy_rate_s"),
        )
        .join(attributes, on="itemid", how="left")
        .with_columns(pl.col("i_categoryid", "i_parent_categoryid", "i_available").fill_null(-1))
        .drop("last_ts", "first_ts")
    )


def category_features(items: pl.DataFrame) -> pl.DataFrame:
    """Popularity of each category, aggregated from the item features of the same snapshot."""
    return items.group_by(pl.col("i_categoryid").alias("categoryid")).agg(
        pl.col("i_w_all").sum().alias("c_w_all"),
        pl.col("i_w_7d").sum().alias("c_w_7d"),
        pl.len().alias("c_n_items"),
    )
