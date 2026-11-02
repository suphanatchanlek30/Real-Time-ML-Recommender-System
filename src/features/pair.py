"""User x item features, computed when candidates are scored.

They need the user's past interactions with the exact item, so they cannot be a precomputed
table. They are the strongest signal for warm users, which is why the candidate set must keep
items the user already saw instead of filtering them out.
"""

from datetime import datetime

import polars as pl

from src.features.user import DAY_MS


def pair_features(
    pairs: pl.DataFrame, events: pl.DataFrame, items: pl.DataFrame, as_of: datetime
) -> pl.DataFrame:
    """Features for each (visitorid, itemid) row of `pairs`, in the same order.

    `items` is the item feature table of the same snapshot (gives each item's category).
    """
    # only the users we score matter: keeps the join small
    h = events.filter(pl.col("ts") < as_of).join(
        pairs.select("visitorid").unique(), on="visitorid", how="semi"
    )
    category = items.select("itemid", pl.col("i_categoryid").alias("cat"))

    touched = h.group_by("visitorid", "itemid").agg(
        (pl.col("event") == "view").sum().alias("p_n_view"),
        (pl.col("event") == "addtocart").sum().alias("p_n_cart"),
        (pl.col("event") == "transaction").sum().alias("p_n_buy"),
        pl.col("ts").max().alias("last_ts"),
    )
    # how much of the user's activity is in each category
    share = (
        h.join(category, on="itemid", how="left")
        .group_by("visitorid", "cat")
        .len()
        .with_columns((pl.col("len") / pl.col("len").sum().over("visitorid")).alias("p_cat_share"))
        .select("visitorid", "cat", "p_cat_share")
    )
    return (
        pairs.join(category, on="itemid", how="left", maintain_order="left")
        .join(touched, on=["visitorid", "itemid"], how="left", maintain_order="left")
        .join(share, on=["visitorid", "cat"], how="left", maintain_order="left")
        .select(
            "visitorid",
            "itemid",
            pl.col("p_n_view", "p_n_cart", "p_n_buy").fill_null(0),
            ((pl.lit(as_of) - pl.col("last_ts")).dt.total_milliseconds() / DAY_MS).alias(
                "p_days_since_touch"
            ),
            pl.col("p_cat_share").fill_null(0.0),
        )
    )
