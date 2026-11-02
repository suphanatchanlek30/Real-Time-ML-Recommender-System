"""Cleaning steps for the raw events table. Each step is a pure function."""

import polars as pl

from src.etl.config import BotFilter

RAW_SCHEMA = {
    "timestamp": pl.Int64,
    "visitorid": pl.Int64,
    "event": pl.String,
    "itemid": pl.Int64,
    "transactionid": pl.Int64,
}


def read_raw(path) -> pl.DataFrame:
    """Read events.csv with an explicit schema so malformed values fail loudly."""
    return pl.read_csv(path, schema=RAW_SCHEMA)


def drop_duplicates(df: pl.DataFrame) -> pl.DataFrame:
    return df.unique(maintain_order=True)


def parse_timestamp(df: pl.DataFrame) -> pl.DataFrame:
    """Convert epoch milliseconds (UTC) to datetime and put columns in their final order."""
    return df.select(
        pl.from_epoch("timestamp", time_unit="ms").cast(pl.Datetime("ms")).alias("ts"),
        "visitorid",
        "itemid",
        "event",
        "transactionid",
    ).sort("ts", "visitorid", "itemid")


def find_bot_users(df: pl.DataFrame, rule: BotFilter) -> tuple[pl.DataFrame, dict]:
    """Return the users to drop and the thresholds used."""
    per_user = df.group_by("visitorid").agg(
        pl.len().alias("n"),
        (pl.col("event") != "view").sum().alias("n_actions"),
    )
    hard_cap = per_user["n"].quantile(rule.hard_cap_quantile)
    view_only_min = per_user["n"].quantile(rule.view_only_quantile)
    bots = per_user.filter(
        (pl.col("n") > hard_cap) | ((pl.col("n") > view_only_min) & (pl.col("n_actions") == 0))
    )
    thresholds = {"hard_cap": hard_cap, "view_only_min": view_only_min}
    return bots.select("visitorid", "n"), thresholds


def drop_users(df: pl.DataFrame, users: pl.DataFrame) -> pl.DataFrame:
    return df.join(users.select("visitorid"), on="visitorid", how="anti")
