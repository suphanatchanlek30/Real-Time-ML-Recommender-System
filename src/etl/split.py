"""Time-based Train / Val / Test split (random splits would leak the future into train)."""

from datetime import datetime, timedelta

import polars as pl

from src.etl.config import SplitConfig


def split_boundaries(df: pl.DataFrame, cfg: SplitConfig) -> tuple[datetime, datetime]:
    """Return (val_start, test_start), counted back from the last event."""
    test_start = df["ts"].max() - timedelta(days=cfg.test_days)
    val_start = test_start - timedelta(days=cfg.val_days)
    return val_start, test_start


def split_by_time(df: pl.DataFrame, cfg: SplitConfig) -> dict[str, pl.DataFrame]:
    val_start, test_start = split_boundaries(df, cfg)
    return {
        "train": df.filter(pl.col("ts") < val_start),
        "val": df.filter((pl.col("ts") >= val_start) & (pl.col("ts") < test_start)),
        "test": df.filter(pl.col("ts") >= test_start),
    }
