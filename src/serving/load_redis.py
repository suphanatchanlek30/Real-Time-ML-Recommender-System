"""Load the batch user state (snapshot) into Redis.

    python -m src.serving.load_redis        (or: make load-redis)

Wipes the Redis database first, then writes the state as of the snapshot time. The stream is
replayed from that same time, so a re-load always gives a clean starting point.
"""

import argparse
import json
from datetime import datetime

import polars as pl
import redis

from src.etl.config import PROCESSED_DIR
from src.serving.config import Settings
from src.serving.feature_store import load_batch
from src.serving.timeutil import to_ms


def batch_tables(events: pl.DataFrame, items: pl.DataFrame, as_of: datetime):
    """Per user x item and per user x category counts of the events before `as_of`.

    `items` is the item feature table of the snapshot (it gives each item's category).
    """
    h = events.filter(pl.col("ts") < as_of)
    touched = h.group_by("visitorid", "itemid").agg(
        (pl.col("event") == "view").sum().alias("n_view"),
        (pl.col("event") == "addtocart").sum().alias("n_cart"),
        (pl.col("event") == "transaction").sum().alias("n_buy"),
        pl.col("ts").max().dt.epoch("ms").alias("last_ms"),
    )
    categories = (
        h.join(items.select("itemid", pl.col("i_categoryid").alias("category")), on="itemid")
        .group_by("visitorid", "category")
        .agg(pl.len().alias("n"))
    )
    return touched, categories


def main() -> dict:
    cfg = Settings()
    base = cfg.features_dir / cfg.snapshot
    as_of = datetime.fromisoformat(
        json.loads((cfg.features_dir / "features_report.json").read_text())[cfg.snapshot]["as_of"]
    )
    events = pl.read_parquet(PROCESSED_DIR / "events.parquet")
    items = pl.read_parquet(base / "item_features.parquet")
    state = pl.read_parquet(base / "user_state.parquet")
    touched, categories = batch_tables(events, items, as_of)

    r = redis.Redis.from_url(cfg.redis_url, decode_responses=True)
    report = load_batch(r, state, touched, categories, to_ms(as_of))
    report["redis_memory"] = r.info("memory")["used_memory_human"]
    report["keys"] = r.dbsize()
    report["clock"] = str(as_of)
    return report


if __name__ == "__main__":
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(main(), indent=2))
