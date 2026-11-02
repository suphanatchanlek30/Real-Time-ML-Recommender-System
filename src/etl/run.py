"""ETL entry point: raw events.csv -> clean Parquet + time-based splits.

    python -m src.etl.run            (or: make etl)

Outputs in <DATA_DIR>/processed: events.parquet, train.parquet, val.parquet,
test.parquet, item_properties.parquet and etl_report.json (counts, thresholds, split dates,
file hash).
"""

import hashlib
import json
import logging
from pathlib import Path

import polars as pl

from src.etl import clean, item_properties
from src.etl.config import PROCESSED_DIR, RAW_DIR, BotFilter, SplitConfig
from src.etl.schema import CleanEvents, RawEvents
from src.etl.split import split_boundaries, split_by_time

log = logging.getLogger("etl")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def write_parquet(df: pl.DataFrame, path: Path) -> None:
    """Validate against the clean schema before writing, and again after reading back."""
    CleanEvents.validate(df)
    df.write_parquet(path, compression="zstd")
    CleanEvents.validate(pl.read_parquet(path))


def run(raw_dir: Path = RAW_DIR, out_dir: Path = PROCESSED_DIR) -> dict:
    bot_rule, split_cfg = BotFilter(), SplitConfig()
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = clean.read_raw(raw_dir / "events.csv")
    RawEvents.validate(raw)
    log.info("raw rows: %d", raw.height)

    deduped = clean.drop_duplicates(raw)
    events = clean.parse_timestamp(deduped)

    bots, thresholds = clean.find_bot_users(events, bot_rule)
    events = clean.drop_users(events, bots)
    log.info(
        "dropped %d duplicate rows, %d bot users (%d events)",
        raw.height - deduped.height,
        bots.height,
        bots["n"].sum(),
    )

    write_parquet(events, out_dir / "events.parquet")
    splits = split_by_time(events, split_cfg)
    for name, part in splits.items():
        write_parquet(part, out_dir / f"{name}.parquet")
        log.info("%s: %d events, %d users", name, part.height, part["visitorid"].n_unique())

    val_start, test_start = split_boundaries(events, split_cfg)
    report = {
        "rows_raw": raw.height,
        "rows_duplicates_removed": raw.height - deduped.height,
        "bot_users_removed": bots.height,
        "bot_events_removed": int(bots["n"].sum()),
        "bot_thresholds": thresholds,
        "rows_clean": events.height,
        "users_clean": events["visitorid"].n_unique(),
        "items_clean": events["itemid"].n_unique(),
        "ts_min": str(events["ts"].min()),
        "ts_max": str(events["ts"].max()),
        "val_start": str(val_start),
        "test_start": str(test_start),
        "splits": {
            k: {"events": v.height, "users": v["visitorid"].n_unique()} for k, v in splits.items()
        },
        "events_parquet_sha256": sha256(out_dir / "events.parquet"),
    }
    report.update(item_properties.run(raw_dir, out_dir))
    (out_dir / "etl_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(json.dumps(run(), indent=2))
