"""Extract the two item properties we use from the two 850 MB property files, once.

Only `categoryid` and `available` are kept (EDA: the other properties are anonymous codes).
Each row is a *change event* with a timestamp, so features can look up the value as it was
at a given time instead of using a value that was set later (which would leak the future).
"""

from pathlib import Path

import pandera.polars as pa
import polars as pl

KEEP = ["categoryid", "available"]
PARTS = ["item_properties_part1.csv", "item_properties_part2.csv"]

ItemProperties = pa.DataFrameSchema(
    {
        "ts": pa.Column(pl.Datetime("ms")),
        "itemid": pa.Column(pl.Int64),
        "property": pa.Column(pl.String, pa.Check.isin(KEEP)),
        "value": pa.Column(pl.Int64, pa.Check.ge(0)),
    },
    # an item has one value per property at a given moment
    unique=["itemid", "ts", "property"],
    strict=True,
)


def extract(raw_dir: Path) -> pl.DataFrame:
    """Lazy scan of both files, keeping only the wanted properties (small in memory)."""
    parts = [
        pl.scan_csv(raw_dir / name, infer_schema=False).filter(pl.col("property").is_in(KEEP))
        for name in PARTS
    ]
    df = pl.concat(parts).collect()
    return (
        df.select(
            pl.from_epoch(pl.col("timestamp").cast(pl.Int64), time_unit="ms")
            .cast(pl.Datetime("ms"))
            .alias("ts"),
            pl.col("itemid").cast(pl.Int64),
            "property",
            pl.col("value").cast(pl.Int64),  # fails loudly if a value is not numeric
        )
        .unique(["itemid", "ts", "property"], keep="last", maintain_order=True)
        .sort("itemid", "ts", "property")
    )


def run(raw_dir: Path, out_dir: Path) -> dict:
    props = extract(raw_dir)
    ItemProperties.validate(props)
    path = out_dir / "item_properties.parquet"
    props.write_parquet(path, compression="zstd")
    ItemProperties.validate(pl.read_parquet(path))
    return {
        "item_properties_rows": props.height,
        "item_properties_items": props["itemid"].n_unique(),
    }
