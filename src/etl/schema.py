"""Pandera schemas validated every time Parquet is read or written."""

import pandera.polars as pa
import polars as pl

from src.etl.config import EVENT_TYPES

# transactionid must be present exactly when the event is a transaction
_transaction_id_matches_event = pa.Check(
    lambda data: data.lazyframe.select(
        ((pl.col("event") == "transaction") == pl.col("transactionid").is_not_null()).all()
    ),
    error="transactionid must be set if and only if event == 'transaction'",
)

RawEvents = pa.DataFrameSchema(
    {
        "timestamp": pa.Column(pl.Int64, pa.Check.gt(0)),
        "visitorid": pa.Column(pl.Int64),
        "event": pa.Column(pl.String, pa.Check.isin(EVENT_TYPES)),
        "itemid": pa.Column(pl.Int64),
        "transactionid": pa.Column(pl.Int64, nullable=True),
    },
    checks=_transaction_id_matches_event,
    strict=True,
)

CleanEvents = pa.DataFrameSchema(
    {
        "ts": pa.Column(pl.Datetime("ms")),
        "visitorid": pa.Column(pl.Int64),
        "itemid": pa.Column(pl.Int64),
        "event": pa.Column(pl.String, pa.Check.isin(EVENT_TYPES)),
        "transactionid": pa.Column(pl.Int64, nullable=True),
    },
    checks=_transaction_id_matches_event,
    strict=True,
    unique=["ts", "visitorid", "itemid", "event", "transactionid"],
    report_duplicates="all",
)
