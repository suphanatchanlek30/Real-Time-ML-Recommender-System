from datetime import datetime, timedelta

import pandera.errors
import polars as pl
import pytest

from src.etl import clean
from src.etl.config import BotFilter, SplitConfig
from src.etl.schema import CleanEvents, RawEvents
from src.etl.split import split_by_time

DAY_MS = 86_400_000


def raw(rows):
    return pl.DataFrame(rows, schema=clean.RAW_SCHEMA, orient="row")


def test_drop_duplicates_removes_only_exact_copies():
    df = raw(
        [
            (1000, 1, "view", 10, None),
            (1000, 1, "view", 10, None),
            (1000, 1, "addtocart", 10, None),
        ]
    )
    assert clean.drop_duplicates(df).height == 2


def test_parse_timestamp_converts_ms_to_utc_datetime_and_sorts():
    df = raw([(2000, 1, "view", 10, None), (1000, 1, "view", 11, None)])
    out = clean.parse_timestamp(df)
    assert out.columns == ["ts", "visitorid", "itemid", "event", "transactionid"]
    assert out["ts"].to_list() == [datetime(1970, 1, 1, 0, 0, 1), datetime(1970, 1, 1, 0, 0, 2)]


def test_bot_filter_drops_view_only_heavy_and_over_cap_but_keeps_heavy_buyers():
    rows = [(1, u, "view", 1, None) for u in range(200) for _ in range(2)]  # 200 normal users
    for u in (300, 301, 302):  # heavy, view-only
        rows += [(i, u, "view", i, None) for i in range(30)]
    rows += [(i, 400, "view", i, None) for i in range(29)] + [(99, 400, "addtocart", 1, None)]
    rows += [(i, 500, "view", i, None) for i in range(505)]  # over the hard cap
    df = clean.parse_timestamp(raw(rows)).unique()
    rule = BotFilter(hard_cap_quantile=0.995, view_only_quantile=0.97)
    bots, _ = clean.find_bot_users(df, rule)
    assert set(bots["visitorid"]) == {300, 301, 302, 500}
    assert 400 in set(clean.drop_users(df, bots)["visitorid"])


def test_split_is_chronological_and_disjoint():
    start = datetime(2015, 1, 1)
    df = pl.DataFrame(
        {
            "ts": [start + timedelta(days=d) for d in range(30)],
            "visitorid": list(range(30)),
            "itemid": list(range(30)),
            "event": ["view"] * 30,
            "transactionid": [None] * 30,
        },
        schema_overrides={"ts": pl.Datetime("ms"), "transactionid": pl.Int64},
    )
    parts = split_by_time(df, SplitConfig(test_days=7, val_days=7))
    assert sum(p.height for p in parts.values()) == df.height
    assert parts["train"]["ts"].max() < parts["val"]["ts"].min()
    assert parts["val"]["ts"].max() < parts["test"]["ts"].min()


def test_raw_schema_rejects_unknown_event_type():
    with pytest.raises(pandera.errors.SchemaError):
        RawEvents.validate(raw([(1000, 1, "click", 10, None)]))


def test_raw_schema_requires_transactionid_only_for_transactions():
    with pytest.raises(pandera.errors.SchemaError):
        RawEvents.validate(raw([(1000, 1, "transaction", 10, None)]))
    with pytest.raises(pandera.errors.SchemaError):
        RawEvents.validate(raw([(1000, 1, "view", 10, 5)]))
    RawEvents.validate(raw([(1000, 1, "transaction", 10, 5), (1000, 1, "view", 10, None)]))


def test_clean_schema_rejects_duplicates():
    df = clean.parse_timestamp(raw([(1000, 1, "view", 10, None)] * 2))
    with pytest.raises((pandera.errors.SchemaError, pandera.errors.SchemaErrors)):
        CleanEvents.validate(df)
