"""Offline == online: Redis state fed by the stream gives the same features as the batch code."""

from datetime import datetime, timedelta

import fakeredis
import numpy as np
import polars as pl
import pytest
from pydantic import ValidationError

from src.features.pair import pair_features
from src.features.user import derive_user_features, user_state
from src.serving.feature_store import (
    CLOCK,
    Event,
    advance_clock,
    handle_event,
    load_batch,
    read_user,
)
from src.serving.load_redis import batch_tables
from src.serving.recommender import pair_feature_matrix, user_feature_values
from src.serving.timeutil import from_ms, to_ms
from tests.test_features import T0, random_events

SNAPSHOT = T0 + timedelta(days=4)  # state is loaded as of this time
NOW = T0 + timedelta(days=9)  # features are compared at this time
ITEMS = pl.DataFrame({"itemid": list(range(1, 31)), "i_categoryid": [i % 4 for i in range(1, 31)]})
ITEM_CAT = dict(zip(ITEMS["itemid"].to_list(), ITEMS["i_categoryid"].to_list()))


def replay(ev: pl.DataFrame, r) -> None:
    """Load the state at SNAPSHOT, then push every later event through the consumer handler."""
    touched, categories = batch_tables(ev, ITEMS, SNAPSHOT)
    load_batch(r, user_state(ev, SNAPSHOT), touched, categories, to_ms(SNAPSHOT))
    later = ev.filter((pl.col("ts") >= SNAPSHOT) & (pl.col("ts") < NOW)).sort("ts", "itemid")
    for ts, user, item, kind in later.select("ts", "visitorid", "itemid", "event").rows():
        handle_event(r, Event(visitor_id=user, item_id=item, event=kind, ts=to_ms(ts)), ITEM_CAT)
        advance_clock(r, to_ms(ts))


@pytest.fixture(scope="module")
def online():
    ev = random_events(n=900, seed=3)
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    replay(ev, r)
    return ev, r


def test_user_features_online_equal_offline(online):
    ev, r = online
    offline = derive_user_features(user_state(ev, NOW), NOW)
    assert offline.height > 5
    for row in offline.iter_rows(named=True):
        data = read_user(r, row["visitorid"])
        got = user_feature_values(row["visitorid"], data.state, to_ms(NOW))
        for name, value in got.items():
            assert value == pytest.approx(row[name], rel=1e-9, abs=1e-12), (row["visitorid"], name)


def test_pair_features_online_equal_offline(online):
    ev, r = online
    users = ev.filter(pl.col("ts") < NOW)["visitorid"].unique().to_list()[:8]
    items = np.arange(1, 31)
    pairs = pl.DataFrame(
        {"visitorid": [u for u in users for _ in items], "itemid": list(items) * len(users)}
    )
    offline = pair_features(pairs, ev, ITEMS, NOW)
    for u in users:
        data = read_user(r, u)
        got = pair_feature_matrix(items, data, ITEM_CAT, to_ms(NOW))
        want = offline.filter(pl.col("visitorid") == u).sort("itemid")
        cols = ["p_n_view", "p_n_cart", "p_n_buy", "p_days_since_touch", "p_cat_share"]
        np.testing.assert_allclose(got, want.select(cols).to_numpy(), rtol=1e-9, equal_nan=True)


def test_clock_follows_the_newest_event_and_never_goes_back(online):
    ev, r = online
    newest = ev.filter((pl.col("ts") >= SNAPSHOT) & (pl.col("ts") < NOW))["ts"].max()
    assert int(r.get(CLOCK)) == to_ms(newest)
    advance_clock(r, to_ms(newest) - 1000)
    assert int(r.get(CLOCK)) == to_ms(newest)


def test_unknown_user_reads_as_empty():
    data = read_user(fakeredis.FakeStrictRedis(decode_responses=True), 12345)
    assert data.state is None and data.items == {} and data.clock_ms is None


def test_new_user_event_creates_state_and_category_counts():
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    ts = to_ms(T0)
    handle_event(r, Event(visitor_id=7, item_id=5, event="addtocart", ts=ts), ITEM_CAT)
    handle_event(r, Event(visitor_id=7, item_id=6, event="view", ts=ts + 1000), ITEM_CAT)
    d = read_user(r, 7)
    assert (d.state["n_cart"], d.state["n_view"], d.state["n_sessions"]) == (1, 1, 1)
    assert d.items["c:5"] == "1" and d.items["v:6"] == "1" and d.items["t:6"] == str(ts + 1000)
    assert d.cats == {str(ITEM_CAT[5]): "1", str(ITEM_CAT[6]): "1"}
    assert r.lrange("user:7:recent", 0, -1) == ["5", "6"]


def test_late_event_is_skipped():
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    ts = to_ms(T0)
    assert handle_event(r, Event(visitor_id=1, item_id=5, event="view", ts=ts + 5000), ITEM_CAT)
    assert not handle_event(r, Event(visitor_id=1, item_id=6, event="view", ts=ts), ITEM_CAT)
    assert read_user(r, 1).state["n_view"] == 1


def test_item_without_category_still_counts_as_an_event():
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    handle_event(r, Event(visitor_id=1, item_id=999, event="view", ts=to_ms(T0)), ITEM_CAT)
    d = read_user(r, 1)
    assert d.state["n_view"] == 1 and d.cats == {}


def test_event_schema_rejects_bad_messages():
    with pytest.raises(ValidationError):
        Event(visitor_id=1, item_id=2, event="click", ts=1)
    with pytest.raises(ValidationError):
        Event(visitor_id=1, item_id=2, event="view", ts=0)
    with pytest.raises(ValidationError):
        Event.model_validate({"visitor_id": "x", "item_id": 2, "event": "view", "ts": 5})


def test_load_batch_resets_redis_so_a_reload_does_not_double_count():
    ev = random_events(n=300, seed=5)
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    replay(ev, r)
    first = {k: r.hgetall(k) for k in r.scan_iter("user:*:state")}
    replay(ev, r)  # a second `make replay`
    assert {k: r.hgetall(k) for k in r.scan_iter("user:*:state")} == first


def test_timeutil_roundtrip():
    assert from_ms(to_ms(datetime(2015, 9, 4, 2, 59, 47, 788000))) == datetime(
        2015, 9, 4, 2, 59, 47, 788000
    )
