import json

import fakeredis
import polars as pl

from src.serving.feature_store import read_user
from src.serving.timeutil import to_ms
from src.streaming.consumer import process_message
from src.streaming.producer import load_events, to_message
from tests.test_features import T0

CATEGORY = {5: 1, 6: 2}


def payload(**fields):
    base = {"visitor_id": 1, "item_id": 5, "event": "view", "ts": to_ms(T0)}
    return json.dumps({**base, **fields}).encode()


def test_message_format_is_keyed_by_visitor_and_round_trips():
    key, value = to_message(42, 7, "addtocart", 1433221332117)
    assert key == b"42"
    assert json.loads(value) == {
        "visitor_id": 42,
        "item_id": 7,
        "event": "addtocart",
        "ts": 1433221332117,
    }


def test_consumer_applies_valid_events_and_counts_the_bad_ones():
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    assert process_message(r, payload(), CATEGORY) == ("ok", to_ms(T0))
    assert process_message(r, payload(item_id=6, ts=to_ms(T0) + 1000), CATEGORY)[0] == "ok"
    assert process_message(r, payload(ts=to_ms(T0) - 5000), CATEGORY)[0] == "late"
    assert process_message(r, payload(event="click"), CATEGORY) == ("invalid", None)
    assert process_message(r, b"not json", CATEGORY) == ("invalid", None)
    assert process_message(r, payload(visitor_id="abc"), CATEGORY) == ("invalid", None)
    state = read_user(r, 1).state
    assert state["n_view"] == 2 and state["n_sessions"] == 1


def test_consumer_skips_a_redelivered_message():
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    pos = ("user-events", 2, 40)
    assert process_message(r, payload(), CATEGORY, pos)[0] == "ok"
    assert process_message(r, payload(), CATEGORY, pos)[0] == "duplicate"
    assert read_user(r, 1).state["n_view"] == 1


def test_replayed_events_are_time_ordered_and_start_at_the_snapshot(tmp_path, monkeypatch):
    import src.streaming.producer as producer

    events = pl.DataFrame(
        {
            "ts": [T0.replace(day=3), T0.replace(day=1), T0.replace(day=2)],
            "visitorid": [1, 2, 3],
            "itemid": [9, 8, 7],
            "event": ["view"] * 3,
            "transactionid": [None] * 3,
        },
        schema_overrides={"ts": pl.Datetime("ms"), "transactionid": pl.Int64},
    )
    (tmp_path / "events.parquet").parent.mkdir(exist_ok=True)
    events.write_parquet(tmp_path / "events.parquet")
    monkeypatch.setattr(producer, "PROCESSED_DIR", tmp_path)
    out = load_events(T0.replace(day=2))
    assert out["visitorid"].to_list() == [3, 1]  # day 1 is before the start; the rest is sorted
    assert load_events(T0.replace(day=1), limit=1).height == 1
