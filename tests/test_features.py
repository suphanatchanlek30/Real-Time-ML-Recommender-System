import json
import math
import random
from datetime import datetime, timedelta

import polars as pl
import pytest
from polars.testing import assert_frame_equal

from src.features.build import build_snapshot
from src.features.item import category_features, item_attributes, item_features
from src.features.pair import pair_features
from src.features.schema import (
    BY_GROUP,
    DECAY_TAU_DAYS,
    FEATURE_COLUMNS,
    RECENT_N,
    cast_to_schema,
    schema_dict,
    table_schema,
)
from src.features.snapshots import make_snapshots
from src.features.user import derive_user_features, update_user_state, user_state

T0 = datetime(2015, 6, 1)
AS_OF = datetime(2015, 6, 10)


def events(rows):
    """rows: (days after T0, user, item, event)"""
    return pl.DataFrame(
        [
            (T0 + timedelta(days=d), u, i, e, 5 if e == "transaction" else None)
            for d, u, i, e in rows
        ],
        schema={
            "ts": pl.Datetime("ms"),
            "visitorid": pl.Int64,
            "itemid": pl.Int64,
            "event": pl.String,
            "transactionid": pl.Int64,
        },
        orient="row",
    )


def random_events(n=400, seed=0):
    rng = random.Random(seed)
    rows, seen = [], set()
    while len(rows) < n:
        ts = rng.randint(0, 9 * 24 * 60)  # minutes: gaps both inside and beyond a session
        key = (ts, rng.randint(1, 15))
        if key in seen:
            continue
        seen.add(key)
        rows.append(
            (
                ts / 1440,
                key[1],
                rng.randint(1, 30),
                rng.choice(["view"] * 6 + ["addtocart", "transaction"]),
            )
        )
    return events(rows)


# ---------- leakage: the headline property of this module ----------


def snapshot_inputs():
    ev = random_events()
    props = pl.DataFrame(
        {
            "ts": [T0, T0 + timedelta(days=2)],
            "itemid": [1, 1],
            "property": ["categoryid", "available"],
            "value": [10, 1],
        },
        schema={
            "ts": pl.Datetime("ms"),
            "itemid": pl.Int64,
            "property": pl.String,
            "value": pl.Int64,
        },
    )
    tree = pl.DataFrame({"categoryid": [10], "parentid": [3]})
    return ev, props, tree


def test_features_ignore_events_at_or_after_as_of():
    ev, props, tree = snapshot_inputs()
    snap = type("S", (), {"as_of": AS_OF})()
    base = build_snapshot(ev.filter(pl.col("ts") < AS_OF), props, tree, snap)
    future = events(
        [(10.0, 1, 1, "transaction"), (11.5, 99, 77, "view"), (12.0, 2, 1, "addtocart")]
    )
    full = build_snapshot(pl.concat([ev, future]), props, tree, snap)
    for name in base:
        assert_frame_equal(
            base[name].sort(base[name].columns[0]), full[name].sort(full[name].columns[0])
        )


def test_event_exactly_at_as_of_is_excluded():
    ev = events([(1, 1, 5, "view"), (9, 1, 6, "view")])  # second event is AS_OF exactly
    state = user_state(ev, AS_OF)
    assert state["n_view"].to_list() == [1]


# ---------- hand-computed values ----------


def test_user_features_hand_computed():
    ev = events([(1, 1, 10, "view"), (1.01, 1, 11, "addtocart"), (5, 1, 12, "transaction")])
    f = derive_user_features(user_state(ev, AS_OF), AS_OF).row(0, named=True)
    assert (f["u_n_events"], f["u_n_view"], f["u_n_cart"], f["u_n_buy"]) == (3, 1, 1, 1)
    assert f["u_n_sessions"] == 2  # first two events are 14 minutes apart, the third 4 days later
    assert f["u_cart_per_view"] == 1.0 and f["u_buy_per_event"] == pytest.approx(1 / 3)
    # AS_OF is day 9: last event on day 5, first on day 1
    assert f["u_days_since_last"] == pytest.approx(4.0)
    assert f["u_days_since_first"] == pytest.approx(8.0)
    # decayed to AS_OF: transaction (w=5) is 4 days old, cart (w=3) 7.99 days, view (w=1) 8 days
    tau = DECAY_TAU_DAYS
    expected = 5 * math.exp(-4 / tau) + 3 * math.exp(-7.99 / tau) + 1 * math.exp(-8 / tau)
    assert f["u_decay_interest"] == pytest.approx(expected)
    assert f["u_decay_actions"] == pytest.approx(math.exp(-4 / tau) + math.exp(-7.99 / tau))


def test_item_features_hand_computed():
    # AS_OF is day 9. Item 10 events, with their age at AS_OF:
    #   view 8d ago (u1) | view 6.5d ago (u2) | addtocart 0.5d ago (u3) | buy 0.2d ago (u2)
    ev = events(
        [
            (1, 1, 10, "view"),
            (2.5, 2, 10, "view"),
            (8.5, 3, 10, "addtocart"),
            (8.8, 2, 10, "transaction"),
            (8.9, 1, 20, "view"),
        ]
    )
    attrs = pl.DataFrame(
        {"itemid": [10], "i_categoryid": [7], "i_parent_categoryid": [3], "i_available": [1]}
    )
    r = item_features(ev, attrs, AS_OF).filter(pl.col("itemid") == 10).row(0, named=True)
    assert r["i_w_all"] == 1 + 1 + 3 + 5
    assert r["i_w_1d"] == 3 + 5  # cart and buy only
    assert r["i_w_7d"] == 1 + 3 + 5  # the 8-day-old view is outside
    assert r["i_w_28d"] == 10
    assert r["i_trend"] == pytest.approx((8 + 1) / (9 / 7 + 1))
    assert (r["i_n_view"], r["i_n_cart"], r["i_n_buy"], r["i_n_users"]) == (2, 1, 1, 3)
    assert r["i_days_since_last"] == pytest.approx(0.2) and r["i_age_days"] == pytest.approx(8.0)
    assert (r["i_categoryid"], r["i_available"]) == (7, 1)
    assert item_features(ev, attrs, AS_OF).filter(pl.col("itemid") == 20)["i_categoryid"][0] == -1


def test_smoothed_rate_pulls_small_items_towards_the_global_rate():
    # item 1: 1 view + 1 cart (raw 100%), item 2: 98 views and no cart; global = 1 / 99
    rows = [(1, 1, 1, "view"), (1, 1, 1, "addtocart")] + [(1, u, 2, "view") for u in range(2, 100)]
    attrs = pl.DataFrame(
        {
            "itemid": [1, 2],
            "i_categoryid": [1, 1],
            "i_parent_categoryid": [-1, -1],
            "i_available": [1, 1],
        }
    )
    out = item_features(events(rows), attrs, AS_OF).sort("itemid")
    assert out["i_cart_rate_s"][0] < 0.1  # a raw rate would be 1.0


def test_item_attributes_use_the_value_known_at_as_of():
    props = pl.DataFrame(
        {
            "ts": [T0, T0 + timedelta(days=5), T0 + timedelta(days=20)],
            "itemid": [1, 1, 1],
            "property": ["categoryid", "categoryid", "categoryid"],
            "value": [10, 11, 12],
        },
        schema={
            "ts": pl.Datetime("ms"),
            "itemid": pl.Int64,
            "property": pl.String,
            "value": pl.Int64,
        },
    )
    tree = pl.DataFrame({"categoryid": [10, 11, 12], "parentid": [1, 2, 3]})
    row = item_attributes(props, tree, AS_OF).row(0, named=True)
    assert row["i_categoryid"] == 11 and row["i_parent_categoryid"] == 2  # not 12 (set later)
    assert row["i_available"] == -1  # never recorded


def test_category_features_aggregate_items():
    items = pl.DataFrame({"i_categoryid": [1, 1, 2], "i_w_all": [5, 3, 7], "i_w_7d": [1, 0, 2]})
    out = category_features(items).sort("categoryid")
    assert out["c_w_all"].to_list() == [8, 7] and out["c_n_items"].to_list() == [2, 1]


def test_pair_features_hand_computed_and_keep_order():
    ev = events(
        [(1, 1, 10, "view"), (2, 1, 10, "addtocart"), (3, 1, 11, "view"), (2, 2, 99, "view")]
    )
    items = pl.DataFrame({"itemid": [10, 11, 12], "i_categoryid": [5, 5, 6]})
    pairs = pl.DataFrame({"visitorid": [1, 1, 2, 3], "itemid": [12, 10, 10, 10]})
    out = pair_features(pairs, ev, items, AS_OF)
    assert out["itemid"].to_list() == [12, 10, 10, 10]  # same order as the input pairs
    r = out.rows(named=True)
    assert (r[0]["p_n_view"], r[0]["p_days_since_touch"], r[0]["p_cat_share"]) == (0, None, 0.0)
    assert (r[1]["p_n_view"], r[1]["p_n_cart"]) == (1, 1)
    assert r[1]["p_days_since_touch"] == pytest.approx(7.0)  # last touch on day 2, AS_OF day 9
    assert r[1]["p_cat_share"] == pytest.approx(1.0)  # all 3 of user 1's events are in category 5
    assert r[3]["p_n_view"] == 0 and r[3]["p_cat_share"] == 0.0  # user 3 has no history


# ---------- offline == online ----------


def test_incremental_update_matches_batch_state():
    ev = random_events().filter(pl.col("ts") < AS_OF)
    batch = user_state(ev, AS_OF).sort("visitorid")
    online = {}
    for ts, user, item, event in (
        ev.sort("ts", "itemid").select("ts", "visitorid", "itemid", "event").rows()
    ):
        online[user] = update_user_state(online.get(user), ts, item, event)
    assert sorted(online) == batch["visitorid"].to_list()
    for row in batch.iter_rows(named=True):
        s = online[row["visitorid"]]
        for col in (
            "n_view",
            "n_cart",
            "n_buy",
            "n_sessions",
            "first_ts",
            "last_ts",
            "recent_items",
        ):
            assert s[col] == row[col], (row["visitorid"], col)
        assert s["decay_interest"] == pytest.approx(row["decay_interest"])
        assert s["decay_actions"] == pytest.approx(row["decay_actions"])
        assert len(s["recent_items"]) <= RECENT_N


# ---------- schema ----------


def test_snapshot_columns_match_the_schema_and_json():
    ev, props, tree = snapshot_inputs()
    snap = type("S", (), {"as_of": AS_OF})()
    tables = build_snapshot(ev, props, tree, snap)
    assert tables["user_features"].columns[1:] == [f.name for f in BY_GROUP["user"]]
    assert set(tables["item_features"].columns) == {"itemid", *[f.name for f in BY_GROUP["item"]]}
    assert tables["category_features"].columns[1:] == [f.name for f in BY_GROUP["category"]]
    spec = schema_dict()
    assert spec["feature_order"] == FEATURE_COLUMNS == [f["name"] for f in spec["features"]]
    json.dumps(spec)  # serialisable


def test_schema_rejects_out_of_range_values():
    bad = (
        pl.DataFrame({"visitorid": [1]})
        .with_columns(
            [
                pl.lit(0).cast(pl.Int64).alias(f.name)
                if f.dtype == "int"
                else pl.lit(0.0).alias(f.name)
                for f in BY_GROUP["user"]
            ]
        )
        .with_columns(pl.lit(2.0).alias("u_buy_per_event"))
    )  # must be within [0, 1]
    with pytest.raises(Exception):
        table_schema("user", "visitorid").validate(cast_to_schema(bad, "user"))


def test_snapshots_follow_the_etl_split_boundaries():
    ev = events([(d, 1, 1, "view") for d in range(0, 40)])
    rank, val, test = make_snapshots(ev)
    assert val.as_of == rank.label_end and test.as_of == val.label_end
    assert rank.as_of == val.as_of - timedelta(days=7)
    assert test.as_of == ev["ts"].max() - timedelta(days=7)
