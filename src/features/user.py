"""User features.

Two layers, so offline and online share one definition:

  state    raw numbers that can be updated one event at a time (this is what Redis stores):
           counts, first/last timestamp, session count, decayed sums, last-20 items
  derive   features computed from the state *at scoring time* (recency, ratios, decay to now)

`user_state` (batch, polars) and `update_user_state` (one event, plain Python) must give the
same state; tests/test_features.py checks that. Never store a derived value such as
"days since last event": it is stale the moment it is written.
"""

import math
from datetime import datetime, timedelta

import polars as pl

from src.config import EVENT_WEIGHTS, SESSION_GAP_MINUTES
from src.features.schema import DECAY_TAU_DAYS, RECENT_N

DAY_MS = 86_400_000
GAP = timedelta(minutes=SESSION_GAP_MINUTES)


def _weight() -> pl.Expr:
    return pl.col("event").replace_strict(EVENT_WEIGHTS, return_dtype=pl.Int64)


def user_state(events: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Raw per-user state from all events strictly before `as_of`."""
    h = (
        events.filter(pl.col("ts") < as_of)
        .sort("visitorid", "ts", "itemid")
        .with_columns(
            _weight().alias("w"),
            pl.col("ts").diff().over("visitorid").alias("gap"),
        )
    )
    # age of every event relative to the user's last event, in decay units
    age = (pl.col("ts").max() - pl.col("ts")).dt.total_milliseconds() / DAY_MS / DECAY_TAU_DAYS
    return h.group_by("visitorid").agg(
        (pl.col("event") == "view").sum().alias("n_view"),
        (pl.col("event") == "addtocart").sum().alias("n_cart"),
        (pl.col("event") == "transaction").sum().alias("n_buy"),
        (pl.col("gap").is_null() | (pl.col("gap") > GAP)).sum().alias("n_sessions"),
        pl.col("ts").min().alias("first_ts"),
        pl.col("ts").max().alias("last_ts"),
        # decayed sums are stored as of last_ts: sum(w * exp(-(last_ts - ts) / tau))
        (pl.col("w") * (-age).exp()).sum().alias("decay_interest"),
        ((pl.col("event") != "view").cast(pl.Float64) * (-age).exp()).sum().alias("decay_actions"),
        pl.col("itemid").tail(RECENT_N).alias("recent_items"),
    )


def update_user_state(state: dict | None, ts: datetime, itemid: int, event: str) -> dict:
    """Apply one event (events of a user must arrive in time order). This is the online path."""
    w = EVENT_WEIGHTS[event]
    action = 0.0 if event == "view" else 1.0
    if state is None:
        state = {
            "n_view": 0,
            "n_cart": 0,
            "n_buy": 0,
            "n_sessions": 1,
            "first_ts": ts,
            "last_ts": ts,
            "decay_interest": 0.0,
            "decay_actions": 0.0,
            "recent_items": [],
        }
        decay = 1.0
    else:
        gap = ts - state["last_ts"]
        if gap > GAP:
            state["n_sessions"] += 1
        # shrink the old sums to the new event's time, then add the event
        decay = math.exp(-gap.total_seconds() / 86400 / DECAY_TAU_DAYS)
    state["decay_interest"] = state["decay_interest"] * decay + w
    state["decay_actions"] = state["decay_actions"] * decay + action
    state[{"view": "n_view", "addtocart": "n_cart", "transaction": "n_buy"}[event]] += 1
    state["last_ts"] = ts
    state["recent_items"] = (state["recent_items"] + [itemid])[-RECENT_N:]
    return state


def derive_user_features(state: pl.DataFrame, as_of: datetime) -> pl.DataFrame:
    """Model features from the raw state, evaluated at time `as_of` (offline) or now (online)."""
    since_last = (pl.lit(as_of) - pl.col("last_ts")).dt.total_milliseconds() / DAY_MS
    n_events = pl.col("n_view") + pl.col("n_cart") + pl.col("n_buy")
    return state.select(
        "visitorid",
        n_events.alias("u_n_events"),
        pl.col("n_view").alias("u_n_view"),
        pl.col("n_cart").alias("u_n_cart"),
        pl.col("n_buy").alias("u_n_buy"),
        pl.col("n_sessions").alias("u_n_sessions"),
        (pl.col("n_cart") / pl.col("n_view").clip(lower_bound=1)).alias("u_cart_per_view"),
        (pl.col("n_buy") / n_events).alias("u_buy_per_event"),
        since_last.alias("u_days_since_last"),
        ((pl.lit(as_of) - pl.col("first_ts")).dt.total_milliseconds() / DAY_MS).alias(
            "u_days_since_first"
        ),
        # bring the decayed sums from last_ts forward to as_of
        (pl.col("decay_interest") * (-since_last / DECAY_TAU_DAYS).exp()).alias("u_decay_interest"),
        (pl.col("decay_actions") * (-since_last / DECAY_TAU_DAYS).exp()).alias("u_decay_actions"),
    )
