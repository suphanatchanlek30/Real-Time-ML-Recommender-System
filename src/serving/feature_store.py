"""Online feature store on Redis: the same user state the offline features are built from.

Layout (all keys per user id `u`; ids are stored as plain integers):

    user:{u}:state   HASH   n_view n_cart n_buy n_sessions first_ts last_ts (epoch ms)
                            decay_interest decay_actions   -> the raw state of features/user.py
    user:{u}:recent  LIST   the last 20 item ids (oldest first)
    user:{u}:items   HASH   v:{item} c:{item} b:{item}  view/cart/buy counts of this exact item
                            t:{item}                     epoch ms of the last touch
                                                         -> pair features p_n_*, p_days_since_touch
    user:{u}:cats    HASH   {category} -> events in that category   -> p_cat_share
    clock            STRING event time of the newest processed event (epoch ms)

Why `clock`: the data is from 2015, so wall-clock time would make every recency 4000 days and
every decayed sum 0. All "now" values are event time: the batch load sets it to the snapshot
time and the consumer moves it forward.

The categories come from the frozen item table of the snapshot (the same table the offline
pair features use), so offline and online agree. Nothing here expires: dropping state with a
TTL would make online features differ from the offline ones the model was trained on.
"""

import itertools
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Literal

import polars as pl
from pydantic import BaseModel, Field

from src.features.spec import RECENT_N
from src.features.user import update_user_state
from src.serving.timeutil import from_ms

CLOCK = "clock"
STATE_INT = ("n_view", "n_cart", "n_buy", "n_sessions")
STATE_FLOAT = ("decay_interest", "decay_actions")
CODE = {"view": "v", "addtocart": "c", "transaction": "b"}


def state_key(u: int) -> str:
    return f"user:{u}:state"


def recent_key(u: int) -> str:
    return f"user:{u}:recent"


def items_key(u: int) -> str:
    return f"user:{u}:items"


def cats_key(u: int) -> str:
    return f"user:{u}:cats"


class Event(BaseModel):
    """One message of the `user-events` topic."""

    visitor_id: int
    item_id: int
    event: Literal["view", "addtocart", "transaction"]
    ts: int = Field(gt=0, description="epoch milliseconds")


@dataclass
class UserData:
    """Everything the API needs about one user, read in a single Redis round trip."""

    state: dict | None  # decoded state, None when the user has no history
    items: dict[str, str]
    cats: dict[str, str]
    clock_ms: int | None


# ---------- write path: consumer ----------


def _decode_state(raw: Mapping[str, str]) -> dict:
    state = {k: int(raw[k]) for k in STATE_INT}
    state.update({k: float(raw[k]) for k in STATE_FLOAT})
    state["first_ts"], state["last_ts"] = from_ms(raw["first_ts"]), from_ms(raw["last_ts"])
    state["recent_items"] = []  # kept in its own list key, not in the hash
    return state


def _encode_state(state: Mapping) -> dict:
    from src.serving.timeutil import to_ms

    out = {k: state[k] for k in (*STATE_INT, *STATE_FLOAT)}
    out["first_ts"], out["last_ts"] = to_ms(state["first_ts"]), to_ms(state["last_ts"])
    return out


def handle_event(r, ev: Event, item_cat: Mapping[int, int]) -> bool:
    """Apply one event to the user's Redis state. Returns False when it was skipped as late.

    Pure with respect to everything but Redis, so it is unit-tested with fakeredis. A user's
    events must arrive in time order (Kafka keeps that per key); an older event is skipped
    because the decayed sums can only move forward in time.
    """
    u, item = ev.visitor_id, ev.item_id
    raw = r.hgetall(state_key(u))
    state = _decode_state(raw) if raw else None
    ts = from_ms(ev.ts)
    if state is not None and ts < state["last_ts"]:
        return False
    state = update_user_state(state, ts, item, ev.event)

    pipe = r.pipeline(transaction=True)
    pipe.hset(state_key(u), mapping=_encode_state(state))
    pipe.rpush(recent_key(u), item)
    pipe.ltrim(recent_key(u), -RECENT_N, -1)
    pipe.hincrby(items_key(u), f"{CODE[ev.event]}:{item}", 1)
    pipe.hset(items_key(u), f"t:{item}", ev.ts)
    category = item_cat.get(item)
    if category is not None:  # items missing from the frozen table have no category
        pipe.hincrby(cats_key(u), str(category), 1)
    pipe.execute()
    return True


def advance_clock(r, ts_ms: int) -> None:
    """Move the event-time clock forward (never back)."""
    current = r.get(CLOCK)
    if current is None or ts_ms > int(current):
        r.set(CLOCK, ts_ms)


# ---------- read path: API ----------


def read_user(r, u: int) -> UserData:
    """One pipelined round trip: state, per-item counts, per-category counts and the clock."""
    pipe = r.pipeline(transaction=False)
    pipe.hgetall(state_key(u))
    pipe.hgetall(items_key(u))
    pipe.hgetall(cats_key(u))
    pipe.get(CLOCK)
    state, items, cats, clock = pipe.execute()
    return UserData(
        state=_decode_state(state) if state else None,
        items=items,
        cats=cats,
        clock_ms=int(clock) if clock is not None else None,
    )


# ---------- batch load: the snapshot state before the stream starts ----------


def _by_user(df: pl.DataFrame):
    """Rows grouped by their first column (the frame must be sorted by it)."""
    for key, rows in itertools.groupby(df.iter_rows(), key=lambda row: row[0]):
        yield key, list(rows)


def load_batch(
    r,
    state: pl.DataFrame,
    touched: pl.DataFrame,
    categories: pl.DataFrame,
    clock_ms: int,
    chunk: int = 2000,
) -> dict:
    """Write the offline user state into Redis (after wiping it), pipelined.

    state       user_state table of the snapshot
    touched     visitorid, itemid, n_view, n_cart, n_buy, last_ms   per user x item
    categories  visitorid, category, n                              per user x category
    """
    r.flushdb()  # a re-load must start from the batch state, or replayed events count twice
    started = time.time()
    state = state.sort("visitorid").with_columns(
        pl.col("first_ts").dt.epoch("ms").alias("first_ms"),
        pl.col("last_ts").dt.epoch("ms").alias("last_ms"),
    )
    pairs = _by_user(touched.sort("visitorid", "itemid"))
    cats = _by_user(categories.sort("visitorid", "category"))
    next_pair, next_cat = next(pairs, None), next(cats, None)

    pipe, n_users = r.pipeline(transaction=False), 0
    for row in state.iter_rows(named=True):
        u = row["visitorid"]
        mapping = {k: row[k] for k in (*STATE_INT, *STATE_FLOAT)}
        mapping.update(first_ts=row["first_ms"], last_ts=row["last_ms"])
        pipe.hset(state_key(u), mapping=mapping)
        if row["recent_items"]:
            pipe.rpush(recent_key(u), *row["recent_items"])
        if next_pair and next_pair[0] == u:
            fields = {}
            for _, item, v, c, b, last_ms in next_pair[1]:
                for code, count in (("v", v), ("c", c), ("b", b)):
                    if count:
                        fields[f"{code}:{item}"] = count
                fields[f"t:{item}"] = last_ms
            pipe.hset(items_key(u), mapping=fields)
            next_pair = next(pairs, None)
        if next_cat and next_cat[0] == u:
            pipe.hset(cats_key(u), mapping={str(c): n for _, c, n in next_cat[1]})
            next_cat = next(cats, None)
        n_users += 1
        if n_users % chunk == 0:
            pipe.execute()
    pipe.execute()
    r.set(CLOCK, clock_ms)
    return {"users": n_users, "seconds": round(time.time() - started, 1)}
