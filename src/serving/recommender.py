"""Online recommendation: retrieve 200 candidates, build the 38 features, let the Ranker order them.

Mirrors the offline path (training/dataset.py + training/ranker.py) step for step; the parity
tests check that both give the same features and the same ranking.
"""

import math
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from src.features.spec import BY_GROUP, DECAY_TAU_DAYS, FEATURE_COLUMNS
from src.serving.bundle import Bundle
from src.serving.feature_store import UserData
from src.serving.tables import ItemTables
from src.serving.timeutil import DAY_MS, to_ms

COL = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
USER_COLS = [f.name for f in BY_GROUP["user"]]


@dataclass
class Result:
    items: list[tuple[int, float]]  # (item id, ranker score), best first
    retrieval: str  # "als" or "popular"
    degraded: bool  # True: Redis was unavailable, user and pair features were left empty


def user_feature_values(user_id: int, state: dict, now_ms: int) -> dict[str, float]:
    """u_* features from the raw state at time `now`.

    Plain Python on purpose: building a one-row polars frame for every request cost ~1 ms, more
    than everything else in the request. The formulas are those of features/user.py
    `derive_user_features`; the parity tests compare the two on random data.
    """
    n_view, n_cart, n_buy = state["n_view"], state["n_cart"], state["n_buy"]
    n_events = n_view + n_cart + n_buy
    since_last = (now_ms - to_ms(state["last_ts"])) / DAY_MS
    fade = math.exp(-since_last / DECAY_TAU_DAYS)  # brings the decayed sums forward to `now`
    return {
        "u_n_events": n_events,
        "u_n_view": n_view,
        "u_n_cart": n_cart,
        "u_n_buy": n_buy,
        "u_n_sessions": state["n_sessions"],
        "u_cart_per_view": n_cart / max(n_view, 1),
        "u_buy_per_event": n_buy / n_events,
        "u_days_since_last": since_last,
        "u_days_since_first": (now_ms - to_ms(state["first_ts"])) / DAY_MS,
        "u_decay_interest": state["decay_interest"] * fade,
        "u_decay_actions": state["decay_actions"] * fade,
    }


def pair_feature_matrix(items: np.ndarray, data: UserData, item_cat: dict, now_ms: int):
    """p_* features of one user against each candidate, from the Redis hashes.

    Walks the user's own hash fields (a few per touched item) instead of looking up 4 keys for
    each of the 200 candidates, and fills the category share one category at a time.
    """
    n = len(items)
    out = np.zeros((n, 5))
    out[:, 3] = np.nan  # days since the last touch: unknown until the user touched the item
    position = {item: i for i, item in enumerate(items.tolist())}
    column = {"v": 0, "c": 1, "b": 2}
    for field, value in data.items.items():
        code, _, item = field.partition(":")
        i = position.get(int(item))
        if i is None:
            continue
        if code == "t":
            out[i, 3] = (now_ms - int(value)) / DAY_MS
        else:
            out[i, column[code]] = int(value)

    n_events = 0
    if data.state:
        n_events = data.state["n_view"] + data.state["n_cart"] + data.state["n_buy"]
    if n_events and data.cats:
        category = np.array([item_cat.get(item, np.nan) for item in items.tolist()], dtype=float)
        for cat, count in data.cats.items():
            out[category == int(cat), 4] = int(count) / n_events
    return out


class Recommender:
    def __init__(self, bundle: Bundle, tables: ItemTables, n_candidates: int = 200):
        self.bundle, self.tables, self.n = bundle, tables, n_candidates
        # event-time clock to use when Redis has none yet: the time of the batch snapshot
        self.default_now_ms = to_ms(datetime.fromisoformat(tables.as_of))
        self._popular_items = np.array(bundle.popular[:n_candidates])
        self._popular_scores = np.full(len(self._popular_items), np.nan)
        # ranked lists that are the same for every user in the same situation, see recommend()
        self._shared: dict[bool, list[tuple[int, float]]] = {}

    def candidates(self, user_id: int):
        """(items, als scores, retrieval name). ALS for users it knows, popular list otherwise."""
        b, n = self.bundle, self.n
        row = b.user_row.get(int(user_id))
        if row is None:
            return self._popular_items, self._popular_scores, "popular"
        scores = b.user_factors[row] @ b.item_factors.T
        top = np.argpartition(-scores, n - 1)[:n]
        top = top[np.argsort(-scores[top])]
        return b.item_ids[top], scores[top], "als"

    def features(self, user_id, items, a_score, retrieval, data: UserData | None):
        X = np.full((len(items), len(FEATURE_COLUMNS)), np.nan)
        X[:, self.tables.columns] = self.tables.lookup(items)
        X[:, COL["a_score"]] = a_score
        X[:, COL["a_rank"]] = np.arange(1, len(items) + 1)
        X[:, COL["a_source"]] = 1.0 if retrieval == "als" else 0.0
        if data is not None:
            now = data.clock_ms if data.clock_ms is not None else self.default_now_ms
            if data.state:
                values = user_feature_values(user_id, data.state, now)
                for name, value in values.items():
                    X[:, COL[name]] = value
            pairs = pair_feature_matrix(items, data, self.tables.item_cat, now)
            for k, name in enumerate(
                ["p_n_view", "p_n_cart", "p_n_buy", "p_days_since_touch", "p_cat_share"]
            ):
                X[:, COL[name]] = pairs[:, k]
        return X

    def recommend(self, user_id: int, k: int, data: UserData | None) -> Result:
        """`data` is None when Redis could not be read: same path with empty user/pair features.

        A user ALS does not know and who has no history gets the same 200 popular candidates and
        the same (empty) features as every such user, so the ranked list is identical for all of
        them: it is computed once per model and reused. This is half of the real traffic.
        """
        items, a_score, retrieval = self.candidates(user_id)
        shared = retrieval == "popular" and (data is None or not (data.state or data.items))
        if shared and (data is None) in self._shared:
            return Result(self._shared[data is None][:k], retrieval, data is None)

        X = self.features(user_id, items, a_score, retrieval, data)
        scores = self.bundle.booster.predict(X)
        order = np.argsort(-scores, kind="stable")
        if shared:  # keep the whole ranking, any k is a slice of it
            self._shared[data is None] = [(int(items[i]), float(scores[i])) for i in order]
            return Result(self._shared[data is None][:k], retrieval, data is None)
        return Result(
            items=[(int(items[i]), float(scores[i])) for i in order[:k]],
            retrieval=retrieval,
            degraded=data is None,
        )
