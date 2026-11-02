"""Online recommendation: retrieve 200 candidates, build the 38 features, let the Ranker order them.

Mirrors the offline path (training/dataset.py + training/ranker.py) step for step; the parity
tests check that both give the same features and the same ranking.
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl

from src.features.spec import BY_GROUP, FEATURE_COLUMNS
from src.features.user import derive_user_features
from src.serving.bundle import Bundle
from src.serving.feature_store import UserData
from src.serving.tables import ItemTables
from src.serving.timeutil import DAY_MS, from_ms, to_ms

COL = {name: i for i, name in enumerate(FEATURE_COLUMNS)}
USER_COLS = [f.name for f in BY_GROUP["user"]]
STATE_SCHEMA = {
    "visitorid": pl.Int64,
    "n_view": pl.Int64,
    "n_cart": pl.Int64,
    "n_buy": pl.Int64,
    "n_sessions": pl.Int64,
    "first_ts": pl.Datetime("ms"),
    "last_ts": pl.Datetime("ms"),
    "decay_interest": pl.Float64,
    "decay_actions": pl.Float64,
}


@dataclass
class Result:
    items: list[tuple[int, float]]  # (item id, ranker score), best first
    retrieval: str  # "als" or "popular"
    degraded: bool  # True: Redis was unavailable, user and pair features were left empty


def user_feature_values(user_id: int, state: dict, now_ms: int) -> dict[str, float]:
    """u_* features from the raw state at time `now` (same code as the offline path)."""
    row = {"visitorid": [user_id], **{k: [state[k]] for k in STATE_SCHEMA if k != "visitorid"}}
    state_df = pl.DataFrame(row, schema=STATE_SCHEMA)
    return derive_user_features(state_df, from_ms(now_ms)).select(USER_COLS).row(0, named=True)


def pair_feature_matrix(items: np.ndarray, data: UserData, item_cat: dict, now_ms: int):
    """p_* features of one user against each candidate, from the Redis hashes."""
    n_events = 0
    if data.state:
        n_events = data.state["n_view"] + data.state["n_cart"] + data.state["n_buy"]
    out = np.zeros((len(items), 5))
    for n, item in enumerate(items.tolist()):
        out[n, 0] = int(data.items.get(f"v:{item}", 0))
        out[n, 1] = int(data.items.get(f"c:{item}", 0))
        out[n, 2] = int(data.items.get(f"b:{item}", 0))
        last = data.items.get(f"t:{item}")
        out[n, 3] = (now_ms - int(last)) / DAY_MS if last is not None else np.nan
        cat = item_cat.get(item)
        if cat is not None and n_events:
            out[n, 4] = int(data.cats.get(str(cat), 0)) / n_events
    return out


class Recommender:
    def __init__(self, bundle: Bundle, tables: ItemTables, n_candidates: int = 200):
        self.bundle, self.tables, self.n = bundle, tables, n_candidates
        # event-time clock to use when Redis has none yet: the time of the batch snapshot
        self.default_now_ms = to_ms(datetime.fromisoformat(tables.as_of))

    def candidates(self, user_id: int):
        """(items, als scores, retrieval name). ALS for users it knows, popular list otherwise."""
        b, n = self.bundle, self.n
        row = b.user_row.get(int(user_id))
        if row is None:
            return np.array(b.popular[:n]), np.full(min(n, len(b.popular)), np.nan), "popular"
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
        """`data` is None when Redis could not be read: same path with empty user/pair features."""
        items, a_score, retrieval = self.candidates(user_id)
        X = self.features(user_id, items, a_score, retrieval, data)
        scores = self.bundle.booster.predict(X)
        order = np.argsort(-scores, kind="stable")[:k]
        return Result(
            items=[(int(items[i]), float(scores[i])) for i in order],
            retrieval=retrieval,
            degraded=data is None,
        )
