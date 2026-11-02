"""Candidate generation with ALS (implicit feedback matrix factorisation).

Training uses the `implicit` library; scoring is a plain dot product of the saved factors, so
serving only needs numpy (no `implicit`, no compiler).
"""

from dataclasses import dataclass
from datetime import datetime

import numpy as np
import polars as pl
import scipy.sparse as sp

from src.config import EVENT_WEIGHTS
from src.training.config import ALS_PARAMS, MIN_EVENTS, N_CANDIDATES


@dataclass
class AlsModel:
    user_ids: np.ndarray  # row i of user_factors belongs to user_ids[i]
    item_ids: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray
    params: dict

    def __post_init__(self):
        self._user_row = {int(u): i for i, u in enumerate(self.user_ids)}

    def knows(self, user: int) -> bool:
        return int(user) in self._user_row

    def save(self, path) -> None:
        np.savez_compressed(
            path,
            user_ids=self.user_ids,
            item_ids=self.item_ids,
            user_factors=self.user_factors,
            item_factors=self.item_factors,
        )

    @classmethod
    def load(cls, path, params: dict | None = None) -> "AlsModel":
        z = np.load(path)
        return cls(z["user_ids"], z["item_ids"], z["user_factors"], z["item_factors"], params or {})


def interaction_matrix(history: pl.DataFrame, as_of: datetime, min_events: int = MIN_EVENTS):
    """Weighted user x item matrix from events before `as_of`, for users/items with enough data."""
    h = history.filter(pl.col("ts") < as_of)
    n_user = h.group_by("visitorid").len().rename({"len": "n_u"})
    n_item = h.group_by("itemid").len().rename({"len": "n_i"})
    h = (
        h.join(n_user, on="visitorid")
        .join(n_item, on="itemid")
        .filter((pl.col("n_u") >= min_events) & (pl.col("n_i") >= min_events))
    )
    weights = h.with_columns(
        pl.col("event").replace_strict(EVENT_WEIGHTS, return_dtype=pl.Float64).alias("w")
    )
    pairs = weights.group_by("visitorid", "itemid").agg(pl.col("w").sum())
    users = pairs["visitorid"].unique().sort()
    items = pairs["itemid"].unique().sort()
    user_index = pl.DataFrame({"visitorid": users, "row": np.arange(users.len())})
    item_index = pl.DataFrame({"itemid": items, "col": np.arange(items.len())})
    coded = pairs.join(user_index, on="visitorid").join(item_index, on="itemid")
    matrix = sp.csr_matrix(
        (coded["w"].to_numpy(), (coded["row"].to_numpy(), coded["col"].to_numpy())),
        shape=(users.len(), items.len()),
    )
    return matrix, users.to_numpy(), items.to_numpy()


def fit_als(history: pl.DataFrame, as_of: datetime, params: dict | None = None) -> AlsModel:
    from implicit.cpu.als import AlternatingLeastSquares  # CPU build: no GPU needed

    p = {**ALS_PARAMS, **(params or {})}
    matrix, users, items = interaction_matrix(history, as_of)
    matrix = matrix.copy()
    matrix.data = 1.0 + p["alpha"] * matrix.data  # confidence of each observed interaction
    model = AlternatingLeastSquares(
        factors=p["factors"],
        regularization=p["regularization"],
        iterations=p["iterations"],
        calculate_training_loss=False,
        random_state=42,
    )
    model.fit(matrix, show_progress=False)
    return AlsModel(users, items, np.asarray(model.user_factors), np.asarray(model.item_factors), p)


def als_candidates(
    model: AlsModel, users, n: int = N_CANDIDATES, batch: int = 2000
) -> pl.DataFrame:
    """Top-n items by dot product for every user the model knows. Already-seen items stay in:
    people re-add and re-buy items, and the Ranker's pair features need them as candidates."""
    known = [int(u) for u in users if model.knows(u)]
    rows = np.array([model._user_row[u] for u in known])
    out = []
    for start in range(0, len(known), batch):
        chunk = slice(start, start + batch)
        scores = model.user_factors[rows[chunk]] @ model.item_factors.T
        top = np.argpartition(-scores, n - 1, axis=1)[:, :n]
        top_scores = np.take_along_axis(scores, top, axis=1)
        order = np.argsort(-top_scores, axis=1)
        top, top_scores = (
            np.take_along_axis(top, order, 1),
            np.take_along_axis(top_scores, order, 1),
        )
        count = top.shape[0]
        out.append(
            pl.DataFrame(
                {
                    "visitorid": np.repeat(known[chunk], n),
                    "itemid": model.item_ids[top.ravel()],
                    "a_score": top_scores.ravel().astype(np.float64),
                    "a_rank": np.tile(np.arange(1, n + 1), count),
                }
            )
        )
    if not out:
        return pl.DataFrame(
            schema={
                "visitorid": pl.Int64,
                "itemid": pl.Int64,
                "a_score": pl.Float64,
                "a_rank": pl.Int64,
            }
        )
    return pl.concat(out).with_columns(pl.col("visitorid", "itemid", "a_rank").cast(pl.Int64))
