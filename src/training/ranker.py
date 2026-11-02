"""LightGBM LambdaRank: reorder each user's 200 candidates."""

import lightgbm as lgb
import numpy as np
import polars as pl

from src.features.schema import FEATURE_COLUMNS
from src.training.config import EARLY_STOPPING, MAX_TREES, RANKER_PARAMS


def to_matrix(frame: pl.DataFrame) -> np.ndarray:
    """Feature matrix in schema.json order; nulls become NaN, which LightGBM handles."""
    return frame.select(FEATURE_COLUMNS).cast(pl.Float64).to_numpy()


def _groups(frame: pl.DataFrame) -> tuple[pl.DataFrame, list[int]]:
    """Keep users with at least one positive (a group of all zeros teaches nothing) and return
    the frame sorted so that each (snapshot, user) group is contiguous, plus the group sizes."""
    frame = frame.with_columns(pl.col("label").sum().over("gid").alias("_pos")).filter(
        pl.col("_pos") > 0
    )
    frame = frame.sort("gid", maintain_order=True).drop("_pos")
    sizes = frame.group_by("gid", maintain_order=True).len()["len"].to_list()
    return frame, sizes


def _dataset(frame: pl.DataFrame) -> lgb.Dataset | None:
    frame, sizes = _groups(frame)
    if frame.is_empty():
        return None
    return lgb.Dataset(
        to_matrix(frame),
        label=frame["label"].to_numpy(),
        group=sizes,
        feature_name=FEATURE_COLUMNS,
        free_raw_data=False,
    )


def fit_ranker(train: pl.DataFrame, holdout: pl.DataFrame) -> tuple[lgb.Booster, dict]:
    """Train with early stopping on `holdout` (a later snapshot, still inside Train), then refit
    on train + holdout with the best number of trees. Returns the booster and a small report.

    Frames need a `gid` column (snapshot + user) and a `label` column.
    """
    d_train, d_hold = _dataset(train), _dataset(holdout)
    if d_train is None or d_hold is None:
        raise ValueError("no user with a positive label in the Ranker training or holdout data")
    kept, sizes = _groups(train)
    stats = {"train_rows": kept.height, "train_users_with_positive": len(sizes)}
    evals = {}
    model = lgb.train(
        RANKER_PARAMS,
        d_train,
        num_boost_round=MAX_TREES,
        valid_sets=[d_hold],
        valid_names=["hold"],
        callbacks=[lgb.early_stopping(EARLY_STOPPING, verbose=False), lgb.record_evaluation(evals)],
    )
    best = max(model.best_iteration, 1)
    final = lgb.train(RANKER_PARAMS, _dataset(pl.concat([train, holdout])), num_boost_round=best)
    stats.update(best_iteration=best, holdout_ndcg10=float(model.best_score["hold"]["ndcg@10"]))
    return final, stats


def rerank(frame: pl.DataFrame, booster: lgb.Booster) -> dict[int, list[int]]:
    """Reorder every user's candidates by the ranker score. user -> items, best first."""
    scored = frame.with_columns(pl.Series("score", booster.predict(to_matrix(frame))))
    ranked = (
        scored.sort(["visitorid", "score"], descending=[False, True])
        .group_by("visitorid", maintain_order=True)
        .agg(pl.col("itemid"))
    )
    return dict(zip(ranked["visitorid"], ranked["itemid"].to_list()))


def retrieved_order(frame: pl.DataFrame) -> dict[int, list[int]]:
    """Candidates in retrieval order (ALS rank, or popularity rank for cold users)."""
    ranked = (
        frame.sort(["visitorid", "a_rank"])
        .group_by("visitorid", maintain_order=True)
        .agg(pl.col("itemid"))
    )
    return dict(zip(ranked["visitorid"], ranked["itemid"].to_list()))
