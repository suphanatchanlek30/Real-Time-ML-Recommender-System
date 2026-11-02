"""Turn (users, snapshot) into the table the Ranker scores: one row per user x candidate item."""

from datetime import datetime

import polars as pl

from src.etl.config import DATA_DIR
from src.evaluation.baselines import fit_popularity
from src.features.pair import pair_features
from src.features.schema import FEATURE_COLUMNS
from src.training.als import AlsModel, als_candidates
from src.training.config import N_CANDIDATES

FEATURES_DIR = DATA_DIR / "features"


def load_tables(snapshot: str) -> dict[str, pl.DataFrame]:
    """Feature tables written by `make features`."""
    return {
        name: pl.read_parquet(FEATURES_DIR / snapshot / f"{name}.parquet")
        for name in ("user_features", "item_features", "category_features")
    }


def build_candidates(
    users: list[int], als: AlsModel, popular: list[int], n: int = N_CANDIDATES
) -> pl.DataFrame:
    """ALS top-n for users the model knows; the popular list for everyone else (cold start)."""
    from_als = als_candidates(als, users, n).with_columns(pl.lit(1).alias("a_source"))
    cold = [u for u in users if not als.knows(u)]
    fallback = pl.DataFrame(
        {
            "visitorid": [u for u in cold for _ in popular[:n]],
            "itemid": popular[:n] * len(cold),
            "a_rank": list(range(1, len(popular[:n]) + 1)) * len(cold),
        }
    ).with_columns(
        pl.col("visitorid", "itemid", "a_rank").cast(pl.Int64),
        pl.lit(None, dtype=pl.Float64).alias("a_score"),
        pl.lit(0).alias("a_source"),
    )
    return pl.concat([from_als, fallback.select(from_als.columns)]).with_columns(
        pl.col("a_source").cast(pl.Int64)
    )


def ranking_frame(
    cands: pl.DataFrame, tables: dict[str, pl.DataFrame], events: pl.DataFrame, as_of: datetime
) -> pl.DataFrame:
    """Candidates + all feature groups, columns in FEATURE_COLUMNS order (after the two keys)."""
    pairs = pair_features(
        cands.select("visitorid", "itemid"), events, tables["item_features"], as_of
    )
    frame = (
        cands.join(tables["user_features"], on="visitorid", how="left")
        .join(tables["item_features"], on="itemid", how="left")
        .join(
            tables["category_features"].rename({"categoryid": "i_categoryid"}),
            on="i_categoryid",
            how="left",
        )
        .join(pairs, on=["visitorid", "itemid"], how="left")
    )
    return frame.select("visitorid", "itemid", *FEATURE_COLUMNS)


def add_labels(frame: pl.DataFrame, truth: dict[int, set[int]]) -> pl.DataFrame:
    """label = 1 if the user added the item to cart or bought it in the label window."""
    positives = pl.DataFrame(
        {
            "visitorid": [u for u, items in truth.items() for _ in items],
            "itemid": [i for items in truth.values() for i in items],
        },
        schema={"visitorid": pl.Int64, "itemid": pl.Int64},
    ).with_columns(pl.lit(1).alias("label"))
    return frame.join(positives, on=["visitorid", "itemid"], how="left").with_columns(
        pl.col("label").fill_null(0)
    )


def make_frame(
    users: list[int],
    events: pl.DataFrame,
    as_of: datetime,
    snapshot: str,
    als: AlsModel,
    n: int = N_CANDIDATES,
) -> pl.DataFrame:
    """Candidates + features for `users` as of the snapshot, using only events before as_of."""
    history = events.filter(pl.col("ts") < as_of)
    popular = fit_popularity(history, n=n, window_days=7)
    cands = build_candidates(users, als, popular, n)
    return ranking_frame(cands, load_tables(snapshot), events, as_of)
