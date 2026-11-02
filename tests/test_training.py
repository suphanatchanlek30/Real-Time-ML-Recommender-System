from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

pytest.importorskip("lightgbm")  # training extras (make install-train); skipped without them

from src.features.schema import FEATURE_COLUMNS  # noqa: E402
from src.training.als import AlsModel, als_candidates, interaction_matrix  # noqa: E402
from src.training.dataset import add_labels, build_candidates  # noqa: E402
from src.training.ranker import fit_ranker, rerank, retrieved_order  # noqa: E402
from src.training.train import gate, window  # noqa: E402

T0 = datetime(2015, 6, 1)
AS_OF = datetime(2015, 6, 10)


def events(rows):
    return pl.DataFrame(
        [(T0 + timedelta(days=d), u, i, e, None) for d, u, i, e in rows],
        schema={
            "ts": pl.Datetime("ms"),
            "visitorid": pl.Int64,
            "itemid": pl.Int64,
            "event": pl.String,
            "transactionid": pl.Int64,
        },
        orient="row",
    )


def test_interaction_matrix_weights_and_min_events_filter():
    ev = events(
        [
            (1, 1, 10, "view"),
            (2, 1, 10, "addtocart"),  # user 1 -> item 10: 1 + 3 = 4
            (1, 1, 11, "view"),
            (1, 2, 10, "view"),
            (1, 2, 11, "view"),
            (1, 3, 99, "view"),  # user 3 and item 99 have a single event: dropped
            (20, 1, 10, "transaction"),  # after AS_OF: ignored
        ]
    )
    matrix, users, items = interaction_matrix(ev, AS_OF, min_events=2)
    assert users.tolist() == [1, 2] and items.tolist() == [10, 11]
    assert matrix[0, 0] == 4 and matrix[1, 0] == 1


def tiny_model():
    # item 2 is the best match for user 1, then item 0; user 2 prefers item 1
    return AlsModel(
        user_ids=np.array([1, 2]),
        item_ids=np.array([100, 101, 102, 103]),
        user_factors=np.array([[1.0, 0.0], [0.0, 1.0]]),
        item_factors=np.array([[0.5, 0.0], [0.0, 0.9], [0.9, 0.0], [0.1, 0.1]]),
        params={},
    )


def test_als_candidates_are_ranked_dot_products():
    out = als_candidates(tiny_model(), [1, 2, 3], n=2)  # user 3 is unknown to the model
    assert out.filter(pl.col("visitorid") == 1)["itemid"].to_list() == [102, 100]
    assert out.filter(pl.col("visitorid") == 2)["itemid"].to_list() == [101, 103]
    assert out["a_rank"].to_list() == [1, 2, 1, 2] and 3 not in out["visitorid"].to_list()


def test_build_candidates_falls_back_to_popular_for_unknown_users():
    out = build_candidates([1, 3], tiny_model(), popular=[7, 8, 9], n=2)
    assert out.filter(pl.col("visitorid") == 1)["a_source"].to_list() == [1, 1]
    cold = out.filter(pl.col("visitorid") == 3).sort("a_rank")
    assert cold["itemid"].to_list() == [7, 8] and cold["a_source"].to_list() == [0, 0]
    assert cold["a_score"].null_count() == 2


def test_add_labels_marks_items_the_user_acted_on():
    frame = pl.DataFrame({"visitorid": [1, 1, 2], "itemid": [10, 11, 10]})
    out = add_labels(frame, {1: {11}, 2: {99}})
    assert out["label"].to_list() == [0, 1, 0]


def test_window_is_half_open():
    ev = events([(1, 1, 1, "view"), (2, 1, 1, "view"), (3, 1, 1, "view")])
    assert window(ev, T0 + timedelta(days=1), T0 + timedelta(days=3)).height == 2
    assert window(ev, T0 + timedelta(days=2), None).height == 2


def test_gate_needs_better_ndcg_and_no_recall_loss():
    bar = {"ndcg@10": 0.02, "recall@200": 0.10}
    assert gate({"ndcg@10": 0.03, "recall@200": 0.10}, bar)[0]
    assert not gate({"ndcg@10": 0.02, "recall@200": 0.10}, bar)[0]  # equal is not better
    assert not gate({"ndcg@10": 0.03, "recall@200": 0.09}, bar)[0]  # recall got worse


def synthetic_frame(n_users=80, n_items=20, seed=0):
    """Every user has 20 candidates; the item feature i_w_all decides which ones are positives."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_users):
        signal = rng.random(n_items)
        for i in range(n_items):
            row = {"visitorid": u, "itemid": i, "gid": f"s-{u}", "a_rank": i + 1}
            row.update({c: float(rng.random()) for c in FEATURE_COLUMNS})
            row.update(i_w_all=float(signal[i]), a_rank=i + 1, label=int(signal[i] > 0.9))
            rows.append(row)
    return pl.DataFrame(rows)


def test_ranker_learns_to_put_positives_first_and_retrieved_order_is_kept():
    frame = synthetic_frame()
    train, hold = frame.filter(pl.col("visitorid") < 60), frame.filter(pl.col("visitorid") >= 60)
    booster, stats = fit_ranker(train, hold)
    assert stats["train_users_with_positive"] > 0
    ranked = rerank(hold, booster)
    positives = hold.filter(pl.col("label") == 1).group_by("visitorid").agg(pl.col("itemid"))
    hits = sum(
        ranked[u][0] in set(items) for u, items in zip(positives["visitorid"], positives["itemid"])
    )
    assert hits / positives.height > 0.8  # top-1 is a positive for most users
    assert retrieved_order(hold)[60] == sorted(hold.filter(pl.col("visitorid") == 60)["itemid"])
