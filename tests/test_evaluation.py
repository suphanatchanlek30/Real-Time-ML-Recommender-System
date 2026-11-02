import math
from datetime import datetime, timedelta

import numpy as np
import polars as pl
import pytest

from src.evaluation.baselines import fit_popularity, random_recs
from src.evaluation.ground_truth import build_ground_truth, split_warm_cold
from src.evaluation.metrics import (
    average_precision_at_k,
    coverage_at_k,
    hit_rate_at_k,
    ndcg_at_k,
    per_user_metrics,
    recall_at_k,
    summarize,
)

# hits at rank 1 and rank 3; the user has 2 relevant items in total
RECS, TRUTH = [1, 2, 3, 4], {1, 3}


def test_recall_and_hit_rate():
    assert recall_at_k(RECS, TRUTH, 3) == 1.0
    assert recall_at_k(RECS, TRUTH, 1) == 0.5
    assert recall_at_k([9, 8], TRUTH, 2) == 0.0
    assert hit_rate_at_k(RECS, TRUTH, 1) == 1.0
    assert hit_rate_at_k([9, 8], TRUTH, 2) == 0.0


def test_ndcg_hand_computed():
    dcg = 1 / math.log2(2) + 1 / math.log2(4)  # hits at ranks 1 and 3
    ideal = 1 / math.log2(2) + 1 / math.log2(3)  # both hits at ranks 1 and 2
    assert ndcg_at_k(RECS, TRUTH, 3) == pytest.approx(dcg / ideal)


def test_ndcg_is_one_for_perfect_ranking_and_lower_when_hit_is_deeper():
    assert ndcg_at_k([1, 3, 7], TRUTH, 3) == pytest.approx(1.0)
    assert ndcg_at_k([7, 8, 1], {1}, 3) < ndcg_at_k([1, 8, 7], {1}, 3)


def test_average_precision_hand_computed():
    # precision@1 = 1, precision@3 = 2/3, divided by min(|truth|, k) = 2
    assert average_precision_at_k(RECS, TRUTH, 3) == pytest.approx((1 + 2 / 3) / 2)


def test_metrics_use_min_truth_k_so_perfect_top_k_scores_one():
    truth = {1, 2, 3, 4, 5}
    assert ndcg_at_k([1, 2], truth, 2) == pytest.approx(1.0)
    assert average_precision_at_k([1, 2], truth, 2) == pytest.approx(1.0)


def test_repeated_items_are_not_counted_twice():
    assert recall_at_k([1, 1, 1], {1, 2}, 3) == 0.5


def test_per_user_metrics_and_coverage():
    recs = {1: [10, 11], 2: [11, 12]}
    truth = {1: {10}, 2: {99}}
    m = per_user_metrics(recs, truth, [1, 2], ks=(2,))
    assert m["recall@2"].tolist() == [1.0, 0.0]
    assert coverage_at_k(recs, [1, 2], 2, n_items=6) == 0.5  # {10, 11, 12} of 6


def test_bootstrap_interval_contains_mean_and_is_reproducible():
    values = np.array([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0])
    a, b = summarize(values), summarize(values)
    assert a == b
    assert a["lo"] <= a["mean"] <= a["hi"]
    assert math.isnan(summarize(np.array([]))["mean"])


def make_events(rows):
    return pl.DataFrame(
        rows,
        schema={
            "ts": pl.Datetime("ms"),
            "visitorid": pl.Int64,
            "itemid": pl.Int64,
            "event": pl.String,
        },
        orient="row",
    )


def test_ground_truth_ignores_views():
    t = datetime(2015, 1, 1)
    df = make_events([(t, 1, 10, "view"), (t, 1, 11, "addtocart"), (t, 2, 12, "view")])
    assert build_ground_truth(df) == {1: {11}}


def test_split_warm_cold():
    t = datetime(2015, 1, 1)
    history = make_events([(t, 1, 10, "view")])
    assert split_warm_cold([1, 2], history) == ([1], [2])


def test_popularity_weights_actions_and_window_pads_with_all_time():
    end = datetime(2015, 3, 1)
    old, new = end - timedelta(days=30), end - timedelta(days=1)
    df = make_events(
        [
            (old, 1, 100, "view"),
            (old, 2, 100, "view"),
            (old, 3, 100, "view"),  # old favourite: 3 points
            (new, 4, 200, "transaction"),  # recent: 5 points
            (new, 5, 300, "view"),  # recent: 1 point
        ]
    )
    assert fit_popularity(df, n=3) == [200, 100, 300]  # whole history: 5, 3, 1
    assert fit_popularity(df, n=3, window_days=7) == [200, 300, 100]  # recent first, padded
    assert fit_popularity(df, n=2, window_days=7) == [200, 300]


def test_random_recs_are_reproducible_and_unique():
    a = random_recs([1, 2], list(range(100)), n=10, seed=1)
    assert a == random_recs([1, 2], list(range(100)), n=10, seed=1)
    assert len(set(a[1])) == 10
