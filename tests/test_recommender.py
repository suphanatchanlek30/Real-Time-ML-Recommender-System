"""The shared-answer cache of the online recommender must never change an answer."""

import pytest

from src.serving.feature_store import UserData
from tests.test_api import synthetic_recommender

NEW_USER = UserData(state=None, items={}, cats={}, clock_ms=1_441_335_587_788)
WITH_STATE = {
    "n_view": 2,
    "n_cart": 0,
    "n_buy": 0,
    "n_sessions": 1,
    "first_ts": None,
    "last_ts": None,
    "decay_interest": 1.0,
    "decay_actions": 0.0,
}


class CountingBooster:
    """Wraps the model and counts how often the ranker really runs."""

    def __init__(self, booster):
        self.booster, self.calls = booster, 0

    def predict(self, X):
        self.calls += 1
        return self.booster.predict(X)


@pytest.fixture
def rec():
    r = synthetic_recommender()
    r.bundle.booster = CountingBooster(r.bundle.booster)
    return r


def test_new_users_share_one_ranking_that_is_computed_once(rec):
    first = rec.recommend(999_999, 10, NEW_USER)
    again = rec.recommend(888_888, 10, NEW_USER)
    assert rec.bundle.booster.calls == 1
    assert first.items == again.items and first.retrieval == "popular" and not first.degraded


@pytest.mark.parametrize("k", [1, 5, 20])
def test_cached_answer_equals_a_fresh_computation_for_any_k(rec, k):
    rec.recommend(999_999, 10, NEW_USER)  # fills the cache with the full ranking
    cached = rec.recommend(999_999, k, NEW_USER)
    fresh = synthetic_recommender()  # same seed, empty cache
    assert cached.items == fresh.recommend(999_999, k, NEW_USER).items


def test_users_with_history_are_never_served_from_the_shared_answer(rec):
    with_history = UserData(state=None, items={"v:105": "2"}, cats={}, clock_ms=NEW_USER.clock_ms)
    rec.recommend(999_999, 10, NEW_USER)
    rec.recommend(777_777, 10, with_history)
    rec.recommend(777_777, 10, with_history)
    assert rec.bundle.booster.calls == 3  # the two users with history were each computed


def test_als_users_are_never_cached(rec):
    rec.recommend(3, 10, NEW_USER)
    rec.recommend(3, 10, NEW_USER)
    assert rec.bundle.booster.calls == 2


def test_redis_down_has_its_own_shared_answer(rec):
    healthy = rec.recommend(999_999, 10, NEW_USER)
    degraded = rec.recommend(999_999, 10, None)
    assert degraded.degraded and not healthy.degraded
    assert rec.recommend(555_555, 10, None).items == degraded.items
    assert rec.bundle.booster.calls == 2


def test_a_new_model_starts_with_an_empty_cache():
    old, new = synthetic_recommender(seed=0), synthetic_recommender(version="2", seed=5)
    assert old.recommend(999_999, 10, NEW_USER).items != new.recommend(999_999, 10, NEW_USER).items
