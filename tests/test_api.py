"""API contract, fallbacks and reload, with a tiny synthetic model and fakeredis (no Docker)."""

import fakeredis
import lightgbm as lgb
import numpy as np
import pytest
import redis
from fastapi.testclient import TestClient

from src.features.spec import FEATURE_COLUMNS
from src.serving.app import create_app
from src.serving.bundle import Bundle, BundleError
from src.serving.feature_store import Event, advance_clock, handle_event
from src.serving.recommender import Recommender
from src.serving.tables import ITEM_COLUMNS, ItemTables
from src.serving.timeutil import to_ms
from tests.test_features import T0

AS_OF = "2015-09-04 02:59:47.788000"


def synthetic_recommender(version="1", n_items=40, n_users=10, seed=0) -> Recommender:
    rng = np.random.default_rng(seed)
    items = np.arange(100, 100 + n_items)
    X = rng.random((600, len(FEATURE_COLUMNS)))
    booster = lgb.train(
        {"objective": "lambdarank", "verbosity": -1, "min_data_in_leaf": 5, "seed": 1},
        lgb.Dataset(X, label=rng.integers(0, 2, 600), group=[20] * 30),
        num_boost_round=5,
    )
    bundle = Bundle(
        user_ids=np.arange(1, n_users + 1),
        item_ids=items,
        user_factors=rng.random((n_users, 8), dtype=np.float32),
        item_factors=rng.random((n_items, 8), dtype=np.float32),
        booster=booster,
        popular=[int(i) for i in items[:30]],
        version=version,
        info={},
    )
    block = np.vstack(
        [rng.random((n_items, len(ITEM_COLUMNS))), np.full((1, len(ITEM_COLUMNS)), np.nan)]
    )
    tables = ItemTables(
        row={int(i): n for n, i in enumerate(items)},
        block=block,
        columns=[FEATURE_COLUMNS.index(c) for c in ITEM_COLUMNS],
        item_cat={int(i): int(i) % 4 for i in items},
        as_of=AS_OF,
    )
    return Recommender(bundle, tables, n_candidates=20)


class BrokenRedis:
    """Every call fails like a Redis that is down."""

    def __getattr__(self, name):
        def fail(*args, **kwargs):
            raise redis.ConnectionError("redis is down")

        return fail


@pytest.fixture
def fake():
    return fakeredis.FakeStrictRedis(decode_responses=True)


@pytest.fixture
def client(fake):
    app = create_app(redis_client=fake, recommender=synthetic_recommender())
    with TestClient(app) as c:
        yield c


def test_recommend_contract(client):
    r = client.get("/recommend", params={"user_id": 3, "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"items", "model_version", "source", "retrieval", "degraded"}
    assert len(body["items"]) == 5 and set(body["items"][0]) == {"item_id", "score"}
    assert body["model_version"] == "1" and r.headers["X-Model-Version"] == "1"
    assert body["source"] == "ranker" and body["retrieval"] == "als" and body["degraded"] is False
    scores = [i["score"] for i in body["items"]]
    assert scores == sorted(scores, reverse=True)
    assert len({i["item_id"] for i in body["items"]}) == 5


def test_unknown_user_gets_popular_candidates_ranked(client):
    body = client.get("/recommend", params={"user_id": 999999, "k": 10}).json()
    assert body["retrieval"] == "popular" and body["source"] == "ranker"
    assert all(100 <= i["item_id"] < 130 for i in body["items"])  # from the popular list


def test_default_k_and_validation(client):
    assert len(client.get("/recommend", params={"user_id": 3}).json()["items"]) == 10
    for params in ({"user_id": 3, "k": 0}, {"user_id": 3, "k": 101}, {"user_id": -1}, {}):
        assert client.get("/recommend", params=params).status_code == 422


def test_redis_down_still_answers_but_is_degraded():
    app = create_app(redis_client=BrokenRedis(), recommender=synthetic_recommender())
    with TestClient(app) as c:
        body = c.get("/recommend", params={"user_id": 3}).json()
        assert body["degraded"] is True and body["source"] == "ranker" and len(body["items"]) == 10
        assert "redis_errors_total 1.0" in c.get("/metrics").text
        health = c.get("/health").json()
        assert health["status"] == "degraded" and health["redis"] is False


def test_ranker_error_falls_back_to_the_popular_list(client, monkeypatch):
    rec = client.app.state.rec
    monkeypatch.setattr(rec, "recommend", lambda *a, **k: 1 / 0)
    body = client.get("/recommend", params={"user_id": 3, "k": 4}).json()
    assert body["source"] == "popular_fallback"
    assert [i["item_id"] for i in body["items"]] == list(range(100, 104))
    assert all(i["score"] is None for i in body["items"])


def test_stream_events_change_the_recommendation(client, fake):
    before = client.get("/recommend", params={"user_id": 999999, "k": 20}).json()["items"]
    ts = to_ms(T0)
    for n, item in enumerate(range(125, 130)):
        handle_event(fake, Event(visitor_id=999999, item_id=item, event="addtocart", ts=ts + n), {})
    advance_clock(fake, ts + 10)
    after = client.get("/recommend", params={"user_id": 999999, "k": 20}).json()["items"]
    assert [i["item_id"] for i in before] != [i["item_id"] for i in after]


def test_health_and_metrics(client):
    health = client.get("/health").json()
    assert health["status"] == "ok" and health["model_version"] == "1"
    assert health["redis"] is True and health["snapshot_as_of"] == AS_OF
    client.get("/recommend", params={"user_id": 3})
    text = client.get("/metrics").text
    assert 'recommend_source_total{retrieval="als",source="ranker"} 1.0' in text
    assert 'http_requests_total{endpoint="/recommend",status="200"} 1.0' in text
    assert "http_request_duration_seconds_bucket" in text and 'model_info{version="1"} 1.0' in text


def test_reload_swaps_the_model_and_a_bad_bundle_keeps_the_old_one(fake):
    state = {"next": synthetic_recommender(version="2", seed=1)}

    def loader():
        if isinstance(state["next"], Exception):
            raise state["next"]
        return state["next"]

    app = create_app(redis_client=fake, recommender=synthetic_recommender(), loader=loader)
    with TestClient(app) as c:
        assert c.post("/admin/reload").json() == {"previous_version": "1", "model_version": "2"}
        assert c.get("/recommend", params={"user_id": 3}).json()["model_version"] == "2"
        state["next"] = BundleError("feature order differs")
        r = c.post("/admin/reload")
        assert r.status_code == 409 and "still serving v2" in r.json()["detail"]
        assert c.get("/recommend", params={"user_id": 3}).json()["model_version"] == "2"
        assert 'model_reloads_total{result="ok"} 1.0' in c.get("/metrics").text


def test_admin_token_is_enforced_when_set(fake, monkeypatch):
    monkeypatch.setenv("ADMIN_TOKEN", "secret")
    app = create_app(
        redis_client=fake,
        recommender=synthetic_recommender(),
        loader=lambda: synthetic_recommender("2"),
    )
    with TestClient(app) as c:
        assert c.post("/admin/reload").status_code == 403
        assert c.post("/admin/reload", headers={"X-Admin-Token": "secret"}).status_code == 200


def test_no_model_is_reported_not_crashed(fake):
    def loader():
        raise BundleError("missing")

    app = create_app(redis_client=fake, loader=loader)
    with TestClient(app) as c:
        assert (
            c.get("/health").status_code == 503 and c.get("/health").json()["status"] == "no_model"
        )
        assert c.get("/recommend", params={"user_id": 1}).status_code == 503


def test_unhandled_errors_are_counted_as_500_and_in_progress_returns_to_zero(fake):
    app = create_app(redis_client=fake, recommender=synthetic_recommender())

    @app.get("/boom")
    def boom():
        raise RuntimeError("boom")

    with TestClient(app, raise_server_exceptions=False) as c:
        assert c.get("/boom").status_code == 500
        text = c.get("/metrics").text
        assert 'http_requests_total{endpoint="/boom",status="500"} 1.0' in text
        assert "http_requests_in_progress 1.0" in text  # only the /metrics request itself
        assert 'http_request_duration_seconds_bucket{endpoint="/boom",le="5.0"}' in text
