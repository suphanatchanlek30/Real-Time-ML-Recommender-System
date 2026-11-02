"""FastAPI service.

    GET  /recommend?user_id=..&k=10   top-k items with the model version and where they came from
    GET  /health                      model version, Redis reachability, event-time clock
    GET  /metrics                     Prometheus metrics
    POST /admin/reload                load the current bundle (models/champion) without a restart

`source` of a recommendation:
    ranker            the Ranker ordered 200 candidates (ALS for known users, popular otherwise)
    popular_fallback  the Ranker could not run (error); the popular list was returned unranked
`retrieval` says where the 200 candidates came from (als / popular) and `degraded` is true when
Redis was slow or down, so user and pair features were left empty (the Ranker still runs).
"""

import logging
import os
import time
from contextlib import asynccontextmanager
from datetime import datetime

import redis
from fastapi import FastAPI, HTTPException, Query, Request, Response
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel

from src.serving.bundle import BundleError, load_bundle
from src.serving.config import Settings
from src.serving.feature_store import read_user
from src.serving.recommender import Recommender
from src.serving.tables import load_tables
from src.serving.timeutil import from_ms

log = logging.getLogger("api")
LATENCY_BUCKETS = (0.005, 0.01, 0.02, 0.035, 0.05, 0.075, 0.1, 0.15, 0.25, 0.5, 1.0)


class ScoredItem(BaseModel):
    item_id: int
    score: float | None


class Recommendation(BaseModel):
    items: list[ScoredItem]
    model_version: str
    source: str
    retrieval: str
    degraded: bool


def build_recommender(settings: Settings) -> Recommender:
    """Load the bundle and the item tables and check that they belong together."""
    bundle = load_bundle(settings.model_dir)
    tables = load_tables(settings.features_dir, settings.snapshot)
    trained_at = bundle.info.get("val_as_of")
    if trained_at and datetime.fromisoformat(trained_at) != datetime.fromisoformat(tables.as_of):
        raise BundleError(
            f"model was trained on the snapshot of {trained_at} but the feature tables are from "
            f"{tables.as_of}: run `make features` and `make pull-model` again"
        )
    return Recommender(bundle, tables, settings.n_candidates)


def create_app(
    settings: Settings | None = None,
    *,
    redis_client=None,
    recommender: Recommender | None = None,
    loader=None,
) -> FastAPI:
    settings = settings or Settings()
    loader = loader or (lambda: build_recommender(settings))
    registry = CollectorRegistry()  # one per app, so tests can create many apps
    m = {
        "latency": Histogram(
            "http_request_duration_seconds",
            "Request latency",
            ["endpoint"],
            buckets=LATENCY_BUCKETS,
            registry=registry,
        ),
        "requests": Counter(
            "http_requests_total", "Requests", ["endpoint", "status"], registry=registry
        ),
        "source": Counter(
            "recommend_source_total", "Recommendations", ["source", "retrieval"], registry=registry
        ),
        "redis_errors": Counter(
            "redis_errors_total", "Redis reads that failed or timed out", registry=registry
        ),
        "redis_latency": Histogram(
            "redis_read_seconds",
            "Redis read time per request",
            buckets=LATENCY_BUCKETS,
            registry=registry,
        ),
        "reloads": Counter("model_reloads_total", "Reload attempts", ["result"], registry=registry),
        "model": Gauge("model_info", "Loaded model version", ["version"], registry=registry),
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.rec = recommender
        app.state.redis = redis_client or redis.Redis.from_url(
            settings.redis_url,
            decode_responses=True,
            socket_timeout=settings.redis_timeout_s,
            socket_connect_timeout=settings.redis_timeout_s,
        )
        if app.state.rec is None:
            try:
                app.state.rec = loader()
            except Exception as err:  # start anyway: /health reports it, /admin/reload can fix it
                log.error("no model loaded at startup: %s", err)
        if app.state.rec:
            m["model"].labels(app.state.rec.bundle.version).set(1)
        yield

    app = FastAPI(title="Recommender API", lifespan=lifespan)

    @app.middleware("http")
    async def observe(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        endpoint = request.url.path
        m["latency"].labels(endpoint).observe(time.perf_counter() - start)
        m["requests"].labels(endpoint, response.status_code).inc()
        return response

    def model() -> Recommender:
        rec = app.state.rec
        if rec is None:
            raise HTTPException(503, "no model loaded")
        return rec

    @app.get("/recommend", response_model=Recommendation)
    def recommend(response: Response, user_id: int = Query(ge=0), k: int = Query(10, ge=1, le=100)):
        rec = model()  # one reference for the whole request, even if a reload happens meanwhile
        data, started = None, time.perf_counter()
        try:
            data = read_user(app.state.redis, user_id)
        except (redis.RedisError, OSError) as err:
            m["redis_errors"].inc()
            log.warning("redis unavailable, serving without user features: %s", err)
        m["redis_latency"].observe(time.perf_counter() - started)

        try:
            result = rec.recommend(user_id, k, data)
            items = [ScoredItem(item_id=i, score=s) for i, s in result.items]
            source, retrieval, degraded = "ranker", result.retrieval, result.degraded
        except Exception:
            log.exception("ranker failed, returning the popular list")
            items = [ScoredItem(item_id=i, score=None) for i in rec.bundle.popular[:k]]
            source, retrieval, degraded = "popular_fallback", "popular", data is None
        m["source"].labels(source, retrieval).inc()
        response.headers["X-Model-Version"] = rec.bundle.version
        return Recommendation(
            items=items,
            model_version=rec.bundle.version,
            source=source,
            retrieval=retrieval,
            degraded=degraded,
        )

    @app.get("/health")
    def health(response: Response):
        rec = app.state.rec
        clock, redis_ok = None, True
        try:
            app.state.redis.ping()
            raw = app.state.redis.get("clock")
            clock = from_ms(int(raw)).isoformat() if raw is not None else None
        except (redis.RedisError, OSError):
            redis_ok = False
        if rec is None:
            response.status_code = 503
        return {
            "status": "ok" if rec and redis_ok else ("no_model" if rec is None else "degraded"),
            "model_version": rec.bundle.version if rec else None,
            "snapshot_as_of": rec.tables.as_of if rec else None,
            "redis": redis_ok,
            "event_clock": clock,
        }

    @app.get("/metrics")
    def metrics():
        return Response(generate_latest(registry), media_type=CONTENT_TYPE_LATEST)

    @app.post("/admin/reload")
    def reload(request: Request):
        token = os.environ.get("ADMIN_TOKEN")
        if token and request.headers.get("X-Admin-Token") != token:
            raise HTTPException(403, "bad admin token")
        old = app.state.rec.bundle.version if app.state.rec else None
        try:
            new = loader()  # built completely first: a bad bundle never replaces a good one
        except Exception as err:
            m["reloads"].labels("failed").inc()
            raise HTTPException(409, f"reload failed, still serving v{old}: {err}") from err
        app.state.rec = new  # one assignment swaps the model
        m["reloads"].labels("ok").inc()
        m["model"].labels(new.bundle.version).set(1)
        if old and old != new.bundle.version:
            m["model"].labels(old).set(0)
        return {"previous_version": old, "model_version": new.bundle.version}

    return app


def app_factory() -> FastAPI:
    """Entry point for uvicorn: `uvicorn src.serving.app:app_factory --factory`."""
    logging.basicConfig(level=logging.INFO)
    return create_app()
