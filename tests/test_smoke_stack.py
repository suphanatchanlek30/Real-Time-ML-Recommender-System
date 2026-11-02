"""End to end against the running stack: Kafka -> consumer -> Redis -> API.

    make up && make replay        # stack running with the batch state loaded
    make smoke                    # RUN_STACK_TESTS=1 pytest tests/test_smoke_stack.py

Skipped unless RUN_STACK_TESTS=1, so `make test` passes without Docker.
"""

import json
import os
import time

import httpx
import pytest
import redis

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_STACK_TESTS") != "1", reason="set RUN_STACK_TESTS=1 with the stack running"
)

from src.serving.config import Settings  # noqa: E402  (also loads .env)
from src.streaming.config import TOPIC  # noqa: E402

API = os.environ.get("API_URL", f"http://localhost:{os.environ.get('API_PORT', '8000')}")
REDIS_URL = Settings().redis_url
BOOTSTRAP = os.environ.get(
    "KAFKA_BOOTSTRAP", f"localhost:{os.environ.get('KAFKA_EXTERNAL_PORT', '9094')}"
)


def test_health_reports_a_model_and_a_reachable_redis():
    body = httpx.get(f"{API}/health").json()
    assert body["status"] == "ok" and body["redis"] is True and body["model_version"]


def test_recommend_contract_for_known_and_new_users():
    for user in (9145, 987654321):
        r = httpx.get(f"{API}/recommend", params={"user_id": user, "k": 10})
        body = r.json()
        assert r.status_code == 200 and len(body["items"]) == 10
        assert body["source"] == "ranker" and r.headers["X-Model-Version"] == body["model_version"]


def test_metrics_are_exposed():
    text = httpx.get(f"{API}/metrics").text
    assert "http_request_duration_seconds_bucket" in text and "recommend_source_total" in text


def test_event_sent_to_kafka_reaches_redis_and_the_api_sees_it():
    from confluent_kafka import Producer

    r = redis.Redis.from_url(REDIS_URL, decode_responses=True)
    visitor = 9_000_000_000 + int(time.time()) % 1_000_000  # a user that exists nowhere else
    ts = int(r.get("clock") or 1) + 1000  # event time after the newest event seen so far
    producer = Producer({"bootstrap.servers": BOOTSTRAP})
    message = {"visitor_id": visitor, "item_id": 5, "event": "view", "ts": ts}
    producer.produce(TOPIC, key=str(visitor), value=json.dumps(message))
    producer.flush(10)

    deadline = time.time() + 20
    while time.time() < deadline and not r.exists(f"user:{visitor}:state"):
        time.sleep(0.3)
    assert r.hget(f"user:{visitor}:state", "n_view") == "1", "event did not reach Redis in 20 s"
    assert r.hget(f"user:{visitor}:items", "v:5") == "1"
    assert httpx.get(f"{API}/recommend", params={"user_id": visitor}).status_code == 200
