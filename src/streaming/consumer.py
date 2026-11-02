"""Read events from Kafka and keep the user state in Redis up to date.

    python -m src.streaming.consumer

Per message: validate the schema, apply the event to the user's Redis state (the same
`update_user_state` the offline features are checked against) and move the event-time clock.
Bad messages are counted and skipped, never allowed to stop the stream. Offsets are committed
after each batch is written, so a crash re-reads at most one batch (replayed events past a
user's last event are skipped as late, which keeps this safe).
"""

import json
import logging
import signal
import time
from collections import Counter

import polars as pl
import redis
from pydantic import ValidationError

from src.serving.config import Settings
from src.serving.feature_store import Event, advance_clock, handle_event
from src.streaming import config as K

log = logging.getLogger("consumer")


def process_message(r, payload: bytes, item_cat) -> tuple[str, int | None]:
    """Apply one raw message. Returns ("ok" | "late" | "invalid", event time in ms)."""
    try:
        ev = Event.model_validate(json.loads(payload))
    except (ValueError, ValidationError):  # bad JSON and schema errors
        return "invalid", None
    return ("ok" if handle_event(r, ev, item_cat) else "late"), ev.ts


def main() -> None:
    from confluent_kafka import Consumer

    settings = Settings()
    items = pl.read_parquet(settings.features_dir / settings.snapshot / "item_features.parquet")
    item_cat = dict(zip(items["itemid"].to_list(), items["i_categoryid"].to_list()))
    r = redis.Redis.from_url(settings.redis_url, decode_responses=True)
    consumer = Consumer(
        {
            "bootstrap.servers": K.BOOTSTRAP,
            "group.id": K.GROUP,
            "enable.auto.commit": False,
            "auto.offset.reset": "earliest",
        }
    )
    consumer.subscribe([K.TOPIC])
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    stats, last_log = Counter(), time.time()
    log.info("consuming %s from %s as group %s", K.TOPIC, K.BOOTSTRAP, K.GROUP)
    while not stop["now"]:
        messages = consumer.consume(num_messages=500, timeout=1.0)
        newest = 0
        for msg in messages:
            if msg.error():
                stats["kafka_error"] += 1
                continue
            outcome, ts = process_message(r, msg.value(), item_cat)
            stats[outcome] += 1
            newest = max(newest, ts or 0)
        if newest:
            advance_clock(r, newest)
        if messages:
            consumer.commit(asynchronous=False)
        if time.time() - last_log > 10:
            log.info("processed %s", dict(stats))
            last_log = time.time()
    consumer.close()
    log.info("stopped: %s", dict(stats))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main()
