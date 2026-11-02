"""Read events from Kafka and keep the user state in Redis up to date.

    python -m src.streaming.consumer

Per message: validate the schema, apply the event to the user's Redis state (the same
`update_user_state` the offline features are checked against) and move the event-time clock.
Bad messages are counted and skipped, never allowed to stop the stream. Offsets are committed
after each batch is written; a crash re-reads at most one batch, and the offsets stored in Redis
(see feature_store.handle_event) make that harmless: no event is applied twice.
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


def process_message(r, payload: bytes, item_cat, position=None) -> tuple[str, int | None]:
    """Apply one raw message. Returns ("ok" | "late" | "duplicate" | "invalid", event time ms)."""
    try:
        ev = Event.model_validate(json.loads(payload))
    except (ValueError, ValidationError):  # bad JSON and schema errors
        return "invalid", None
    return handle_event(r, ev, item_cat, position), ev.ts


def ensure_topic() -> None:
    """Create the topic if it does not exist yet (the consumer may start before the producer)."""
    from confluent_kafka.admin import AdminClient, NewTopic

    admin = AdminClient({"bootstrap.servers": K.BOOTSTRAP})
    if K.TOPIC not in admin.list_topics(timeout=30).topics:
        futures = admin.create_topics([NewTopic(K.TOPIC, K.PARTITIONS, replication_factor=1)])
        try:
            futures[K.TOPIC].result()
        except Exception as err:  # created by someone else in the meantime
            log.info("topic not created: %s", err)


def main() -> None:
    from confluent_kafka import Consumer, KafkaError, KafkaException

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
    ensure_topic()
    consumer.subscribe([K.TOPIC])
    stop = {"now": False}
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))

    stats, last_log, commit_failures = Counter(), time.time(), 0
    log.info("consuming %s from %s as group %s", K.TOPIC, K.BOOTSTRAP, K.GROUP)
    while not stop["now"]:
        messages = consumer.consume(num_messages=500, timeout=1.0)
        newest, handled = 0, 0
        for msg in messages:
            if msg.error():
                stats["kafka_error"] += 1
                if stats["kafka_error"] <= 3 or stats["kafka_error"] % 100 == 0:
                    log.warning("kafka: %s", msg.error())
                continue
            handled += 1
            position = (msg.topic(), msg.partition(), msg.offset())
            outcome, ts = process_message(r, msg.value(), item_cat, position)
            stats[outcome] += 1
            newest = max(newest, ts or 0)
        if newest:
            advance_clock(r, newest)
        if handled:
            try:
                consumer.commit(asynchronous=False)
                commit_failures = 0
            except KafkaException as err:
                # the offsets stored in Redis keep processing idempotent either way, so a
                # failed commit is not fatal -- unless it keeps failing (stale partition state
                # after the topic was recreated): then exit and let the container restart
                if err.args[0].code() != KafkaError._NO_OFFSET:
                    raise
                commit_failures += 1
                if commit_failures >= 20:
                    raise RuntimeError("offsets cannot be committed, restarting") from err
        if time.time() - last_log > 10:
            log.info("processed %s", dict(stats))
            last_log = time.time()
    consumer.close()
    log.info("stopped: %s", dict(stats))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main()
