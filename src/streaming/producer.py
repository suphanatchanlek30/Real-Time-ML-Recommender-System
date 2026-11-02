"""Replay recorded events into Kafka to imitate live traffic.

    python -m src.streaming.producer [--rate 2000] [--limit N] [--reset-topic]

Events from the snapshot time onwards (default: the time the batch state in Redis stops at) are
sent in time order, keyed by visitor_id, as JSON {visitor_id, item_id, event, ts}.
`--reset-topic` deletes and recreates the topic so a second replay does not mix with the first.
"""

import argparse
import json
import time
from datetime import datetime

import polars as pl

from src.etl.config import PROCESSED_DIR
from src.serving.config import Settings
from src.streaming import config as K


def snapshot_start(settings: Settings) -> datetime:
    report = json.loads((settings.features_dir / "features_report.json").read_text())
    return datetime.fromisoformat(report[settings.snapshot]["as_of"])


def load_events(start: datetime, limit: int | None = None) -> pl.DataFrame:
    """Events at or after `start`, ordered by time (ties by item) like the batch state."""
    ev = pl.read_parquet(PROCESSED_DIR / "events.parquet").filter(pl.col("ts") >= start)
    ev = ev.sort("ts", "itemid")
    return ev.head(limit) if limit else ev


def to_message(visitor_id: int, item_id: int, event: str, ts_ms: int) -> tuple[bytes, bytes]:
    """(key, value) of one Kafka message."""
    value = {"visitor_id": visitor_id, "item_id": item_id, "event": event, "ts": ts_ms}
    return str(visitor_id).encode(), json.dumps(value).encode()


def reset_topic(admin) -> None:
    from confluent_kafka.admin import NewTopic

    if K.TOPIC in admin.list_topics(timeout=10).topics:
        admin.delete_topics([K.TOPIC])[K.TOPIC].result()
        for _ in range(50):  # deletion is asynchronous: wait until the topic is gone
            if K.TOPIC not in admin.list_topics(timeout=10).topics:
                break
            time.sleep(0.2)
    admin.create_topics([NewTopic(K.TOPIC, num_partitions=K.PARTITIONS, replication_factor=1)])[
        K.TOPIC
    ].result()


def main(rate: float = 0, limit: int | None = None, reset: bool = False) -> dict:
    from confluent_kafka import Producer
    from confluent_kafka.admin import AdminClient

    settings = Settings()
    start = snapshot_start(settings)
    events = load_events(start, limit)
    if reset:
        reset_topic(AdminClient({"bootstrap.servers": K.BOOTSTRAP}))

    producer = Producer(
        {"bootstrap.servers": K.BOOTSTRAP, "linger.ms": 20, "compression.type": "lz4"}
    )
    rows = events.select("visitorid", "itemid", "event", pl.col("ts").dt.epoch("ms")).iter_rows()
    began, sent = time.time(), 0
    for visitor, item, kind, ts_ms in rows:
        key, value = to_message(visitor, item, kind, ts_ms)
        while True:
            try:
                producer.produce(K.TOPIC, key=key, value=value)
                break
            except BufferError:  # local queue full: let the client send, then retry
                producer.poll(0.1)
        producer.poll(0)
        sent += 1
        if rate and sent % 100 == 0:  # keep the average speed at `rate` events per second
            ahead = sent / rate - (time.time() - began)
            if ahead > 0:
                time.sleep(ahead)
    producer.flush()
    seconds = round(time.time() - began, 1)
    return {
        "sent": sent,
        "from": str(start),
        "seconds": seconds,
        "topic": K.TOPIC,
        "events_per_second": round(sent / max(seconds, 1e-9)),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--rate", type=float, default=0, help="events per second (0 = as fast as possible)"
    )
    parser.add_argument("--limit", type=int, default=None, help="stop after this many events")
    parser.add_argument("--reset-topic", action="store_true")
    args = parser.parse_args()
    print(json.dumps(main(args.rate, args.limit, args.reset_topic), indent=2))
