"""Prometheus metrics of the stream consumer (served on its own port, see CONSUMER_METRICS_PORT)."""

from prometheus_client import Counter, Gauge, Histogram

EVENTS = Counter("consumer_events_total", "Messages handled", ["outcome"])
EVENT_CLOCK = Gauge(
    "consumer_event_clock_seconds", "Event time of the newest processed event (epoch seconds)"
)
KAFKA_LAG = Gauge("consumer_kafka_lag", "Messages not yet consumed", ["partition"])
BATCH_SECONDS = Histogram(
    "consumer_batch_seconds",
    "Time to apply one batch of messages to Redis",
    buckets=(0.01, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)


def kafka_lag(consumer) -> dict[int, int]:
    """Per assigned partition: newest offset in the topic minus the next offset to read."""
    lag = {}
    for tp in consumer.assignment():
        position = consumer.position([tp])[0].offset
        _, high = consumer.get_watermark_offsets(tp, cached=True)
        if position >= 0:  # negative = no position yet
            lag[tp.partition] = max(high - position, 0)
    return lag
