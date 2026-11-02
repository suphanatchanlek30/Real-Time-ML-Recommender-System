"""Kafka settings shared by the producer and the consumer (from .env or Compose, see src/env.py)."""

import os

from src.env import load_env

load_env()

# from the Mac the broker is reached on the published port; Compose sets kafka:9092 in containers
BOOTSTRAP = os.environ.get(
    "KAFKA_BOOTSTRAP", f"localhost:{os.environ.get('KAFKA_EXTERNAL_PORT', '9094')}"
)
TOPIC = os.environ.get("KAFKA_TOPIC", "user-events")
GROUP = os.environ.get("KAFKA_GROUP", "feature-updater")
# events of one visitor share a key, hence a partition, hence stay in order
PARTITIONS = int(os.environ.get("KAFKA_PARTITIONS", "3"))
# the consumer serves Prometheus metrics on this port (inside its container)
METRICS_PORT = int(os.environ.get("CONSUMER_METRICS_PORT", "9100"))
