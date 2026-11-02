"""Kafka settings shared by the producer and the consumer."""

import os

BOOTSTRAP = os.environ.get("KAFKA_BOOTSTRAP", "localhost:9094")
TOPIC = os.environ.get("KAFKA_TOPIC", "user-events")
GROUP = os.environ.get("KAFKA_GROUP", "feature-updater")
PARTITIONS = 3  # events of one visitor share a key, hence a partition, hence stay in order
