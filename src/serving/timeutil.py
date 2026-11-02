"""Epoch-millisecond helpers. All times are naive UTC, like the ETL output."""

from datetime import datetime, timedelta

EPOCH = datetime(1970, 1, 1)
DAY_MS = 86_400_000


def to_ms(dt: datetime) -> int:
    return (dt - EPOCH) // timedelta(milliseconds=1)


def from_ms(ms: int) -> datetime:
    return EPOCH + timedelta(milliseconds=int(ms))
