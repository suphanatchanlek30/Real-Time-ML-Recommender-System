"""Point-in-time snapshots.

A feature table is always "what was known before `as_of`": every feature function keeps only
events with ts < as_of, so nothing from the period we later predict can leak in.

    rank_train   as_of = val_start - 7 days   labels: the last 7 days of Train
                 (the Ranker learns on this: features from the past, labels from the "future")
    val          as_of = val_start            labels: the Val period (model selection)
    test         as_of = test_start           labels: the Test period (history = Train + Val)

Boundaries come from the same function the ETL split uses, so features and evaluation
cannot drift apart. `as_of` is a plain parameter: more rolling snapshots can be added later.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import polars as pl

from src.etl.config import SplitConfig
from src.etl.split import split_boundaries


@dataclass(frozen=True)
class Snapshot:
    name: str
    as_of: datetime  # features use events strictly before this
    label_start: datetime  # labels (what the user does next) in [label_start, label_end)
    label_end: datetime | None  # None = until the last event


def make_snapshots(
    events: pl.DataFrame, cfg: SplitConfig = SplitConfig(), label_days: int = 7
) -> list[Snapshot]:
    val_start, test_start = split_boundaries(events, cfg)
    rank_start = val_start - timedelta(days=label_days)
    return [
        Snapshot("rank_train", rank_start, rank_start, val_start),
        Snapshot("val", val_start, val_start, test_start),
        Snapshot("test", test_start, test_start, None),
    ]
