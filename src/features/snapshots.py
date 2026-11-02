"""Point-in-time snapshots.

A feature table is always "what was known before `as_of`": every feature function keeps only
events with ts < as_of, so nothing from the period we later predict can leak in.

    rank_1..N    as_of = val_start - 7k days  labels: the 7 days after as_of, all inside Train
                 (the Ranker learns on these: features from the past, labels from the "future";
                 several rolling snapshots give it enough users with a positive label)
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
    events: pl.DataFrame,
    cfg: SplitConfig = SplitConfig(),
    label_days: int = 7,
    n_rank: int = 4,
) -> list[Snapshot]:
    val_start, test_start = split_boundaries(events, cfg)
    rank = []
    for k in range(n_rank, 0, -1):  # oldest first: rank_N ... rank_1
        start = val_start - timedelta(days=label_days * k)
        rank.append(Snapshot(f"rank_{k}", start, start, start + timedelta(days=label_days)))
    return [
        *rank,
        Snapshot("val", val_start, val_start, test_start),
        Snapshot("test", test_start, test_start, None),
    ]
