"""Paths and tunable parameters for the ETL step."""

import os
from dataclasses import dataclass
from pathlib import Path

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

EVENT_TYPES = ("view", "addtocart", "transaction")


@dataclass(frozen=True)
class BotFilter:
    """Rules for dropping abnormal users (decided from notebooks/01_eda.ipynb).

    A user is dropped if either rule matches (event counts are over the whole dataset):
      1. hard cap: more events than the `hard_cap_quantile` of events-per-user
      2. view-only: more events than the `view_only_quantile` and no addtocart/transaction
    Heavy users who do buy are kept below the hard cap: the EDA shows they are not bots.
    """

    hard_cap_quantile: float = 0.9999
    view_only_quantile: float = 0.999


@dataclass(frozen=True)
class SplitConfig:
    """Time-based split counted back from the last event. Never split randomly."""

    test_days: int = 7
    val_days: int = 7
