"""Item and category features as numpy arrays, built once at startup.

A request must not run polars joins over 221K items: it looks up 200 rows of this matrix.
"""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from src.features.spec import BY_GROUP, FEATURE_COLUMNS

ITEM_COLUMNS = [f.name for f in BY_GROUP["item"]] + [f.name for f in BY_GROUP["category"]]


@dataclass
class ItemTables:
    row: dict[int, int]  # item id -> row of `block`
    block: np.ndarray  # one row per item, plus a last all-NaN row for unknown items
    columns: list[int]  # position of each block column inside FEATURE_COLUMNS
    item_cat: dict[int, int]  # item id -> category id (-1 unknown), for p_cat_share
    as_of: str

    def lookup(self, items: np.ndarray) -> np.ndarray:
        unknown = len(self.block) - 1
        return self.block[[self.row.get(int(i), unknown) for i in items]]


def load_tables(directory: Path, snapshot: str) -> ItemTables:
    base = Path(directory) / snapshot
    items = pl.read_parquet(base / "item_features.parquet")
    cats = pl.read_parquet(base / "category_features.parquet").rename(
        {"categoryid": "i_categoryid"}
    )
    joined = items.join(cats, on="i_categoryid", how="left")
    matrix = joined.select(ITEM_COLUMNS).cast(pl.Float64).to_numpy()
    block = np.vstack([matrix, np.full((1, matrix.shape[1]), np.nan)])
    report = (Path(directory) / "features_report.json").read_text()
    import json

    as_of = json.loads(report)[snapshot]["as_of"]
    return ItemTables(
        row={int(i): n for n, i in enumerate(joined["itemid"])},
        block=block,
        columns=[FEATURE_COLUMNS.index(c) for c in ITEM_COLUMNS],
        item_cat=dict(zip(items["itemid"].to_list(), items["i_categoryid"].to_list())),
        as_of=as_of,
    )
