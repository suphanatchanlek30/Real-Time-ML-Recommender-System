"""Build the feature tables for every snapshot.

    python -m src.features.build        (or: make features)

Writes to <DATA_DIR>/features:
    schema.json                     feature names, types, order, parameters (saved with the model)
    <snapshot>/user_state.parquet   raw per-user state (what Redis holds)
    <snapshot>/user_features.parquet, item_features.parquet, category_features.parquet
    features_report.json            row counts and as_of of each snapshot
User x item pair features are computed at candidate-scoring time (src/features/pair.py).
"""

import json
import logging

import polars as pl

from src.etl.config import DATA_DIR, PROCESSED_DIR, RAW_DIR
from src.features.item import category_features, item_attributes, item_features
from src.features.schema import cast_to_schema, table_schema, write_schema
from src.features.snapshots import Snapshot, make_snapshots
from src.features.user import derive_user_features, user_state

FEATURES_DIR = DATA_DIR / "features"
log = logging.getLogger("features")


def build_snapshot(
    events: pl.DataFrame, props: pl.DataFrame, tree: pl.DataFrame, snap: Snapshot
) -> dict[str, pl.DataFrame]:
    """All feature tables of one snapshot, using only events before snap.as_of."""
    state = user_state(events, snap.as_of)
    users = cast_to_schema(derive_user_features(state, snap.as_of), "user")
    attrs = item_attributes(props, tree, snap.as_of)
    items = cast_to_schema(item_features(events, attrs, snap.as_of), "item")
    cats = cast_to_schema(category_features(items), "category")
    return {
        "user_state": state,
        "user_features": table_schema("user", "visitorid").validate(users),
        "item_features": table_schema("item", "itemid").validate(items),
        "category_features": table_schema("category", "categoryid").validate(cats),
    }


def main() -> dict:
    events = pl.read_parquet(PROCESSED_DIR / "events.parquet")
    props = pl.read_parquet(PROCESSED_DIR / "item_properties.parquet")
    tree = pl.read_csv(RAW_DIR / "category_tree.csv")

    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    write_schema(FEATURES_DIR / "schema.json")

    report = {}
    for snap in make_snapshots(events):
        tables = build_snapshot(events, props, tree, snap)
        out = FEATURES_DIR / snap.name
        out.mkdir(exist_ok=True)
        for name, df in tables.items():
            df.write_parquet(out / f"{name}.parquet", compression="zstd")
        report[snap.name] = {
            "as_of": str(snap.as_of),
            "label_start": str(snap.label_start),
            "label_end": str(snap.label_end),
            **{name: df.height for name, df in tables.items()},
        }
        log.info(
            "%s as_of=%s %s",
            snap.name,
            snap.as_of,
            {k: v for k, v in report[snap.name].items() if isinstance(v, int)},
        )
    (FEATURES_DIR / "features_report.json").write_text(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    print(json.dumps(main(), indent=2))
