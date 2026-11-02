"""Pandera schemas generated from the feature list in spec.py."""

import pandera.polars as pa
import polars as pl

from src.features.spec import (  # noqa: F401  (re-exported: the rest of the code imports from here)
    BY_GROUP,
    DECAY_TAU_DAYS,
    FEATURE_COLUMNS,
    FEATURES,
    GROUPS,
    PRIOR_STRENGTH,
    RECENT_N,
    WINDOWS_DAYS,
    Feature,
    schema_dict,
    write_schema,
)

DTYPES = {"int": pl.Int64, "float": pl.Float64}


def table_schema(group: str, key: str | None = None) -> pa.DataFrameSchema:
    """Pandera schema of a feature table: optional unique key + the group's features."""
    columns = {}
    for f in BY_GROUP[group]:
        checks = []
        if f.min is not None:
            checks.append(pa.Check.ge(f.min))
        if f.max is not None:
            checks.append(pa.Check.le(f.max))
        columns[f.name] = pa.Column(DTYPES[f.dtype], checks, nullable=f.nullable)
    if key:
        columns[key] = pa.Column(pl.Int64)
    return pa.DataFrameSchema(columns, unique=[key] if key else None, strict=True)


def cast_to_schema(df: pl.DataFrame, group: str) -> pl.DataFrame:
    """Cast the group's feature columns to their declared types (polars counts are UInt32)."""
    return df.with_columns(pl.col(f.name).cast(DTYPES[f.dtype]) for f in BY_GROUP[group])
