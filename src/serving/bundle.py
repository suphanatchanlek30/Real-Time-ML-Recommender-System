"""The model bundle written by training (`make train`) and fetched by `make pull-model`.

models/champion/
    als.npz  ranker.txt  schema.json  popular.json  etl_report.json   (from MLflow, alias champion)
    VERSION.json                                                     (written by pull-model)
"""

import json
from dataclasses import dataclass
from pathlib import Path

import lightgbm as lgb
import numpy as np

from src.features.spec import FEATURE_COLUMNS


class BundleError(RuntimeError):
    pass


@dataclass
class Bundle:
    user_ids: np.ndarray
    item_ids: np.ndarray
    user_factors: np.ndarray
    item_factors: np.ndarray
    booster: lgb.Booster
    popular: list[int]
    version: str
    info: dict

    def __post_init__(self):
        self.user_row = {int(u): i for i, u in enumerate(self.user_ids)}


def load_bundle(path: Path) -> Bundle:
    """Read everything and check it before returning, so a bad bundle never replaces a good one."""
    path = Path(path)
    try:
        als = np.load(path / "als.npz")
        schema = json.loads((path / "schema.json").read_text())
        info = json.loads((path / "VERSION.json").read_text())
        popular = json.loads((path / "popular.json").read_text())
        booster = lgb.Booster(model_file=str(path / "ranker.txt"))
    except FileNotFoundError as err:
        raise BundleError(f"incomplete model bundle in {path}: {err.filename} missing") from err
    # the model was trained on these columns in this order: refuse anything else
    if schema["feature_order"] != FEATURE_COLUMNS:
        raise BundleError("feature order in the bundle differs from src/features/spec.py")
    if booster.num_feature() != len(FEATURE_COLUMNS):
        raise BundleError("ranker was trained with a different number of features")
    return Bundle(
        user_ids=als["user_ids"],
        item_ids=als["item_ids"],
        user_factors=als["user_factors"],
        item_factors=als["item_factors"],
        booster=booster,
        popular=[int(i) for i in popular],
        version=str(info["version"]),
        info=info,
    )
