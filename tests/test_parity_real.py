"""The real champion model and the real data: the API path equals the offline training path.

Skipped unless `make pull-model` and `make features` have been run. Uses fakeredis loaded with
the state of a sample of validation users.
"""

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest

fakeredis = pytest.importorskip("fakeredis")
pytest.importorskip("lightgbm")
pytest.importorskip("scipy")

MODEL_DIR, FEATURES = Path("models/champion"), Path("data/features")
EVENTS = Path("data/processed/events.parquet")
pytestmark = pytest.mark.skipif(
    not (MODEL_DIR / "VERSION.json").exists()
    or not (FEATURES / "val").exists()
    or not EVENTS.exists(),
    reason="needs models/champion (make pull-model) and data/features (make features)",
)

from src.evaluation.ground_truth import build_ground_truth  # noqa: E402
from src.features.spec import FEATURE_COLUMNS  # noqa: E402
from src.serving.bundle import load_bundle  # noqa: E402
from src.serving.feature_store import load_batch, read_user  # noqa: E402
from src.serving.load_redis import batch_tables  # noqa: E402
from src.serving.recommender import Recommender  # noqa: E402
from src.serving.tables import load_tables  # noqa: E402
from src.serving.timeutil import to_ms  # noqa: E402
from src.training.als import AlsModel  # noqa: E402
from src.training.dataset import make_frame  # noqa: E402
from src.training.ranker import rerank  # noqa: E402
from src.training.train import window  # noqa: E402


def test_online_features_and_ranking_equal_offline():
    bundle = load_bundle(MODEL_DIR)
    tables = load_tables(FEATURES, "val")
    as_of = datetime.fromisoformat(tables.as_of)
    events = pl.read_parquet(EVENTS)

    val_end = datetime.fromisoformat(
        json.loads((FEATURES / "features_report.json").read_text())["val"]["label_end"]
    )
    truth = build_ground_truth(window(events, as_of, val_end))  # the Val week
    ranked_users = sorted(truth)
    known = [u for u in ranked_users if u in bundle.user_row][:60]
    others = [u for u in ranked_users if u not in bundle.user_row][:60]
    users = known + others
    assert known and others

    # online side: Redis holds the batch state of these users as of the snapshot
    items = pl.read_parquet(FEATURES / "val" / "item_features.parquet")
    state = pl.read_parquet(FEATURES / "val" / "user_state.parquet").filter(
        pl.col("visitorid").is_in(users)
    )
    touched, categories = batch_tables(events, items, as_of)
    touched = touched.filter(pl.col("visitorid").is_in(users))
    categories = categories.filter(pl.col("visitorid").is_in(users))
    r = fakeredis.FakeStrictRedis(decode_responses=True)
    load_batch(r, state, touched, categories, to_ms(as_of))
    rec = Recommender(bundle, tables)

    # offline side: exactly what training/evaluation does
    als = AlsModel(bundle.user_ids, bundle.item_ids, bundle.user_factors, bundle.item_factors, {})
    frame = make_frame(users, events, as_of, "val", als)
    offline_order = rerank(frame, bundle.booster)

    for u in users:
        data = read_user(r, u)
        cand_items, a_score, retrieval = rec.candidates(u)
        X = rec.features(u, cand_items, a_score, retrieval, data)
        want = frame.filter(pl.col("visitorid") == u).sort("a_rank")
        assert want["itemid"].to_list() == cand_items.tolist(), f"candidates differ for {u}"
        expected = want.select(FEATURE_COLUMNS).cast(pl.Float64).to_numpy()
        # ALS factors are float32: a matrix-vector and a matrix-matrix product add in a different
        # order and differ by ~1e-7 (relative), so a_score gets a looser tolerance than the rest
        score = FEATURE_COLUMNS.index("a_score")
        rest = [i for i in range(len(FEATURE_COLUMNS)) if i != score]
        np.testing.assert_allclose(
            X[:, rest],
            expected[:, rest],
            rtol=1e-9,
            atol=1e-12,
            equal_nan=True,
            err_msg=f"features differ for user {u}",
        )
        np.testing.assert_allclose(X[:, score], expected[:, score], rtol=1e-5, equal_nan=True)
        online = [i for i, _ in rec.recommend(u, 200, data).items]
        assert online == offline_order[u], f"ranking differs for {u}"


def test_bundle_matches_the_feature_snapshot():
    info = json.loads((MODEL_DIR / "VERSION.json").read_text())
    report = json.loads((FEATURES / "features_report.json").read_text())
    assert info["val_as_of"] == report["val"]["as_of"].replace("T", " ")
