"""Hyperparameters and paths of the training step."""

import os
from pathlib import Path

from src.env import load_env

load_env()  # values from .env (variables already set take precedence)

N_CANDIDATES = 200  # items retrieved per user before the Ranker reorders them
MIN_EVENTS = 2  # ALS only learns users/items with at least this many events (EDA: 62.6% of data)
N_RANK_SNAPSHOTS = 4  # rolling snapshots the Ranker is trained on

ALS_PARAMS = {
    "factors": 64,
    "regularization": 0.1,
    "iterations": 15,
    "alpha": 20.0,  # confidence = 1 + alpha * (weighted interactions)
}

RANKER_PARAMS = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "eval_at": [10],
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "lambdarank_truncation_level": 20,
    "verbosity": -1,
    "seed": 42,
}
MAX_TREES = 500
EARLY_STOPPING = 30

REPORT_DIR = Path("reports")
ALS_TUNING_FILE = REPORT_DIR / "als_best.json"  # written by `make tune-als`, read by `make train`

# MLflow: a local SQLite file gives tracking AND a model registry (aliases) without a server.
# Override with MLFLOW_TRACKING_URI to log to a remote server instead.
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db")
MLFLOW_EXPERIMENT = "recsys"
REGISTERED_MODEL = "recsys-bundle"
