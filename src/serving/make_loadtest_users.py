"""Pick the user ids the load test requests, so the traffic looks like the real thing.

    python -m src.serving.make_loadtest_users        (or: make loadtest-users)

A uniform random number would almost always hit an unknown user. The mix below follows the data:
most traffic is users the model has never seen, a part has history in Redis but is unknown to ALS,
and a part is known to ALS. Unknown ids are drawn above the largest real id, so they are new.
"""

import json
import random
from pathlib import Path

import numpy as np
import polars as pl

from src.serving.config import Settings

OUT = Path("loadtest/users.json")
SHARE = {"als": 0.15, "history": 0.35, "new": 0.50}  # of the requests
SIZE = 6000


def main(seed: int = 0) -> dict:
    cfg = Settings()
    rng = random.Random(seed)
    als = np.load(cfg.model_dir / "als.npz")["user_ids"]
    state = pl.read_parquet(cfg.features_dir / cfg.snapshot / "user_state.parquet")["visitorid"]
    known_to_als = set(als.tolist())
    history_only = [u for u in state.to_list() if u not in known_to_als]

    n = {k: round(SIZE * v) for k, v in SHARE.items()}
    top = int(state.max())
    users = (
        rng.sample(als.tolist(), n["als"])
        + rng.sample(history_only, n["history"])
        + [top + 1 + rng.randrange(10_000_000) for _ in range(n["new"])]
    )
    rng.shuffle(users)
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(json.dumps(users))
    return {"file": str(OUT), "users": len(users), **{f"{k}_users": v for k, v in n.items()}}


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
