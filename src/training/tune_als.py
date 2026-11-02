"""Small grid search for the ALS hyperparameters, scored on Validation.

    python -m src.training.tune_als       (or: make tune-als)

Metric: Recall@200 of the ALS candidates over validation users that ALS knows (only they get
ALS candidates). Writes reports/als_tuning.md and the best setting to reports/als_best.json,
which `make train` then uses. Test is not used.
"""

import itertools
import json
import logging
import time

import numpy as np
import polars as pl

from src.etl.config import PROCESSED_DIR
from src.evaluation.ground_truth import build_ground_truth
from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.features.snapshots import make_snapshots
from src.training import config as C
from src.training.als import als_candidates, fit_als
from src.training.train import window

log = logging.getLogger("tune")
GRID = {"factors": [32, 64, 128], "regularization": [0.01, 0.1], "alpha": [10.0, 40.0]}


def main() -> list[dict]:
    events = pl.read_parquet(PROCESSED_DIR / "events.parquet")
    val = next(s for s in make_snapshots(events) if s.name == "val")
    truth = build_ground_truth(window(events, val.label_start, val.label_end))

    rows = []
    for values in itertools.product(*GRID.values()):
        params = dict(zip(GRID, values))
        start = time.time()
        als = fit_als(events, val.as_of, params)
        users = [u for u in sorted(truth) if als.knows(u)]
        cands = als_candidates(als, users, C.N_CANDIDATES)
        recs = {
            u: items
            for u, items in zip(
                *cands.group_by("visitorid", maintain_order=True)
                .agg(pl.col("itemid"))
                .get_columns()
            )
        }
        rows.append(
            {
                **params,
                "users": len(users),
                "recall@200": float(np.mean([recall_at_k(recs[u], truth[u], 200) for u in users])),
                "ndcg@10": float(np.mean([ndcg_at_k(recs[u], truth[u], 10) for u in users])),
                "seconds": round(time.time() - start, 1),
            }
        )
        log.info("%s", rows[-1])

    rows.sort(key=lambda r: r["recall@200"], reverse=True)
    C.REPORT_DIR.mkdir(exist_ok=True)
    best = {k: rows[0][k] for k in GRID}
    C.ALS_TUNING_FILE.write_text(json.dumps(best, indent=2))
    header = list(rows[0])
    lines = [
        "# ALS tuning on Validation",
        "",
        "Recall@200 / NDCG@10 over validation users that ALS knows (ALS candidates only).",
        "",
        "| " + " | ".join(header) + " |",
        "|" + " --- |" * len(header),
    ]
    lines += [
        "| "
        + " | ".join(
            str(r[h]) if h not in ("recall@200", "ndcg@10") else f"{r[h]:.4f}" for h in header
        )
        + " |"
        for r in rows
    ]
    (C.REPORT_DIR / "als_tuning.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines), f"\n\nbest: {best}")
    return rows


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    main()
