"""Train ALS + Ranker, compare with the baseline on Validation, gate and register in MLflow.

    python -m src.training.train          (or: make train)

Steps
  1. Ranker training data: for each rolling snapshot rank_k, fit ALS on events before its as_of,
     retrieve 200 candidates per user and label them with what the user did in the next 7 days.
  2. Fit the Ranker (early stopping on the newest snapshot, then refit on all).
  3. Validation: ALS fitted on all of Train; score popular baseline, ALS only and ALS + Ranker.
  4. Evaluation gate: the new model must beat the current `champion` (or the popularity
     baseline when there is none) on NDCG@10 without hurting Recall@200.
  5. Log everything to MLflow; if the gate passes, register a version and move `champion`.
The Test set is never touched here.
"""

import argparse
import json
import logging
import subprocess
import tempfile
from pathlib import Path

import polars as pl

from src.etl.config import PROCESSED_DIR
from src.evaluation.baselines import fit_popularity
from src.evaluation.ground_truth import build_ground_truth, split_warm_cold
from src.evaluation.run_baseline import evaluate_model, to_markdown
from src.features.schema import write_schema
from src.features.snapshots import make_snapshots
from src.training import config as C
from src.training.als import fit_als
from src.training.dataset import add_labels, make_frame
from src.training.ranker import fit_ranker, rerank, retrieved_order

log = logging.getLogger("train")


def window(events: pl.DataFrame, start, end) -> pl.DataFrame:
    cond = pl.col("ts") >= start
    return events.filter(cond & (pl.col("ts") < end) if end is not None else cond)


def ranker_training_frames(events, snaps, als_params):
    """One labelled frame per rolling snapshot (rank_N ... rank_1), oldest first."""
    frames = []
    for snap in snaps:
        truth = build_ground_truth(window(events, snap.label_start, snap.label_end))
        users = sorted(truth)
        als = fit_als(events, snap.as_of, als_params)
        frame = make_frame(users, events, snap.as_of, snap.name, als)
        frame = add_labels(frame, truth).with_columns(
            (pl.lit(snap.name) + "-" + pl.col("visitorid").cast(pl.String)).alias("gid")
        )
        log.info(
            "%s: %d users, %d rows, %d positives (ALS knows %d users)",
            snap.name,
            len(users),
            frame.height,
            frame["label"].sum(),
            sum(als.knows(u) for u in users),
        )
        frames.append(frame)
    return frames


def champion_metrics(client) -> tuple[dict | None, str | None]:
    """Validation metrics of the current champion run, if there is one."""
    try:
        version = client.get_model_version_by_alias(C.REGISTERED_MODEL, "champion")
    except Exception:  # no registered model or no alias yet
        return None, None
    return client.get_run(version.run_id).data.metrics, version.version


def gate(new: dict, bar: dict) -> tuple[bool, str]:
    ok_ndcg = new["ndcg@10"] > bar["ndcg@10"]
    ok_recall = new["recall@200"] >= bar["recall@200"]
    msg = (
        f"NDCG@10 {new['ndcg@10']:.4f} vs bar {bar['ndcg@10']:.4f} "
        f"({'ok' if ok_ndcg else 'FAIL'}); Recall@200 {new['recall@200']:.4f} vs "
        f"{bar['recall@200']:.4f} ({'ok' if ok_recall else 'FAIL'})"
    )
    return ok_ndcg and ok_recall, msg


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def main(use_mlflow: bool = True) -> dict:
    events = pl.read_parquet(PROCESSED_DIR / "events.parquet")
    etl_report = json.loads((PROCESSED_DIR / "etl_report.json").read_text())
    snaps = make_snapshots(events, n_rank=C.N_RANK_SNAPSHOTS)
    rank_snaps = [s for s in snaps if s.name.startswith("rank_")]
    val = next(s for s in snaps if s.name == "val")
    als_params = json.loads(C.ALS_TUNING_FILE.read_text()) if C.ALS_TUNING_FILE.exists() else {}
    log.info("ALS params: %s", {**C.ALS_PARAMS, **als_params})

    # 1-2. Ranker
    frames = ranker_training_frames(events, rank_snaps, als_params)
    booster, ranker_stats = fit_ranker(pl.concat(frames[:-1]), frames[-1])
    log.info("ranker: %s", ranker_stats)

    # 3. Validation
    truth = build_ground_truth(window(events, val.label_start, val.label_end))
    users = sorted(truth)
    history = events.filter(pl.col("ts") < val.as_of)
    warm, cold = split_warm_cold(users, history)
    groups = {"all": users, "warm": warm, "cold": cold}
    als = fit_als(events, val.as_of, als_params)
    frame = make_frame(users, events, val.as_of, "val", als)
    popular = fit_popularity(history, n=C.N_CANDIDATES, window_days=7)
    models = {
        "popular_7d": dict.fromkeys(users, popular),
        "als": retrieved_order(frame),
        "als_ranker": rerank(frame, booster),
    }
    catalog = history["itemid"].n_unique()
    results = {name: evaluate_model(recs, truth, groups, catalog) for name, recs in models.items()}

    new = {k: results["als_ranker"]["all"][k]["mean"] for k in ("ndcg@10", "recall@200")}
    client = None
    if use_mlflow:
        import mlflow
        from mlflow import MlflowClient

        mlflow.set_tracking_uri(C.MLFLOW_URI)
        mlflow.set_experiment(C.MLFLOW_EXPERIMENT)
        client = MlflowClient()
    champ, champ_version = champion_metrics(client) if client else (None, None)
    if champ:
        bar = {"ndcg@10": champ["val_ndcg10"], "recall@200": champ["val_recall200"]}
        bar_name = f"champion v{champ_version}"
    else:
        base = results["popular_7d"]["all"]
        bar = {k: base[k]["mean"] for k in ("ndcg@10", "recall@200")}
        bar_name = "popular_7d baseline"
    passed, msg = gate(new, bar)
    log.info("gate vs %s: %s -> %s", bar_name, msg, "PASS" if passed else "FAIL")

    meta = {
        "n_users": len(users),
        "n_warm": len(warm),
        "n_cold": len(cold),
        "history_end": str(history["ts"].max()),
        "catalog": catalog,
    }
    report = to_markdown("val", results, meta)
    C.REPORT_DIR.mkdir(exist_ok=True)
    (C.REPORT_DIR / "train_val.md").write_text(report)
    print(report)

    summary = {
        "gate_passed": passed,
        "gate": msg,
        "bar": bar_name,
        "ranker": ranker_stats,
        "results": results,
    }
    if not client:
        return summary

    with mlflow.start_run(run_name="als+ranker") as run, tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        mlflow.log_params({f"als_{k}": v for k, v in {**C.ALS_PARAMS, **als_params}.items()})
        mlflow.log_params({f"ranker_{k}": v for k, v in C.RANKER_PARAMS.items()})
        mlflow.log_params(
            {
                "n_candidates": C.N_CANDIDATES,
                "min_events": C.MIN_EVENTS,
                "n_rank_snapshots": C.N_RANK_SNAPSHOTS,
                "ranker_trees": ranker_stats["best_iteration"],
            }
        )
        mlflow.set_tags(
            {
                "git_commit": git_commit(),
                "data_sha256": etl_report["events_parquet_sha256"],
                "data_from": etl_report["ts_min"],
                "data_to": etl_report["ts_max"],
                "val_as_of": str(val.as_of),
                "gate": "pass" if passed else "fail",
            }
        )
        for model, per_group in results.items():
            for group, stats in per_group.items():
                for key, value in stats.items():
                    if isinstance(value, dict):
                        name = f"{model}.{group}.{key}".replace("@", "_at_")  # MLflow rejects "@"
                        mlflow.log_metric(name, value["mean"])
        # the two numbers the next run's gate reads back
        mlflow.log_metric("val_ndcg10", new["ndcg@10"])
        mlflow.log_metric("val_recall200", new["recall@200"])
        mlflow.log_metric("ranker_holdout_ndcg10", ranker_stats["holdout_ndcg10"])

        # bundle = everything serving needs, versioned together
        bundle = tmp / "bundle"
        bundle.mkdir()
        als.save(bundle / "als.npz")
        booster.save_model(str(bundle / "ranker.txt"))
        write_schema(bundle / "schema.json")
        (bundle / "popular.json").write_text(json.dumps(popular))
        (bundle / "etl_report.json").write_text(json.dumps(etl_report, indent=2))
        mlflow.log_artifacts(str(bundle), artifact_path="bundle")
        mlflow.log_dict(results, "val_results.json")

        if passed:
            name = C.REGISTERED_MODEL
            try:
                client.create_registered_model(name)
            except Exception:
                pass  # already exists
            mv = client.create_model_version(
                name, f"{run.info.artifact_uri}/bundle", run.info.run_id
            )
            client.set_registered_model_alias(name, "champion", mv.version)
            log.info("registered %s v%s and moved alias 'champion'", name, mv.version)
            summary["registered_version"] = mv.version
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mlflow", action="store_true", help="train and evaluate only")
    args = parser.parse_args()
    out = main(use_mlflow=not args.no_mlflow)
    print(json.dumps({k: v for k, v in out.items() if k != "results"}, indent=2, default=str))
