"""Score the baselines on the Validation set.

    python -m src.evaluation.run_baseline            (or: make eval-baseline)

The Test set is held back: it is reported once, for the final chosen model only
(`--split test` exists for that moment; it fits on Train + Val).
Writes reports/baseline_<split>.json and reports/baseline_<split>.md.
"""

import argparse
import json
from pathlib import Path

import polars as pl

from src.etl.config import PROCESSED_DIR
from src.evaluation.baselines import fit_popularity, random_recs
from src.evaluation.ground_truth import build_ground_truth, split_warm_cold
from src.evaluation.metrics import coverage_at_k, per_user_metrics, summarize

REPORT_DIR = Path("reports")
KS = (10, 200)
# columns of the report table: (metric key, header)
TABLE = [
    ("recall@10", "Recall@10"),
    ("ndcg@10", "NDCG@10"),
    ("map@10", "MAP@10"),
    ("hit@10", "Hit@10"),
    ("recall@200", "Recall@200"),
]


def evaluate_model(recs, truth, groups, catalog_size):
    """Metrics (mean + 95% CI) for each user group: all / warm / cold."""
    result = {}
    for group, users in groups.items():
        if not users:
            continue
        values = per_user_metrics(recs, truth, users, KS)
        stats = {name: summarize(v) for name, v in values.items()}
        stats["coverage@10"] = coverage_at_k(recs, users, 10, catalog_size)
        stats["n_users"] = len(users)
        result[group] = stats
    return result


def to_markdown(split: str, results: dict, meta: dict) -> str:
    lines = [f"# Results on {split}", ""]
    lines += [
        f"- Evaluated users: {meta['n_users']:,} "
        f"(warm {meta['n_warm']:,}, cold {meta['n_cold']:,})",
        f"- Ground truth: items added to cart or bought during the {split} period",
        f"- Fitted on events before {meta['history_end']}; catalog = {meta['catalog']:,} items",
        "- Format: mean [95% bootstrap CI over users]",
        "",
    ]
    for group in ("all", "warm", "cold"):
        if not any(group in r for r in results.values()):
            continue
        n = next(r[group]["n_users"] for r in results.values() if group in r)
        lines += [f"## {group.capitalize()} users (n = {n:,})", ""]
        lines += ["| Model | " + " | ".join(h for _, h in TABLE) + " | Coverage@10 |"]
        lines += ["| --- |" + " --- |" * (len(TABLE) + 1)]
        for model, res in results.items():
            s = res[group]
            cells = [f"{s[k]['mean']:.4f} [{s[k]['lo']:.4f}, {s[k]['hi']:.4f}]" for k, _ in TABLE]
            lines += [f"| {model} | " + " | ".join(cells) + f" | {s['coverage@10']:.3%} |"]
        lines.append("")
    return "\n".join(lines)


def main(split: str = "val") -> None:
    train = pl.read_parquet(PROCESSED_DIR / "train.parquet")
    val = pl.read_parquet(PROCESSED_DIR / "val.parquet")
    if split == "val":
        history, period = train, val
    else:  # test: the final model may use Train + Val
        history, period = pl.concat([train, val]), pl.read_parquet(PROCESSED_DIR / "test.parquet")

    truth = build_ground_truth(period)
    users = sorted(truth)
    warm, cold = split_warm_cold(users, history)
    groups = {"all": users, "warm": warm, "cold": cold}
    catalog = history["itemid"].unique().to_list()

    popular_all = fit_popularity(history, n=200)
    popular_7d = fit_popularity(history, n=200, window_days=7)
    models = {
        "random": random_recs(users, catalog),
        "popular_all": dict.fromkeys(users, popular_all),
        "popular_7d": dict.fromkeys(users, popular_7d),
    }
    results = {name: evaluate_model(r, truth, groups, len(catalog)) for name, r in models.items()}

    meta = {
        "n_users": len(users),
        "n_warm": len(warm),
        "n_cold": len(cold),
        "history_end": str(history["ts"].max()),
        "catalog": len(catalog),
    }
    REPORT_DIR.mkdir(exist_ok=True)
    (REPORT_DIR / f"baseline_{split}.json").write_text(
        json.dumps({"meta": meta, "results": results}, indent=2)
    )
    md = to_markdown(split, results, meta)
    (REPORT_DIR / f"baseline_{split}.md").write_text(md)
    print(md)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["val", "test"], default="val")
    main(parser.parse_args().split)
