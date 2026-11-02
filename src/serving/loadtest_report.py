"""Turn a load-test run into numbers and charts.

    python -m src.serving.loadtest_report results/baseline --title "Before"

Reads results/<label>/{load,stress}.json (k6 summaries) and the matching *.window.json (start and
end time of the run), asks Prometheus for the server-side series of that window and writes
docs/loadtest/<label>-<scenario>.png plus a markdown summary on stdout.
"""

import argparse
import json
import os
import urllib.parse
import urllib.request
from pathlib import Path

PROM = os.environ.get(
    "PROMETHEUS_URL", f"http://localhost:{os.environ.get('PROMETHEUS_PORT', '9090')}"
)
OUT_DIR = Path("docs/loadtest")
REC = 'endpoint="/recommend"'
W = "10s"
LATENCY = f"rate(http_request_duration_seconds_bucket{{{REC}}}[{W}])"
REQS = f"http_requests_total{{{REC}}}"
QUERIES = {
    "rps": f"sum(rate({REQS}[{W}]))",
    "p50": f"1000 * histogram_quantile(0.50, sum by (le) ({LATENCY}))",
    "p95": f"1000 * histogram_quantile(0.95, sum by (le) ({LATENCY}))",
    "p99": f"1000 * histogram_quantile(0.99, sum by (le) ({LATENCY}))",
    "errors": (
        f'100 * (sum(rate(http_requests_total{{{REC},status=~"5.."}}[{W}])) or vector(0))'
        f" / sum(rate({REQS}[{W}]))"
    ),
    "cpu": f'sum(rate(process_cpu_seconds_total{{job="api"}}[{W}]))',
    "inflight": "sum(http_requests_in_progress)",
    "vus": "max(k6_vus)",
}


def prom_range(query: str, start: float, end: float, step: int = 2) -> list[tuple[float, float]]:
    url = f"{PROM}/api/v1/query_range?" + urllib.parse.urlencode(
        {"query": query, "start": start, "end": end, "step": step}
    )
    result = json.load(urllib.request.urlopen(url, timeout=30))["data"]["result"]
    if not result:
        return []
    return [(float(t), float(v)) for t, v in result[0]["values"] if v not in ("NaN", "+Inf")]


def summarize_k6(path: Path) -> dict:
    """The numbers of one k6 summary file."""
    data = json.loads(path.read_text())
    m = data["metrics"]
    t = m["http_req_duration"]["values"]
    return {
        "requests": int(m["http_reqs"]["values"]["count"]),
        "rate": m["http_reqs"]["values"]["rate"],
        "failed_pct": m["http_req_failed"]["values"]["rate"] * 100,
        "avg": t["avg"],
        "med": t["med"],
        "p90": t["p(90)"],
        "p95": t["p(95)"],
        "p99": t["p(99)"],
        "max": t["max"],
        "max_vus": int(m.get("vus_max", {}).get("values", {}).get("max", 0)),
        "dropped": int(m.get("dropped_iterations", {}).get("values", {}).get("count", 0)),
        "thresholds": {
            f"{name}: {expr}": res["ok"]
            for name, metric in m.items()
            if "thresholds" in metric
            for expr, res in metric["thresholds"].items()
        },
    }


def find_knee(
    series: dict[str, list[tuple[float, float]]], p95_limit=100.0, error_limit=1.0, run=3
):
    """First moment the service stops meeting its targets, and the rate it sustained before that.

    `series` holds aligned (time, value) lists for rps, p95 and errors. "Stops meeting" means the
    p95 or the error rate is over its limit for `run` points in a row (a single spike is noise).
    Returns None if the service never broke.
    """
    times = [t for t, _ in series["p95"]]
    p95 = dict(series["p95"])
    err = dict(series["errors"])
    rps = dict(series["rps"])
    bad = [p95.get(t, 0) > p95_limit or err.get(t, 0) > error_limit for t in times]
    for i in range(len(bad) - run + 1):
        if all(bad[i : i + run]):
            before = [rps[t] for t in times[:i] if t in rps]
            return {
                "at": times[i],
                "sustained_rps": max(before) if before else 0.0,
                "p95_ms": p95[times[i]],
                "rps_when_broke": rps.get(times[i], 0.0),
            }
    return None


def plot(series: dict, title: str, out: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def xy(name, scale=1.0):
        pts = series.get(name) or []
        if not pts:
            return [], []
        t0 = series["_t0"]
        return [(t - t0) for t, _ in pts], [v * scale for _, v in pts]

    fig, ax = plt.subplots(2, 2, figsize=(13, 7.5), sharex=True)
    ax[0, 0].plot(*xy("rps"), label="requests / s", color="#2a6fdb")
    ax[0, 0].set(title="Throughput", ylabel="requests / s")
    twin = ax[0, 0].twinx()
    twin.plot(*xy("vus"), label="k6 VUs", color="#999", linestyle="--")
    twin.set_ylabel("k6 virtual users")
    ax[0, 0].legend(loc="upper left")

    for name, color in [("p50", "#3aa655"), ("p95", "#e69f00"), ("p99", "#d6403f")]:
        ax[0, 1].plot(*xy(name), label=name, color=color)
    ax[0, 1].axhline(100, color="red", linestyle=":", linewidth=1)
    ax[0, 1].set(title="Latency (server side)", ylabel="ms")
    ax[0, 1].legend(loc="upper left")
    peak = max([v for _, v in series.get("p99", [])] or [100])
    if peak > 1000:
        ax[0, 1].set_yscale("log")

    ax[1, 0].plot(*xy("errors"), color="#d6403f", label="error rate (%)")
    ax[1, 0].set(title="Errors and requests in flight", ylabel="% of requests", xlabel="seconds")
    twin = ax[1, 0].twinx()
    twin.plot(*xy("inflight"), color="#7b5ea7", label="in flight")
    twin.set_ylabel("requests in flight")

    ax[1, 1].plot(*xy("cpu"), color="#444", label="API CPU")
    ax[1, 1].set(title="API CPU", ylabel="cores", xlabel="seconds")
    for a in ax.flat:
        a.grid(alpha=0.3)
    fig.suptitle(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=110)
    plt.close(fig)


def report_scenario(folder: Path, scenario: str, title: str) -> dict | None:
    summary, window = folder / f"{scenario}.json", folder / f"{scenario}.window.json"
    if not (summary.exists() and window.exists()):
        return None
    span = json.loads(window.read_text())
    start, end = span["start"] - 5, span["end"] + 5
    series = {name: prom_range(q, start, end) for name, q in QUERIES.items()}
    series["_t0"] = span["start"]
    image = OUT_DIR / f"{folder.name}-{scenario}.png"
    plot(series, f"{title}: {scenario}", image)
    result = {"scenario": scenario, "image": str(image), **summarize_k6(summary)}
    result["knee"] = find_knee(series) if scenario == "stress" else None
    result["peak_cpu"] = max([v for _, v in series["cpu"]] or [0])
    result["peak_rps"] = max([v for _, v in series["rps"]] or [0])
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--title", default=None)
    args = parser.parse_args()
    title = args.title or args.folder.name
    results = [r for s in ("load", "stress") if (r := report_scenario(args.folder, s, title))]
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
