import json

from src.serving.loadtest_report import find_knee, summarize_k6


def series(points):
    """points: list of (rps, p95, error%) at t = 0, 2, 4 ..."""
    times = [2.0 * i for i in range(len(points))]
    return {
        "rps": list(zip(times, [p[0] for p in points])),
        "p95": list(zip(times, [p[1] for p in points])),
        "errors": list(zip(times, [p[2] for p in points])),
    }


def test_knee_is_the_first_sustained_break_and_reports_the_rate_before_it():
    healthy = [(100, 10, 0), (300, 20, 0), (600, 40, 0), (900, 80, 0)]
    broken = [(1000, 150, 0), (1000, 400, 2), (950, 900, 5)]
    knee = find_knee(series(healthy + broken))
    assert knee["at"] == 8.0 and knee["sustained_rps"] == 900 and knee["p95_ms"] == 150


def test_a_single_spike_is_not_a_knee():
    points = [(100, 10, 0), (200, 500, 0), (300, 12, 0), (400, 15, 0), (500, 18, 0)]
    assert find_knee(series(points)) is None


def test_errors_alone_can_break_the_service():
    points = [(100, 10, 0), (200, 20, 0)] + [(300, 30, 5)] * 3
    assert find_knee(series(points))["sustained_rps"] == 200


def test_k6_summary_numbers_and_thresholds(tmp_path):
    summary = {
        "metrics": {
            "http_reqs": {"values": {"count": 1000, "rate": 50.0}},
            "http_req_failed": {"values": {"rate": 0.002}},
            "http_req_duration": {
                "values": {"avg": 5, "med": 4, "p(90)": 8, "p(95)": 9, "p(99)": 20, "max": 40},
                "thresholds": {"p(95)<100": {"ok": True}},
            },
            "vus_max": {"values": {"max": 100}},
        }
    }
    path = tmp_path / "load.json"
    path.write_text(json.dumps(summary))
    out = summarize_k6(path)
    assert (out["requests"], out["failed_pct"], out["p95"], out["max_vus"], out["dropped"]) == (
        1000,
        0.2,
        9,
        100,
        0,
    )
    assert out["thresholds"] == {"http_req_duration: p(95)<100": True}
