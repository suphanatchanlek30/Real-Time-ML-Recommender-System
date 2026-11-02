"""Skip test files whose optional dependencies are not installed (instead of failing to import).

Serving and training tests need the extras of `make install-all`; the core tests (ETL,
features, evaluation) run with the base requirements only.
"""

import importlib.util

NEEDS = {
    "test_feature_store.py": ["fakeredis", "redis", "pydantic", "lightgbm"],
    "test_streaming.py": ["fakeredis", "redis", "pydantic", "lightgbm"],
    "test_api.py": ["fakeredis", "redis", "fastapi", "httpx", "lightgbm", "prometheus_client"],
    "test_parity_real.py": ["fakeredis", "lightgbm", "scipy"],
    "test_training.py": ["lightgbm", "scipy"],
    "test_loadtest_report.py": ["matplotlib"],
    "test_smoke_stack.py": ["redis", "httpx", "confluent_kafka"],
}


def _missing(module: str) -> bool:
    return importlib.util.find_spec(module) is None


collect_ignore = [f for f, mods in NEEDS.items() if any(_missing(m) for m in mods)]


def pytest_report_header(config):
    if collect_ignore:
        return f"skipped (missing packages, see make install-all): {', '.join(collect_ignore)}"
