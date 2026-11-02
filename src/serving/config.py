"""Settings of the API, the batch loader and the stream consumer (environment variables)."""

import os
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    redis_url: str = field(
        default_factory=lambda: os.environ.get("REDIS_URL", "redis://localhost:6379/0")
    )
    # a slow Redis must not make the API slow: give up quickly and serve without user features
    redis_timeout_s: float = field(
        default_factory=lambda: float(os.environ.get("REDIS_TIMEOUT_S", "0.05"))
    )
    model_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("MODEL_DIR", "models/champion"))
    )
    features_dir: Path = field(
        default_factory=lambda: Path(os.environ.get("FEATURES_DIR", "data/features"))
    )
    snapshot: str = field(default_factory=lambda: os.environ.get("FEATURES_SNAPSHOT", "val"))
    n_candidates: int = 200
