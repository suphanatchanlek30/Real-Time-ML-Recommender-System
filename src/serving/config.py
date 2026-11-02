"""Settings of the API, the batch loader and the stream consumer.

Values come from the environment; on the Mac `.env` fills it (see src/env.py), in a container
Docker Compose does. The defaults below are only a fallback when neither is present.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.env import load_env

load_env()


def _redis_url() -> str:
    """REDIS_URL if given (Compose sets it for containers), else built from the .env parts."""
    if os.environ.get("REDIS_URL"):
        return os.environ["REDIS_URL"]
    host = os.environ.get("REDIS_HOST", "localhost")
    port = os.environ.get("REDIS_PORT", "6379")
    return f"redis://{host}:{port}/{os.environ.get('REDIS_DB', '0')}"


@dataclass(frozen=True)
class Settings:
    redis_url: str = field(default_factory=_redis_url)
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
