"""Load settings from the project's `.env` file into the environment.

    KEY=value          one per line; `#` starts a comment line; surrounding quotes are removed

Variables that are already set (a shell export, or Docker Compose inside a container) win over
the file, so the same code works on the Mac, where `.env` supplies the values, and in a
container, where Compose supplies them and there is no `.env`. Copy `.env.example` to `.env`.
"""

import os
from pathlib import Path

ENV_FILE = Path(__file__).resolve().parent.parent / ".env"


def parse_env(text: str) -> dict[str, str]:
    values = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        values[key.strip()] = value
    return values


def load_env(path: Path = ENV_FILE) -> dict[str, str]:
    """Read `path` (if it exists) and set each variable that is not already set."""
    if not path.exists():
        return {}
    values = parse_env(path.read_text())
    for key, value in values.items():
        os.environ.setdefault(key, value)
    return values
