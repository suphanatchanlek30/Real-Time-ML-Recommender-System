import importlib
import re
from pathlib import Path

from src.env import load_env, parse_env
from src.serving.config import Settings

ROOT = Path(__file__).resolve().parent.parent


def test_parse_env_handles_comments_blank_lines_quotes_and_equals_in_values():
    text = """
    # a comment
    API_PORT=8000
    NAME = "quoted value"
    URL=sqlite:///mlflow.db
    EMPTY=
    TOKEN='a=b'
    not a setting
    """
    assert parse_env(text) == {
        "API_PORT": "8000",
        "NAME": "quoted value",
        "URL": "sqlite:///mlflow.db",
        "EMPTY": "",
        "TOKEN": "a=b",
    }


def test_variables_already_set_win_over_the_file(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text("FROM_FILE=file\nALREADY_SET=file\n")
    monkeypatch.delenv("FROM_FILE", raising=False)
    monkeypatch.setenv("ALREADY_SET", "shell")
    load_env(env_file)
    import os

    assert os.environ["FROM_FILE"] == "file" and os.environ["ALREADY_SET"] == "shell"
    monkeypatch.delenv("FROM_FILE")


def test_missing_env_file_is_fine(tmp_path):
    assert load_env(tmp_path / "nope.env") == {}


def test_settings_build_the_redis_url_from_the_parts(monkeypatch):
    monkeypatch.delenv("REDIS_URL", raising=False)
    monkeypatch.setenv("REDIS_HOST", "cache")
    monkeypatch.setenv("REDIS_PORT", "6380")
    monkeypatch.setenv("REDIS_DB", "2")
    assert Settings().redis_url == "redis://cache:6380/2"
    monkeypatch.setenv("REDIS_URL", "redis://redis:6379/0")  # Compose sets this in containers
    assert Settings().redis_url == "redis://redis:6379/0"


def test_settings_read_paths_snapshot_and_timeout(monkeypatch):
    monkeypatch.setenv("MODEL_DIR", "m/x")
    monkeypatch.setenv("FEATURES_SNAPSHOT", "test")
    monkeypatch.setenv("REDIS_TIMEOUT_S", "0.2")
    s = Settings()
    assert (str(s.model_dir), s.snapshot, s.redis_timeout_s) == ("m/x", "test", 0.2)


def test_kafka_config_follows_the_environment(monkeypatch):
    import src.streaming.config as kafka

    monkeypatch.delenv("KAFKA_BOOTSTRAP", raising=False)
    monkeypatch.setenv("KAFKA_EXTERNAL_PORT", "9199")
    monkeypatch.setenv("KAFKA_PARTITIONS", "5")
    monkeypatch.setenv("KAFKA_TOPIC", "t1")
    importlib.reload(kafka)
    assert (kafka.BOOTSTRAP, kafka.PARTITIONS, kafka.TOPIC) == ("localhost:9199", 5, "t1")
    monkeypatch.undo()
    importlib.reload(kafka)


def test_every_variable_used_by_compose_is_in_env_example():
    """Compose would silently use an empty string for a variable missing from .env."""
    lines = (ROOT / "docker-compose.yml").read_text().splitlines()
    code = "\n".join(line for line in lines if not line.lstrip().startswith("#"))
    used = set(re.findall(r"\$\{(\w+)", code))
    defined = set(parse_env((ROOT / ".env.example").read_text()))
    assert used and used <= defined, f"missing from .env.example: {sorted(used - defined)}"


def test_images_contain_no_settings():
    for name in ("Dockerfile.api", "Dockerfile.consumer"):
        text = (ROOT / "docker" / name).read_text()
        assert "REDIS_URL" not in text and "FEATURES_DIR" not in text and "MODEL_DIR" not in text


def test_env_file_is_not_committed_but_the_template_is():
    ignored = (ROOT / ".gitignore").read_text().split()
    assert ".env" in ignored and ".env.example" not in ignored
