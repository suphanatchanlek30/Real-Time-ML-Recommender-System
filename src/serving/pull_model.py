"""Fetch the `champion` bundle from the MLflow registry into models/champion/.

    python -m src.serving.pull_model        (or: make pull-model)

Runs on the host (MLflow is a local SQLite file, not reachable from the containers). The API
reads the folder through a Docker volume, so upgrading or rolling back the model is: move the
`champion` alias in MLflow, run this, then POST /admin/reload. No image rebuild.
"""

import json
import shutil
import tempfile
from pathlib import Path

from src.serving.config import Settings
from src.training import config as C


def main(target: Path | None = None) -> dict:
    import mlflow
    from mlflow import MlflowClient

    target = Path(target or Settings().model_dir)
    mlflow.set_tracking_uri(C.MLFLOW_URI)
    client = MlflowClient()
    version = client.get_model_version_by_alias(C.REGISTERED_MODEL, "champion")
    run = client.get_run(version.run_id)

    target.parent.mkdir(parents=True, exist_ok=True)
    # download next to the target, then swap: a failed download never leaves a half-written bundle
    with tempfile.TemporaryDirectory(dir=target.parent) as tmp:
        downloaded = Path(
            mlflow.artifacts.download_artifacts(
                run_id=version.run_id, artifact_path="bundle", dst_path=tmp
            )
        )
        info = {
            "version": version.version,
            "run_id": version.run_id,
            "val_as_of": run.data.tags.get("val_as_of"),
            "git_commit": run.data.tags.get("git_commit"),
            "data_sha256": run.data.tags.get("data_sha256"),
            "val_ndcg10": run.data.metrics.get("val_ndcg10"),
            "val_recall200": run.data.metrics.get("val_recall200"),
        }
        (downloaded / "VERSION.json").write_text(json.dumps(info, indent=2))
        if target.exists():
            shutil.rmtree(target)
        shutil.move(str(downloaded), str(target))
    return info


if __name__ == "__main__":
    print(json.dumps(main(), indent=2))
