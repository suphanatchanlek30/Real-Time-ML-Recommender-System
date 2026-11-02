.PHONY: install install-train lint format test etl eval-baseline features tune-als train mlflow-ui

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .

format:
	ruff format .

test:
	pytest

etl:
	python -m src.etl.run

eval-baseline:
	python -m src.evaluation.run_baseline

features:
	python -m src.features.build

install-train:
	pip install -r requirements.txt -r requirements-train.txt

tune-als:
	python -m src.training.tune_als

train:
	python -m src.training.train

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db
