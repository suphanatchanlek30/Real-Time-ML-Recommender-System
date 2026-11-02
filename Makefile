.PHONY: install lint format test etl eval-baseline

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
