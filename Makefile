.PHONY: install lint format test etl

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
