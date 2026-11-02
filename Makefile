.PHONY: install lint format test

install:
	pip install -r requirements-dev.txt

lint:
	ruff check .

format:
	ruff format .

test:
	pytest
