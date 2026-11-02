.PHONY: install install-train lint format test etl eval-baseline features tune-als train mlflow-ui pull-model load-redis up down replay logs smoke install-all

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

install-all:
	pip install -r requirements.txt -r requirements-train.txt -r requirements-serve.txt -r requirements-test.txt

tune-als:
	python -m src.training.tune_als

train:
	python -m src.training.train

mlflow-ui:
	mlflow ui --backend-store-uri sqlite:///mlflow.db

pull-model:
	python -m src.serving.pull_model

up:
	docker compose up -d --build

down:
	docker compose down

load-redis:
	REDIS_URL=redis://localhost:6379/0 python -m src.serving.load_redis

# start from a clean state, then replay the events through Kafka:
#   stop the consumer, reset Redis to the batch state, recreate the topic, restart the consumer
replay:
	docker compose stop consumer
	$(MAKE) load-redis
	docker compose run --rm producer python -m src.streaming.producer --reset-only
	docker compose up -d consumer
	docker compose run --rm producer

logs:
	docker compose logs -f api consumer

# end-to-end check against the running stack (make up && make replay first)
smoke:
	RUN_STACK_TESTS=1 pytest tests/test_smoke_stack.py -v
