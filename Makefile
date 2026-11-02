.PHONY: install install-train lint format test etl eval-baseline features tune-als train mlflow-ui pull-model load-redis up down replay logs smoke install-all env loadtest loadtest-smoke loadtest-stress loadtest-users api-test

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

# create .env from the template once; edit .env to change ports, images, topic, token, ...
env:
	@test -f .env && echo ".env already exists" || (cp .env.example .env && echo "created .env from .env.example")

up: env
	docker compose up -d --build

down:
	docker compose down

load-redis:
	python -m src.serving.load_redis

# start from a clean state, then replay the events through Kafka:
#   stop the consumer, reset Redis to the batch state, recreate the topic, restart the consumer
replay:
	docker compose stop consumer
	$(MAKE) load-redis
	docker compose run --rm --build producer python -m src.streaming.producer --reset-only
	docker compose up -d consumer
	docker compose run --rm --build producer

logs:
	docker compose logs -f api consumer

# end-to-end check against the running stack (make up && make replay first)
smoke:
	RUN_STACK_TESTS=1 pytest tests/test_smoke_stack.py -v

# ---------- load test (needs: make up, make replay) ----------
# user ids that look like the real traffic -> loadtest/users.json
loadtest-users:
	python -m src.serving.make_loadtest_users

loadtest-smoke:
	K6_SCENARIO=smoke docker compose --profile loadtest run --rm k6 run --quiet --out experimental-prometheus-rw /loadtest/k6.js

loadtest:
	K6_SCENARIO=load docker compose --profile loadtest run --rm k6 run --quiet --out experimental-prometheus-rw /loadtest/k6.js

loadtest-stress:
	K6_SCENARIO=stress docker compose --profile loadtest run --rm k6 run --quiet --out experimental-prometheus-rw /loadtest/k6.js

# run the Postman collection from the terminal (Newman in Docker, nothing to install); needs the stack up
api-test:
	docker run --rm -v "$(PWD)/postman":/etc/newman postman/newman run recsys.postman_collection.json -e local.postman_environment.json --env-var base_url=http://host.docker.internal:$${API_PORT:-8000}
