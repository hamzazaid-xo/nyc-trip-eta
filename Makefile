# The only interface you type. Everything runs in Docker.

DC      := docker compose
RUN     := $(DC) run --rm
DIRS    := data/raw data/processed data/interim models mlruns reports docs/notes

.DEFAULT_GOAL := help
.PHONY: help init build shell lock download data notebook mlflow baselines train explain api test lint fmt down clean

help: ## show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk -F':.*?## ' '{printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

init: ## create gitignored working directories
	@mkdir -p $(DIRS)

lock: ## regenerate uv.lock inside the container (run after editing pyproject.toml)
	docker run --rm -v "$(CURDIR)":/app -w /app ghcr.io/astral-sh/uv:0.12.21-python3.14-trixie-slim uv lock

build: init ## build the dev and runtime images
	$(DC) build dev api

shell: init ## bash inside the dev container
	$(RUN) dev bash

download: init ## fetch raw NYC TLC data into data/raw/   [Phase 1]
	$(RUN) dev python scripts/download_data.py

data: init ## clean + split -> data/processed/ + cleaning report   [Phase 1]
	$(RUN) dev python -m tripeta.data

notebook: init ## JupyterLab at http://localhost:8888   [Phase 2]
	$(DC) up notebook

mlflow: init ## MLflow UI at http://localhost:5000
	$(DC) up -d mlflow
	@echo "MLflow UI -> http://localhost:5000"

baselines: init ## run the baseline models   [Phase 3]
	$(RUN) dev python -m tripeta.train --baselines

train: init ## train LightGBM   [Phase 4]
	$(RUN) dev python -m tripeta.train

explain: init ## generate SHAP plots   [Phase 5]
	$(RUN) dev python -m tripeta.evaluate --explain

api: init ## build + serve the production image at http://localhost:8000   [Phase 6]
	$(DC) up --build api

test: init ## run pytest
	$(RUN) test

lint: init ## ruff check + format check
	$(RUN) dev ruff check .
	$(RUN) dev ruff format --check .

fmt: init ## ruff autofix + format
	$(RUN) dev ruff check --fix .
	$(RUN) dev ruff format .

down: ## stop and remove all containers
	$(DC) down --remove-orphans

clean: ## remove caches (keeps data and models)
	$(RUN) dev bash -c "rm -rf .pytest_cache .ruff_cache && find . -name __pycache__ -prune -exec rm -rf {} +"
