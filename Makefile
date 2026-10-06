ENV_FILE ?= .env
QUESTION ?=
export ENV_FILE QUESTION

.PHONY: run chat postgres db-init up down logs check format

run:
	uv run --locked --no-editable stackoverflow-assistant "$$QUESTION"

chat:
	uv run --locked --no-editable streamlit run app.py

postgres:
	docker compose --env-file "$$ENV_FILE" up -d postgres

db-init:
	uv run --locked --no-editable stackoverflow-db-init

up:
	docker compose --env-file "$$ENV_FILE" up --build -d

down:
	docker compose --env-file "$$ENV_FILE" down

logs:
	docker compose --env-file "$$ENV_FILE" logs -f streamlit

check:
	uv run --locked --no-editable ruff check .
	uv run --locked --no-editable ruff format --check .

format:
	uv run --locked --no-editable ruff format .
