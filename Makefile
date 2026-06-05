.PHONY: install run-ui run-api test lint format typecheck eval-baseline eval-enhanced eval-report docker-up

install:
	pip install -e ".[dev]"

run-ui:
	streamlit run app.py

run-api:
	uvicorn api.main:app --reload

test:
	pytest

lint:
	ruff check .

format:
	ruff format .

typecheck:
	mypy .

eval-baseline:
	python -m rag.eval --pipeline baseline

eval-enhanced:
	python -m rag.eval --pipeline enhanced

eval-report:
	python -m rag.eval_report

docker-up:
	docker compose up --build
