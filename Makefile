.PHONY: install test lint check seed api eval

install:
	uv sync

test:
	.venv/bin/pytest -q

lint:
	.venv/bin/ruff check .

check: lint test

seed:
	.venv/bin/retailops seed

api:
	.venv/bin/retailops serve

eval:
	.venv/bin/retailops eval-run
