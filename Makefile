.PHONY: format lint test test_integration typing

format:
	uv run --group lint ruff format .
	uv run --group lint ruff check --fix .

lint:
	uv run --group lint ruff check .
	uv run --group lint ruff format --check .

test:
	uv run --group test pytest tests/unit_tests

test_integration:
	uv run --group test pytest tests/integration_tests

typing:
	uv run --group typing mypy langchain_diffbot
