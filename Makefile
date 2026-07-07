.PHONY: test lint typecheck pytest

test: lint typecheck pytest

lint:
	ruff check src tests

typecheck:
	mypy src

pytest:
	pytest -q
