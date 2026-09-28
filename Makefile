# Developer shortcuts for rapidtools. Run `make help` to list the targets.
#
# The Python interpreter can be overridden, e.g. `make test PYTHON=python3.12`.

PYTHON ?= python
SRC     = rapidtools tests
DOCS    = docs/source
SITE    = docs/build/html

.DEFAULT_GOAL := help

.PHONY: help install lint format typecheck check test coverage docs docs-serve clean

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## Editable install with the test and docs extras
	$(PYTHON) -m pip install -e ".[dev]"

lint:  ## Ruff lint and formatting check (the style gate used in CI)
	$(PYTHON) -m ruff check $(SRC)
	$(PYTHON) -m ruff format --check $(SRC)

format:  ## Apply Ruff fixes and formatting
	$(PYTHON) -m ruff check $(SRC) --fix
	$(PYTHON) -m ruff format $(SRC)

typecheck:  ## Static type check of the package (advisory: not yet clean, so not part of `check`)
	$(PYTHON) -m mypy rapidtools

check: lint test  ## What a pull request must pass: lint and tests

test:  ## Run the test suite (coverage summary comes from pyproject's pytest options)
	$(PYTHON) -m pytest

coverage:  ## Run the test suite and write an HTML coverage report to htmlcov/
	$(PYTHON) -m pytest --cov-report=html

docs:  ## Build the documentation site into docs/build/html
	$(PYTHON) -m sphinx -b html $(DOCS) $(SITE)

docs-serve: docs  ## Build the docs and serve them at http://localhost:8000
	$(PYTHON) -m http.server --directory $(SITE) 8000

clean:  ## Remove build, cache and coverage artifacts
	rm -rf build dist *.egg-info .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov
	rm -rf $(SITE) docs/build/doctrees $(DOCS)/api/generated
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
