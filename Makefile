PYTHON ?= python3

.PHONY: install check lint test smoke

install:
	$(PYTHON) -m pip install -e ".[dev]"

# CI-style target: lint, then the full test suite. Must pass before any commit that closes a phase.
check: lint test

lint:
	ruff check src tests
	ruff format --check src tests

test:
	$(PYTHON) -m pytest

# Dummy end-to-end run: random scores -> metrics -> results/<run_id>.jsonl
smoke:
	$(PYTHON) -m indispoof.cli smoke --config configs/phase0_smoke.yaml
