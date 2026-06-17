PYTHON ?= python3

.PHONY: install run test lint typecheck

install:
	$(PYTHON) -m pip install -r requirements.txt

run:
	PYTHONPATH=src $(PYTHON) -m analytics_app.presentation.cli

test:
	PYTHONPATH=src $(PYTHON) -m pytest -q

lint:
	PYTHONPATH=src $(PYTHON) -m ruff check src tests

typecheck:
	PYTHONPATH=src $(PYTHON) -m mypy src
