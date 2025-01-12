.PHONY: setup clean test format lint

PYTHON := python3
VENV := venv
BIN := $(VENV)/bin

setup: $(VENV)/touch

$(VENV)/touch: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -r requirements.txt
	touch $(VENV)/touch

clean:
	rm -rf $(VENV)
	rm -rf __pycache__
	rm -rf .pytest_cache
	rm -rf *.pyc
	rm -rf .coverage
	rm -rf htmlcov

test:
	$(BIN)/pytest -v tests/

format:
	$(BIN)/black .

lint:
	$(BIN)/flake8 .

run:
	$(BIN)/python team_optimizer.py

.DEFAULT_GOAL := setup