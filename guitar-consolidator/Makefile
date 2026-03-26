.PHONY: install serve test

PYTHON ?= python3

install:
	$(PYTHON) -m venv .venv
	.venv/bin/pip install -e ".[dev]"

PORT ?= 8080

serve:
	.venv/bin/python server.py $(PORT)

test:
	.venv/bin/python -m pytest tests/test_suite.py -v
