.PHONY: install serve test

install:
	python -m venv .venv
	.venv/bin/pip install -e ".[dev]"

serve:
	.venv/bin/python server.py

test:
	.venv/bin/python -m pytest tests/test_suite.py -v
