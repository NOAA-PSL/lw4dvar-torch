# `make check` is the merge gate: run it before opening a PR. CI runs the
# same target (.github/workflows/check.yml). CPU only; see tests/README.md.
PYTHON ?= python

.PHONY: check lint test

check: lint test

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -m "not gpu"
