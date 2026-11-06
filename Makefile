# `make check` is the merge gate: run it before opening a PR. CI runs the
# same target (.github/workflows/check.yml). CPU only; see tests/README.md.
PYTHON ?= python

.PHONY: check check-ace2 lint test

check: lint test

lint:
	$(PYTHON) -m ruff check .

test:
	$(PYTHON) -m pytest -m "not gpu and not ace2"

# The mini-ACE2 solver smoke test (tests/ace2/): needs fme
# (requirements-ci-ace2.txt) and downloads its checkpoint on first use.
check-ace2:
	$(PYTHON) -m pytest -m ace2
