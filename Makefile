PYTHON ?= python
RUFF_PATHS := scripts/verify_repo.py .github/scripts/pr_hygiene.py

.PHONY: help lint verify-source verify-historical verify-final verify-paper verify-all

help:
	@echo "make lint               lint repository-maintained Python tooling"
	@echo "make verify-source      verify delivered source files against the manifest"
	@echo "make verify-historical  verify carried-forward evidence and recorded results"
	@echo "make verify-final       verify final evidence inputs"
	@echo "make verify-paper       verify report and presentation readiness"
	@echo "make verify-all         run every available verification phase"

lint:
	ruff check $(RUFF_PATHS)
	ruff format --check $(RUFF_PATHS)

verify-source:
	$(PYTHON) scripts/verify_repo.py --source --require-complete

verify-historical:
	$(PYTHON) scripts/verify_repo.py --historical --require-complete

verify-final:
	$(PYTHON) scripts/verify_repo.py --final-evidence --require-complete

verify-paper:
	$(PYTHON) scripts/verify_repo.py --paper --require-complete

verify-all:
	$(PYTHON) scripts/verify_repo.py --all --require-complete
