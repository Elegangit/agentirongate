PYTHON ?= python

.PHONY: test lint validate identity smoke
test:
	$(PYTHON) -m pytest
lint:
	$(PYTHON) -m ruff check src tests
validate:
	$(PYTHON) -m gatebench validate
identity:
	$(PYTHON) -m gatebench identity
smoke:
	$(PYTHON) -m gatebench run --output outputs/smoke
	$(PYTHON) -m gatebench analyze outputs/smoke
	$(PYTHON) -m gatebench plot outputs/smoke
