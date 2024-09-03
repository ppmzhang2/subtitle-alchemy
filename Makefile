###############################################################################
# COMMANDS
###############################################################################
.PHONY: clean
## Clean python cache file.
clean:
	find . -name '*.pyo' -delete
	find . -name '*.pyc' -delete
	find . -name __pycache__ -delete
	find . -name '*~' -delete
	find . -name .coverage -delete
	find . -name 'coverage.*' -delete
	find . -name 'requirements*.txt' -delete
	find . -type d -name .pytest_cache -exec rm -r {} +
	find . -type d -name .ruff_cache -exec rm -r {} +
	find . -type d -name .mypy_cache -exec rm -r {} +

.PHONY: lint
## pylint check
lint:
	ruff check src/rcnn --show-source --show-fixes \
	    --exit-zero

.PHONY: test
test:
	PYTHONPATH=. \
	    pytest -s -v --cov=src --cov-config=pyproject.toml \
	    > coverage.txt