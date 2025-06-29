###############################################################################
# COMMANDS
###############################################################################
.PHONY: clean
## Clean python cache file.
clean:
	find . -path './.venv' -prune -o -name '*.pyo' -delete
	find . -path './.venv' -prune -o -name '*.pyc' -delete
	find . -path './.venv' -prune -o -name '__pycache__' -delete
	find . -path './.venv' -prune -o -name '*~' -delete
	find . -path './.venv' -prune -o -name '.coverage' -delete
	find . -path './.venv' -prune -o -name 'coverage.*' -delete
	find . -path './.venv' -prune -o -name 'requirements*.txt' -delete
	find . -path './.venv' -prune -o -name 'ruff.log' -delete
	find . -path './.venv' -prune -o -type d -name '.pytest_cache' -exec rm -r {} +
	find . -path './.venv' -prune -o -type d -name '.ruff_cache' -exec rm -r {} +
	find . -path './.venv' -prune -o -type d -name '.mypy_cache' -exec rm -r {} +
	find . -path './.venv' -prune -o -type d -name 'dist' -exec rm -r {} +
	find . -path './.venv' -prune -o -type d -name 'build' -exec rm -r {} +
	find . -path './.venv' -prune -o -type d -name '*.egg-info' -exec rm -r {} +

.PHONY: lint
## pylint check
lint:
	ruff check src/subtitle_alchemy --show-source --show-fixes \
	    --exit-zero

.PHONY: test
test:
	PYTHONPATH=src \
	    pytest -s -v tests > coverage.txt
