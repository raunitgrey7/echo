# echo: speech intelligence for lectures and meetings.
# Every target runs through uv so the same interpreter and lock file are used everywhere.

UV ?= uv
PROFILE ?= full
BACKEND ?= auto
MODEL ?= base
PORT ?= 8000

.PHONY: setup lint format test bench bench-quick data serve docker clean

setup:            ## install the locked environment (Python 3.13 via uv)
	$(UV) sync --all-groups

lint:             ## ruff lint and format check
	$(UV) run ruff check src tests
	$(UV) run ruff format --check src tests

format:           ## apply ruff formatting
	$(UV) run ruff format src tests
	$(UV) run ruff check --fix src tests

test:             ## full test suite
	$(UV) run pytest

data:             ## build the synthetic corpus (needs macOS say)
	$(UV) run python -m echo.cli build-data --profile $(PROFILE)

bench:            ## build data if needed, run the benchmark, write benchmarks/results.{json,md}, check thresholds
	$(UV) run python -m echo.bench --profile $(PROFILE) --backend $(BACKEND) --model $(MODEL) --check benchmarks/thresholds.json

bench-quick:      ## small grid for development and CI
	$(UV) run python -m echo.bench --profile quick --backend $(BACKEND) --model $(MODEL) --check benchmarks/thresholds.json

serve:            ## API and dashboard on http://127.0.0.1:$(PORT)
	$(UV) run uvicorn echo.api.app:app --host 127.0.0.1 --port $(PORT)

docker:           ## build the container image
	docker build -t echo:latest .

clean:            ## remove caches, jobs and generated recordings (keeps the cached utterances)
	rm -rf .pytest_cache .ruff_cache data/jobs data/recordings data/manifest_*.json
	find . -name __pycache__ -type d -prune -exec rm -rf {} +
