.DEFAULT_GOAL := help
CONFIG ?= configs/default.yaml
CLI    := renewcast
ARGS   ?=

.PHONY: help install install-tf lint format test test-keras check-env \
        download targets preprocess eda baseline train evaluate report all smoke clean docker-build

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --- setup -------------------------------------------------------------------------------------
install:  ## Install the package with dev tools (CPU only: data prep, baselines, tests)
	pip install -e ".[dev]"

install-tf:  ## Additionally install TensorFlow/Keras (needed for train / evaluate)
	pip install -e ".[dev,tf,viz]"

check-env:  ## Print library versions and visible GPUs
	$(CLI) check-env

# --- quality -----------------------------------------------------------------------------------
lint:  ## Static checks (notebooks are excluded on purpose)
	ruff check src tests scripts
	ruff format --check src tests scripts

format:  ## Auto-format and fix lint issues
	ruff check --fix src tests scripts
	ruff format src tests scripts

test:  ## Fast CPU tests
	pytest

test-keras:  ## Tests that need a Keras backend (tiny synthetic data, no GPU needed)
	pytest -m keras

# --- pipeline stages ---------------------------------------------------------------------------
download:  ## Fetch HOSTRADA NetCDF files from DWD (several GB)
	$(CLI) download -c $(CONFIG) $(ARGS)

targets:  ## Clean the ENTSO-E CSV into hourly UTC targets
	$(CLI) prepare-targets -c $(CONFIG) $(ARGS)

preprocess:  ## NetCDF -> memory-mapped .npy cubes
	$(CLI) preprocess -c $(CONFIG) $(ARGS)

eda:  ## Missing-value chart and weather maps
	$(CLI) eda -c $(CONFIG) $(ARGS)

baseline:  ## CPU-only Ridge / gradient-boosting reference models
	$(CLI) baseline -c $(CONFIG) $(ARGS)

train:  ## Train the CNNs (GPU strongly recommended)
	$(CLI) train -c $(CONFIG) $(ARGS)

evaluate:  ## Score the latest runs and write figures
	$(CLI) evaluate -c $(CONFIG) $(ARGS)

report:  ## Collect metrics and figures into reports/
	$(CLI) report -c $(CONFIG) $(ARGS)

all:  ## Run every stage in order
	$(CLI) run-all -c $(CONFIG) $(ARGS)

smoke:  ## Small, fast end-to-end run to validate a new machine
	$(CLI) run-all -c configs/smoke_test.yaml --skip eda

# --- housekeeping ------------------------------------------------------------------------------
clean:  ## Remove caches (never touches data/ or artifacts/)
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov build dist src/*.egg-info
	find . -name __pycache__ -type d -prune -exec rm -rf {} +

docker-build:  ## Build the GPU image
	docker build -t renewcast:latest .
