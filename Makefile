#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = Deduplication-experiments
PYTHON_VERSION = 3.12

# Use the project's virtual environment when it exists, otherwise the Python on PATH.
# The path is quoted because on Windows make hands recipes that contain quotes to
# cmd.exe, which cannot run an unquoted path with forward slashes.
ifeq ($(OS),Windows_NT)
VENV_PYTHON = .venv/Scripts/python.exe
else
VENV_PYTHON = .venv/bin/python
endif
PYTHON_INTERPRETER = $(if $(wildcard $(VENV_PYTHON)),"$(VENV_PYTHON)",python)

# Python randomises string hashing per process, which changes the iteration
# order of sets of strings; a fixed hash seed keeps runs repeatable across processes.
export PYTHONHASHSEED = 0

#################################################################################
# COMMANDS                                                                      #
#################################################################################

## Install the package with its pinned dependencies and all extras
.PHONY: requirements
requirements:
	$(PYTHON_INTERPRETER) -m pip install -U pip
	$(PYTHON_INTERPRETER) -m pip install -e ".[dev,embeddings,pyjedai]"

## Delete Python caches
.PHONY: clean
clean:
	$(PYTHON_INTERPRETER) -c "import pathlib, shutil; [shutil.rmtree(p) for p in pathlib.Path('.').rglob('__pycache__') if '.venv' not in p.parts]; [shutil.rmtree(p, ignore_errors=True) for p in ('.pytest_cache', '.ruff_cache')]"

## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	$(PYTHON_INTERPRETER) -m ruff format --check
	$(PYTHON_INTERPRETER) -m ruff check

## Format source code with ruff
.PHONY: format
format:
	$(PYTHON_INTERPRETER) -m ruff check --fix
	$(PYTHON_INTERPRETER) -m ruff format

## Run tests
.PHONY: test
test:
	$(PYTHON_INTERPRETER) -m pytest

## Set up the Python virtual environment in .venv
.PHONY: create_environment
create_environment:
	python -m venv .venv
	@echo Created .venv. Run make requirements next; the Makefile uses .venv automatically.

#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Download the datasets and prepare them in the common format
.PHONY: data
data:
	$(PYTHON_INTERPRETER) -m dedup_bench.dataset download
	$(PYTHON_INTERPRETER) -m dedup_bench.dataset prepare

## Run every library once on Febrl 3 and the restaurants (needs make data)
.PHONY: smoke
smoke:
	$(PYTHON_INTERPRETER) -m dedup_bench.smoke run

## Split every dataset and build its frozen candidate pairs (needs make data)
.PHONY: candidates
candidates:
	$(PYTHON_INTERPRETER) -m dedup_bench.blocking.candidates

## Tune and test one method on one dataset, all seeds (DATASET=... METHOD=section.model)
.PHONY: experiment
experiment:
	$(PYTHON_INTERPRETER) -m dedup_bench.experiment run $(DATASET) $(METHOD)

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = open(sys.argv[1], encoding='utf-8').read(); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" $(MAKEFILE_LIST)
