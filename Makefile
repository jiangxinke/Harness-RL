SHELL := /usr/bin/env bash

PROJECT_CACHE ?= $(HOME)/.cache/harness-rl
SP_ENV_PREFIX ?= $(PROJECT_CACHE)/training-env
SP_SOURCE_ROOT ?= $(PROJECT_CACHE)/sources
SP_BUILD_CACHE ?= $(PROJECT_CACHE)/build
SP_BUILD_LOG ?= $(SP_BUILD_CACHE)/setup.log
SP_SEARCH_R1_ROOT ?= $(PROJECT_CACHE)/search-r1
CONFIG ?= configs/experiment_a_central.yaml
PYTHON ?= python3

.PHONY: help doctor setup resume verify status test lint prepare-data prepare-eval \
	search-setup search-verify search-start configure-runtime frozen-start ray-start \
	preflight train lock-training-runtime

help:
	@echo "Harness-RL"
	@echo "  make doctor            Check host requirements"
	@echo "  make setup             Create the locked training environment"
	@echo "  make resume            Resume an interrupted environment build"
	@echo "  make verify            Verify dependencies and run repository checks"
	@echo "  make prepare-data      Download and convert the training dataset"
	@echo "  make prepare-eval      Build the evaluation manifest"
	@echo "  make search-setup      Create the Search-R1 environment"
	@echo "  make search-start      Start the Search-R1 retriever"
	@echo "  make configure-runtime Write StackPlanner2.0/config.yaml"
	@echo "  make frozen-start      Start the frozen specialist endpoint"
	@echo "  make ray-start         Start a single-node Ray runtime"
	@echo "  make preflight         Print and validate CONFIG"
	@echo "  make train             Launch CONFIG"
	@echo "  make test              Run Harness-RL tests"
	@echo "  make lint              Run Ruff on Harness-RL code"

lock-training-runtime:
	@uv pip compile \
	  StackPlanner2.0/backend/packages/harness/pyproject.toml \
	  requirements/training-runtime.in \
	  --constraints requirements/training-runtime-constraints.txt \
	  --python-version 3.12 \
	  --python-platform x86_64-manylinux_2_28 \
	  --generate-hashes --no-annotate \
	  --custom-compile-command 'make lock-training-runtime' \
	  --output-file requirements/training-runtime.lock

doctor:
	@SP_ENV_PREFIX="$(SP_ENV_PREFIX)" SP_SOURCE_ROOT="$(SP_SOURCE_ROOT)" \
	  SP_BUILD_CACHE="$(SP_BUILD_CACHE)" bash scripts/doctor_training_host.sh

setup: doctor
	@SP_ENV_PREFIX="$(SP_ENV_PREFIX)" SP_SOURCE_ROOT="$(SP_SOURCE_ROOT)" \
	  SP_BUILD_CACHE="$(SP_BUILD_CACHE)" SP_BUILD_LOG="$(SP_BUILD_LOG)" \
	  bash scripts/setup_training_environment.sh

resume: doctor
	@SP_ENV_PREFIX="$(SP_ENV_PREFIX)" SP_SOURCE_ROOT="$(SP_SOURCE_ROOT)" \
	  SP_BUILD_CACHE="$(SP_BUILD_CACHE)" SP_BUILD_LOG="$(SP_BUILD_LOG)" \
	  SP_RESUME_BUILD=1 bash scripts/setup_training_environment.sh

verify:
	@SP_ENV_PREFIX="$(SP_ENV_PREFIX)" SP_SOURCE_ROOT="$(SP_SOURCE_ROOT)" \
	  bash scripts/verify_training_environment.sh

status:
	@SP_ENV_PREFIX="$(SP_ENV_PREFIX)" SP_SOURCE_ROOT="$(SP_SOURCE_ROOT)" \
	  bash scripts/training_environment_status.sh

prepare-data:
	@$(PYTHON) scripts/prepare_asearcher.py

prepare-eval:
	@$(PYTHON) scripts/prepare_evaluation.py

search-setup:
	@SP_SEARCH_R1_ROOT="$(SP_SEARCH_R1_ROOT)" \
	  bash scripts/setup_search_r1_environment.sh

search-verify:
	@SP_SEARCH_R1_ROOT="$(SP_SEARCH_R1_ROOT)" \
	  bash scripts/verify_search_r1_environment.sh

search-start:
	@SP_SEARCH_R1_ROOT="$(SP_SEARCH_R1_ROOT)" \
	  bash scripts/start_search_r1_retriever.sh

configure-runtime:
	@bash scripts/configure_stackplanner_runtime.sh

frozen-start:
	@SP_ENV_PREFIX="$(SP_ENV_PREFIX)" bash scripts/start_frozen_sglang_server.sh

ray-start:
	@SP_ENV_PREFIX="$(SP_ENV_PREFIX)" bash scripts/start_local_training_ray.sh

preflight:
	@$(PYTHON) train_stackplanner.py --config "$(CONFIG)" --print-command

train:
	@$(PYTHON) train_stackplanner.py --config "$(CONFIG)"

test:
	@$(PYTHON) -m pytest tests -q

lint:
	@$(PYTHON) -m ruff check train_stackplanner.py scripts \
	  slime/slime_plugins/stackplanner tests
