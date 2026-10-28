#!/usr/bin/env bash
# Whole-model tests run in the e2e jobs.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run_pytest tests/vllm -m 'not model_compile' -v --durations 25 "$@"
