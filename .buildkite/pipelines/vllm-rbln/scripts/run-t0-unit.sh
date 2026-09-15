#!/usr/bin/env bash
# Without --model-compile the whole-model tests skip themselves.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run_pytest tests/vllm -v --durations 25 "$@"
