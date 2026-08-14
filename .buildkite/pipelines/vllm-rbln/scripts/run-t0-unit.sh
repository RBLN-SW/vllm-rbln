#!/usr/bin/env bash
# Without --model-compile the whole-model tests skip themselves.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

run_pytest tests/native -v --durations 25 "$@"
