#!/usr/bin/env bash
# T0 -- unit lane. No --model-compile, so the whole-model lane stays skipped.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

sync_env
run_pytest tests/native -v --durations 25 "$@"
