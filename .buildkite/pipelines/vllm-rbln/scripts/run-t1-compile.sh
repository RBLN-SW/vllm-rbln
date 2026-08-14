#!/usr/bin/env bash
# --model-compile enables these tests; -m keeps the T0 tests from running again.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_env HF_TOKEN

args=(
  tests/native
  -m model_compile
  --model-compile
  -x
  -v
  --durations 25
)

# Passing it unconditionally would override each spec's own count.
if [ -n "${NUM_HIDDEN_LAYERS:-}" ]; then
  args+=(--num-hidden-layers "${NUM_HIDDEN_LAYERS}")
fi

run_pytest "${args[@]}" "$@"
