#!/usr/bin/env bash
# --model-compile enables these tests; -m keeps the T0 tests from running again.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# HF_HOME matters more under HF_HUB_OFFLINE: an unset one silently becomes a
# cold ~/.cache/huggingface, which offline mode turns into a confusing
# LocalEntryNotFoundError deep inside a test instead of a slow first download.
require_env HF_TOKEN HF_HOME

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
