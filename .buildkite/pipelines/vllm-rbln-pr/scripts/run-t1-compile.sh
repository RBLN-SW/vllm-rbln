#!/usr/bin/env bash
# T1 -- whole-model lane. --model-compile enables it; -m selects only it, or
# every T0 test would run again on top.

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

# The option is the lane overruling every spec, so passing it unconditionally
# would silence each model's own count. T2 sets it to 0 on purpose.
if [ -n "${NUM_HIDDEN_LAYERS:-}" ]; then
  args+=(--num-hidden-layers "${NUM_HIDDEN_LAYERS}")
fi

sync_env
run_pytest "${args[@]}" "$@"
