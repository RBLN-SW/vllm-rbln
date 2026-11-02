#!/usr/bin/env bash
# --model-compile enables these tests; -m excludes the unit tests.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_env HF_TOKEN

declare -A group_paths=(
  [basic_correctness]=tests/vllm/basic_correctness
  [compile]=tests/vllm/compile
  [dp]=tests/vllm/distributed
  [specdec]=tests/vllm/v1/spec_decode
  [core]=tests/vllm/v1/core
)

paths=(tests/vllm)
group=""
if [ "${1:-}" = "--group" ]; then
  if [ "$#" -lt 2 ]; then
    echo "--group requires a group name" >&2
    exit 2
  fi
  group="$2"
  if [ "$group" = "others" ]; then
    for path in "${group_paths[@]}"; do
      paths+=(--ignore "$path")
    done
  elif [ -n "$group" ] && [ -n "${group_paths[$group]:-}" ]; then
    paths=("${group_paths[$group]}")
  else
    echo "unknown e2e group: $group" >&2
    exit 2
  fi
  LANE_NAME="${LANE_NAME}-$group"
  shift 2
fi

args=(
  "${paths[@]}"
  -m model_compile
  --model-compile
  -x
  -v
  --durations 25
)

status=0
run_pytest "${args[@]}" "$@" || status=$?
# pytest returns 5 when no tests match; the remaining group can be empty.
if [ "$group" = "others" ] && [ "$status" -eq 5 ]; then
  echo "No e2e tests outside the named groups."
  exit 0
fi
exit "$status"
