#!/usr/bin/env bash
# --model-compile enables these tests; -m excludes the unit tests.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

require_env HF_TOKEN

paths=(tests/vllm)
if [ "${1:-}" = "--group" ]; then
  if [ "$#" -lt 2 ]; then
    echo "--group requires a group name" >&2
    exit 2
  fi
  case "$2" in
    basic_correctness) paths=(tests/vllm/basic_correctness) ;;
    model_compile) paths=(tests/vllm/compile) ;;
    dp) paths=(tests/vllm/distributed) ;;
    specdec) paths=(tests/vllm/v1/spec_decode) ;;
    core) paths=(tests/vllm/v1/core) ;;
    *)
      echo "unknown e2e group: $2" >&2
      exit 2
      ;;
  esac
  LANE_NAME="${LANE_NAME}-$2"
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

run_pytest "${args[@]}" "$@"
