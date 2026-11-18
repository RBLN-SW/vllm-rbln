#!/usr/bin/env bash
# Whole-model tests run in the e2e jobs.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

paths=(tests/vllm)
if [ "${1:-}" = "--group" ]; then
  if [ "$#" -lt 2 ]; then
    echo "--group requires a group name" >&2
    exit 2
  fi
  group="$2"
  case "$group" in
    sample)
      paths=(
        tests/vllm/v1/sample
        --shard-count "${BUILDKITE_PARALLEL_JOB_COUNT:-1}"
        --shard-index "${BUILDKITE_PARALLEL_JOB:-0}"
      )
      ;;
    others) paths+=(--ignore tests/vllm/v1/sample) ;;
    *) echo "unknown unit group: $group" >&2; exit 2 ;;
  esac
  LANE_NAME="${LANE_NAME}-$group"
  shift 2
fi

run_pytest "${paths[@]}" -m 'not model_compile' -v --durations 25 "$@"
