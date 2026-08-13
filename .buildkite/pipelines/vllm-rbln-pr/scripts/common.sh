#!/usr/bin/env bash
# Sourced by the lane scripts, never executed.

set -euo pipefail

LANE_PWD="$PWD"
cd "$(dirname "${BASH_SOURCE[0]}")/../../../.."
[ -f pyproject.toml ] || { echo "not the repo root: $PWD" >&2; exit 1; }
# artifact_paths are relative to the checkout, so results have to land there.
[ -z "${BUILDKITE:-}" ] || LANE_PWD="$PWD"

LANE_NAME="${LANE_NAME:-$(basename "$0" .sh | sed 's/^run-//')}"

section() {
  # Buildkite folds output at these markers: --- collapsed, +++ expanded.
  if [ -n "${BUILDKITE:-}" ]; then
    echo "--- $1"
  else
    echo "==> $1"
  fi
}

sync_env() {
  [ -n "${BUILDKITE:-}" ] || return 0
  section ":package: uv sync"
  uv sync --locked --python 3.12 --extra test --extra dev --extra runtime
}

require_env() {
  local name
  for name in "$@"; do
    if [ -n "${!name:-}" ]; then
      continue
    fi
    if [ -n "${BUILDKITE:-}" ]; then
      echo "$name is not set; this lane needs it." >&2
      exit 1
    fi
    echo "warning: $name is not set; anything that downloads will fail." >&2
  done
}

# cmd 2>&1 | log_to "$log"
log_to() {
  if [ -n "${BUILDKITE:-}${LANE_ECHO:-}" ]; then
    tee "$1"
  else
    cat >"$1"
  fi
}

run_pytest() {
  if [ -n "${BUILDKITE:-}" ]; then
    echo "--- :pytest: pytest (${LANE_NAME})"
  fi
  uv run --no-sync pytest "$@"
}
