#!/usr/bin/env bash
# Sourced by the lane scripts, never executed. Anything after a lane script's
# name is appended to its pytest command.

set -euo pipefail

LANE_PWD="$PWD"
cd "$(dirname "${BASH_SOURCE[0]}")/../../.."
# On CI the checkout instead, whatever directory the step was invoked from:
# artifact_paths are relative to the checkout, so a step that cd'd first would
# otherwise leave results where the upload cannot see them.
[ -z "${BUILDKITE:-}" ] || LANE_PWD="$PWD"

section() {
  # Buildkite folds output on these markers: --- collapsed, +++ expanded.
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

# Keep a command's output in a file: `cmd 2>&1 | log_to "$log"`. On CI the stream
# is also the build log, so show it there as well; by hand the file is enough, and
# LANE_ECHO=1 shows it anyway.
log_to() {
  if [ -n "${BUILDKITE:-}${LANE_ECHO:-}" ]; then
    tee "$1"
  else
    cat >"$1"
  fi
}

run_pytest() {
  if [ -n "${BUILDKITE:-}" ]; then
    echo "+++ :pytest: pytest"
  fi
  uv run --no-sync pytest "$@"
}
