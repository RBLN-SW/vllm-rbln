#!/usr/bin/env bash
# Performance sweep.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_here}/../common.sh"

require_env HF_TOKEN

sync_env
section ":chart_with_upwards_trend: perf"
# The driver reads this, so where results land has one source. Defaulted here
# rather than there because only this side knows where the caller was.
export PERF_OUTPUT_DIR="${PERF_OUTPUT_DIR:-${LANE_PWD}/perf-results}"

uv run --no-sync python "${_here}/run_perf.py" "$@"
