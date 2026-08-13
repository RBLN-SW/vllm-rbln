#!/usr/bin/env bash
# T2 is T1's tests at full depth, so T1 is not repeated. A failed lane stops
# before perf, which is the gate on releasing.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_here}/common.sh"

bash "${_here}/run-t0-unit.sh"
bash "${_here}/run-t2-compile-full.sh"
bash "${_here}/perf/run-perf.sh"
