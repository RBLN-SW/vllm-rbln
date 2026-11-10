#!/usr/bin/env bash
# lm-eval comes from an overlay rather than the locked environment: it
# is only needed here, and pinning it keeps a score comparable across runs, since
# the harness version decides the prompt and the filters.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_here}/../common.sh"

require_env HF_TOKEN

section ":dart: lm-eval"

export LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-${LANE_PWD}/lm-eval-results}"
log_dir="${LM_EVAL_OUTPUT_DIR}/logs"
mkdir -p "${log_dir}"
log="${log_dir}/${BUILDKITE_JOB_ID:-$(date +%Y%m%d_%H%M%S)}.log"

uv run --no-sync --with "lm_eval[api]==0.4.12" python "${_here}/run_lm_eval.py" \
  run "$@" 2>&1 | log_to "${log}"
