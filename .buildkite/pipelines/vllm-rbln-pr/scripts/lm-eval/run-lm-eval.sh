#!/usr/bin/env bash
# lm-eval comes from an overlay rather than the locked environment: it
# is only needed here, and pinning it keeps a score comparable across runs, since
# the harness version decides the prompt and the filters.

_here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${_here}/../common.sh"

require_env HF_TOKEN

sync_env
section ":dart: lm-eval"

run_id="$(date +%Y%m%d_%H%M%S)"
# The driver reads this, so where results land has one source. Defaulted here
# rather than there because only this side knows where the caller was.
export LM_EVAL_OUTPUT_DIR="${LM_EVAL_OUTPUT_DIR:-${LANE_PWD}/lm-eval-results}"
out="${LM_EVAL_OUTPUT_DIR}"

# Arguments are target names. A flag would otherwise become one and write a
# directory named after it.
for arg in "$@"; do
  case "$arg" in
    -*)
      echo "$arg: arguments are target names; set the output with LM_EVAL_OUTPUT_DIR" >&2
      exit 2
      ;;
  esac
done

# One target per invocation of the driver: that is what lets the log live with
# the results it describes, since the server's output is inherited rather than
# piped and only this level knows which target is running.
targets=("$@")
if [ ${#targets[@]} -eq 0 ]; then
  # A glob that matches nothing is left as the pattern itself, which would
  # become a target named "*".
  for f in "${_here}"/targets/*.yaml "${_here}"/targets/*.yml; do
    [ -e "$f" ] && targets+=("$(basename "${f%.*}")")
  done
  # Otherwise a targets directory that has moved or been emptied is a run that
  # does nothing and reports success.
  if [ ${#targets[@]} -eq 0 ]; then
    echo "no targets under ${_here}/targets" >&2
    exit 1
  fi
fi

status=0
for target in "${targets[@]}"; do
  log_dir="${out}/${run_id}/${target}"
  mkdir -p "${log_dir}"
  log="${log_dir}/lm-eval.log"
  echo "  ${target} -> ${log}"

  uv run --no-sync --with "lm_eval[api]==0.4.12" python "${_here}/run_lm_eval.py" \
    --run-id "${run_id}" "${target}" 2>&1 | log_to "${log}" ||
    {
      echo "  ${target} failed; see ${log}" >&2
      status=1
    }
done
exit "${status}"
