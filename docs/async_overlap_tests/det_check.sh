#!/usr/bin/env bash
# Determinism check: run ONE config N times (repro) in a single process and
# report whether every run's per-input output TEXT is bit-identical.
#
# Env:
#   OUTDIR   output dir (default ./_test_out)
#   NAME     label for this run (default run)
#   CFGENV   config env, e.g.:
#              full async: "VLLM_RBLN_OPTIMISTIC_SCHED=1 RBLN_DYNAMO_ASYNC=defer"
#              sync (ref): "VLLM_RBLN_DISABLE_ASYNC=1"
#   BATCH    per-rank max_num_seqs (default 4)
#   REPRO    number of repro runs to compare (default 12)
#   MAXTOK   max output tokens (default 128)
#   DP       data-parallel size (default 4)
#   RBLN_DEVICES  comma list of device ids (default 4,5,6,7)
#   VENV     path to the test venv (default ~/codebase/vllm-executor/.venv)
set -u
VENV=${VENV:-$HOME/codebase/vllm-executor/.venv}
source "$VENV/bin/activate"

OUTDIR=${OUTDIR:-./_test_out}; mkdir -p "$OUTDIR/det_logs"
NAME=${NAME:-run}; CFGENV=${CFGENV:-}; BATCH=${BATCH:-4}
REPRO=${REPRO:-12}; MAXTOK=${MAXTOK:-128}; DP=${DP:-4}
export RBLN_DEVICES=${RBLN_DEVICES:-4,5,6,7}
LOG="$OUTDIR/det_logs/${NAME}.log"

pkill -9 -u "$(id -u)" -f "vllm_rbln_exec.parity_runner" 2>/dev/null || true
find /dev/shm -maxdepth 1 -uid "$(id -u)" -delete 2>/dev/null || true
sleep 3

COMMON="VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error VLLM_RBLN_AUTO_PORT=1 RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SPDLOG_LEVEL=warning RBLN_VERBOSE=warning VLLM_LOGGING_LEVEL=WARNING RBLN_DEVICES=$RBLN_DEVICES"
ARGS="--task r --model gpt-oss-120b --ep --dp $DP --rsd 1 --trust-remote-code --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 --batch $BATCH --max-num-blocks 129 --max-tokens $MAXTOK --num-prompts $BATCH --logprobs 0 --no-inspect-logits --repro-run $REPRO --use-cached-models --cache-ignore --skip-validation --mode 0"

echo "=== det_check NAME=$NAME BATCH=$BATCH REPRO=$REPRO CFGENV='$CFGENV' devices=$RBLN_DEVICES ==="
# shellcheck disable=SC2086
( env $COMMON $CFGENV python3 -m vllm_rbln_exec.parity_runner $ARGS ) >"$LOG" 2>&1
rc=$?
if grep -q "Reproduce Test Done" "$LOG"; then repro=PASS
elif grep -q "Non-deterministic" "$LOG"; then repro=FAIL
else repro=NA; fi
abort=$(grep -c "SYS_TASK_ABORTED" "$LOG")
mm=$(grep -c "mismatch in" "$LOG")
echo "DET_RESULT NAME=$NAME batch=$BATCH repro_run=$REPRO rc=$rc repro=$repro abort=$abort mismatches=$mm log=$LOG"
