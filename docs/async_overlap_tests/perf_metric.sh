#!/usr/bin/env bash
# Throughput sweep. Metric = total_output_tokens / generate wall-clock, WARM runs
# only (cold/compile run = call 1 is dropped). Per-rank batch B = max_num_seqs;
# num_prompts = B*dp so each DP rank runs B concurrent sequences (DP shards the
# prompts across ranks). Global tok/s is summed across ranks by perf_parse.py.
# Requires the RBLN_PERF_TIMING instrumentation in _run_llm (see the .patch here).
#
# Env: OUTDIR (default ./_test_out), BATCHES (default "1 4 8 16"), REPRO (default
#   2 = 1 cold + 2 warm), MAXTOK (128), DP (4), RBLN_DEVICES (4,5,6,7), VENV.
set -u
VENV=${VENV:-$HOME/codebase/vllm-executor/.venv}
source "$VENV/bin/activate"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OUTDIR=${OUTDIR:-./_test_out}; mkdir -p "$OUTDIR/perf_logs"
SUMMARY="$OUTDIR/perf_metric_summary.txt"; : > "$SUMMARY"
DP=${DP:-4}; MAXTOK=${MAXTOK:-128}; REPRO=${REPRO:-2}
BATCHES=${BATCHES:-"1 4 8 16"}
export RBLN_DEVICES=${RBLN_DEVICES:-4,5,6,7}

COMMON="VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error VLLM_RBLN_AUTO_PORT=1 RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 SPDLOG_LEVEL=warning RBLN_VERBOSE=warning VLLM_LOGGING_LEVEL=WARNING RBLN_PERF_TIMING=1 RBLN_DEVICES=$RBLN_DEVICES"

cleanup(){ pkill -9 -u "$(id -u)" -f "vllm_rbln_exec.parity_runner" 2>/dev/null; find /dev/shm -maxdepth 1 -uid "$(id -u)" -delete 2>/dev/null; sleep 4; }

run(){ # $1=cfgname $2=cfgenv $3=B
  local name="$1" cfgenv="$2" B="$3"; local nprompts=$((B * DP))
  local logf="$OUTDIR/perf_logs/${name}_b${B}.log"; cleanup
  local ARGS="--task r --model gpt-oss-120b --ep --dp $DP --rsd 1 --trust-remote-code --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 --batch $B --max-num-blocks 129 --max-tokens $MAXTOK --num-prompts $nprompts --logprobs 0 --no-inspect-logits --repro-run $REPRO --use-cached-models --cache-ignore --skip-validation --mode 0"
  echo "=== RUN cfg=$name B=$B num_prompts=$nprompts ==="
  # shellcheck disable=SC2086
  ( env $COMMON $cfgenv python3 -m vllm_rbln_exec.parity_runner $ARGS ) >"$logf" 2>&1
  echo "  rc=$? log=$(basename "$logf")"
}

for b in $BATCHES; do run async "VLLM_RBLN_OPTIMISTIC_SCHED=1 RBLN_DYNAMO_ASYNC=defer" "$b"; done
for b in $BATCHES; do run sync  "VLLM_RBLN_DISABLE_ASYNC=1" "$b"; done

echo "=== PARSING ==="
python3 "$HERE/perf_parse.py" "$OUTDIR/perf_logs" "$DP" | tee "$SUMMARY"
