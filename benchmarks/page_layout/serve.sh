#!/usr/bin/env bash
# Launch MiniMax-M2.5 in one of the two prefix-caching modes under comparison.
#
# Both modes run the same kernel block (8192) and the same prefill chunk (512);
# only the match unit and how it is reached differ, which is the whole point --
# anything else changing would make the comparison meaningless.
#
#   subblock    --block-size 8192, sub_block = chunk = 512  (VLLM_RBLN_SUB_BLOCK_CACHE)
#   pagelayout  --block-size 512 (page), attn_block_size 8192 (VLLM_RBLN_PAGE_LAYOUT)
#
# No KV connector: this compares vLLM-local mechanisms only.
#
# Usage: serve.sh <subblock|pagelayout>
# Env:   RBLN_DEVICES (default 4,5,6,7), PORT (8102), DP (4), HF_HOME,
#        MAX_MODEL_LEN (16384), MAX_NUM_SEQS (8), MODEL
set -uo pipefail
MODE=${1:?usage: serve.sh <subblock|pagelayout>}
REPO=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$REPO"
source "$REPO/.venv/bin/activate"
unset PYTHONPATH

MODEL=${MODEL:-MiniMaxAI/MiniMax-M2.5}
PORT=${PORT:-8102}
DP=${DP:-4}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-16384}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-8}
KERNEL_BLOCK=${KERNEL_BLOCK:-8192}
PAGE=${PAGE:-512}
CHUNK=${CHUNK:-512}

# MoE / EP runtime flags, shared with the e2e launchers.
export VLLM_RBLN_COMPILE_STRICT_MODE=1
export VLLM_RBLN_METRICS=1
export VLLM_RBLN_USE_VLLM_MODEL=1
export VLLM_RBLN_USE_MOE_TOKENS_MASK=1
export VLLM_RBLN_MOE_USE_OPT_KERNEL=1
export VLLM_RBLN_SAMPLER=1
export VLLM_RBLN_ENABLE_WARM_UP=1
export VLLM_RBLN_MOE_REDUCE_SCATTER=1
export VLLM_USE_V1=1
export VLLM_ENGINE_READY_TIMEOUT_S=14400
export VLLM_RPC_TIMEOUT=1800000
export RBLN_RSD_EP=1
export RBLN_MOE_OPT=1
export RBLN_FORCE_CCL_ASYNC=1
export RBLN_ROOT_IP=127.0.0.1
export RBLN_LOCAL_IP=127.0.0.1
export VLLM_RBLN_USE_DEVICE_TENSOR=1
export VLLM_DISABLE_COMPILE_CACHE=${VLLM_DISABLE_COMPILE_CACHE:-0}
# Block hashes must be reproducible across the two serves being compared.
export PYTHONHASHSEED=0
export RBLN_DEVICES=${RBLN_DEVICES:-4,5,6,7}

case "$MODE" in
  subblock)
    export VLLM_RBLN_SUB_BLOCK_CACHE=1 VLLM_RBLN_PAGE_LAYOUT=0
    BLOCK_ARGS=(--block-size "$KERNEL_BLOCK")
    ;;
  pagelayout)
    export VLLM_RBLN_SUB_BLOCK_CACHE=0 VLLM_RBLN_PAGE_LAYOUT=1
    BLOCK_ARGS=(--block-size "$PAGE"
                --additional-config "{\"attn_block_size\": $KERNEL_BLOCK}")
    ;;
  *) echo "unknown mode $MODE (expected subblock or pagelayout)"; exit 2 ;;
esac

echo "[launch] $(date '+%F %T') mode=$MODE devices=$RBLN_DEVICES dp=$DP" \
     "kernel_block=$KERNEL_BLOCK args=${BLOCK_ARGS[*]}"
exec vllm serve "$MODEL" \
    --port "$PORT" --served-model-name MiniMax \
    --data-parallel-size "$DP" --enable-expert-parallel \
    --max-model-len "$MAX_MODEL_LEN" "${BLOCK_ARGS[@]}" \
    --enable-chunked-prefill --max-num-batched-tokens "$CHUNK" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --gpu-memory-utilization 0.8 --trust-remote-code
