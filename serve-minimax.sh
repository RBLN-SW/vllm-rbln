#!/usr/bin/env bash
export HF_HOME=/mnt/shared_data/.cache/huggingface
export PYTHONHASHSEED=42
export VLLM_RBLN_DISABLE_OFFLOAD=1
export VLLM_ENGINE_READY_TIMEOUT_S=4800
export VLLM_RPC_TIMEOUT=1800000
export VLLM_RBLN_USE_DEVICE_TENSOR=1
export VLLM_RBLN_COMPILE_STRICT_MODE=1
export VLLM_DISABLE_COMPILE_CACHE=1
export VLLM_RBLN_USE_VLLM_MODEL=1
export VLLM_RBLN_AUTO_PORT=1
export VLLM_RBLN_SUB_BLOCK_CACHE=0
export VLLM_RBLN_SAMPLER="${VLLM_RBLN_SAMPLER:-1}"
export VLLM_RBLN_MOE_REDUCE_SCATTER=1
export VLLM_RBLN_BATCH_ATTN_OPT=1
export VLLM_RBLN_METRICS=1
export VLLM_RBLN_SORT_BATCH=1
export RBLN_WEIGHT_FREE=1
export RBLN_FORCE_CCL_ASYNC=1
export RBLN_RUNTIME_FORCE_SYNC=1
export SAFETENSORS_FAST_GPU=1
export RBLN_COMPILER_LOG_LEVEL=3
export VLLM_LOGGING_LEVEL=DEBUG
export RBLN_ROOT_IP=127.0.0.1
export RBLN_LOCAL_IP=127.0.0.1
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.8}"

SPEC="${SPEC:-on}"
SPEC_ARGS=()
if [ "$SPEC" = "on" ]; then
  SPEC_ARGS=(--speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}')
fi

# Name the ITL metrics JSON after the run's knobs (max-num-seqs / spec / sampler)
# so parallel sweeps don't clobber each other. Flags normalize to 1/0.
MNS="${MAX_NUM_SEQS:-8}"
SPEC_FLAG=0
[ "$SPEC" = "on" ] && SPEC_FLAG=1
SAMPLER_FLAG=0
case "$VLLM_RBLN_SAMPLER" in 1 | true | True | TRUE) SAMPLER_FLAG=1 ;; esac
export VLLM_RBLN_METRICS_JSON_FILE="max-num-seqs=${MNS}_spec_=${SPEC_FLAG}_sampler=${SAMPLER_FLAG}.json"

exec vllm serve MiniMaxAI/MiniMax-M2.5 \
  --port 8001 \
  --trust-remote-code \
  --disable-uvicorn-access-log \
  --enable-expert-parallel \
  --gpu-memory-utilization 0.8 \
  --max-model-len 65536 \
  --max-num-seqs "${MNS}" \
  --max-num-batched-tokens 512 \
  --block-size 1024 \
  --enable-prefix-caching \
  --safetensors-load-strategy prefetch \
  --enable-chunked-prefill \
  --enable-auto-tool-choice \
  --tool-call-parser minimax_m2 \
  --reasoning-parser minimax_m2 \
  "${SPEC_ARGS[@]}" \
  --data-parallel-size 4
