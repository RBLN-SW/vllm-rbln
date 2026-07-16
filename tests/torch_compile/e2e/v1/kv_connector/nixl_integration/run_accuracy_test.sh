#!/bin/bash
set -xe

# Parse command line arguments
KV_BUFFER_DEVICE="cpu"  # Default to cpu
while [[ $# -gt 0 ]]; do
  case $1 in
    --kv_buffer_device)
      KV_BUFFER_DEVICE="$2"
      shift 2
      ;;
    *)
      echo "Unknown option $1"
      echo "Usage: $0 [--kv_buffer_device <rbln|cpu>]"
      exit 1
      ;;
  esac
done

echo "Running accuracy tests with kv_buffer_device=$KV_BUFFER_DEVICE"

# Build the kv-transfer-config once. kv_buffer_device must be set explicitly
# for BOTH host-bounce ("cpu") and D2D ("rbln"): omitting it lets the base
# KVTransferConfig default (not "rbln") win, so the rbln case would silently
# run the wrong transport.
case "$KV_BUFFER_DEVICE" in
  cpu | rbln) ;;
  *)
    echo "Invalid --kv_buffer_device: '$KV_BUFFER_DEVICE' (expected cpu|rbln)"
    exit 1
    ;;
esac
KV_CONFIG="{\"kv_connector\":\"RblnNixlConnector\",\"kv_role\":\"kv_both\",\"kv_buffer_device\":\"$KV_BUFFER_DEVICE\"}"

# Models to run
MODEL_NAMES=${MODEL_NAMES:-}
if [[ -n "$MODEL_NAMES" ]]; then
  MODELS=("$MODEL_NAMES")
else
  MODELS=(
      "Qwen/Qwen3-0.6B"
  )
fi

# Number of prefill and decode instances to create
NUM_PREFILL_INSTANCES=${NUM_PREFILL_INSTANCES:-1} # Default to 1
NUM_DECODE_INSTANCES=${NUM_DECODE_INSTANCES:-1}   # Default to 1
PREFILLER_TP_SIZE=${PREFILLER_TP_SIZE:-1}
DECODER_TP_SIZE=${DECODER_TP_SIZE:-1}
# Pipeline-parallel sizes (native path only). Prefill and decode may use
# different pipeline-parallel sizes: e.g. a pipelined prefill feeding a
# full-model decode. The side-channel listener is per-engine (one port
# regardless of PP/TP), so PP adds no extra ports; it only consumes TP*PP
# devices per instance.
PREFILLER_PP_SIZE=${PREFILLER_PP_SIZE:-1}
DECODER_PP_SIZE=${DECODER_PP_SIZE:-1}
PREFILL_BLOCK_SIZE=${PREFILL_BLOCK_SIZE:-1024}
DECODE_BLOCK_SIZE=${DECODE_BLOCK_SIZE:-$PREFILL_BLOCK_SIZE}
NUM_GPU_BLOCKS_OVERRIDE=${NUM_GPU_BLOCKS_OVERRIDE:-256}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-4096}
MAX_NUM_BATCHED_TOKENS=${MAX_NUM_BATCHED_TOKENS:-512}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-32}
# TODO: fix this
RBLN_ROOT_IP=${RBLN_ROOT_IP:-127.0.0.1}
RBLN_LOCAL_IP=${RBLN_LOCAL_IP:-127.0.0.1}

# Find the git repository root directory
GIT_ROOT=$(git rev-parse --show-toplevel)

SMI_BIN=$(which rbln-stat || echo "")

# Trap the SIGINT signal (triggered by Ctrl+C)
trap 'kill $(jobs -pr)' SIGINT SIGTERM EXIT

# Waits for vLLM to start.
wait_for_server() {
  local port=$1
  timeout 1200 bash -c "
    until curl -s localhost:${port}/v1/completions > /dev/null; do
      sleep 1
    done" && return 0 || return 1
}

# Function to clean up previous instances
cleanup_instances() {
  echo "Cleaning up any running vLLM instances..."
  pkill -f "vllm serve" || true
  sleep 2
}

# Handle to get model-specific arguments for deepseek
get_model_args() {
  local model_name=$1
  local extra_args=""

  if [[ "$model_name" == "deepseek-ai/deepseek-vl2-tiny" ]]; then
    extra_args="--hf_overrides '{\"architectures\": [\"DeepseekVLV2ForCausalLM\"]}' --trust-remote-code"
  fi

  echo "$extra_args"
}

get_num_gpus() {
  if [[ "$SMI_BIN" == *"rbln"* ]]; then
    # `rbln-stat --list` prints one "NPU <n> : ..." line per device.
    # (The old `--l` short flag was removed, which silently yielded 0 and
    # a later divide-by-zero.)
    "$SMI_BIN" --list 2>/dev/null | grep -cE '^NPU [0-9]+ :'
  else
    # works for non-cuda platforms,
    # assuming at least 2 device and
    # let system to decide which card to use
    echo "2"
  fi
}

# Function to run tests for a specific model
run_tests_for_model() {
  local model_name=$1
  echo "================================"
  echo "Testing model: $model_name"
  echo "================================"

  # Get model-specific arguments
  local model_args=$(get_model_args "$model_name")

  # Arrays to store all hosts and ports
  PREFILL_HOSTS=()
  PREFILL_PORTS=()
  DECODE_HOSTS=()
  DECODE_PORTS=()

  # Start prefill instances
  # Each instance consumes TP*PP devices (one rank per (pp_rank, tp_rank)).
  PREFILL_DEVICES_PER_INSTANCE=$((PREFILLER_TP_SIZE * PREFILLER_PP_SIZE))
  for i in $(seq 0 $((NUM_PREFILL_INSTANCES-1))); do
    # Calculate GPU ID - we'll distribute across available GPUs
    GPU_ID=$(( (i * PREFILL_DEVICES_PER_INSTANCE) % $(get_num_gpus) ))
    NEXT_GPU=${GPU_ID}
    # Add the remaining TP*PP-1 devices for this instance
    for (( j=1; j < PREFILL_DEVICES_PER_INSTANCE; j++ )); do
      NEXT_GPU=$(((GPU_ID + j) % $(get_num_gpus)))
      GPU_ID="${GPU_ID},${NEXT_GPU}"
    done

    # Calculate port number (base port + instance number)
    PORT=$((8100 + i))
    # One side-channel listener per engine (port = base + data_parallel_index;
    # no TP/PP offset), so instances need only stride 1. Use --data-parallel-size
    # -> stride by that instead.
    SIDE_CHANNEL_PORT=$((5559 + i))

    echo "Starting prefill instance $i on GPU $GPU_ID, port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="RBLN_DEVICES=$GPU_ID \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    RBLN_USE_CUSTOM_KERNEL=0 \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    VLLM_RBLN_COMPILE_STRICT_MODE=1 \
    VLLM_RBLN_USE_VLLM_MODEL=1 \
    VLLM_RBLN_USE_DEVICE_TENSOR=1 \
    VLLM_RBLN_AUTO_PORT=1 \
    VLLM_RBLN_BATCH_ATTN_OPT=1 \
    VLLM_RBLN_SORT_BATCH=1 \
    RBLN_ROOT_IP=$RBLN_ROOT_IP RBLN_LOCAL_IP=$RBLN_LOCAL_IP \
    vllm serve $model_name \
    --port $PORT \
    --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE \
    --max-model-len $MAX_MODEL_LEN \
    --block-size ${PREFILL_BLOCK_SIZE} \
    --enable-chunked-prefill \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-seqs $MAX_NUM_SEQS \
    --tensor-parallel-size $PREFILLER_TP_SIZE \
    --pipeline-parallel-size $PREFILLER_PP_SIZE \
    --kv-transfer-config '$KV_CONFIG'"

    if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
    else
    FULL_CMD="$BASE_CMD"
    fi

    eval "$FULL_CMD &"

    # Store host and port for proxy configuration
    PREFILL_HOSTS+=("localhost")
    PREFILL_PORTS+=($PORT)
  done

  # Start decode instances
  # Each instance consumes TP*PP devices (one rank per (pp_rank, tp_rank)).
  DECODE_DEVICES_PER_INSTANCE=$((DECODER_TP_SIZE * DECODER_PP_SIZE))
  for i in $(seq 0 $((NUM_DECODE_INSTANCES-1))); do
    # Calculate GPU ID - we'll distribute across available GPUs, starting from after prefill GPUs
    GPU_ID=$(( (i * DECODE_DEVICES_PER_INSTANCE + NEXT_GPU + 1) % $(get_num_gpus) ))
    NEXT_GPU=${GPU_ID}
    # Add the remaining TP*PP-1 devices for this instance
    for (( j=1; j < DECODE_DEVICES_PER_INSTANCE; j++ )); do
      NEXT_GPU=$(((GPU_ID + j) % $(get_num_gpus)))
      GPU_ID="${GPU_ID},${NEXT_GPU}"
    done
    # Calculate port number (base port + instance number)
    PORT=$((8200 + i))
    # One side-channel listener per engine (port = base + data_parallel_index;
    # no TP/PP offset), so instances need only stride 1. Use --data-parallel-size
    # -> stride by that instead.
    SIDE_CHANNEL_PORT=$((5659 + i))

    echo "Starting decode instance $i on GPU $GPU_ID, port $PORT"

    # Build the command with or without model-specific args
    BASE_CMD="RBLN_DEVICES=$GPU_ID \
    UCX_NET_DEVICES=all \
    VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT \
    RBLN_USE_CUSTOM_KERNEL=0 \
    VLLM_DISABLE_COMPILE_CACHE=1 \
    VLLM_RBLN_COMPILE_STRICT_MODE=1 \
    VLLM_RBLN_USE_VLLM_MODEL=1 \
    VLLM_RBLN_USE_DEVICE_TENSOR=1 \
    VLLM_RBLN_AUTO_PORT=1 \
    VLLM_RBLN_BATCH_ATTN_OPT=1 \
    VLLM_RBLN_SORT_BATCH=1 \
    RBLN_ROOT_IP=$RBLN_ROOT_IP RBLN_LOCAL_IP=$RBLN_LOCAL_IP \
    vllm serve $model_name \
    --port $PORT \
    --num-gpu-blocks-override $NUM_GPU_BLOCKS_OVERRIDE \
    --max-model-len $MAX_MODEL_LEN \
    --block-size ${DECODE_BLOCK_SIZE} \
    --enable-chunked-prefill \
    --max-num-batched-tokens $MAX_NUM_BATCHED_TOKENS \
    --max-num-seqs $MAX_NUM_SEQS \
    --tensor-parallel-size $DECODER_TP_SIZE \
    --pipeline-parallel-size $DECODER_PP_SIZE \
    --kv-transfer-config '$KV_CONFIG'"

    if [ -n "$model_args" ]; then
    FULL_CMD="$BASE_CMD $model_args"
    else
    FULL_CMD="$BASE_CMD"
    fi

    eval "$FULL_CMD &"

    # Store host and port for proxy configuration
    DECODE_HOSTS+=("localhost")
    DECODE_PORTS+=($PORT)
  done

  # Wait for all instances to start
  for PORT in "${PREFILL_PORTS[@]}"; do
    echo "Waiting for prefill instance on port $PORT to start..."
    wait_for_server $PORT
  done

  for PORT in "${DECODE_PORTS[@]}"; do
    echo "Waiting for decode instance on port $PORT to start..."
    wait_for_server $PORT
  done

  # Build the command for the proxy server with all the hosts and ports
  PROXY_CMD="python3 ${GIT_ROOT}/tests/torch_compile/e2e/v1/kv_connector/nixl_integration/toy_proxy_server.py --port 8192"

  # Add all prefill hosts and ports
  PROXY_CMD+=" --prefiller-hosts ${PREFILL_HOSTS[@]}"
  PROXY_CMD+=" --prefiller-ports ${PREFILL_PORTS[@]}"

  # Add all decode hosts and ports
  PROXY_CMD+=" --decoder-hosts ${DECODE_HOSTS[@]}"
  PROXY_CMD+=" --decoder-ports ${DECODE_PORTS[@]}"

  # Start the proxy server
  echo "Starting proxy server with command: $PROXY_CMD"
  $PROXY_CMD &

  # Wait for the proxy to start
  sleep 5

  # Run lm eval for this model
  echo "Running tests for $model_name"
  TEST_MODEL=$model_name python3 -m pytest -s -x ${GIT_ROOT}/tests/torch_compile/e2e/v1/kv_connector/nixl_integration/test_accuracy.py

  # Clean up before running next model
  cleanup_instances
  sleep 3
}

# Run tests for each model
for model in "${MODELS[@]}"; do
  run_tests_for_model "$model"
done

echo "All tests completed!"
