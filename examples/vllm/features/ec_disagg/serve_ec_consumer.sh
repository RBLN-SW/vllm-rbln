#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Start the EC consumer: the prefill+decode instance on the vllm model path. It
# reads each image's encoder output from the producer over NIXL instead of
# encoding the image itself.
#
# Usage:
#   RBLN_DEVICES=4 bash serve_ec_consumer.sh [MODEL] [PORT]
#
# MODEL must be the same string as on the producer. With TENSOR_PARALLEL_SIZE
# above 1, list that many devices in RBLN_DEVICES.
# When sharing a host, run the producer on a separate device.
set -euo pipefail

MODEL="${1:-google/gemma-4-26B-A4B}"
PORT="${2:-8200}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-8192}"
IMAGE_LIMIT="${IMAGE_LIMIT:-1}"
EC_CPU_BYTES="${EC_CPU_BYTES:-1073741824}"

export RBLN_DEVICES="${RBLN_DEVICES:-4}"

chat_args=()
if [[ "$MODEL" == "google/gemma-4-26B-A4B" ]]; then
    chat_args=(--chat-template "$(dirname -- "${BASH_SOURCE[0]}")/gemma4_chat_template.jinja" --chat-template-content-format openai)
fi

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --model-impl vllm \
    --tensor-parallel-size "$TENSOR_PARALLEL_SIZE" \
    --block-size "$BLOCK_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS" \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --enable-request-id-headers \
    --limit-mm-per-prompt "{\"image\": $IMAGE_LIMIT, \"video\": 0}" \
    --ec-transfer-config "{
        \"ec_connector\": \"RblnECCPUConnector\",
        \"ec_role\": \"ec_consumer\",
        \"ec_connector_extra_config\": {\"ec_cpu_bytes\": $EC_CPU_BYTES}
    }" \
    "${chat_args[@]}"
