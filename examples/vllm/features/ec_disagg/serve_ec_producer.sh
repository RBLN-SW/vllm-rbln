#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Start the EC producer: an encoder-only instance on the vllm model path that
# encodes images and serves the outputs to consumers over NIXL.
#
# Usage (same host as the consumer's default devices):
#   RBLN_DEVICES=5 bash serve_ec_producer.sh [MODEL] [PORT]
#
# MODEL must be the same string on the producer and the consumer: it is part of
# the compatibility hash the two compare. A consumer on another host needs
# VLLM_EC_SIDE_CHANNEL_HOST set to an address of this host it can reach;
# producers sharing a host need distinct VLLM_EC_SIDE_CHANNEL_PORT values.
set -euo pipefail

MODEL="${1:-google/gemma-4-26B-A4B}"
PORT="${2:-8100}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-1}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-2048}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-4}"
BLOCK_SIZE="${BLOCK_SIZE:-8192}"
IMAGE_LIMIT="${IMAGE_LIMIT:-1}"
EC_CPU_BYTES="${EC_CPU_BYTES:-1073741824}"

export RBLN_DEVICES="${RBLN_DEVICES:-5}"

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
        \"ec_role\": \"ec_producer\",
        \"ec_connector_extra_config\": {\"ec_cpu_bytes\": $EC_CPU_BYTES}
    }" \
    "${chat_args[@]}"
