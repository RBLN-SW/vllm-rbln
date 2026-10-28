#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Start the EC producer: an encoder-only instance on the vllm model path that
# encodes images and serves the outputs to consumers over NIXL.
#
# Usage:
#   RBLN_DEVICES=0 bash serve_ec_producer.sh [MODEL] [PORT]
#
# MODEL must be the same string on the producer and the consumer: it is part of
# the compatibility hash the two compare. A consumer on another host needs
# VLLM_EC_SIDE_CHANNEL_HOST set to an address of this host it can reach;
# producers sharing a host need distinct VLLM_EC_SIDE_CHANNEL_PORT values.
set -euo pipefail

MODEL="${1:-nvidia/MiniMax-M3-NVFP4}"
PORT="${2:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
BLOCK_SIZE="${BLOCK_SIZE:-8192}"
IMAGE_LIMIT="${IMAGE_LIMIT:-1}"
EC_CPU_BYTES="${EC_CPU_BYTES:-1073741824}"

export RBLN_DEVICES="${RBLN_DEVICES:-0}"

exec vllm serve "$MODEL" \
    --port "$PORT" \
    --model-impl vllm \
    --block-size "$BLOCK_SIZE" \
    --max-model-len "$MAX_MODEL_LEN" \
    --max-num-batched-tokens 8192 \
    --no-enable-prefix-caching \
    --no-async-scheduling \
    --enable-request-id-headers \
    --limit-mm-per-prompt "{\"image\": $IMAGE_LIMIT, \"video\": 0}" \
    --ec-transfer-config "{
        \"ec_connector\": \"RblnECCPUConnector\",
        \"ec_role\": \"ec_producer\",
        \"ec_connector_extra_config\": {\"ec_cpu_bytes\": $EC_CPU_BYTES}
    }"
