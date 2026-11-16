#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Start the EC producer: an encoder-only instance on the vllm model path that
# encodes images and serves device-resident outputs over NIXL/RBLN VRAM.
#
# Usage (same host as the consumer's default devices):
#   RBLN_DEVICES=4 bash serve_ec_producer.sh [MODEL] [PORT]
#
# MODEL and its revision must match on the producer and the consumer.
# A consumer on another host needs
# EC_PRODUCER_HOST set to an address of this host it can reach. Producers
# sharing a host need distinct EC_PRODUCER_PORT values. EC_LISTEN_IP selects
# the RoCE interface used by the RBLN backend; "auto" selects a data NIC.
set -euo pipefail

MODEL="${1:-nvidia/MiniMax-M3-NVFP4}"
PORT="${2:-8100}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"
BLOCK_SIZE="${BLOCK_SIZE:-8192}"
IMAGE_LIMIT="${IMAGE_LIMIT:-1}"
EC_BUFFER_BYTES="${EC_BUFFER_BYTES:-1073741824}"
EC_PRODUCER_HOST="${EC_PRODUCER_HOST:-127.0.0.1}"
EC_PRODUCER_PORT="${EC_PRODUCER_PORT:-16100}"
EC_LISTEN_IP="${EC_LISTEN_IP:-auto}"

export RBLN_DEVICES="${RBLN_DEVICES:-4}"

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
        \"ec_connector\": \"RblnECDeviceConnector\",
        \"ec_role\": \"ec_producer\",
        \"ec_connector_extra_config\": {
            \"device_buffer_bytes\": $EC_BUFFER_BYTES,
            \"producer_host\": \"$EC_PRODUCER_HOST\",
            \"producer_port\": $EC_PRODUCER_PORT,
            \"listen_ip\": \"$EC_LISTEN_IP\"
        }
    }"
