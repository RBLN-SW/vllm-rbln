#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Start EC disaggregation consumer using RblnECNixlPushConnector.
#
# Usage:
#   bash serve_ec_push_consumer.sh
#
# The consumer binds a ZMQ PULL socket and receives NIXL metadata
# directly from producers. Start this BEFORE the producers.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

MODEL_ID="${1:-Qwen3-VL-8B-Instruct}"
PORT="${2:-9100}"

# ZMQ PULL bind address (producers connect here)
PULL_HOST="${PULL_HOST:-0.0.0.0}"
PULL_PORT="${PULL_PORT:-16100}"

# Consumer devices
CONSUMER_DEVICES="${CONSUMER_DEVICES:-22,23,24,25,26,27,28,29}"

export RBLN_DEVICES=$CONSUMER_DEVICES
exec vllm serve "$MODEL_ID" \
    --port "$PORT" \
    --mm-processor-kwargs '{"max_pixels": 802816}' \
    --ec-transfer-config "{
        \"ec_connector\": \"RblnECNixlPushConnector\",
        \"ec_role\": \"ec_consumer\",
        \"ec_buffer_device\": \"cpu\",
        \"ec_connector_extra_config\": {
            \"pull_host\": \"$PULL_HOST\",
            \"pull_port\": $PULL_PORT
        }
    }"
