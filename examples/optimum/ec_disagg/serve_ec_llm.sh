#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Start EC disaggregation llm using RblnECNixlConnector.
#
# Usage:
#   bash serve_ec_llm.sh
#
# The llm binds a ZMQ PULL socket and receives NIXL metadata
# directly from encoders. Start this BEFORE the encoders.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

MODEL_ID="${1:-Qwen3-VL-8B-Instruct}"
PORT="${2:-9100}"

# ZMQ PULL bind address (encoders connect here)
PULL_HOST="${PULL_HOST:-0.0.0.0}"
PULL_PORT="${PULL_PORT:-16100}"

# LLM devices
LLM_DEVICES="${LLM_DEVICES:-22,23,24,25,26,27,28,29}"

export RBLN_DEVICES=$LLM_DEVICES
exec vllm serve "$MODEL_ID" \
    --port "$PORT" \
    --mm-processor-kwargs '{"max_pixels": 802816}' \
    --ec-transfer-config "{
        \"ec_connector\": \"RblnECNixlConnector\",
        \"ec_role\": \"ec_consumer\",
        \"ec_connector_extra_config\": {
            \"pull_host\": \"$PULL_HOST\",
            \"pull_port\": $PULL_PORT
        }
    }"
