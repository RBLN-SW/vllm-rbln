#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Start EC disaggregation encoder(s) using RblnECNixlConnector.
#
# Usage:
#   NUM_ENCODERS=2 bash serve_ec_encoder.sh
#
# Each encoder gets 1 device and connects to the llm's PULL port.
# Start the llm FIRST with: bash serve_ec_llm.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

MODEL_ID="${1:-Qwen3-VL-8B-Instruct}"
BASE_PORT="${2:-8100}"
BASE_DEVICE="${BASE_DEVICE:-20}"
NUM_ENCODERS="${NUM_ENCODERS:-2}"

# LLM PULL endpoint (where encoders push metadata)
LLM_HOST="${LLM_HOST:-127.0.0.1}"
LLM_PULL_PORT="${LLM_PULL_PORT:-16100}"

# Device list: either explicit or auto-generated
if [ -n "$RBLN_DEVICE_LIST" ]; then
    IFS=',' read -ra DEVICES <<< "$RBLN_DEVICE_LIST"
    NUM_ENCODERS="${#DEVICES[@]}"
else
    DEVICES=()
    for i in $(seq 0 $((NUM_ENCODERS - 1))); do
        DEVICES+=($((BASE_DEVICE + i)))
    done
fi

launch_encoder() {
    local idx=$1
    local device=${DEVICES[$idx]}
    local port=$((BASE_PORT + idx))

    echo "Starting encoder $idx (device=$device, port=$port, push→$LLM_HOST:$LLM_PULL_PORT)"
    RBLN_DEVICES=$device vllm serve "$MODEL_ID" \
        --port "$port" \
        --mm-processor-kwargs '{"max_pixels": 802816}' \
        --ec-transfer-config "{
            \"ec_connector\": \"RblnECNixlConnector\",
            \"ec_role\": \"ec_producer\",
            \"ec_connector_extra_config\": {
                \"llm_host\": \"$LLM_HOST\",
                \"llm_pull_port\": $LLM_PULL_PORT
            }
        }" &
}

for i in $(seq 0 $((NUM_ENCODERS - 1))); do
    launch_encoder "$i"
done

echo "Launched $NUM_ENCODERS encoder(s). Press Ctrl+C to stop all."
trap 'kill $(jobs -p) 2>/dev/null; wait' INT TERM
wait
