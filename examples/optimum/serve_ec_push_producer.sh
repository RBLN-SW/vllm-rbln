#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Start EC disaggregation producer(s) using RblnECNixlPushConnector.
#
# Usage:
#   NUM_PRODUCERS=2 bash serve_ec_push_producer.sh
#
# Each producer gets 1 device and connects to the consumer's PULL port.
# Start the consumer FIRST with: bash serve_ec_push_consumer.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

MODEL_ID="${1:-Qwen3-VL-8B-Instruct}"
BASE_PORT="${2:-8100}"
BASE_DEVICE="${BASE_DEVICE:-20}"
NUM_PRODUCERS="${NUM_PRODUCERS:-2}"

# Consumer PULL endpoint (where producers push metadata)
CONSUMER_HOST="${CONSUMER_HOST:-127.0.0.1}"
CONSUMER_PULL_PORT="${CONSUMER_PULL_PORT:-16100}"

# Device list: either explicit or auto-generated
if [ -n "$RBLN_DEVICE_LIST" ]; then
    IFS=',' read -ra DEVICES <<< "$RBLN_DEVICE_LIST"
    NUM_PRODUCERS="${#DEVICES[@]}"
else
    DEVICES=()
    for i in $(seq 0 $((NUM_PRODUCERS - 1))); do
        DEVICES+=($((BASE_DEVICE + i)))
    done
fi

launch_producer() {
    local idx=$1
    local device=${DEVICES[$idx]}
    local port=$((BASE_PORT + idx))

    echo "Starting producer $idx (device=$device, port=$port, push→$CONSUMER_HOST:$CONSUMER_PULL_PORT)"
    RBLN_DEVICES=$device vllm serve "$MODEL_ID" \
        --port "$port" \
        --mm-processor-kwargs '{"max_pixels": 802816}' \
        --ec-transfer-config "{
            \"ec_connector\": \"RblnECNixlPushConnector\",
            \"ec_role\": \"ec_producer\",
            \"ec_buffer_device\": \"cpu\",
            \"ec_connector_extra_config\": {
                \"consumer_host\": \"$CONSUMER_HOST\",
                \"consumer_pull_port\": $CONSUMER_PULL_PORT
            }
        }" &
}

for i in $(seq 0 $((NUM_PRODUCERS - 1))); do
    launch_producer "$i"
done

echo "Launched $NUM_PRODUCERS producer(s). Press Ctrl+C to stop all."
trap 'kill $(jobs -p) 2>/dev/null; wait' INT TERM
wait
