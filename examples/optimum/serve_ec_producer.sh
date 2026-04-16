#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Start EC disaggregation producer(s) (vision encoder) via vllm serve.
#
# Usage (6 producers on devices 0-5):
#   NUM_PRODUCERS=6 bash serve_ec_producer.sh
#
# Usage (6 producers on devices 8-13):
#   NUM_PRODUCERS=6 BASE_DEVICE=8 bash serve_ec_producer.sh
#
# Usage (custom device list):
#   RBLN_DEVICE_LIST="2,5,7,9" bash serve_ec_producer.sh
#
# Usage (single producer, backward-compatible):
#   bash serve_ec_producer.sh [MODEL_ID] [PORT] [NIXL_HOST] [NIXL_PORT]
#
# Each producer gets 1 device and its own vllm port + NIXL side-channel port.
# Start the consumer with: NUM_PRODUCERS=6 bash serve_ec_consumer.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

MODEL_ID="${1:-Qwen2-VL-7B-Instruct}"
BASE_PORT="${2:-8000}"
NIXL_HOST="${3:-127.0.0.1}"
NIXL_BASE_PORT="${4:-25300}"
BASE_DEVICE="${BASE_DEVICE:-0}"

# Device list: either explicit (RBLN_DEVICE_LIST="2,5,7") or auto-generated
if [ -n "$RBLN_DEVICE_LIST" ]; then
    IFS=',' read -ra DEVICES <<< "$RBLN_DEVICE_LIST"
    NUM_PRODUCERS="${#DEVICES[@]}"
else
    NUM_PRODUCERS="${NUM_PRODUCERS:-1}"
    DEVICES=()
    for i in $(seq 0 $((NUM_PRODUCERS - 1))); do
        DEVICES+=($((BASE_DEVICE + i)))
    done
fi

launch_producer() {
    local idx=$1
    local device=${DEVICES[$idx]}
    local port=$((BASE_PORT + idx))
    local nixl_port=$((NIXL_BASE_PORT + idx * 2))

    echo "Starting producer $idx (device=$device, port=$port, nixl_port=$nixl_port)"
    RBLN_DEVICES=$device vllm serve "$MODEL_ID" \
        --port "$port" \
        --mm-processor-kwargs '{"max_pixels": 802816}' \
        --ec-transfer-config "{
            \"ec_connector\": \"RblnECNixlConnector\",
            \"ec_role\": \"ec_producer\",
            \"ec_buffer_device\": \"cpu\",
            \"ec_connector_extra_config\": {
                \"side_channel_host\": \"$NIXL_HOST\",
                \"side_channel_port\": $nixl_port
            }
        }" &
}

for i in $(seq 0 $((NUM_PRODUCERS - 1))); do
    launch_producer "$i"
done

echo "Launched $NUM_PRODUCERS producer(s). Press Ctrl+C to stop all."
trap 'kill $(jobs -p) 2>/dev/null; wait' INT TERM
wait
