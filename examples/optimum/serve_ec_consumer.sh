#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Start an EC disaggregation consumer (decoder) via vllm serve.
#
# Usage (single producer, backward-compatible):
#   bash serve_ec_consumer.sh Qwen2-VL-7B-Instruct 8001 127.0.0.1 25300
#
# Usage (multi-producer, 8 producers on ports 25300-25307):
#   NUM_PRODUCERS=8 bash serve_ec_consumer.sh
#
# The consumer pulls encoder cache from producer(s) via NIXL and runs
# prefill_decoder + decode. Start producers first with serve_ec_producer.sh.

MODEL_ID="${1:-Qwen2-VL-7B-Instruct}"
PORT="${2:-8001}"
NIXL_HOST="${3:-127.0.0.1}"
NIXL_BASE_PORT="${4:-25300}"
NUM_PRODUCERS="${NUM_PRODUCERS:-1}"

# Build producer_endpoints JSON array
if [ "$NUM_PRODUCERS" -gt 1 ]; then
    ENDPOINTS="["
    for i in $(seq 0 $((NUM_PRODUCERS - 1))); do
        [ "$i" -gt 0 ] && ENDPOINTS+=","
        ENDPOINTS+="{\"host\":\"$NIXL_HOST\",\"port\":$((NIXL_BASE_PORT + i))}"
    done
    ENDPOINTS+="]"

    exec vllm serve "$MODEL_ID" \
        --port "$PORT" \
        --mm-processor-kwargs '{"max_pixels": 802816}' \
        --ec-transfer-config "{
            \"ec_connector\": \"RblnECNixlConnector\",
            \"ec_role\": \"ec_consumer\",
            \"ec_buffer_device\": \"cpu\",
            \"ec_connector_extra_config\": {
                \"producer_endpoints\": $ENDPOINTS
            }
        }"
else
    exec vllm serve "$MODEL_ID" \
        --port "$PORT" \
        --mm-processor-kwargs '{"max_pixels": 802816}' \
        --ec-transfer-config "{
            \"ec_connector\": \"RblnECNixlConnector\",
            \"ec_role\": \"ec_consumer\",
            \"ec_buffer_device\": \"cpu\",
            \"ec_connector_extra_config\": {
                \"side_channel_host\": \"$NIXL_HOST\",
                \"side_channel_port\": $NIXL_BASE_PORT
            }
        }"
fi
