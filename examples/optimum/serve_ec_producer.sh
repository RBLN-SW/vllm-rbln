#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Start an EC disaggregation producer (vision encoder) via vllm serve.
#
# Usage:
#   bash serve_ec_producer.sh [MODEL_ID] [PORT] [NIXL_HOST] [NIXL_PORT]
#
# The producer runs the vision encoder and exposes encoder cache via NIXL.
# Start the consumer with serve_ec_consumer.sh, then send requests to both
# using client_ec_disaggregated.py.

MODEL_ID="${1:-Qwen2-VL-7B-Instruct}"
PORT="${2:-8000}"
NIXL_HOST="${3:-127.0.0.1}"
NIXL_PORT="${4:-25300}"

exec vllm serve "$MODEL_ID" \
    --port "$PORT" \
    --mm-processor-kwargs '{"max_pixels": 802816}' \
    --ec-transfer-config "{
        \"ec_connector\": \"RblnECNixlConnector\",
        \"ec_role\": \"ec_producer\",
        \"ec_buffer_device\": \"cpu\",
        \"ec_connector_extra_config\": {
            \"side_channel_host\": \"$NIXL_HOST\",
            \"side_channel_port\": $NIXL_PORT
        }
    }"
