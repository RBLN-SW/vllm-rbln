#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${VIRTUAL_ENV:-$SCRIPT_DIR/../../../.venv}}"
role="$1"
device="$2"
model="$3"
port="$4"

export RBLN_VISIBLE_DEVICES="${RBLN_VISIBLE_DEVICES:-${RBLN_DEVICES:-$device}}"
export RBLN_DEVICES="$RBLN_VISIBLE_DEVICES"
export UCX_TLS="${UCX_TLS:-tcp,sm,self}"
export UCX_NET_DEVICES="${UCX_NET_DEVICES:-all}"

if [[ $model == google/gemma-4-26B-A4B ]]; then
    CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/../gemma4_chat_template.jinja}"
fi
chat_template_args=()
if [[ -n ${CHAT_TEMPLATE:-} ]]; then
    chat_template_args+=(--chat-template "$CHAT_TEMPLATE")
fi

exec "$VENV_DIR/bin/vllm" serve "$model" \
    --host "${SERVE_HOST:-127.0.0.1}" --port "$port" \
    "${chat_template_args[@]}" \
    --model-impl vllm \
    --tensor-parallel-size "${TENSOR_PARALLEL_SIZE:-1}" \
    --block-size "${BLOCK_SIZE:-8192}" \
    --max-model-len "${MAX_MODEL_LEN:-262144}" \
    --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS:-2048}" \
    --max-num-seqs "${MAX_NUM_SEQS:-1}" \
    --no-enable-prefix-caching --no-async-scheduling --enable-request-id-headers \
    --limit-mm-per-prompt "{\"image\": ${IMAGE_LIMIT:-4}, \"video\": 0}" \
    --ec-transfer-config "{
        \"ec_connector\": \"RblnECCPUConnector\",
        \"ec_role\": \"$role\",
        \"ec_connector_extra_config\": {\"ec_cpu_bytes\": ${EC_CPU_BYTES:-1073741824}}
    }"
