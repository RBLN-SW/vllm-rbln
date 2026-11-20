#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${VIRTUAL_ENV:-$SCRIPT_DIR/../../../.venv}}"
MODEL="${1:-google/gemma-4-26B-A4B}"
DEVICES="${DEVICES:-0,1,2,3}"
PROXY_HOST="${PROXY_HOST:-127.0.0.1}"
PROXY_PORT="${PROXY_PORT:-8000}"
FIRST_REPLICA_PORT="${FIRST_REPLICA_PORT:-8300}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-262144}"
PREFILL_CHUNK_SIZE="${PREFILL_CHUNK_SIZE:-2048}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-$PREFILL_CHUNK_SIZE}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-1}"
BLOCK_SIZE="${BLOCK_SIZE:-8192}"
IMAGE_LIMIT="${IMAGE_LIMIT:-4}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/../logs/baseline}"

if (( PREFILL_CHUNK_SIZE > MAX_NUM_BATCHED_TOKENS || MAX_NUM_BATCHED_TOKENS > BLOCK_SIZE )); then
    echo "Require PREFILL_CHUNK_SIZE <= MAX_NUM_BATCHED_TOKENS <= BLOCK_SIZE." >&2
    exit 2
fi
if [[ ! $DEVICES =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "DEVICES must be a comma-separated list, for example 0,1,2,3." >&2
    exit 2
fi

IFS=, read -r -a devices <<< "$DEVICES"
declare -A seen_devices=()
for device in "${devices[@]}"; do
    if [[ -v "seen_devices[$device]" ]]; then
        echo "Device $device appears more than once." >&2
        exit 2
    fi
    seen_devices[$device]=1
done
if (( PROXY_PORT >= FIRST_REPLICA_PORT && PROXY_PORT < FIRST_REPLICA_PORT + ${#devices[@]} )); then
    echo "Proxy port overlaps the replica ports." >&2
    exit 2
fi

mkdir -p "$LOG_DIR"
pids=()
upstreams=()
cleanup() {
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

serve_args=(
    --host 127.0.0.1
    --model-impl vllm
    --tensor-parallel-size 1
    --max-model-len "$MAX_MODEL_LEN"
    --block-size "$BLOCK_SIZE"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --additional-config "{\"prefill_chunk_size\":[$PREFILL_CHUNK_SIZE]}"
    --limit-mm-per-prompt "{\"image\":$IMAGE_LIMIT,\"video\":0}"
    --no-enable-prefix-caching
    --no-async-scheduling
    --enable-request-id-headers
)
if [[ -n ${SERVED_MODEL_NAME:-} ]]; then
    serve_args+=(--served-model-name "$SERVED_MODEL_NAME")
fi
if [[ $MODEL == google/gemma-4-26B-A4B ]]; then
    CHAT_TEMPLATE="${CHAT_TEMPLATE:-$SCRIPT_DIR/../gemma4_chat_template.jinja}"
fi
if [[ -n ${CHAT_TEMPLATE:-} ]]; then
    serve_args+=(--chat-template "$CHAT_TEMPLATE")
fi

for index in "${!devices[@]}"; do
    device="${devices[$index]}"
    port=$((FIRST_REPLICA_PORT + index))
    RBLN_VISIBLE_DEVICES="$device" RBLN_DEVICES="$device" \
        "$VENV_DIR/bin/vllm" serve "$MODEL" "${serve_args[@]}" \
        --port "$port" > "$LOG_DIR/device-$device.log" 2>&1 &
    pids+=("$!")
    upstreams+=("http://127.0.0.1:$port")
    echo "Starting device $device on port $port. Log: $LOG_DIR/device-$device.log"
done

deadline=$((SECONDS + STARTUP_TIMEOUT))
for index in "${!upstreams[@]}"; do
    until curl --max-time 2 -fsS "${upstreams[$index]}/health" >/dev/null 2>&1; do
        if ! kill -0 "${pids[$index]}" 2>/dev/null || (( SECONDS >= deadline )); then
            echo "Replica on device ${devices[$index]} failed to become ready. See $LOG_DIR/device-${devices[$index]}.log." >&2
            exit 1
        fi
        sleep 2
    done
done

"$VENV_DIR/bin/python" "$SCRIPT_DIR/proxy.py" \
    --host "$PROXY_HOST" --port "$PROXY_PORT" "${upstreams[@]}" &
pids+=("$!")
echo "Starting baseline endpoint at http://$PROXY_HOST:$PROXY_PORT"
wait -n || true
echo "A baseline process exited. Check the replica logs in $LOG_DIR." >&2
exit 1
