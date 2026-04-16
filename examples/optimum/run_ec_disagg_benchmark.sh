#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# EC Disaggregated Encoder Benchmark
#
# Launches producers, consumer, proxy, waits for readiness, runs benchmark,
# then cleans up everything.
#
# Usage:
#   bash examples/optimum/run_ec_disagg_benchmark.sh
#
# Override any variable via env:
#   NUM_PRODUCERS=4 CONSUMER_DEVICES="4,5,6,7" NUM_PROMPTS=50 \
#       bash examples/optimum/run_ec_disagg_benchmark.sh
#
set -euo pipefail

###############################################################################
# Configuration
###############################################################################
MODEL="${MODEL:-Qwen2-VL-7B-Instruct}"
NUM_PRODUCERS="${NUM_PRODUCERS:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-300}"
REQUEST_RATE="${REQUEST_RATE:-2.0}"

# Ports
PRODUCER_BASE_PORT="${PRODUCER_BASE_PORT:-8000}"
CONSUMER_PORT="${CONSUMER_PORT:-9000}"
PROXY_PORT="${PROXY_PORT:-1800}"
NIXL_BASE_PORT="${NIXL_BASE_PORT:-25300}"
NIXL_HOST="${NIXL_HOST:-127.0.0.1}"

# Devices
PRODUCER_BASE_DEVICE="${PRODUCER_BASE_DEVICE:-0}"       # producers use devices 0..N-1
CONSUMER_DEVICES="${CONSUMER_DEVICES:-10,11,12,13,14,15,16,17}"

# Logging
LOG_PATH="${LOG_PATH:-./logs/ec_disagg}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"               # wait_for_server timeout

# Benchmark
BENCH_DATASET="${BENCH_DATASET:-lmarena-ai/VisionArena-Chat}"
BENCH_BACKEND="${BENCH_BACKEND:-openai-chat}"

###############################################################################
# Derived
###############################################################################
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
START_TIME="$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_PATH"

declare -a PIDS=()

###############################################################################
# Helpers
###############################################################################
wait_for_server() {
    local port=$1
    local name=${2:-"server on :$port"}
    echo "[wait] Waiting for $name ..."
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -sf http://127.0.0.1:$port/health > /dev/null 2>&1; do
            sleep 2
        done" && echo "[wait] $name is ready." && return 0
    echo "[wait] TIMEOUT waiting for $name" >&2
    return 1
}

cleanup() {
    echo ""
    echo "[cleanup] Stopping all processes..."
    trap - INT TERM

    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done

    echo "[cleanup] Done."
    exit 0
}

trap cleanup INT TERM

###############################################################################
# Print config
###############################################################################
cat <<EOF
============================================================
  EC Disaggregated Encoder Benchmark
============================================================
  Model:            $MODEL
  Producers:        $NUM_PRODUCERS (devices $PRODUCER_BASE_DEVICE..$((PRODUCER_BASE_DEVICE + NUM_PRODUCERS - 1)))
  Consumer:         port $CONSUMER_PORT (devices $CONSUMER_DEVICES)
  Proxy:            port $PROXY_PORT
  NIXL:             $NIXL_HOST:$NIXL_BASE_PORT-$((NIXL_BASE_PORT + NUM_PRODUCERS - 1))
  Prompts:          $NUM_PROMPTS @ ${REQUEST_RATE} req/s
  Logs:             $LOG_PATH/
============================================================
EOF

###############################################################################
# 1. Start producers
###############################################################################
echo ""
echo "[1/4] Starting $NUM_PRODUCERS producer(s)..."

PRODUCER_LOG="$LOG_PATH/producer_${START_TIME}.log"
NUM_PRODUCERS=$NUM_PRODUCERS \
BASE_DEVICE=$PRODUCER_BASE_DEVICE \
    bash "$SCRIPT_DIR/serve_ec_producer.sh" \
        "$MODEL" "$PRODUCER_BASE_PORT" "$NIXL_HOST" "$NIXL_BASE_PORT" \
    > "$PRODUCER_LOG" 2>&1 &
PIDS+=($!)

for i in $(seq 0 $((NUM_PRODUCERS - 1))); do
    wait_for_server $((PRODUCER_BASE_PORT + i)) "producer $i"
done

###############################################################################
# 2. Start consumer
###############################################################################
echo ""
echo "[2/4] Starting consumer..."

CONSUMER_LOG="$LOG_PATH/consumer_${START_TIME}.log"
RBLN_DEVICES=$CONSUMER_DEVICES \
NUM_PRODUCERS=$NUM_PRODUCERS \
    bash "$SCRIPT_DIR/serve_ec_consumer.sh" \
        "$MODEL" "$CONSUMER_PORT" "$NIXL_HOST" "$NIXL_BASE_PORT" \
    > "$CONSUMER_LOG" 2>&1 &
PIDS+=($!)

wait_for_server "$CONSUMER_PORT" "consumer"

###############################################################################
# 3. Start proxy
###############################################################################
echo ""
echo "[3/4] Starting proxy..."

# Build encode-servers-urls
ENCODE_URLS=""
for i in $(seq 0 $((NUM_PRODUCERS - 1))); do
    [ -n "$ENCODE_URLS" ] && ENCODE_URLS+=","
    ENCODE_URLS+="http://127.0.0.1:$((PRODUCER_BASE_PORT + i))"
done

PROXY_LOG="$LOG_PATH/proxy_${START_TIME}.log"
python "$SCRIPT_DIR/client_ec_disaggregated.py" \
    --host 0.0.0.0 \
    --port "$PROXY_PORT" \
    --encode-servers-urls "$ENCODE_URLS" \
    --decode-servers-urls "http://127.0.0.1:$CONSUMER_PORT" \
    > "$PROXY_LOG" 2>&1 &
PIDS+=($!)

wait_for_server "$PROXY_PORT" "proxy"

echo ""
echo "============================================================"
echo "  All services are up!"
echo "============================================================"

###############################################################################
# 4. Run benchmark
###############################################################################
echo ""
echo "[4/4] Running benchmark ($NUM_PROMPTS prompts, rate=$REQUEST_RATE)..."

BENCH_LOG="$LOG_PATH/bench_${START_TIME}.log"

vllm bench serve \
    --model "$MODEL" \
    --backend "$BENCH_BACKEND" \
    --endpoint /v1/chat/completions \
    --dataset-name hf \
    --dataset-path "$BENCH_DATASET" \
    --seed 0 \
    --num-prompts "$NUM_PROMPTS" \
    --request-rate "$REQUEST_RATE" \
    --port "$PROXY_PORT" \
    2>&1 | tee "$BENCH_LOG"

echo ""
echo "============================================================"
echo "  Benchmark complete."
echo "  Logs: $LOG_PATH/*_${START_TIME}.log"
echo "============================================================"

###############################################################################
# Cleanup
###############################################################################
cleanup
