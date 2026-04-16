#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# EC Disaggregated Benchmark with RblnECNixlPushConnector
#
# Launches producers, consumer, proxy, waits for readiness, runs benchmark.
#
# Usage:
#   bash examples/optimum/run_ec_push_benchmark.sh
#
# Override any variable via env:
#   NUM_PRODUCERS=4 NUM_PROMPTS=10 bash examples/optimum/run_ec_push_benchmark.sh
#
set -euo pipefail

###############################################################################
# Activate venv
###############################################################################
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

###############################################################################
# Configuration
###############################################################################
MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"
NUM_PRODUCERS="${NUM_PRODUCERS:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
REQUEST_RATE="${REQUEST_RATE:-0.5}"

# Ports
PRODUCER_BASE_PORT="${PRODUCER_BASE_PORT:-8100}"
CONSUMER_PORT="${CONSUMER_PORT:-9100}"
PROXY_PORT="${PROXY_PORT:-1900}"

# ZMQ PULL port (consumer binds, producers connect)
CONSUMER_PULL_PORT="${CONSUMER_PULL_PORT:-16100}"
CONSUMER_HOST="${CONSUMER_HOST:-127.0.0.1}"

# Devices (using free devices 20-29)
PRODUCER_BASE_DEVICE="${PRODUCER_BASE_DEVICE:-20}"
CONSUMER_DEVICES="${CONSUMER_DEVICES:-22,23,24,25,26,27,28,29}"

# Logging
LOG_PATH="${LOG_PATH:-./logs/ec_push}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

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
  EC Disaggregated Benchmark (RblnECNixlPushConnector)
============================================================
  Model:            $MODEL
  Connector:        RblnECNixlPushConnector (ZMQ PUSH/PULL + NIXL)
  Producers:        $NUM_PRODUCERS (devices $PRODUCER_BASE_DEVICE..$((PRODUCER_BASE_DEVICE + NUM_PRODUCERS - 1)))
  Consumer:         port $CONSUMER_PORT (devices $CONSUMER_DEVICES)
  PULL port:        $CONSUMER_HOST:$CONSUMER_PULL_PORT
  Proxy:            port $PROXY_PORT
  Prompts:          $NUM_PROMPTS @ ${REQUEST_RATE} req/s
  Logs:             $LOG_PATH/
============================================================
EOF

###############################################################################
# 1. Start consumer FIRST (binds PULL socket)
###############################################################################
echo ""
echo "[1/4] Starting consumer (must be up before producers)..."

CONSUMER_LOG="$LOG_PATH/consumer_${START_TIME}.log"
CONSUMER_DEVICES=$CONSUMER_DEVICES \
PULL_HOST="0.0.0.0" \
PULL_PORT=$CONSUMER_PULL_PORT \
    bash "$SCRIPT_DIR/serve_ec_push_consumer.sh" \
        "$MODEL" "$CONSUMER_PORT" \
    > "$CONSUMER_LOG" 2>&1 &
PIDS+=($!)

wait_for_server "$CONSUMER_PORT" "consumer"

###############################################################################
# 2. Start producers (connect to consumer's PULL port)
###############################################################################
echo ""
echo "[2/4] Starting $NUM_PRODUCERS producer(s)..."

PRODUCER_LOG="$LOG_PATH/producer_${START_TIME}.log"
NUM_PRODUCERS=$NUM_PRODUCERS \
BASE_DEVICE=$PRODUCER_BASE_DEVICE \
CONSUMER_HOST=$CONSUMER_HOST \
CONSUMER_PULL_PORT=$CONSUMER_PULL_PORT \
    bash "$SCRIPT_DIR/serve_ec_push_producer.sh" \
        "$MODEL" "$PRODUCER_BASE_PORT" \
    > "$PRODUCER_LOG" 2>&1 &
PIDS+=($!)

for i in $(seq 0 $((NUM_PRODUCERS - 1))); do
    wait_for_server $((PRODUCER_BASE_PORT + i)) "producer $i"
done

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
