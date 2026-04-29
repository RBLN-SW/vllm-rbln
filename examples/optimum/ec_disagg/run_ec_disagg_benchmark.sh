#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# EC Disaggregated Benchmark with RblnECNixlConnector
#
# Launches encoders, llm, proxy, waits for readiness, runs benchmark.
#
# Usage:
#   bash examples/optimum/ec_disagg/run_ec_disagg_benchmark.sh
#
# Override any variable via env:
#   NUM_ENCODERS=4 NUM_PROMPTS=10 bash examples/optimum/ec_disagg/run_ec_disagg_benchmark.sh
#
set -euo pipefail

###############################################################################
# Activate venv
###############################################################################
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

###############################################################################
# Configuration
###############################################################################
MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"
NUM_ENCODERS="${NUM_ENCODERS:-2}"
NUM_PROMPTS="${NUM_PROMPTS:-4}"
REQUEST_RATE="${REQUEST_RATE:-0.5}"
BENCH_NUM_WARMUPS="${BENCH_NUM_WARMUPS:-0}"

# Ports
ENCODER_BASE_PORT="${ENCODER_BASE_PORT:-8100}"
LLM_PORT="${LLM_PORT:-9200}"
PROXY_PORT="${PROXY_PORT:-1900}"

# ZMQ PULL port (llm binds, encoders connect)
LLM_PULL_PORT="${LLM_PULL_PORT:-16100}"
LLM_HOST="${LLM_HOST:-127.0.0.1}"

# Devices (using free devices 20-29)
ENCODER_BASE_DEVICE="${ENCODER_BASE_DEVICE:-16}"
LLM_DEVICES="${LLM_DEVICES:-24,25,26,27,28,29,30,31}"

# Logging
LOG_PATH="${LOG_PATH:-./logs/ec_disagg}"
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
  EC Disaggregated Benchmark (RblnECNixlConnector)
============================================================
  Model:        $MODEL
  Connector:    RblnECNixlConnector (ZMQ PUSH/PULL + NIXL)
  Encoders:     $NUM_ENCODERS on devices $ENCODER_BASE_DEVICE..$((ENCODER_BASE_DEVICE + NUM_ENCODERS - 1))
  LLM:          port $LLM_PORT, devices $LLM_DEVICES
  PULL port:    $LLM_HOST:$LLM_PULL_PORT
  Proxy:        port $PROXY_PORT
  Prompts:      $NUM_PROMPTS @ ${REQUEST_RATE} req/s
  Logs:         $LOG_PATH/
============================================================
EOF

###############################################################################
# 1. Start llm FIRST (binds PULL socket)
###############################################################################
echo ""
echo "[1/4] Starting llm (must be up before encoders)..."

LLM_LOG="$LOG_PATH/llm_${START_TIME}.log"
LLM_DEVICES=$LLM_DEVICES \
PULL_HOST="0.0.0.0" \
PULL_PORT=$LLM_PULL_PORT \
    bash "$SCRIPT_DIR/serve_ec_llm.sh" \
        "$MODEL" "$LLM_PORT" \
    > "$LLM_LOG" 2>&1 &
PIDS+=($!)

wait_for_server "$LLM_PORT" "llm"

###############################################################################
# 2. Start encoders (connect to llm's PULL port)
###############################################################################
echo ""
echo "[2/4] Starting $NUM_ENCODERS encoder(s)..."

ENCODER_LOG="$LOG_PATH/encoder_${START_TIME}.log"
NUM_ENCODERS=$NUM_ENCODERS \
BASE_DEVICE=$ENCODER_BASE_DEVICE \
LLM_HOST=$LLM_HOST \
LLM_PULL_PORT=$LLM_PULL_PORT \
    bash "$SCRIPT_DIR/serve_ec_encoder.sh" \
        "$MODEL" "$ENCODER_BASE_PORT" \
    > "$ENCODER_LOG" 2>&1 &
PIDS+=($!)

for i in $(seq 0 $((NUM_ENCODERS - 1))); do
    wait_for_server $((ENCODER_BASE_PORT + i)) "encoder $i"
done

###############################################################################
# 3. Start proxy
###############################################################################
echo ""
echo "[3/4] Starting proxy..."

# Build encode-servers-urls
ENCODE_URLS=""
for i in $(seq 0 $((NUM_ENCODERS - 1))); do
    [ -n "$ENCODE_URLS" ] && ENCODE_URLS+=","
    ENCODE_URLS+="http://127.0.0.1:$((ENCODER_BASE_PORT + i))"
done

PROXY_LOG="$LOG_PATH/proxy_${START_TIME}.log"
python "$SCRIPT_DIR/client_ec_disaggregated.py" \
    --host 0.0.0.0 \
    --port "$PROXY_PORT" \
    --encode-servers-urls "$ENCODE_URLS" \
    --decode-servers-urls "http://127.0.0.1:$LLM_PORT" \
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
if [ "${BENCH_SKIP:-0}" != "1" ]; then
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
        --num-warmups "$BENCH_NUM_WARMUPS" \
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
else
    echo ""
    echo "============================================================"
    echo "  BENCH_SKIP=1 — services running; drive externally."
    echo "  Proxy:     http://127.0.0.1:$PROXY_PORT"
    echo "  Logs:      $LOG_PATH/{llm,encoder,proxy}_${START_TIME}.log"
    echo "  Ctrl+C or \`kill $$\` to stop."
    echo "============================================================"
    # Block until signal; trap invokes cleanup().
    while true; do sleep 3600; done
fi
