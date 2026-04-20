#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Disaggregated encoder vs monolithic (non-disaggregated) A/B benchmark
# on an identical workload.
#
#   non-disagg:  single vllm serve; decoder TP8 + visual co-located via
#                `--additional-config '{"rbln_config":{"device":[...],
#                "visual":{"device":...}}}'`
#   disagg:      N independent encoders + TP8 llm + proxy, using
#                RblnECNixlConnector (the only connector that works
#                with Qwen3-VL today).
#
# Same MODEL / NUM_PROMPTS / RATES / dataset for both, with warm-up run
# per mode before measurement.
#
# Usage:
#   bash examples/optimum/compare_disagg_vs_nondisagg.sh
#
# Override:
#   MODES="nondisagg disagg" NUM_ENCODERS=8 RATES="0.3 0.5" \
#   NUM_PROMPTS=40 bash examples/optimum/compare_disagg_vs_nondisagg.sh
#
set -euo pipefail

###############################################################################
# Activate venv
###############################################################################
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$REPO_ROOT/.venv/bin/activate" ]; then
    source "$REPO_ROOT/.venv/bin/activate"
fi

###############################################################################
# Configuration (override via env)
###############################################################################
MODEL="${MODEL:-Qwen3-VL-8B-Instruct}"
MODES="${MODES:-nondisagg disagg}"

NUM_ENCODERS="${NUM_ENCODERS:-8}"
NUM_PROMPTS="${NUM_PROMPTS:-40}"
NUM_WARMUP="${NUM_WARMUP:-8}"
RATES="${RATES:-0.5}"

# Devices — pick ranges that don't collide across modes since we run
# sequentially and tear each stack down. Default: 16–31 (free on this host).
#
# non-disagg: 8 decoder devices (TP8) + 1 visual device
NONDISAGG_DECODER_DEVICES="${NONDISAGG_DECODER_DEVICES:-16,17,18,19,20,21,22,23}"
NONDISAGG_VISUAL_DEVICE="${NONDISAGG_VISUAL_DEVICE:-24}"

# disagg: encoder_base..+NUM_ENCODERS-1, llm TP8 on 8 devices
DISAGG_ENCODER_BASE_DEVICE="${DISAGG_ENCODER_BASE_DEVICE:-16}"
DISAGG_LLM_DEVICES="${DISAGG_LLM_DEVICES:-24,25,26,27,28,29,30,31}"

# Ports
NONDISAGG_PORT="${NONDISAGG_PORT:-9300}"
DISAGG_ENCODER_BASE_PORT="${DISAGG_ENCODER_BASE_PORT:-8000}"
DISAGG_LLM_PORT="${DISAGG_LLM_PORT:-9000}"
DISAGG_PROXY_PORT="${DISAGG_PROXY_PORT:-1800}"
DISAGG_PULL_PORT="${DISAGG_PULL_PORT:-16100}"

# Bench
BENCH_DATASET="${BENCH_DATASET:-lmarena-ai/VisionArena-Chat}"
BENCH_BACKEND="${BENCH_BACKEND:-openai-chat}"

# Paths
START_TIME="$(date +%Y%m%d_%H%M%S)"
LOG_PATH="${LOG_PATH:-$REPO_ROOT/logs/ec_vs_nondisagg/$START_TIME}"
RESULT_DIR="$LOG_PATH/results"
mkdir -p "$LOG_PATH" "$RESULT_DIR"

TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

declare -a PIDS=()

###############################################################################
# Helpers
###############################################################################
wait_for_server() {
    local port=$1
    local name=${2:-"server on :$port"}
    echo "[wait] $name ..."
    timeout "$TIMEOUT_SECONDS" bash -c "
        until curl -sf http://127.0.0.1:$port/health > /dev/null 2>&1; do
            sleep 2
        done" && echo "[wait] $name ready." && return 0
    echo "[wait] TIMEOUT waiting for $name" >&2
    return 1
}

cleanup_pids() {
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    sleep 2
    for pid in "${PIDS[@]:-}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill -9 "$pid" 2>/dev/null || true
        fi
    done
    PIDS=()
}

kill_leftovers() {
    pgrep -u "$USER" -af "vllm serve|client_ec_disaggregated" \
        | grep -v pgrep | awk '{print $1}' | xargs -r kill -9 2>/dev/null || true
    sleep 3
}

trap 'echo ""; echo "[trap] interrupted — cleaning up"; cleanup_pids; kill_leftovers; exit 130' INT TERM

###############################################################################
# Launch functions
###############################################################################
launch_nondisagg() {
    local tag=$1
    local log="$LOG_PATH/${tag}_serve.log"

    # Build the rbln_config JSON: decoder TP8 + visual on a separate device.
    local decoder_devices_json="[$NONDISAGG_DECODER_DEVICES]"
    local rbln_config_json
    rbln_config_json=$(cat <<EOF
{"rbln_config": {"device": $decoder_devices_json, "visual": {"device": $NONDISAGG_VISUAL_DEVICE}}}
EOF
)
    echo "[launch:$tag] vllm serve (decoder=$NONDISAGG_DECODER_DEVICES, visual=$NONDISAGG_VISUAL_DEVICE, port=$NONDISAGG_PORT)"
    vllm serve "$MODEL" \
        --port "$NONDISAGG_PORT" \
        --mm-processor-kwargs '{"max_pixels": 802816}' \
        --additional-config "$rbln_config_json" \
        > "$log" 2>&1 &
    PIDS+=($!)
    wait_for_server "$NONDISAGG_PORT" "$tag"
}

launch_disagg() {
    local tag=$1
    local encoder_log="$LOG_PATH/${tag}_encoder.log"
    local llm_log="$LOG_PATH/${tag}_llm.log"
    local proxy_log="$LOG_PATH/${tag}_proxy.log"

    echo "[launch:$tag] llm (TP8 on $DISAGG_LLM_DEVICES) ..."
    LLM_DEVICES="$DISAGG_LLM_DEVICES" \
    PULL_HOST="0.0.0.0" \
    PULL_PORT="$DISAGG_PULL_PORT" \
        bash "$SCRIPT_DIR/serve_ec_llm.sh" \
            "$MODEL" "$DISAGG_LLM_PORT" \
        > "$llm_log" 2>&1 &
    PIDS+=($!)
    wait_for_server "$DISAGG_LLM_PORT" "$tag llm"

    echo "[launch:$tag] $NUM_ENCODERS encoder(s) on devices $DISAGG_ENCODER_BASE_DEVICE..$((DISAGG_ENCODER_BASE_DEVICE + NUM_ENCODERS - 1)) ..."
    NUM_ENCODERS="$NUM_ENCODERS" \
    BASE_DEVICE="$DISAGG_ENCODER_BASE_DEVICE" \
    LLM_HOST="127.0.0.1" \
    LLM_PULL_PORT="$DISAGG_PULL_PORT" \
        bash "$SCRIPT_DIR/serve_ec_encoder.sh" \
            "$MODEL" "$DISAGG_ENCODER_BASE_PORT" \
        > "$encoder_log" 2>&1 &
    PIDS+=($!)
    for i in $(seq 0 $((NUM_ENCODERS - 1))); do
        wait_for_server $((DISAGG_ENCODER_BASE_PORT + i)) "$tag encoder $i"
    done

    # Proxy fanning out encodes and forwarding decode to the TP8 llm.
    echo "[launch:$tag] proxy ..."
    local encode_urls=""
    for i in $(seq 0 $((NUM_ENCODERS - 1))); do
        [ -n "$encode_urls" ] && encode_urls+=","
        encode_urls+="http://127.0.0.1:$((DISAGG_ENCODER_BASE_PORT + i))"
    done
    python "$SCRIPT_DIR/client_ec_disaggregated.py" \
        --host 0.0.0.0 \
        --port "$DISAGG_PROXY_PORT" \
        --encode-servers-urls "$encode_urls" \
        --decode-servers-urls "http://127.0.0.1:$DISAGG_LLM_PORT" \
        > "$proxy_log" 2>&1 &
    PIDS+=($!)
    wait_for_server "$DISAGG_PROXY_PORT" "$tag proxy"
}

###############################################################################
# Bench one (mode, rate)
###############################################################################
run_bench() {
    local tag=$1
    local port=$2
    local rate=$3
    local num_prompts=$4
    local result_filename=$5    # "" → warmup / discarded
    local bench_log=$6

    local save_args=()
    if [ -n "$result_filename" ]; then
        save_args=(--save-result --result-dir "$RESULT_DIR" --result-filename "$result_filename")
    fi

    vllm bench serve \
        --model "$MODEL" \
        --backend "$BENCH_BACKEND" \
        --endpoint /v1/chat/completions \
        --dataset-name hf \
        --dataset-path "$BENCH_DATASET" \
        --seed 0 \
        --num-prompts "$num_prompts" \
        --request-rate "$rate" \
        --port "$port" \
        "${save_args[@]}" \
        2>&1 | tee "$bench_log"
}

###############################################################################
# Main
###############################################################################
cat <<EOF
============================================================
  Disaggregated vs non-disaggregated — A/B benchmark
============================================================
  Model:            $MODEL
  Modes:            $MODES
  Rates:            $RATES req/s
  Prompts:          $NUM_PROMPTS  (warmup: $NUM_WARMUP)

  [non-disagg] decoder TP8 = $NONDISAGG_DECODER_DEVICES  |  visual = $NONDISAGG_VISUAL_DEVICE
               port = $NONDISAGG_PORT
  [disagg]     encoders = $NUM_ENCODERS on $DISAGG_ENCODER_BASE_DEVICE..$((DISAGG_ENCODER_BASE_DEVICE + NUM_ENCODERS - 1))
               llm TP8 = $DISAGG_LLM_DEVICES
               proxy = $DISAGG_PROXY_PORT

  Logs + results:   $LOG_PATH
============================================================
EOF

kill_leftovers

for mode in $MODES; do
    echo ""
    echo "########################################################"
    echo "  [$mode]  stack up"
    echo "########################################################"
    case "$mode" in
        nondisagg)
            launch_nondisagg "$mode"
            bench_port="$NONDISAGG_PORT"
            ;;
        disagg)
            launch_disagg "$mode"
            bench_port="$DISAGG_PROXY_PORT"
            ;;
        *)
            echo "[error] unknown mode: $mode" >&2
            exit 1
            ;;
    esac

    echo ""
    echo "[$mode] warm-up ($NUM_WARMUP prompts) ..."
    run_bench "$mode" "$bench_port" "1.0" "$NUM_WARMUP" "" \
        "$LOG_PATH/${mode}_warmup.log" \
        || true

    for rate in $RATES; do
        echo ""
        echo "[$mode] measure: rate=${rate} req/s, prompts=$NUM_PROMPTS"
        fname="${mode}_rate${rate}.json"
        run_bench "$mode" "$bench_port" "$rate" "$NUM_PROMPTS" "$fname" \
            "$LOG_PATH/${mode}_rate${rate}_bench.log"
    done

    echo ""
    echo "[$mode] stack down"
    cleanup_pids
    kill_leftovers
done

###############################################################################
# Summary — reuse the connector comparison script (it parses files named
# `<tag>_rate<RATE>.json`, which matches our layout).
###############################################################################
echo ""
echo "========================================================"
echo "  Comparison summary"
echo "========================================================"
python "$SCRIPT_DIR/compare_ec_results.py" \
    --result-dir "$RESULT_DIR" \
    || echo "[warn] comparison script failed; JSONs are under $RESULT_DIR"

echo ""
echo "Done. Artifacts: $LOG_PATH"
