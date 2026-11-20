#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
topology="${1:-1e3pd}"
model="${2:-google/gemma-4-26B-A4B}"
case "$topology" in
    1e1pd) producers=1; consumers=1; default_devices=0,1 ;;
    1e3pd) producers=1; consumers=3; default_devices=0,1,2,3 ;;
    2e2pd) producers=2; consumers=2; default_devices=0,1,2,3 ;;
    *)
        echo "Usage: $0 {1e1pd|1e3pd|2e2pd} [MODEL]" >&2
        exit 2
        ;;
esac

DEVICES="${DEVICES:-$default_devices}"
if [[ ! $DEVICES =~ ^[0-9]+(,[0-9]+)*$ ]]; then
    echo "DEVICES must be a comma-separated list of devices." >&2
    exit 2
fi
IFS=, read -r -a devices <<< "$DEVICES"
if (( ${#devices[@]} != producers + consumers )); then
    echo "Expected $((producers + consumers)) devices for $topology." >&2
    exit 2
fi
declare -A seen=()
for device in "${devices[@]}"; do
    if [[ -v "seen[$device]" ]]; then
        echo "Device $device appears more than once." >&2
        exit 2
    fi
    seen[$device]=1
done

LOG_DIR="${LOG_DIR:-$SCRIPT_DIR/../logs/$topology}"
STARTUP_TIMEOUT="${STARTUP_TIMEOUT:-1800}"
launch=(bash)
if [[ -n ${CPU_LIST:-} ]]; then
    launch=(taskset -c "$CPU_LIST" bash)
fi

mkdir -p "$LOG_DIR"
pids=()
cleanup() {
    if (( ${#pids[@]} )); then
        kill "${pids[@]}" 2>/dev/null || true
        wait "${pids[@]}" 2>/dev/null || true
    fi
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_server() {
    local role="$1" device="$2" port="$3" side_port="$4"
    local instance_dir="$LOG_DIR/$role-$device"
    mkdir -p "$instance_dir"
    (
        cd "$instance_dir"
        export RBLN_VISIBLE_DEVICES="$device" RBLN_DEVICES="$device"
        export VLLM_EC_SIDE_CHANNEL_PORT="$side_port"
        exec "${launch[@]}" "$SCRIPT_DIR/$role.sh" "$model" "$port"
    ) > "$LOG_DIR/$role-$device.log" 2>&1 &
    pids+=("$!")
    echo "Starting $role on device $device and port $port. Log: $LOG_DIR/$role-$device.log"
}

producer_urls=()
consumer_urls=()
for ((index = 0; index < producers; index++)); do
    port=$((${FIRST_PRODUCER_PORT:-8100} + index))
    side_port=$((${FIRST_EC_PORT:-5601} + index))
    start_server producer "${devices[$index]}" "$port" "$side_port"
    producer_urls+=("http://127.0.0.1:$port")
done
for ((index = 0; index < consumers; index++)); do
    port=$((${FIRST_CONSUMER_PORT:-8200} + index))
    start_server consumer "${devices[$((producers + index))]}" "$port" "${FIRST_EC_PORT:-5601}"
    consumer_urls+=("http://127.0.0.1:$port")
done

deadline=$((SECONDS + STARTUP_TIMEOUT))
urls=("${producer_urls[@]}" "${consumer_urls[@]}")
for index in "${!urls[@]}"; do
    until curl --max-time 2 -fsS "${urls[$index]}/health" >/dev/null 2>&1; do
        if ! kill -0 "${pids[$index]}" 2>/dev/null || (( SECONDS >= deadline )); then
            echo "${urls[$index]} failed to become ready. See $LOG_DIR." >&2
            exit 1
        fi
        sleep 2
    done
done

(
    IFS=,
    PRODUCER_URLS="${producer_urls[*]}" CONSUMER_URLS="${consumer_urls[*]}" \
        exec bash "$SCRIPT_DIR/proxy.sh"
) > "$LOG_DIR/proxy.log" 2>&1 &
pids+=("$!")
echo "Starting $topology proxy at http://127.0.0.1:${PROXY_PORT:-10002}. Log: $LOG_DIR/proxy.log"
wait -n || true
echo "An EPD process exited. Check the logs in $LOG_DIR." >&2
exit 1
