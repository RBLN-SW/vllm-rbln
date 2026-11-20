#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${VIRTUAL_ENV:-$SCRIPT_DIR/../../.venv}}"
topology="${1:-}"
case "$topology" in
    baseline) port=8000 ;;
    1e1pd|1e3pd|2e2pd) port=10002 ;;
    *)
        echo "Usage: $0 {baseline|1e1pd|1e3pd|2e2pd} [benchmark options]" >&2
        exit 2
        ;;
esac
shift

exec "$VENV_DIR/bin/python" "$SCRIPT_DIR/bench_gemma4_encoder.py" \
    --topology "$topology" --base-url "${BASE_URL:-http://127.0.0.1:$port}" \
    --output-dir "${OUTPUT_DIR:-$SCRIPT_DIR/results/gemma4-smoke-uuid}" \
    --qps 2 --images 1 4 --input-tokens 400 --min-prompts 20 \
    --seconds 10 --warmups 4 "$@"
