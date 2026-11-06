#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Start upstream vLLM's EPD proxy in front of the producers and the consumers
# (E+PD: the consumers run both prefill and decode).
#
# Usage:
#   bash serve_ec_proxy.sh [proxy flags, e.g. --no-rewrite]
#
# The proxy ships with vLLM's examples, not with the package. PROXY_SCRIPT
# points at a local copy; without it the one tagged with the installed vLLM
# version is fetched once and cached.
set -euo pipefail

PRODUCER_URLS="${PRODUCER_URLS:-http://127.0.0.1:8100}"
CONSUMER_URLS="${CONSUMER_URLS:-http://127.0.0.1:8200}"
PROXY_PORT="${PROXY_PORT:-8000}"

if [ -z "${PROXY_SCRIPT:-}" ]; then
    VLLM_VERSION="$(python -c 'from importlib.metadata import version; print(version("vllm").split("+")[0])')"
    PROXY_SCRIPT="${XDG_CACHE_HOME:-$HOME/.cache}/vllm-rbln/disagg_epd_proxy-v$VLLM_VERSION.py"
    if [ ! -f "$PROXY_SCRIPT" ]; then
        mkdir -p "$(dirname "$PROXY_SCRIPT")"
        curl -fsSL -o "$PROXY_SCRIPT.tmp" \
            "https://raw.githubusercontent.com/vllm-project/vllm/v$VLLM_VERSION/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py"
        mv "$PROXY_SCRIPT.tmp" "$PROXY_SCRIPT"
    fi
fi

exec python "$PROXY_SCRIPT" \
    --host 0.0.0.0 \
    --port "$PROXY_PORT" \
    --encode-servers-urls "$PRODUCER_URLS" \
    --prefill-servers-urls disable \
    --decode-servers-urls "$CONSUMER_URLS" \
    "$@"
