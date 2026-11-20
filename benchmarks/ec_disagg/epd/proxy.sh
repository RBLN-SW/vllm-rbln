#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Rebellions Inc. All rights reserved.
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${VENV_DIR:-${VIRTUAL_ENV:-$SCRIPT_DIR/../../../.venv}}"

if [[ -z ${PROXY_SCRIPT:-} ]]; then
    cache="${XDG_CACHE_HOME:-$HOME/.cache}/vllm-rbln"
    upstream="$cache/disagg_epd_proxy-v0.31.0.py"
    patch_hash="$(sha256sum "$SCRIPT_DIR/proxy.patch")"
    PROXY_SCRIPT="$cache/ec-disagg-benchmark/disagg_epd_proxy-v0.31.0-${patch_hash%% *}.py"
    if [[ ! -f $PROXY_SCRIPT ]]; then
        mkdir -p "$(dirname -- "$PROXY_SCRIPT")"
        source_file="$(mktemp "${PROXY_SCRIPT}.XXXXXX")"
        trap 'rm -f -- "$source_file"' EXIT
        if [[ -f $upstream ]]; then
            cp -- "$upstream" "$source_file"
        else
            source_url=https://raw.githubusercontent.com/vllm-project/vllm/v0.31.0
            curl -fsSL "$source_url/examples/disaggregated/disaggregated_encoder/disagg_epd_proxy.py" \
                -o "$source_file"
        fi
        checksum=59c842c5b127f9008c64cc770c450a2d4b9a9f63c7139603bb20a95b80845460
        printf '%s  %s\n' "$checksum" "$source_file" | sha256sum --check --status
        patch --batch --forward --no-backup-if-mismatch --reject-file=- "$source_file" \
            "$SCRIPT_DIR/proxy.patch" >&2
        mv -- "$source_file" "$PROXY_SCRIPT"
    fi
fi

exec "$VENV_DIR/bin/python" "$PROXY_SCRIPT" \
    --host "${PROXY_HOST:-127.0.0.1}" --port "${PROXY_PORT:-10002}" \
    --encode-servers-urls "${PRODUCER_URLS:-http://127.0.0.1:8100}" \
    --prefill-servers-urls disable \
    --decode-servers-urls "${CONSUMER_URLS:-http://127.0.0.1:8200,http://127.0.0.1:8201,http://127.0.0.1:8202}" \
    "$@"
