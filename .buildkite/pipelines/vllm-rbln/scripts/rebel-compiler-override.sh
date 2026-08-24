#!/usr/bin/env bash
# Reinstall rebel-compiler at REBEL_COMPILER_VERSION (set by compiler CI) from
# pypi.rebellions.in over the locked env. No-op for normal vllm-rbln PRs.
# Kept in a script so obedients does not interpolate the runtime vars at upload time.
set -euo pipefail
[ -n "${REBEL_COMPILER_VERSION:-}" ] || exit 0

creds="${UV_INDEX_REBELLIONS_USERNAME}:${UV_INDEX_REBELLIONS_PASSWORD}"
host="${REBEL_PYPI_ENDPOINT%/}"
uv pip install \
  --extra-index-url "https://${creds}@${host#https://}/simple" \
  "rebel-compiler==${REBEL_COMPILER_VERSION}"
