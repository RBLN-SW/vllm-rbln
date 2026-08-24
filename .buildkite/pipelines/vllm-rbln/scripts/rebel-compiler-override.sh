#!/usr/bin/env bash
set -euo pipefail
[ -n "${REBEL_COMPILER_VERSION:-}" ] || exit 0

creds="${UV_INDEX_REBELLIONS_USERNAME}:${UV_INDEX_REBELLIONS_PASSWORD}"
host="${REBEL_PYPI_ENDPOINT%/}"
index="https://${creds}@${host#https://}/simple"

echo "+++ :package: override rebel-compiler==${REBEL_COMPILER_VERSION}"
uv pip install --extra-index-url "$index" "rebel-compiler==${REBEL_COMPILER_VERSION}"
buildkite-agent annotate --style success --context rebel-compiler \
  "rebel-compiler overridden to \`${REBEL_COMPILER_VERSION}\` (pypi.rebellions.in/simple)" || true
