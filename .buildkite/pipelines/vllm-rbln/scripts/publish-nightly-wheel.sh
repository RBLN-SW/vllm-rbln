#!/usr/bin/env bash
# Build the nightly wheel and publish it to the UDC Nexus vllm nightly index.
#
# Twin of fsw-integration's job-vllm-rbln-publish-wheel.yaml minus the artifact
# hand-off: GitHub Actions built the wheel in one job and downloaded it in
# another, while Buildkite builds and publishes in the same job.

set -euo pipefail

: "${UV_PUBLISH_URL:?not set -- fix the env block in .buildkite/vllm-rbln-nightly.yml}"
: "${UV_PUBLISH_USERNAME:?not set -- fix the vault name in .buildkite/vllm-rbln-nightly.yml}"
: "${UV_PUBLISH_PASSWORD:?not set -- fix the vault name in .buildkite/vllm-rbln-nightly.yml}"

echo "--- :package: build wheel"
# setuptools-scm derives the version from the nearest tag, so a checkout without
# tags would publish 0.1.devN instead of the real nightly version.
git fetch --tags --force --unshallow 2>/dev/null || git fetch --tags --force
rm -rf dist
uv build --wheel
ls -lh dist/

echo "--- :outbox_tray: publish to ${UV_PUBLISH_URL}"
# Rebuilding a commit can change the wheel hash, so skip by filename before
# uv's content-hash check.
wheel_name=$(basename dist/*.whl)
index_page=$(mktemp)
trap 'rm -f "$index_page"' EXIT
status=$(curl --silent --show-error --location --connect-timeout 10 --max-time 60 \
  --user "${UV_PUBLISH_USERNAME}:${UV_PUBLISH_PASSWORD}" \
  --output "$index_page" --write-out '%{http_code}' \
  "${UV_PUBLISH_URL%/}/simple/vllm-rbln/")
case "$status" in
  200)
    if grep -Fq ">${wheel_name}</a>" "$index_page"; then
      echo "${wheel_name} already exists; skipping publish."
      exit 0
    fi
    ;;
  404) ;;
  *)
    echo "Failed to check the package index: HTTP ${status}" >&2
    exit 1
    ;;
esac

# Keep uv's hash check for concurrent uploads after the existence check.
scheme="${UV_PUBLISH_URL%%://*}"
host_path="${UV_PUBLISH_URL#*://}"
export UV_PUBLISH_CHECK_URL="${scheme}://${UV_PUBLISH_USERNAME}:${UV_PUBLISH_PASSWORD}@${host_path%/}/simple/"
uv publish dist/*.whl
