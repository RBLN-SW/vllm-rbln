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
# --check-url is what makes a re-run on an unchanged commit a no-op instead of a
# 409; it replaces the GHA job's check-nexus-wheel.sh probe. uv reads it from
# UV_PUBLISH_CHECK_URL, and the index needs auth, so the credentials go inline --
# there is no separate UV_PUBLISH_CHECK_* pair.
scheme="${UV_PUBLISH_URL%%://*}"
host_path="${UV_PUBLISH_URL#*://}"
export UV_PUBLISH_CHECK_URL="${scheme}://${UV_PUBLISH_USERNAME}:${UV_PUBLISH_PASSWORD}@${host_path%/}/simple/"
uv publish dist/*.whl
