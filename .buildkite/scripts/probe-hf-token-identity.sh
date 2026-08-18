#!/usr/bin/env bash
# THROWAWAY -- this whole branch exists to answer one question and is not meant
# to merge.
#
# Do FSW_HF_TOKEN and HF_TOKEN belong to the same Hugging Face account?
#
# It matters because the rate limit that killed build 142 (429, "you hit the
# quota of 2500 api requests per 5 minutes period") is charged per account, not
# per job. The vllm-rbln lanes authenticate with FSW_HF_TOKEN and the optimum
# lanes with HF_TOKEN; if both resolve to the same user or org, every lane in
# every concurrent build shares one bucket and capping concurrency is the only
# lever. If they resolve to different accounts, the two lane families are
# already isolated and the vllm-rbln side alone is over budget.
#
# whoami-v2 is a read-only endpoint. Nothing here prints a token.

set -euo pipefail

: "${PROBE_FSW_HF_TOKEN:?not set -- fix the vault name in .buildkite/pr.yml}"
: "${PROBE_HF_TOKEN:?not set -- fix the vault name in .buildkite/pr.yml}"

command -v curl >/dev/null 2>&1 || { apt-get update && apt-get install -y curl; }

identify() {
  local label="$1" token="$2"
  local headers body
  headers=$(mktemp)
  body=$(mktemp)

  local code
  code=$(curl -s -D "${headers}" -o "${body}" -w '%{http_code}' \
    -H "Authorization: Bearer ${token}" \
    "https://huggingface.co/api/whoami-v2")

  echo "--- ${label}: HTTP ${code}"
  if [ "${code}" != "200" ]; then
    echo "    could not identify this token; body follows"
    head -c 300 "${body}"; echo
    return
  fi

  python3 - "${body}" "${label}" <<'PY'
import json, sys

with open(sys.argv[1]) as fh:
    me = json.load(fh)

orgs = [o.get("name") for o in me.get("orgs", [])]
print(f"    name : {me.get('name')}")
print(f"    type : {me.get('type')}")
print(f"    orgs : {orgs or '(none)'}")
PY

  # The quota is a moving window, so these only line up if the two tokens are
  # charged to the same bucket -- a weaker signal than the identity above, but
  # free, and it reads the actual counter rather than an account name.
  grep -i '^x-ratelimit' "${headers}" | sed 's/^/    /' || echo "    (no x-ratelimit headers)"
}

identify "FSW_HF_TOKEN" "${PROBE_FSW_HF_TOKEN}"
identify "HF_TOKEN" "${PROBE_HF_TOKEN}"

echo
echo "Same name/orgs on both  -> one shared quota bucket; only concurrency control helps."
echo "Different name/orgs     -> the lane families are isolated; vllm-rbln alone is over budget."
