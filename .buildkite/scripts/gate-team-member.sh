#!/usr/bin/env bash
# vllm-rbln team-member policy gate (moved off the GHA check-team-member job).

set -euo pipefail

if [ "${BUILDKITE_PULL_REQUEST:-false}" = "false" ]; then
  echo "Not a pull-request build; skipping team-member gate."
  exit 0
fi

command -v jq >/dev/null 2>&1 || { apt-get update && apt-get install -y jq curl; }

if [ -z "${GIT_PAT:-}" ]; then
  echo "GIT_PAT is not set; cannot verify PR author." >&2
  exit 1
fi

repo=$(printf '%s' "${BUILDKITE_REPO}" | sed -E 's#^.*github\.com[:/]##; s#\.git$##')

api="https://api.github.com"
auth="Authorization: Bearer ${GIT_PAT}"

author=$(curl -fsS -H "${auth}" \
  "${api}/repos/${repo}/pulls/${BUILDKITE_PULL_REQUEST}" | jq -r '.user.login')
if [ -z "${author}" ] || [ "${author}" = "null" ]; then
  echo "Could not resolve author of PR #${BUILDKITE_PULL_REQUEST} in ${repo}." >&2
  exit 1
fi
echo "PR #${BUILDKITE_PULL_REQUEST} author: ${author}"

query='{"query":"query { organization(login: \"rbln-sw\") { team1: team(slug: \"sw\") { members(first: 100) { nodes { login } } } team2: team(slug: \"fsw\") { members(first: 100) { nodes { login } } } } }"}'
members=$(curl -fsS -H "${auth}" -H "Content-Type: application/json" -d "${query}" "${api}/graphql" \
  | jq -r '.data.organization.team1.members.nodes[].login, .data.organization.team2.members.nodes[].login' \
  | sort -u)

if printf '%s\n' "${members}" | grep -qx "${author}"; then
  echo "✅ ${author} is on the sw/fsw team -- allowed."
  exit 0
fi

code=$(curl -sS -o /dev/null -w '%{http_code}' -H "${auth}" \
  "${api}/repos/${repo}/collaborators/${author}" || echo "000")
if [ "${code}" = "204" ]; then
  echo "✅ ${author} is a repo collaborator -- allowed."
  exit 0
fi

echo "❌ ${author} is neither an sw/fsw team member nor a collaborator -- blocking CI." >&2
exit 1
