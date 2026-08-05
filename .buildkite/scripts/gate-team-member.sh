#!/usr/bin/env bash
# Policy gate (moved off the GHA check-team-member job): only run CI for a PR
# whose author is on the rbln-sw `sw`/`fsw` team, or is a repo collaborator.
#
# SECURITY NOTE: this is a *policy* gate for same-repo PRs, NOT a fork boundary.
# A forked PR controls this very script and could bypass it, so forked-PR builds
# MUST stay disabled on the Obedients pipeline -- that setting is the real
# security boundary. This gate just narrows same-repo (push-access) authors down
# to the sw/fsw teams / collaborators.
#
# Runs in the bootstrap step BEFORE `buildkite-agent pipeline upload`, so an
# untrusted author uploads no test steps at all:
#   bash .buildkite/scripts/gate-team-member.sh && buildkite-agent pipeline upload ...
#
# Needs: GIT_PAT secret; curl + jq on the agent. BK env: BUILDKITE_PULL_REQUEST,
# BUILDKITE_REPO.
set -euo pipefail

# Not a pull request (e.g. a dev-branch push) -> nothing to gate.
if [ "${BUILDKITE_PULL_REQUEST:-false}" = "false" ]; then
  echo "Not a pull-request build; skipping team-member gate."
  exit 0
fi

command -v jq >/dev/null 2>&1 || { apt-get update && apt-get install -y jq curl; }

if [ -z "${GIT_PAT:-}" ]; then
  echo "GIT_PAT is not set; cannot verify PR author." >&2
  exit 1
fi

# Base repo "owner/name" from the git URL
# (git@github.com:RBLN-SW/vllm-rbln.git or https://github.com/RBLN-SW/vllm-rbln.git).
repo=$(printf '%s' "${BUILDKITE_REPO}" | sed -E 's#^.*github\.com[:/]##; s#\.git$##')

api="https://api.github.com"
auth="Authorization: Bearer ${GIT_PAT}"

# PR author's GitHub login.
author=$(curl -fsS -H "${auth}" \
  "${api}/repos/${repo}/pulls/${BUILDKITE_PULL_REQUEST}" | jq -r '.user.login')
if [ -z "${author}" ] || [ "${author}" = "null" ]; then
  echo "Could not resolve author of PR #${BUILDKITE_PULL_REQUEST} in ${repo}." >&2
  exit 1
fi
echo "PR #${BUILDKITE_PULL_REQUEST} author: ${author}"

# sw + fsw team members (org rbln-sw), first 100 each.
query='{"query":"query { organization(login: \"rbln-sw\") { team1: team(slug: \"sw\") { members(first: 100) { nodes { login } } } team2: team(slug: \"fsw\") { members(first: 100) { nodes { login } } } } }"}'
members=$(curl -fsS -H "${auth}" -H "Content-Type: application/json" -d "${query}" "${api}/graphql" \
  | jq -r '.data.organization.team1.members.nodes[].login, .data.organization.team2.members.nodes[].login' \
  | sort -u)

if printf '%s\n' "${members}" | grep -qx "${author}"; then
  echo "✅ ${author} is on the sw/fsw team -- allowed."
  exit 0
fi

# Fallback: is the author a repo collaborator? (204 = yes, 404 = no)
code=$(curl -sS -o /dev/null -w '%{http_code}' -H "${auth}" \
  "${api}/repos/${repo}/collaborators/${author}" || echo "000")
if [ "${code}" = "204" ]; then
  echo "✅ ${author} is a repo collaborator -- allowed."
  exit 0
fi

echo "❌ ${author} is neither an sw/fsw team member nor a collaborator -- blocking CI." >&2
exit 1
