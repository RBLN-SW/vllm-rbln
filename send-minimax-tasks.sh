#!/usr/bin/env bash
# serve-minimax.sh 로 로컬 8001 포트에 뜬 MiniMax-M2.5 에 대해
# mini-swe-agent(run_lite_batch.py)로 SWE-bench Lite 32개 인스턴스를
# step_limit=10 으로 요청 보낸다.
#
#   agent(LLM 루프)는 로컬, tool call(bash)만 REBEL_SANDBOX_URL 원격 샌드박스에서 실행.
#   LLM 엔드포인트는 로컬 vLLM(http://localhost:8001/v1)을 바라본다.
#
# 사용:
#   ./serve-minimax.sh          # (다른 셸에서) 서버 먼저 띄우고
#   ./send-minimax-tasks.sh     # 이 스크립트로 32개 작업 요청
#
# 조정 가능한 env (기본값):
#   N=32 STEP_LIMIT=10 WORKERS=8
#   PORT=8001 MODEL=MiniMaxAI/MiniMax-M2.5
#   REBEL_SANDBOX_URL=https://rebel-sandbox.sandbox.udc.rbln.in
#   OUTPUT=preds.json
set -euo pipefail

# ── 파라미터 ────────────────────────────────────────────────────────────────
N="${N:-50}"                       # 요청할 작업(인스턴스) 개수
STEP_LIMIT="${STEP_LIMIT:-10}"     # 인스턴스당 최대 스텝
WORKERS="${WORKERS:-50}"            # 동시 실행 수 (서버 max-num-seqs=8 에 맞춤)
PORT="${PORT:-8001}"
MODEL="${MODEL:-MiniMaxAI/MiniMax-M2.5}"
OUTPUT="${OUTPUT:-preds.json}"

# ── LLM 엔드포인트: 로컬 vLLM(8001) ─────────────────────────────────────────
export OPENAI_API_BASE="${OPENAI_API_BASE:-http://localhost:${PORT}/v1}"
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy}"

# ── tool call(bash) 실행용 원격 샌드박스 control plane ──────────────────────
export REBEL_SANDBOX_URL="${REBEL_SANDBOX_URL:-https://rebel-sandbox.sandbox.udc.rbln.in}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MINI_SWE_DIR="${MINI_SWE_DIR:-${SCRIPT_DIR}/.mini_swe_agent}"
RUNNER="${MINI_SWE_DIR}/run_lite_batch.py"

# 프로젝트 .venv 의 python 을 그대로 사용
PYTHON="${PYTHON:-${SCRIPT_DIR}/.venv/bin/python}"
[ -x "${PYTHON}" ] || { echo "[!] ${PYTHON} 를 찾을 수 없습니다." >&2; exit 1; }

# ── 서버 살아있는지 확인 ────────────────────────────────────────────────────
if ! curl -sf -m 5 "http://localhost:${PORT}/v1/models" >/dev/null; then
  echo "[!] http://localhost:${PORT}/v1/models 에 응답이 없습니다. 먼저 ./serve-minimax.sh 로 서버를 띄우세요." >&2
  exit 1
fi

# ── client SDK(mini-swe-agent 어댑터) 설치 확인 ─────────────────────────────
if ! "${PYTHON}" -c "import minisweagent, rebel_sandbox_client" 2>/dev/null; then
  echo "[*] rebel-sandbox-client[minisweagent] 설치 중 (사내 Nexus)..."
  "${PYTHON}" -m pip install --extra-index-url https://nexus.mgmt.rbln.in/repository/pypi-rebel-sandbox-dev/simple \
    "rebel-sandbox-client[minisweagent]"
fi

# ── 배치 러너(run_lite_batch.py) 준비 — 없으면 레포에서 가져옴 ──────────────
if [ ! -f "${RUNNER}" ]; then
  echo "[*] run_lite_batch.py 를 coding-assistant-sandbox(dev)에서 가져옵니다..."
  mkdir -p "${MINI_SWE_DIR}"
  gh api "repos/rebellions-sw/coding-assistant-sandbox/contents/examples/mini_swe_agent/run_lite_batch.py?ref=dev" \
    --jq '.content' | base64 -d > "${RUNNER}"
fi

echo "[batch] N=${N} step_limit=${STEP_LIMIT} workers=${WORKERS} model=openai/${MODEL}"
echo "[batch] LLM=${OPENAI_API_BASE}  sandbox=${REBEL_SANDBOX_URL}  output=${OUTPUT}"

# ── 32개 작업 요청 (step_limit=10, 최대 300초) ──────────────────────────────
exec timeout 300 "${PYTHON}" "${RUNNER}" \
  --n "${N}" \
  --step-limit "${STEP_LIMIT}" \
  --workers "${WORKERS}" \
  --model "openai/${MODEL}" \
  --output "${OUTPUT}" \
  "$@"
