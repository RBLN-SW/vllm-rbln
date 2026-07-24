#!/usr/bin/env bash
# ITL breakdown 스윕 오케스트레이터.
#
# max-num-seqs ∈ {8, 4, 16, 2} × spec ∈ {off, on} × sampler ∈ {on, off} 의
# 16개 조합마다:
#   1. serve-minimax.sh 로 서버 기동 (해당 조합 env 주입, 별도 process group)
#   2. /v1/models 가 응답할 때까지 대기
#   3. send-minimax-tasks.sh 로 부하 전송
#   4. SIGINT 로 graceful 종료 → 각 DP rank 가 ITL breakdown JSON 을 덤프
#   5. JSON 을 결과 디렉터리로 수집
#
# 조정 가능한 env (기본값):
#   RESULTS_DIR=itl-sweep-results  PORT=8001
#   READY_TIMEOUT=3600  SHUTDOWN_TIMEOUT=180
#   N=50  WORKERS=<mns>  STEP_LIMIT=10
#   DRY_RUN=0   # 1 이면 서버/부하 없이 로직만 검증
#
# 사용:  ./run-minimax-sweep.sh
set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 대화형 셸이 아니어도 프로젝트 venv 의 vllm/python 을 쓰도록 PATH 앞에 추가.
# (serve-minimax.sh 는 `exec vllm serve` 를 그대로 사용 → PATH 에서 해석됨)
if [ -d "$SCRIPT_DIR/.venv/bin" ]; then
  export PATH="$SCRIPT_DIR/.venv/bin:$PATH"
fi

RESULTS_DIR="${RESULTS_DIR:-itl-sweep-results}"
PORT="${PORT:-8001}"
READY_TIMEOUT="${READY_TIMEOUT:-3600}"       # 서버 ready 최대 대기(초)
SHUTDOWN_TIMEOUT="${SHUTDOWN_TIMEOUT:-180}"   # graceful 종료 최대 대기(초)
N="${N:-50}"
STEP_LIMIT="${STEP_LIMIT:-10}"
DRY_RUN="${DRY_RUN:-0}"

LOG_DIR="$RESULTS_DIR/logs"
mkdir -p "$RESULTS_DIR" "$LOG_DIR"

# 리스트는 env 로 부분 실행 가능(공백 구분). 기본은 전체 스윕.
#   예) 스모크 테스트 1회:  MNS_LIST=8 SPEC_LIST=on SAMPLER_LIST=1 ./run-minimax-sweep.sh
read -r -a MNS_LIST <<<"${MNS_LIST:-8 4 16 2}"
read -r -a SPEC_LIST <<<"${SPEC_LIST:-off on}"
read -r -a SAMPLER_LIST <<<"${SAMPLER_LIST:-1 0}"   # 1=on, 0=off

SERVER_PID=""

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# SIGINT 을 vLLM 이 SystemExit 로 변환 → worker finally → shutdown() → JSON dump.
# setsid 로 띄웠으므로 pid 는 세션 리더이고 pgid==pid. 프로세스 그룹 전체에 신호를
# 보낸 뒤, 메인(세션 리더)이 아니라 세션의 "모든" 프로세스가 사라질 때까지 기다린다.
# (메인이 먼저 죽고 DP worker 가 아직 JSON 을 flush 중이면 수집이 그걸 놓치는
#  경쟁 상태를 막기 위함.)
stop_server() {
  local pid="$1"
  [ -z "$pid" ] && return 0
  kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid" 2>/dev/null || true
  local waited=0
  # 세션 내 프로세스가 하나라도 살아있는 동안 대기
  while pgrep -s "$pid" >/dev/null 2>&1; do
    sleep 1
    waited=$((waited + 1))
    if [ "$waited" -ge "$SHUTDOWN_TIMEOUT" ]; then
      log "graceful 종료 시간 초과 → SIGTERM/SIGKILL"
      kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null || true
      sleep 5
      kill -KILL -- "-$pid" 2>/dev/null || kill -KILL "$pid" 2>/dev/null || true
      break
    fi
  done
  # 워커가 JSON 을 디스크에 flush 할 여유
  sleep 2
}

wait_ready() {
  local deadline=$((SECONDS + READY_TIMEOUT))
  while [ "$SECONDS" -lt "$deadline" ]; do
    # 서버 프로세스가 죽었으면 즉시 실패
    if [ -n "$SERVER_PID" ] && ! kill -0 "$SERVER_PID" 2>/dev/null; then
      return 1
    fi
    if curl -sf -m 5 "http://localhost:${PORT}/v1/models" >/dev/null 2>&1; then
      return 0
    fi
    sleep 5
  done
  return 1
}

cleanup() {
  log "인터럽트 감지 → 실행 중인 서버 정리"
  stop_server "$SERVER_PID"
  exit 130
}
trap cleanup INT TERM

run_one() {
  local mns="$1" spec="$2" sampler="$3"
  local specflag=0
  [ "$spec" = "on" ] && specflag=1
  local tag="mns=${mns}_spec=${spec}_sampler=${sampler}"
  # serve-minimax.sh 가 만드는 파일 접두어 (pid 는 뒤에 붙음)
  local prefix="max-num-seqs=${mns}_spec_=${specflag}_sampler=${sampler}"
  local outdir="$RESULTS_DIR/$tag"

  log "==== RUN ${tag} ===="
  mkdir -p "$outdir"

  # 이미 수집된 조합은 건너뜀(중단 후 재개 가능). SKIP_EXISTING=0 이면 강제 재실행.
  if [ "${SKIP_EXISTING:-1}" = "1" ] && compgen -G "$outdir/${prefix}.*.json" >/dev/null; then
    log "이미 결과 있음 → 건너뜀: $outdir/ (SKIP_EXISTING=0 으로 재실행)"
    return 0
  fi

  rm -f ./"${prefix}".*.json "$outdir/${prefix}".*.json 2>/dev/null || true

  if [ "$DRY_RUN" = "1" ]; then
    log "[dry-run] MAX_NUM_SEQS=$mns SPEC=$spec VLLM_RBLN_SAMPLER=$sampler ./serve-minimax.sh"
    # rank 4개 분량의 더미 JSON 생성 후 수집 검증
    for r in 100 101 102 103; do
      printf '{"name":"ITL","pid":%d,"tag":"%s"}\n' "$r" "$tag" > "./${prefix}.${r}.json"
    done
    log "[dry-run] send-minimax-tasks.sh (N=$N WORKERS=${WORKERS:-$mns})"
  else
    MAX_NUM_SEQS="$mns" SPEC="$spec" VLLM_RBLN_SAMPLER="$sampler" \
      setsid ./serve-minimax.sh >"$LOG_DIR/serve_${tag}.log" 2>&1 &
    SERVER_PID=$!
    log "server PID=$SERVER_PID, ready 대기 (max ${READY_TIMEOUT}s)"

    if ! wait_ready; then
      log "[!] 서버 ready 실패 → 건너뜀 (로그: $LOG_DIR/serve_${tag}.log)"
      stop_server "$SERVER_PID"
      SERVER_PID=""
      return 1
    fi
    log "server ready. 부하 전송 시작"
    # 어떤 RBLN 장치에 배치됐는지 기록(공유 머신 충돌 확인용)
    if command -v rbln-stat >/dev/null 2>&1; then
      rbln-stat -j 2>/dev/null | python3 -c "
import json,sys
try:
    d=json.load(sys.stdin)
except Exception:
    sys.exit(0)
busy=[x['npu'] for x in d['devices'] if int(x['memory']['used'])>0]
print('[devices] mem>0 인 npu:', busy)
" >>"$LOG_DIR/serve_${tag}.log" 2>&1 || true
    fi

    N="$N" STEP_LIMIT="$STEP_LIMIT" WORKERS="${WORKERS:-$mns}" \
      OUTPUT="$outdir/preds.json" \
      ./send-minimax-tasks.sh >"$LOG_DIR/tasks_${tag}.log" 2>&1 || \
      log "[!] send-minimax-tasks.sh 비정상 종료(계속 진행): $LOG_DIR/tasks_${tag}.log"

    log "부하 완료. 서버 graceful 종료 → JSON dump"
    stop_server "$SERVER_PID"
    SERVER_PID=""
    sleep 5   # 장치 해제 여유
  fi

  # JSON 수집
  if compgen -G "./${prefix}.*.json" >/dev/null; then
    mv ./"${prefix}".*.json "$outdir/"
    log "수집 완료: $(ls "$outdir"/*.json | wc -l) 개 → $outdir/"
  else
    log "[!] JSON 없음: ${tag} (부하가 없었거나 shutdown 미도달)"
  fi
}

log "스윕 시작: ${#MNS_LIST[@]}×${#SPEC_LIST[@]}×${#SAMPLER_LIST[@]} = $((${#MNS_LIST[@]} * ${#SPEC_LIST[@]} * ${#SAMPLER_LIST[@]})) 회 (DRY_RUN=$DRY_RUN)"
for mns in "${MNS_LIST[@]}"; do
  for spec in "${SPEC_LIST[@]}"; do
    for sampler in "${SAMPLER_LIST[@]}"; do
      run_one "$mns" "$spec" "$sampler" || true
    done
  done
done
log "스윕 종료. 결과: $RESULTS_DIR/"
