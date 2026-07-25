#!/usr/bin/env bash
# ctx-len vs phase 실험: greedy, sampler=0, batch size(max-num-seqs) 2/4/8/16 에서
# 세 조건 — spec off / spec on(num_spec=1) / spec on(num_spec=3) — 을 각각 스윕.
# per-step context length 기록(신규)이 켜진 상태로 측정하여 산점도용 데이터를 만든다.
#
# 조건별로 JSON 파일명이 같으므로(예 max-num-seqs=8_spec_=1_sampler=0) 반드시
# 서로 다른 RESULTS_DIR 서브폴더로 분리한다.
#
# 결과: itl-ctx-results/{specoff,ns1,ns3}/mns=*/...
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

BASE="${BASE:-itl-ctx-results}"
mkdir -p "$BASE"

# label  SPEC  NUM_SPEC
run_cond() {
  local label="$1" spec="$2" ns="$3"
  echo "############ CONDITION: $label (spec=$spec num_spec=$ns) ############"
  RESULTS_DIR="$BASE/$label" NUM_SPEC="$ns" SPEC_LIST="$spec" \
    MNS_LIST="${MNS_SWEEP:-2 4 8 16}" SAMPLER_LIST="0" \
    ./run-minimax-sweep.sh
}

run_cond specoff off 3   # spec off: num_spec 무의미(speculative-config 자체가 없음)
run_cond ns1     on  1
run_cond ns3     on  3
echo "############ CTX EXPERIMENT DONE → $BASE ############"
