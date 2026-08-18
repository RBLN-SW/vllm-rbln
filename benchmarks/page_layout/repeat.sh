#!/usr/bin/env bash
# N replays of one fixed workload against a cold prefix cache, on one serve.
#
# The suite buys cache freshness with a fresh seed per repetition, because this
# benchmark used to have no way to clear the cache. It does: dev mode exposes
# POST /reset_prefix_cache. Resetting instead of reseeding removes the largest
# term in the noise -- the hit rate of a single run swings 15.6 points between
# seeds, which is far more than anything worth comparing -- and drops the cost of
# a repetition from a 20-minute serve restart to a 3-minute run.
#
# Run this with identical code on both sides first (an A/A null). The spread of
# the results *is* the resolution of the instrument; no A/B claim below it means
# anything. That check is why this script exists.
#
# The first runs on a fresh serve are not comparable to the rest: an A/A null
# measured 0.6619, 0.6836, then 0.79-0.81 for the remainder, and hit rate tracked
# runtime at r=-0.892. Something outside the prefix cache warms up -- lazily
# compiled shapes, most likely -- and until it has, batches run thinner and
# conversations overlap less, so less gets shared. WARMUP runs are therefore run
# and discarded, not reported.
#
# Usage: repeat.sh <tag> [n] [workload] [seed]
# Env:   PORT (8102), OUT_DIR, WARMUP (3)
set -uo pipefail
TAG=${1:?usage: repeat.sh <tag> [n] [workload] [seed]}
N=${2:-10}
WL=${3:-long}
SEED=${4:-1}
WARMUP=${WARMUP:-3}
PORT=${PORT:-8102}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
OUT_DIR=${OUT_DIR:-$HERE/results_repeat}
DRIVER=${VLLM_REPO:-$HOME/workspace/vllm}/benchmarks/multi_turn/benchmark_serving_multi_turn.py
MODEL=${MODEL:-MiniMaxAI/MiniMax-M2.5}

[ -f "$DRIVER" ] || { echo "missing $DRIVER"; exit 2; }
mkdir -p "$OUT_DIR"
cp -n "$HERE/results/pg1184.txt" "$OUT_DIR/" 2>/dev/null
source "$HERE/../../.venv/bin/activate"
unset PYTHONPATH
cd "$OUT_DIR"

# Prove the reset endpoint is really there before trusting a single number:
# without it every repetition would silently inherit the previous one's cache.
code=$(curl -s -m 10 -o /dev/null -w "%{http_code}" -X POST "localhost:$PORT/reset_prefix_cache")
[ "$code" = "200" ] || { echo "FAIL: /reset_prefix_cache returned $code (VLLM_SERVER_DEV_MODE=1?)"; exit 3; }

snap() {
  curl -s -m 10 "localhost:$PORT/metrics" \
    | grep -E 'vllm:prefix_cache_(hits|queries)_total\{' \
    | python3 -c "
import sys
h = q = 0.0
for line in sys.stdin:
    value = float(line.rsplit(' ', 1)[1])
    if 'hits' in line:
        h += value
    else:
        q += value
print(h, q)"
}

echo "tag=$TAG n=$N (+$WARMUP warmup) workload=$WL seed=$SEED port=$PORT"
for i in $(seq 1 $((WARMUP + N))); do
  curl -s -m 30 -X POST "localhost:$PORT/reset_prefix_cache" > /dev/null
  read -r H0 Q0 <<< "$(snap)"
  python3 "$DRIVER" \
      --model "$MODEL" --served-model-name MiniMax \
      --url "http://localhost:$PORT" \
      --input-file "$HERE/workloads/multi_turn_${WL}.json" \
      --num-clients 4 --max-active-conversations 8 \
      --seed "$SEED" --request-timeout-sec 1800 \
      --stats-json-output "$OUT_DIR/st_${TAG}_${i}.json" \
      > "$OUT_DIR/run_${TAG}_${i}.log" 2>&1
  read -r H1 Q1 <<< "$(snap)"
  RUNTIME=$(grep -oE 'benchmark runtime: [0-9.]+' "$OUT_DIR/run_${TAG}_${i}.log" | tail -1 | grep -oE '[0-9.]+')
  if [ "$i" -le "$WARMUP" ]; then KEEP="warmup"; else KEEP="keep"; fi
  python3 -c "
h = $H1 - $H0
q = $Q1 - $Q0
rate = h / q if q else float('nan')
print(f'  ${TAG} #${i}  hit_rate={rate:.4f}  queries={q:.0f}  runtime=${RUNTIME:-0}s  ${KEEP}')
" | tee -a "$OUT_DIR/summary_${TAG}.txt"
done
echo "done: $OUT_DIR/summary_${TAG}.txt"
