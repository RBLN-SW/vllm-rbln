#!/usr/bin/env bash
# Paired multi-turn suite against a live serve.sh instance.
#
# The controls exist because the first, uncontrolled run of this comparison
# reported the opposite sign from the truth:
#
#   - one discarded warmup run absorbs cold-start effects
#   - a distinct seed per repetition, so every measured run sees conversations
#     whose prefixes the KV cache has never held (this vLLM exposes no HTTP
#     reset-prefix-cache, so freshness is bought with seeds)
#   - the same seeds, workloads and order for both modes, so runs pair up
#   - workloads interleaved (short, long, short, long, ...) so machine drift
#     does not alias onto one workload
#   - device state stamped before each run, to catch a neighbour job appearing
#     on the other NPUs -- it biases wall clock and has done so here
#
# Run it once per mode, then `analyze.py` pairs the two by (workload, seed).
#
# Usage: run_suite.sh <subblock|pagelayout>
# Env:   OUT_DIR (default ./results), VLLM_REPO (required), SEEDS, WORKLOADS,
#        PORT, MODEL
set -uo pipefail
MODE=${1:?usage: run_suite.sh <subblock|pagelayout>}
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
REPO=$(cd "$HERE/../.." && pwd)
OUT_DIR=${OUT_DIR:-$PWD/results}
VLLM_REPO=${VLLM_REPO:?set VLLM_REPO to a vLLM source checkout (see README)}
DRIVER=$VLLM_REPO/benchmarks/multi_turn/benchmark_serving_multi_turn.py
MODEL=${MODEL:-MiniMaxAI/MiniMax-M2.5}
PORT=${PORT:-8102}
SEEDS=${SEEDS:-"1 2 3"}
WORKLOADS=${WORKLOADS:-"short long"}

[ -f "$DRIVER" ] || { echo "missing $DRIVER"; exit 2; }
mkdir -p "$OUT_DIR"
source "$REPO/.venv/bin/activate"
unset PYTHONPATH
# The generator reads its text corpus relative to the working directory.
cd "$OUT_DIR"
[ -f pg1184.txt ] || { echo "missing $OUT_DIR/pg1184.txt (see README)"; exit 2; }

run() {
  local wl=$1 seed=$2 tag=$3
  echo "=== $(date '+%F %T') mode=$MODE workload=$wl seed=$seed tag=$tag ==="
  rbln-stat 2>/dev/null | sed -n '7,14p' | awk '{print "    " $0}'
  python "$DRIVER" \
      --model "$MODEL" --served-model-name MiniMax \
      --url "http://localhost:$PORT" \
      --input-file "$HERE/workloads/multi_turn_${wl}.json" \
      --num-clients 4 --max-active-conversations 8 \
      --seed "$seed" --request-timeout-sec 1800 \
      --stats-json-output "$OUT_DIR/st_${MODE}_${wl}_${seed}.json" \
      > "$OUT_DIR/run_${MODE}_${wl}_${tag}${seed}.log" 2>&1
  echo "    exit=$? $(grep -oE 'benchmark runtime: [0-9.]+ sec' \
      "$OUT_DIR/run_${MODE}_${wl}_${tag}${seed}.log" | tail -1)"
}

run short 900 warmup_      # discarded
for s in $SEEDS; do
  for wl in $WORKLOADS; do run "$wl" "$s" ""; done
done
echo "=== $(date '+%F %T') suite done for $MODE ==="
