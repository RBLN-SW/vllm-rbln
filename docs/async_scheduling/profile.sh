#!/usr/bin/env bash
# Capture a torch/perfetto trace for one config.
#
#   ./profile.sh async     # --async-scheduling
#   ./profile.sh sync      # nothing - the executor defaults to --no-async-scheduling
#
# Keep MAXTOK small: an 8 s run already produces ~4 M events and ~60 MB per rank,
# and profiling roughly halves throughput - the trace is for structure, never for
# absolute timing.
#
# The trace is CPU-only: TorchProfilerActivityMap has no RBLN entry, so there is
# no NPU track. Cross-thread overlap IS visible (gloo runs on pt_gloo_runloop,
# get_output on the async output thread).
#
# This is a structural capture, not a correctness run: --run-iter is 1 (no
# determinism loop) and MAXTOK is deliberately off the golden's 256, so no
# native golden is matched. Use ab_throughput.sh for anything judged.
#
# Env: OUTDIR, BATCH(8), MAXTOK(32), DP(4), RBLN_DEVICES(4,5,6,7), VENV,
#      LOCAL_CACHE, HF_HOME_DIR, COMPILED_MODEL_PATH.
set -u
CFG=${1:?usage: profile.sh async|sync}
VENV=${VENV:-$HOME/codebase/vllm-rbln-executor/.venv}
source "$VENV/bin/activate"

OUTDIR=${OUTDIR:-./_prof}; mkdir -p "$OUTDIR"
DP=${DP:-4}; BATCH=${BATCH:-8}; MAXTOK=${MAXTOK:-32}
export RBLN_DEVICES=${RBLN_DEVICES:-4,5,6,7}

export VLLM_RBLN_EXEC_CONFIG_HOME="$(cd "$OUTDIR" && pwd)/.config/vllm-rbln-exec"
mkdir -p "$VLLM_RBLN_EXEC_CONFIG_HOME"
LOCAL_CACHE=${LOCAL_CACHE:-$HOME/.cache/vllm-rbln-executor}
HF_HOME_DIR=${HF_HOME_DIR:-/mnt/shared_data/groups/sw_dev/.cache/huggingface}
python3 -m vllm_rbln_executor.cli config set local_cache "$LOCAL_CACHE"
python3 -m vllm_rbln_executor.cli config set hf_home "$HF_HOME_DIR"

case "$CFG" in
  async) CFGENV=""; CFGARG="--async-scheduling" ;;
  sync)  CFGENV=""; CFGARG="" ;;
  *) echo "unknown config $CFG (async|sync)"; exit 1 ;;
esac

# VLLM_RBLN_SAMPLER=1 and the absence of the two OFFLINE flags are load-bearing;
# see the long note in ab_throughput.sh for why either one silently ruins a run.
COMMON="VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error \
VLLM_RBLN_SAMPLER=1 \
VLLM_RBLN_AUTO_PORT=1 RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 \
VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1 \
SPDLOG_LEVEL=warning RBLN_VERBOSE=warning \
VLLM_LOGGING_LEVEL=INFO RBLN_DISABLE_AUTO_RDMA_IP=1 RBLN_DEVICES=$RBLN_DEVICES"

cleanup() {
  pkill -9 -u "$(id -u)" -f "vllm_rbln_executor.cli" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "VLLM::" 2>/dev/null
  find /dev/shm -maxdepth 1 -uid "$(id -u)" -delete 2>/dev/null
  sleep 5
}
trap cleanup EXIT

PDIR="$OUTDIR/$CFG"; rm -rf "$PDIR"; mkdir -p "$PDIR"
cleanup
ARGS="vllm-decoderonly gpt-oss -m gpt-oss-120b -ep -dp $DP -rsd 1 \
-s 131072 --block-size 1024 -pcs 512 -b $BATCH -nblk 129 \
--max-tokens $MAXTOK --num-prompts $BATCH --run-iter 1 --cache-ignore ${CFGARG:-} \
${COMPILED_MODEL_PATH:+--compiled-model-path $COMPILED_MODEL_PATH} \
--torch-profile --profile-dir $PDIR rbln-run"

echo "=== PROF $CFG b=$BATCH maxtok=$MAXTOK $(date +%T) ==="
# shellcheck disable=SC2086
( cd "$OUTDIR" && env $COMMON $CFGENV python3 -m vllm_rbln_executor.cli $ARGS ) \
  >"$OUTDIR/${CFG}.log" 2>&1
echo "  rc=$? $(date +%T)"
find "$PDIR" -name "*.pt.trace.json*" -printf "  %p (%s bytes)\n"
echo "analyse: python3 $(dirname "$0")/trace_analyze.py <trace.json.gz> --step"
