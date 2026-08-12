#!/usr/bin/env bash
# Throughput A/B for async vs sync scheduling.
#
# Metric = vLLM's own tqdm "est. speed output" (wall clock, total_out_toks /
# elapsed). That is only comparable across arms when every request emits exactly
# max_tokens, which needs VLLM_RBLN_EXEC_IGNORE_EOS=1 (see COMMON). It is NOT a
# suite default - suite.py `_ignore_eos` reads that variable and defaults to 0.
#
# Read the results with agg_tokps.py - never by eyeballing the last tok/s line.
# Run-to-run spread is 1-3%. NOTE: RUNITER is NOT the old REPRO. --repro-run N
# added N extra full generate calls (more warm tok/s samples per process);
# --run-iter N instead runs a separate 6-token full-vocab-logits determinism
# loop (workers.py _run_rbln_repro) and leaves the single free generate alone.
# One process therefore yields ONE free generate, so ROUNDS is now the only
# lever on sample count and defaults to 3 to keep the old REPRO=2 sample budget.
# That is not free: every extra round is a process restart, 13-15 min of
# recompilation each, so a 2-arm ROUNDS=3 sweep is ~6 restarts.
#
# MAXTOK defaults to 256 so the run lands on a cached native golden.
# native_results_cache_key excludes dp, ep, block_size, max_model_len, pcs and
# nblk, so the golden key is only model_id + max_num_seqs + mt: b8/mt256 picks
# the gpt-oss-120b golden even under DP4+EP. BATCH and MAXTOK are therefore the
# two knobs that select it - off-key values compare against nothing, silently.
#
# Env: OUTDIR, BATCH(8), MAXTOK(256), RUNITER(2), ROUNDS(3), DP(4), CONFIGS,
#      RBLN_DEVICES(4,5,6,7), VENV, LOCAL_CACHE,
#      REMOTE_CACHE, HF_HOME_DIR, COMPILED_MODEL_PATH.
set -u
VENV=${VENV:-$HOME/codebase/vllm-rbln-executor/.venv}
source "$VENV/bin/activate"

OUTDIR=${OUTDIR:-./_ab}; mkdir -p "$OUTDIR"
DP=${DP:-4}; BATCH=${BATCH:-8}; MAXTOK=${MAXTOK:-256}; RUNITER=${RUNITER:-2}
ROUNDS=${ROUNDS:-3}
export RBLN_DEVICES=${RBLN_DEVICES:-4,5,6,7}

# Scope the executor config to this run so the user's global
# ~/.config/vllm-rbln-exec is never overwritten (CI does the same with a
# per-slug CONFIG_HOME).
export VLLM_RBLN_EXEC_CONFIG_HOME="$(cd "$OUTDIR" && pwd)/.config/vllm-rbln-exec"
mkdir -p "$VLLM_RBLN_EXEC_CONFIG_HOME"
LOCAL_CACHE=${LOCAL_CACHE:-$HOME/.cache/vllm-rbln-executor}
REMOTE_CACHE=${REMOTE_CACHE:-/mnt/shared_data/users/yunseong.kim/nas_data/vllm-rbln-exec-golden}
HF_HOME_DIR=${HF_HOME_DIR:-/mnt/shared_data/groups/sw_dev/.cache/huggingface}
python3 -m vllm_rbln_executor.cli config set local_cache "$LOCAL_CACHE"
python3 -m vllm_rbln_executor.cli config set remote_cache "$REMOTE_CACHE"
python3 -m vllm_rbln_executor.cli config set hf_home "$HF_HOME_DIR"
echo "--- vllm-rbln-exec config"
python3 -m vllm_rbln_executor.cli config list

# MKL_NUM_THREADS=1 is a correctness setting, not a tuning knob: with more than
# one MKL thread the generated tokens vary run to run, which makes the runner's
# reproducibility check meaningless. Applies to baseline too.
#
# VLLM_RBLN_SAMPLER=1 is mandatory and must be explicit. The executor's
# _setup_rbln_common_env (suite.py) setdefaults it to 0 on *every* rbln task,
# rbln-run included, because CI wants the CPU sampler. At 0 the runner picks
# vLLM's Sampler instead of RBLNSampler (rbln_model_runner.py:362), so the
# sampled tokens land on the host and the on-device path this branch optimises
# is not in the run at all. Not setting it silently measures the wrong build.
#
# Neither HF_HUB_OFFLINE nor TRANSFORMERS_OFFLINE may be set. The prompt set
# comes from sample_sharegpt, which calls load_dataset(..., streaming=True);
# streaming resolution goes through dataset_module_factory, and this datasets
# version cannot satisfy that from the local hub cache, so offline mode aborts
# the run before the model is even built. TRANSFORMERS_OFFLINE is not a
# narrower alternative - it flips huggingface_hub.constants.HF_HUB_OFFLINE to
# True on its own, producing the identical OfflineModeIsEnabled failure.
# Everything else is already in HF_HOME, so online mode only costs etag checks.
#
# VLLM_RBLN_EXEC_IGNORE_EOS=1 is what makes tok/s an arm-to-arm comparison.
# suite.py `_ignore_eos` defaults it to 0, so requests stop at EOS: measured at
# mt=1024, only 27 of 32 prompts reached the cap and the arms emitted different
# totals (async 29411 vs sync 29511 tokens). Worse than the numerator drift is
# the straggler tail the docstring warns about - finished slots idle while one
# long request runs on, for a different stretch in each arm. Note this does not
# touch the text comparison: ignore_eos decides when generation stops, never
# which token is picked, so two correct arms still produce identical text.
COMMON="MKL_NUM_THREADS=1 VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error \
VLLM_RBLN_SAMPLER=1 VLLM_RBLN_EXEC_IGNORE_EOS=1 \
VLLM_RBLN_AUTO_PORT=1 RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 \
VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1 \
SPDLOG_LEVEL=warning RBLN_VERBOSE=warning \
VLLM_LOGGING_LEVEL=INFO RBLN_DISABLE_AUTO_RDMA_IP=1 RBLN_DEVICES=$RBLN_DEVICES \
${EXTRA_ENV:-}"

# pkill on the cli alone leaves the VLLM::Worker_DP / VLLM::EngineCore children
# holding device memory; they must be killed by name too.
cleanup() {
  pkill -9 -u "$(id -u)" -f "vllm_rbln_executor.cli" 2>/dev/null
  pkill -9 -u "$(id -u)" -f "VLLM::" 2>/dev/null
  find /dev/shm -maxdepth 1 -uid "$(id -u)" -delete 2>/dev/null
  sleep 5
}
trap cleanup EXIT

run() { # $1=name $2=cfgenv $3=round
  # A config may override BATCH (CFG_BATCH). Sweeping batch across separate
  # invocations is not trustworthy here, so any arm that needs comparing has to
  # alternate in time inside one invocation.
  local name="$1" cfgenv="$2" r="$3"
  local eb=${CFG_BATCH:-$BATCH}
  # --num-prompts is per-DP-rank: the suite samples num_prompts * dp prompts,
  # so this keeps the old batch*dp scenario. It does NOT have to equal the
  # golden's prompt count - compare_results zips to the shorter side and
  # sample_sharegpt fills a seed-fixed shuffle from the front, so the first N
  # are the same for any N. CI relies on exactly this (16 prompts vs a
  # 128-prompt golden, Pearsonr 0.9965).
  local np=$eb
  local logf="$OUTDIR/${name}_r${r}.log"
  cleanup
  local ARGS="vllm-decoderonly gpt-oss -m gpt-oss-120b -ep -dp $DP -rsd 1 \
-s 131072 --block-size 1024 -pcs 512 -b $eb -nblk 129 \
--max-tokens $MAXTOK --num-prompts $np --run-iter $RUNITER --cache-ignore \
--cache-results ${EXTRA_ARGS:-} \
${COMPILED_MODEL_PATH:+--compiled-model-path $COMPILED_MODEL_PATH} \
-o $OUTDIR/${name}_r${r}.result.json rbln-run"
  echo "=== RUN $name round=$r b=$eb maxtok=$MAXTOK nprompts=$((np * DP)) $(date +%T) ==="
  # shellcheck disable=SC2086
  ( env $COMMON $cfgenv python3 -m vllm_rbln_executor.cli $ARGS ) >"$logf" 2>&1
  local rc=$?
  echo "  rc=$rc $(date +%T) -> $logf"
  save_outputs "$name" "$r"
}

# The only correctness question that applies to async is "same prompt -> same
# text": async does not emit logprobs, so its Pearsonr against the golden is
# structurally absent (sync and schedonly do emit them and both score 0.99566,
# which is precisely why that gate cannot see the async defect).
#
# result.json is not enough for that comparison - sample_comparison.py stores
# only rbln_run_text[:128], a preview. --cache-results writes the real
# outputs.json into the local cache and records the directory in cache_path,
# so lift it out here. No --cache-tag: that would also retarget
# compile_results_cache_key and send rbln-run looking for a compiled model
# under a tagged path that does not exist. Untagged means every arm shares one
# cache directory, so this copy must happen before the next arm overwrites it.
save_outputs() {
  python3 - "$OUTDIR/$1_r$2.result.json" "$OUTDIR/$1_r$2.outputs.json" <<'PY'
import json, shutil, sys
from pathlib import Path
res, dst = sys.argv[1], sys.argv[2]
try:
    cache = json.loads(Path(res).read_text())["rbln-run"]["cache_path"]
    shutil.copyfile(Path(cache) / "outputs.json", dst)
    print(f"  outputs.json -> {dst}")
except Exception as exc:
    print(f"  outputs.json NOT saved: {exc.__class__.__name__}: {exc}")
PY
}

CONFIGS=${CONFIGS:-"async sync"}
for r in $(seq 1 "$ROUNDS"); do
  for cfg in $CONFIGS; do
    CFG_BATCH=""
    case "$cfg" in
      async) E="" ;;                            # async is the default (see docs section 1)
      sync)  E="VLLM_RBLN_DISABLE_ASYNC=1" ;;
      fsync) E="RBLN_RUNTIME_FORCE_SYNC=1" ;;   # async sched + rebel drains after every op
      # Variable isolation: async_scheduling switches the scheduler class and the
      # runner's output path together. These split them.
      schedonly) E="VLLM_RBLN_ASYNC_RUNNER=0" ;;                            # async sched, sync runner
      runneronly) E="VLLM_RBLN_DISABLE_ASYNC=1 VLLM_RBLN_ASYNC_RUNNER=1" ;;  # sync sched, async runner
      b1) E=""; CFG_BATCH=1 ;;                  # batch as a config, so arms alternate in time
      b8) E=""; CFG_BATCH=8 ;;
      noextcache) E="RBLN_EXTERNAL_ONLY_CACHE_BYTES=0" ;;   # disable rebel ext-ref recon cache
      *) echo "unknown config $cfg"; exit 1 ;;
    esac
    run "$cfg" "$E" "$r"
  done
done
echo "=== DONE: $OUTDIR ==="
echo "now: python3 $(dirname "$0")/agg_tokps.py $OUTDIR/*.log"
echo "     python3 $(dirname "$0")/judge_pearson.py $OUTDIR/*.result.json"
