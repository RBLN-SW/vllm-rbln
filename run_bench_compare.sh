#!/usr/bin/env bash
# kv_cache_access — torch hook vs runtime hook 'bench' 모드 성능 비교.
#
# 두 path 모두 같은 model / block-size / DMA 환경에서 fresh-alloc per-layer
# round-trip 을 측정한다.
#
#   - torch path  : VLLM_RBLN_USE_DEVICE_TENSOR=1 + VLLM_RBLN_KV_CACHE_HOOK_MODE=bench
#   - runtime path: VLLM_RBLN_USE_DEVICE_TENSOR=0 + VLLM_RBLN_KV_CACHE_RT_HOOK_MODE=bench
#
# Usage:
#   ./run_bench_compare.sh                 # default: device 12, max-num-blocks 3
#   RBLN_DEVICES=14 ./run_bench_compare.sh # 다른 NPU
#   MAX_BLOCKS=30 ./run_bench_compare.sh   # 더 큰 layer (62.9 MB)
#   ONLY=torch ./run_bench_compare.sh      # 한 쪽만
#   ONLY=runtime ./run_bench_compare.sh
#   QUIET=1 ./run_bench_compare.sh         # stdout 없이 log 파일에만
#   MODE=bench_1block ./run_bench_compare.sh         # hook 모드 변경
#   MODEL=qwen3-1.7b ./run_bench_compare.sh          # 다른 model alias
#
# Model alias 는 vllm-executor/src/vllm_rbln_exec/utils/registry.py 의
# `KNOWN_MODELS` 에 등록된 것. 예: llama3.2-1b (default), llama3-8b,
# qwen3-0.6b, qwen3-1.7b, qwen3-4b, qwen3-8b, ...
# 출력 폴더가 모델별 다르면 좋음 — OUT_DIR 같이 지정 권장:
#   MODEL=qwen3-1.7b OUT_DIR=/tmp/kv_cache_access/qwen3_1.7b ./run_bench_compare.sh
#
# 기본 동작: stdout + 파일 동시 출력 (tee). 진행 상황 실시간으로 보임.
# 실행 전: `rbln-smi` 로 RBLN_DEVICES 가 idle (Memory 0.0B) 인지 확인.
# 1회 ~3분 (--cache-ignore 라 재컴파일).

set -euo pipefail

# ---- 파라미터 (env 로 override 가능) ----
: "${RBLN_DEVICES:=12}"
: "${MAX_BLOCKS:=3}"
: "${MODEL:=llama3.2-1b}"
: "${ONLY:=}"                              # "" | torch | runtime
: "${OUT_DIR:=/tmp/kv_cache_access}"
: "${QUIET:=0}"                            # 1 → stdout 안 보여줌, log 파일만
: "${MODE:=bench}"                         # bench | bench_1block | bench_reused | ...
                                           # 양쪽 hook 동시 적용 (이름이 같은 모드만 의미 있음)

export RBLN_DEVICES

TASK_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXEC_DIR="${TASK_DIR}/vllm-executor"

mkdir -p "${OUT_DIR}"

echo "================================================================="
echo "  kv_cache_access — torch vs runtime bench compare"
echo "================================================================="
echo "  RBLN_DEVICES : ${RBLN_DEVICES}"
echo "  MAX_BLOCKS   : ${MAX_BLOCKS}  (layer 크기 = blocks × 2.10 MB)"
echo "  MODEL        : ${MODEL}"
echo "  OUT_DIR      : ${OUT_DIR}"
echo "  ONLY         : ${ONLY:-<both>}"
echo "  MODE         : ${MODE}"
echo "================================================================="

cd "${EXEC_DIR}"
# env_rebel.sh 가 venv 활성화 + REBEL_HOME 등 설정.
# shellcheck disable=SC1091
source ./env_rebel.sh

run_one() {
    local label="$1"
    local dt_flag="$2"          # 1 (torch) or 0 (runtime)
    local mode_env="$3"         # VLLM_RBLN_KV_CACHE_HOOK_MODE or _RT_HOOK_MODE
    local trace_env="$4"        # VLLM_RBLN_KV_CACHE_HOOK_TRACE or _RT_HOOK_TRACE
    local log="${OUT_DIR}/bench_${label}.log"

    echo
    echo "[run_bench_compare] >>> ${label}  (USE_DEVICE_TENSOR=${dt_flag})"
    echo "[run_bench_compare]     log: ${log}"

    # 사용자가 준 baseline command 그대로. hook 환경변수만 mode 별 분기.
    # 두 hook 모드 다 'bench' 인 점은 의도 — fresh-alloc per-layer round-trip 측정.
    # 기본은 tee 로 stdout + 파일 동시 출력 (진행 상황 실시간). QUIET=1 이면 파일만.
    local rc=0
    set +e
    if [[ "${QUIET}" == "1" ]]; then
        env \
            "${mode_env}=${MODE}" \
            "${trace_env}=1" \
            RBLN_DUMMY_DEVICE=0 \
            "VLLM_RBLN_USE_DEVICE_TENSOR=${dt_flag}" \
            RBLN_VERBOSE=2 \
            TORCH_RBLN_DISABLE_FALLBACK=compile_error \
            python3 -u -m vllm_rbln_exec.parity_runner \
            --task r --model "${MODEL}" \
            --tp 1 --dp 1 --rsd 1 \
            --max-model-len 2048 --block-size 1024 \
            --max-num-blocks "${MAX_BLOCKS}" \
            --cache-results --cache-ignore \
            --threshold 0.99 --max-num-seqs 1 --mode 0 \
            --logprobs 0 --use-cached-models --skip-validation \
            > "${log}" 2>&1
        rc=$?
    else
        # `python3 -u` 로 unbuffered 출력. stdbuf -oL 로 line-buffered tee.
        env \
            "${mode_env}=${MODE}" \
            "${trace_env}=1" \
            RBLN_DUMMY_DEVICE=0 \
            "VLLM_RBLN_USE_DEVICE_TENSOR=${dt_flag}" \
            RBLN_VERBOSE=2 \
            TORCH_RBLN_DISABLE_FALLBACK=compile_error \
            stdbuf -oL -eL \
            python3 -u -m vllm_rbln_exec.parity_runner \
            --task r --model "${MODEL}" \
            --tp 1 --dp 1 --rsd 1 \
            --max-model-len 2048 --block-size 1024 \
            --max-num-blocks "${MAX_BLOCKS}" \
            --cache-results --cache-ignore \
            --threshold 0.99 --max-num-seqs 1 --mode 0 \
            --logprobs 0 --use-cached-models --skip-validation \
            2>&1 | tee "${log}"
        rc=${PIPESTATUS[0]}
    fi
    set -e

    echo "[run_bench_compare]     exit=${rc}"
    echo "[run_bench_compare]     summary:"
    summarize "${log}" "${label}"
}

summarize() {
    local log="$1"
    local label="$2"
    # torch hook → BENCH] step= 라인
    # runtime hook → BENCH] step= 또는 RT_BENCH] step= 라인
    # 둘 다 동일 prefix 사용중이므로 단일 grep 으로 처리.
    local last
    last="$(grep -E "BENCH[_A-Z0-9]*\] step=" "${log}" | tail -5 || true)"
    if [[ -z "${last}" ]]; then
        echo "      (no BENCH lines found — check ${log} for errors)"
        tail -10 "${log}" | sed 's/^/      | /'
        return
    fi
    echo "${last}" | sed 's/^/      /'
    local final
    final="$(grep -E "FINAL/(LAYER|SPLIT)|CUMULATIVE/LAYER" "${log}" | tail -3 || true)"
    if [[ -n "${final}" ]]; then
        echo "${final}" | sed 's/^/      /'
    fi
}

if [[ -z "${ONLY}" || "${ONLY}" == "torch" ]]; then
    run_one torch   1 VLLM_RBLN_KV_CACHE_HOOK_MODE     VLLM_RBLN_KV_CACHE_HOOK_TRACE
fi
if [[ -z "${ONLY}" || "${ONLY}" == "runtime" ]]; then
    run_one runtime 0 VLLM_RBLN_KV_CACHE_RT_HOOK_MODE  VLLM_RBLN_KV_CACHE_RT_HOOK_TRACE
fi

echo
echo "================================================================="
echo "  done. logs in ${OUT_DIR}/"
echo "  비교:"
echo "    grep -E 'BENCH[_A-Z0-9]*\\] step=' ${OUT_DIR}/bench_torch.log   | tail -10"
echo "    grep -E 'BENCH[_A-Z0-9]*\\] step=' ${OUT_DIR}/bench_runtime.log | tail -10"
echo "    grep -E 'FINAL/LAYER' ${OUT_DIR}/bench_torch.log ${OUT_DIR}/bench_runtime.log"
echo "================================================================="
