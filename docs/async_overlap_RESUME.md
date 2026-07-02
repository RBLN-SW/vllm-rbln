# async overlap — RESUME (full-layer 서버 이어받기용 단일 진입점)

> 2026-07-02. 이 파일 하나로 **full-layer 가능한 서버에서 바로 재현·이어서 진행**할 수 있게 정리.
> 배경/설계: `async_overlap_report.md`, `async_overlap_deferred_design.md`(상세·시행착오 로그). 이 파일은 **현재 상태 + 재현 레시피 + 남은 일**.

## 0. 목표
gpt-oss-120b EP+DP4 decode에서 매 step 도는 DP `gloo:all_reduce`(N+1)를 NPU `forward`(N)/sampler(N) 실행 구간에 **겹쳐** step latency를 줄인다. 그러면서 **정합성(parity) 유지.**

## 1. 현재 상태 (branch `async-overlap-prototype`, origin push됨)

| 항목 | 상태 |
|---|---|
| C7/C8/C9a/C9b deferral 구현 | ✅ (forward+sampler를 device 스레드로, EM/ST 즉시 리턴, AsyncOutput) |
| 정합성 (6/12/18-layer A/B vs sync) | ✅ **0 mismatch** |
| **all_reduce(N+1) ↔ forward(N) overlap** | ✅ **decode 74%** (host span 실측, §4) |
| deferral 실제 engage | ✅ forward가 device executor 스레드에서 실행(thread-id 확인) |
| full-layer(36층) | ⛔ 이 박스 KMD/TDR timeout (§5) — **full-layer 서버에서 확인 필요** |

**결론**: reduced-layer(≤18)에서 목표 달성 확인됨. full-layer 서버에서 남은 건 (a) full-layer 정합성·overlap 재측정, (b) §6 잔여 이슈.

## 2. ⚠️ 측정 함정 — torch.profile/Perfetto로는 overlap 안 보임 (반드시 숙지)
- RBLN forward는 **동기 host-blocking**(`DynamoRuntime.run`)이라 GPU처럼 async 스트림이 없음 → overlap을 내려면 forward를 **별도 device 스레드**(`_DeviceForwardExecutor`)로 밀어야 함(그게 C9의 전부).
- 그 device 스레드는 torch.profiler가 enter 안 한 raw thread → **`record_function`이 thread-local이라 no-op**(트레이스에 forward 이벤트 0개, 검증됨). kineto도 RBLN device 작업을 device-track으로 못 잡음(CUPTI 등가물 없음).
- ⇒ **Perfetto엔 forward가 안 보이거나 메인 tid에 오귀속돼 직렬처럼 보임. 자동 `gloo∩rebel/sync_run` quant도 거짓 0%.** (GPU였다면 CUPTI가 스트림에 잡아서 그냥 보였을 것 — 이건 RBLN 제약.)
- **overlap은 host `perf_counter` span으로만 신뢰성 있게 측정** (§4). Perfetto로 눈 확인이 필요하면 그 span을 Chrome-trace JSON으로 변환(`docs/async_overlap_scripts/span_to_perfetto.py`).

## 3. 재현 레시피

### 3.0 환경
- `~/codebase/vllm-executor`(parity 하네스 + editable venv), `vllm-rbln` = 이 브랜치, `torch_rbln` 설치됨, gpt-oss-120b 접근, **exclusive RBLN 박스**.
- 공통 env:
```bash
cd ~/codebase/vllm-executor && source .venv/bin/activate
export VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error VLLM_RBLN_AUTO_PORT=1
export RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1
export RBLN_VERBOSE=2 SPDLOG_LEVEL=warning VLLM_LOGGING_LEVEL=INFO RBLN_DEVICES=0,1,2,3
```
- 게이트: `VLLM_RBLN_OPTIMISTIC_SCHED=1 VLLM_RBLN_ASYNC_FORWARD=1` (deferral). sync baseline은 `VLLM_RBLN_DISABLE_ASYNC=1`.
- **레이어**: reduced 검증은 `--num-hidden-layers 18`. full-layer는 그 옵션 빼기(+ §5 timeout).

### 3.1 정합성 A/B (예: 18층)
```bash
# sync baseline
rm -f ~/.cache/vllm-rbln-exec/rbln_results_*L18_T16*P4_*
VLLM_RBLN_DISABLE_ASYNC=1 python3 -m vllm_rbln_exec.parity_runner --task r --model gpt-oss-120b \
  --ep --dp 4 --rsd 1 --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 --batch 1 \
  --num-hidden-layers 18 --max-num-blocks 129 --cache-results --cache-ignore --max-tokens 16 --num-prompts 4
cp ~/.cache/vllm-rbln-exec/rbln_results_*L18_T16*P4_*.json /tmp/base18.json
# async
rm -f ~/.cache/vllm-rbln-exec/rbln_results_*L18_T16*P4_*
VLLM_RBLN_OPTIMISTIC_SCHED=1 VLLM_RBLN_ASYNC_FORWARD=1 python3 -m vllm_rbln_exec.parity_runner ...(동일)...
# token_ids 비교 → 0 mismatch 기대
```

### 3.2 overlap 실측 (host span — 신뢰 소스)
```bash
VLLM_RBLN_SPAN_LOG=1 VLLM_RBLN_OPTIMISTIC_SCHED=1 VLLM_RBLN_ASYNC_FORWARD=1 \
python3 -m vllm_rbln_exec.parity_runner ...(async, --profile 불필요)... 2>&1 \
  | grep -aE "SPAN (fwd|allreduce)" > /tmp/spans.log
python3 docs/async_overlap_scripts/overlap_from_spans.py /tmp/spans.log
#  -> "decode all_reduce overlapped by forward = 74.x%"  (18층 기준)
```
`VLLM_RBLN_SPAN_LOG=1`이면 `SPAN fwd <pid> <t0> <t1>`(device 스레드) / `SPAN allreduce <pid> <t0> <t1>`(main)를 stderr로 찍음. all_reduce span은 `forward_context.num_tokens_across_dp`(DP gloo all_reduce)를 감쌈.

### 3.3 Perfetto로 눈 확인 (선택)
```bash
python3 docs/async_overlap_scripts/span_to_perfetto.py /tmp/spans.log   # -> /tmp/c9_overlap_perfetto.json
# ui.perfetto.dev 에 드래그. rank별 "forward (device thread)" 트랙과 "DP all_reduce (main)" 트랙이 겹치는 것 확인.
```

## 4. overlap 판독법 (왜 74%가 진실인가)
- decode 스텝만(<10ms span) 본다. prefill forward + straggler all_reduce **wait**(수십~수백ms)를 빼야 함 — 이게 안 빼면 full-run isect가 3.6%로 희석돼 오판(과거 실수).
- all_reduce(N)은 EM(N)의 `_prepare_inputs`에서 forward(N) submit **전**에 나감 → forward(N) **내부**에서 도는 all_reduce는 **N+1번째** → all_reduce(N+1)이 forward(N)/sampler(N) 도중 실행, sampler 안 기다림. ✅ 목표.
- 18층 실측: decode all_reduce의 **74%가 forward 안**, rank별 39/56 span이 forward에 완전 포함.

## 5. full-layer 실행 (이 박스 이슈 — 새 서버에서 재확인)
- 이 박스는 KMD `3.3.0~rc2`, TDR `timeout_s=6s`. full-layer warmup forward(120B weight-free transform)가 6s 초과 → `SYS_TASK_ABORTED(504)`/`SYS_BUSY(502)`. **CI(udc-08, KMD 3.2.x)는 같은 config full-layer PASS**(obedients `schedule-rebel-rebel-compiler-ci` build 36/37) → **박스 환경 문제, async 무관**(plain dev·rebel dev508/dev519 모두 재현).
- full-layer 서버에서 timeout 나면:
  - 커널 TDR/runtime: `sudo rbln-smi tdr --group 0 --value 600` + `sudo rbln-smi timeout --group 0 --value 600`.
  - 앱(502, optimum 기본 60s): `optimum/rbln/modeling.py`의 `rebel.Runtime(timeout=rbln_config.timeout)` → `... or 600`.
  - 가능하면 CI와 같은 KMD(3.2.x)/docker image 권장.
- **device 위생 (중요)**: `kill -9`로 run을 죽이면 "killed" context가 ASID를 안 놓고 누수 → 이후 `SYS_ENODEV`/`No free ASID`로 device wedge. **정상 종료 유도**, 죽였으면 `rbln-stat`로 killed context 확인 후 필요시 reboot. `find /dev/shm -maxdepth 1 -user $USER -delete`로 shm 정리.

## 6. 남은 일
1. **full-layer 정합성 + overlap 재측정** (full-layer 서버, §3 레시피 그대로 `--num-hidden-layers` 빼고). forward가 더 길어 overlap %가 더 높을 것으로 기대.
2. **`RUN_INTERNAL EnsureSyncedOnPhysicalView` 버그**: 18층에서 간헐적으로 3/4 지점 크래시(get_output D2H 수명버그 `6491df7`와 **다른** vmem 경로). deferred 경로의 또 다른 device-tensor 수명 이슈로 추정 — 재현·수정 필요.
3. **overlap 74% → 잔여 25%**: forward가 짧거나 all_reduce-wait가 긴 스텝. forward가 길어지는 full-layer/큰 배치에서 개선 여지.
4. 최종 정량: full-layer overlap% + step cadence(sync 대비) 감소.

## 7. 핵심 파일·커밋 (이 브랜치)
- deferral: `vllm_rbln/v1/worker/rbln_model_runner.py` — `_DeviceForwardExecutor`, `execute_model`의 `_fast_defer`/`_run_forward`, `sample_tokens`의 `_sample_task`/`_bookkeeping_async_fast`, `AsyncRBLNModelRunnerOutput`.
- DP all_reduce: `vllm_rbln/forward_context.py` `num_tokens_across_dp` (`c9_all_reduce` record_function + SPAN 로그).
- 커밋: `96acef3`(C9a) `d55098b`(C9b) `6491df7`(D2H 수명 fix) `97cd641`(span 계측) + docs. (`git log --oneline origin/dev..HEAD`)
- 측정 스크립트: `docs/async_overlap_scripts/overlap_from_spans.py`, `span_to_perfetto.py`.
- 동시성 주의(구현 시 재확인): thread-local `inference_mode`(device 스레드에서 재진입), sampling_metadata 버퍼 clone(EM(N+1) race), deferred 출력 D2H는 sample_task 안에서(get_output 지연 시 vmem 회수).
