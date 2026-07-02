# async overlap — RESUME (다음 세션 단일 진입점)

> 2026-07-02 갱신 (STEP 1 판정 완료). 이 파일 하나로 다음 세션에서 바로 이어서 진행한다.
> **방향 확정: 단일 host 스레드로 GPU처럼 overlap을 낸다. multi-threading(구 `_DeviceForwardExecutor`) 방식은 기각.**
> **★ STEP 1 결과(§2): (가) 확정 — decode forward가 GIL을 ~68% 점유. 단일 스레드로는 forward의 ~⅓만
> 겹침 가능 → vLLM Python 레벨만으론 불가. 다음 결정은 유저 몫: (A) rebel 런타임 glue 경량화 vs (B) threading 재검토.**
> 측정 도구: `docs/async_overlap_scripts/{overlap_from_spans.py, span_to_perfetto.py}`.
> GPU vs RBLN 방향성 Q&A(왜 GPU는 되고 RBLN은 안 되나, GIL/stream/Event 역할): `docs/async_overlap_gpu_vs_rbln_QA.md`.

## 0. 목표
gpt-oss-120b EP+DP4 decode에서 매 step 도는 DP `gloo:all_reduce`(N+1)를 NPU forward(N) 실행 구간에
**겹쳐** step latency를 줄인다. 정합성(parity) 유지. 그리고 그 overlap이 **torch profile 결과물에서
GPU처럼(후처리 없이) 보이게** 한다. **반드시 단일 스레드**로 — 별도 device 스레드 방식은 기각됨.

## 1. 지금까지의 핵심 발견 (측정 근거 포함)

### (F1) RBLN 제출은 이미 non-blocking — µs 단위
18층 sync 런에서 `DynamoRuntime.run()`(제출 호출) wall time을 계측(임시 프로브, revert됨):
**중앙값 0.01ms, p90 ~0.25ms** (rank×decode 2496콜). 즉 `run()`은 device 완료를 안 기다리고 드라이버
커맨드 큐에 넣고 즉시 리턴한다. 출력도 device-tensor를 **D2H 없이** 리턴(`sync_runtime.py:274`).
→ **rebel_compiler C++/`run_async` 신설 불필요.** 드라이버 큐는 이미 GPU 스트림처럼 FIFO 비동기
(`runtime_instance.cc:1163-1231`, `rblnSubmitJob`/`rblnWaitJob`).

### (F2) 그런데 `model_executable()` 전체는 decode당 ~3.5ms (host)
SPAN fwd(= `_run_forward` = 컴파일 그래프 전체 실행) 중앙값 **3.5ms**. 개별 `.run()`은 µs인데 전체가
3.5ms인 건 컴파일 그래프를 걸어가는 torch/Python dispatch + subgraph별 prepare_inputs/address-patch
+ 그래프 중간 host op 등에서 host 시간이 쌓이기 때문. **이 3.5ms의 정체가 전체 실현가능성을 가른다(아래 미해결).**

### (F3) inline-async(스레드 없음)는 overlap 0% — D2H 지연만으론 부족
`VLLM_RBLN_OPTIMISTIC_SCHED=1` + `VLLM_RBLN_ASYNC_FORWARD=0`(= inline forward+sampler, D2H는
`AsyncRBLNModelRunnerOutput` immediate 경로로 get_output까지 지연) 실측: **decode overlap 0.0%**.
타임라인이 엄격히 직렬 — `F(3.5ms) … ~9ms 갭 … A(1~2ms) … F …`. forward가 끝나고 한참 뒤 all_reduce가
시작돼 겹칠 게 없음. 즉 "출력 D2H만 지연하면 단일 스레드로 overlap 된다"는 가설은 **틀림**.
- 참고: 이 immediate 경로 자체는 vmem race 없음(D2H copy를 sampler 직후 큐에 넣고 get_output에서
  `torch.rbln.synchronize`). race는 구 C9의 "async 스레드로 D2H 지연"에서만 났던 것.

### (F4) torch profile에서 device-thread 작업 보이게 (참고, threading 기각으로 현재는 무의미)
`rbln_worker.py`에 `_ExperimentalConfig(profile_all_threads=True)` 넣으면 모든 스레드가 트레이스에
잡힘(검증됨). 단일 스레드로 가면 forward가 어차피 main에 찍히므로 특별 설정 없이도 보인다.
GPU식 "device 트랙"까지 원하면 kineto plugin이 필요하나(torch 2.11에 `ProfilerActivity.PrivateUse1`
+ `IActivityProfiler`/`GenericTraceActivity` API 완비, torch_rbln엔 미구현), **overlap이 실제로 나기
전엔 불필요**. 후처리 방식도 기각(유저).

## 2. ★ STEP 1 판정 완료 (2026-07-02) — **(가) 확정**: forward가 GIL의 ~68%를 점유

GIL_PROBE 실측(gpt-oss-120b EP+DP4 18층 decode, 2회 실행: 오염본 /tmp/gil.log + 클린본
`RBLN_VERBOSE=warning` /tmp/gil2.log). warm(steady) 244샘플, **클린 캘리브레이션(free_rate≥85M/s)
154샘플에서 `gil_free_ratio` 중앙값 0.33 (mean 0.32, p10–p90 0.30–0.34)**. during 30.6M/s vs
무경합 95.2M/s → **forward 중 GIL이 ~68% 점유됨(=(가))**. 근거 3:
1. 디버그 로깅(dev 빌드 `RBLN_VERBOSE=debug` 기본값이 켜놓은 `[D] [runtime]` 스퓸, 로그의 80%)을
   `RBLN_VERBOSE=warning`으로 꺼도 ratio 동일 ~0.32 → 낮은 during_rate은 CPU 경합이 아니라 진짜 GIL 점유.
2. spin 스레드 붙이면 forward가 (F2의) 3.5ms→~65ms로 18배 폭증. GIL-free forward라면 안 느려짐.
   이 폭증 자체가 subgraph별 fine-grained Python dispatch(prepare_inputs/address-patch)가 GIL을
   잡았다 놨다 반복(switch-interval ping-pong)한다는 signature.
3. 4 rank 전부 일관.

**함의(=RESUME §2 (가) 브랜치)**: 단일 host 스레드로는 DP all_reduce를 forward 창의 **최대 ~⅓**에만
겹칠 수 있음 → **vLLM Python 레벨만으로는 GPU식 overlap 불가**. 남은 길 두 갈래(유저 결정 필요):
- **(A) 런타임 submission glue 경량화**: rebel_compiler에서 그래프 실행의 host-side dispatch/
  prepare_inputs/address-patch 비용을 줄여 GIL 점유 시간 자체를 축소. §3의 drain 계측과 함께,
  "3.5ms forward의 host time 분해"를 rebel 런타임 프로파일로 재현해야 함(어디서 GIL을 잡는지).
- **(B) threading 재검토**: 단일 스레드 제약(유저가 기각했던 방향)을 다시 열지 여부. GIL을 놓는 별도
  스레드에서 all_reduce를 돌리면 그 ~68% 구간에도 겹칠 수 있으나, torch profile GPU식 표시/정합성
  재작업 부담이 큼.

### STEP 1 재현 레시피 (~15분 첫 실행. **캐시가 decode-bucket JIT은 못 건너뜀** — 매 프로세스 재컴파일)
```bash
cd ~/codebase/vllm-executor && source .venv/bin/activate
export VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error VLLM_RBLN_AUTO_PORT=1 \
  RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1 \
  SPDLOG_LEVEL=warning RBLN_VERBOSE=warning VLLM_LOGGING_LEVEL=INFO RBLN_DEVICES=0,1,2,3
# RBLN_VERBOSE=warning 필수: venv _C.so는 dev 빌드라 기본 debug → [D] 런타임 스퓸이 로그 80% 차지.
# SPDLOG_LEVEL은 런타임 로거에 무관(레벨은 RBLN_VERBOSE/RBLN_VERBOSITY가 결정, flags.cc:613).
VLLM_RBLN_OPTIMISTIC_SCHED=1 VLLM_RBLN_GIL_PROBE=1 \
python3 -m vllm_rbln_exec.parity_runner --task r --model gpt-oss-120b --ep --dp 4 --rsd 1 \
  --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 --batch 1 \
  --num-hidden-layers 18 --max-num-blocks 129 --max-tokens 16 --num-prompts 4 2>&1 | tee /tmp/gil2.log
# 분석: forward_ms>1000(=컴파일 오염) 버리고, free_rate>=85M(=무경합 캘리브)만 골라 ratio 중앙값.
# ratio ~0.32 재확인되면 (가). (판정 스크립트는 세션 히스토리 참고: fwd<=1000 & free>=85e6 필터.)
```

## 3. (나)로 판명 시 — drain 지점과 다음 작업
`model_executable` 안의 host-sync(drain)는 `RuntimeInstance::Run`에서:
- 그래프 중간 host op 실행 전 (`runtime_instance.cc:788` `EnsureAllTasksCompleted`)
- const-buffer device op 전 (`:878`), multi-output copy 전 (`:1204`)
- Run() 자체는 끝에서 drain 안 함(`:1198`만 Record). `RBLN_RUNTIME_FORCE_SYNC=1`이면 매 op drain(`:1194`).
계측: 이 지점에 env-gated 카운터 추가(리빌드 필요 — `_C.so`는 venv build이므로 `~/codebase/rebel_compiler`
편집 후 재설치. `.py`는 editable로 즉시 반영). 목표: decode step당 실제 drain 횟수 = 0으로 만들면
forward 제출이 온전히 async가 되어 그 사이 all_reduce(N+1) 삽입 가능.
그 후 vllm-rbln에서 EM(N+1) prep(=all_reduce N+1)을 forward(N) 제출 직후로 파이프라인
(낙관적 스케줄러 batch_queue depth2 활용), D2H는 immediate AsyncOutput으로 지연(F3에서 race 없음 확인).

## 4. 정합성/overlap 검증 레시피
```bash
# sync baseline
rm -f ~/.cache/vllm-rbln-exec/rbln_results_*L18_T16*P4_*
VLLM_RBLN_DISABLE_ASYNC=1 python3 -m vllm_rbln_exec.parity_runner <위 STEP1 동일 인자>
cp ~/.cache/vllm-rbln-exec/rbln_results_*L18_T16*P4_*.json /tmp/base18.json
# 후보 모드로 다시 → token_ids 0 mismatch 기대 (파일 비교)
# overlap 정량(host span, 신뢰 소스):
VLLM_RBLN_SPAN_LOG=1 <후보 모드> 2>&1 | grep -aE "SPAN (fwd|allreduce)" > /tmp/spans.log
python3 docs/async_overlap_scripts/overlap_from_spans.py /tmp/spans.log
# Perfetto 눈 확인: span_to_perfetto.py /tmp/spans.log → ui.perfetto.dev
```
파일명 glob `*L18_T16*P4_*`는 실제 P16 파일과 `DP4_`로 매칭됨. parity `exit=1`은 baseline 없음이라 양성.

## 5. 환경/운영 주의
- **exclusive RBLN 박스 필요**: 동시 DP4/EP 잡 불가(`rcclCommInitRank ret=-12`). 실행 전
  `ps -eo user,args|grep VLLM::`로 남의 잡 0, `rbln-stat` free NPU≥4 확인.
- **정상 종료**: `kill -9`는 ASID 누수로 device wedge(`SYS_ENODEV`/`No free ASID`). 반드시
  `pkill -TERM -f vllm_rbln_exec.parity_runner` 후 `find /dev/shm -maxdepth 1 -user $USER -delete`.
  내 프로세스만. 죽였으면 `rbln-stat`로 killed context 확인.
- **full-layer(36층)**: 이 박스 KMD 3.3.0~rc2 + TDR 6s로 warmup timeout(`SYS_TASK_ABORTED`). async 무관,
  박스 환경 문제(CI udc-08 KMD 3.2.x는 PASS). 필요시 `sudo rbln-smi tdr/timeout --group 0 --value 600`.
  검증은 18층(이 박스 OK)으로 충분.
- `--cache-ignore`는 매번 재컴파일(~15분). 반복 실험엔 빼서 컴파일 캐시 재사용.

## 5.5 --profile overlap 트레이스 확인 (2026-07-02) — 현 상태 overlap 0% 재확정
`--profile`(torch profiler) 18층 decode 트레이스 분석(`docs/async_overlap_scripts/analyze_trace2.py`,
`distill_trace.py`). torch_rbln엔 kineto device 백엔드가 없어 **device 트랙은 없고 host/CPU 트랙만**
나옴(그래서 device-tail overlap은 이 도구로 안 보임 — host 직렬 여부만 확인 가능).
- 메인 스레드 steady-state(중앙값): `execute_model` ~7ms [맨 앞 all_reduce ~0.8ms → forward-walk ~6ms]
  → `sample_tokens` ~8.4ms. **overlap(all_reduce, forward/sample)=0.00ms — 완전 직렬.**
- gloo는 `pt_gloo_runloop` 별도 스레드에서 돌지만 메인 스레드가 all_reduce 결과를 blocking 대기.
  첫 step all_reduce 1717ms는 rank 간 startup barrier 대기(steady 아님).
- **근본 원인(왜 32% GIL 여유가 안 쓰이나)**: GIL 가용성이 아니라 **프로그램 순서**. all_reduce는
  execute_model 맨 앞(forward_context 셋업)에서 forward보다 먼저 실행되고, AR(N+1)은 sample_tokens(N)
  뒤에 옴 → forward의 GIL-free 창에 스케줄되는 게 없음. 그 창을 쓰려면 (a) forward host-walk를 얇게((A)) +
  (b) 스케줄러가 AR(N+1)을 그 창에 삽입. → (A)를 뒷받침.
- 증류 Perfetto: `docs/async_overlap_scripts/rbln_overlap_dp0_distilled.json`(ui.perfetto.dev 드롭).
- 운영 발견: decode-bucket 컴파일 캐시는 **재사용됨**(gil2 실행이 캐시 채운 뒤 --profile 실행은 컴파일
  ~236s로 단축). 첫 콜드 컴파일만 ~50분.

## 6. 현재 코드 상태 (branch `async-overlap-prototype`)
- **threading 코드 전부 제거됨(2026-07-02, 단일 스레드 (가) 확정 + 유저가 threading 완전 배제)**:
  `_DeviceForwardExecutor` 클래스, `_async_forward`/`_device_executor` 필드, execute_model의
  `_fast_defer`/`if self._async_forward` 분기, sample_tokens의 `_defer_sampler`(C9b)/`fwd_future`(C9a)
  블록, `_bookkeeping_async_fast`, `ExecuteModelState.fwd_future`, `AsyncRBLNModelRunnerOutput.sample_future`,
  rbln_worker의 `profile_all_threads`, 미사용 import(`queue`/`Future`/`dataclasses`) 모두 삭제.
  py_compile OK + 레포 전체 dangling ref 0. **런타임 parity 검증은 별도 실행으로 확인.**
- `vllm_rbln/forward_context.py`: DP all_reduce에 `VLLM_RBLN_SPAN_LOG` host span 로그. **유지(측정용)**.
- `vllm_rbln/v1/worker/rbln_model_runner.py`: `_run_forward`의 SPAN fwd 로그 + GIL_PROBE 블록 유지(측정용).
  실행 경로는 이제 inline forward + inline sampler 하나뿐(async-scheduling immediate AsyncOutput은 유지).
- 게이트 플래그(현재): `VLLM_RBLN_OPTIMISTIC_SCHED`(낙관 스케줄러 depth2), `VLLM_RBLN_DISABLE_ASYNC`
  (sync baseline), `VLLM_RBLN_SPAN_LOG`, `VLLM_RBLN_GIL_PROBE`. (`VLLM_RBLN_ASYNC_FORWARD` 제거됨.)
- **필수 env**: `RBLN_VERBOSE=warning`(dev _C.so 기본 debug 스퓸 차단).

## 7. 핵심 파일:라인
- forward + SPAN: `rbln_model_runner.py` `_run_forward()`(~3910), `execute_model`(~3509), `sample_tokens`(~4182).
- immediate AsyncOutput(단일 스레드 D2H 지연): `AsyncRBLNModelRunnerOutput`(270-326, `sample_future=None` 경로).
- DP all_reduce: `forward_context.py` `num_tokens_across_dp`.
- GIL 프로브: `rbln_model_runner.py:4005` (`VLLM_RBLN_GIL_PROBE`).
- rebel 드라이버 큐/drain: `~/codebase/rebel_compiler/rebel/src/runtime/core/runtime_instance.cc`
  `Run()`(1163), `EnsureAllTasksCompleted()`(1046), 제출 `sync_runtime.py:204-274`.
- torch_rbln: device-wide `synchronize()`만 있고 per-op event 없음(`device/device.py:87`).
