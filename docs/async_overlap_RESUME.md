# async overlap — RESUME (다음 세션 단일 진입점)

> 2026-07-02 갱신. 이 파일 하나로 다음 세션에서 바로 이어서 진행한다.
> **방향 확정: 단일 host 스레드로 GPU처럼 overlap을 낸다. multi-threading(구 `_DeviceForwardExecutor`) 방식은 기각.**
> 측정 도구: `docs/async_overlap_scripts/{overlap_from_spans.py, span_to_perfetto.py}`.

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

## 2. ★ 미해결 — 다음 세션 첫 작업: 3.5ms forward host 시간의 정체 판정

단일 스레드 overlap 가능 여부는 (F2)의 3.5ms가 무엇이냐에 달림:
- **(가) GIL 잡는 Python glue** → 그 시간 동안 같은 스레드의 all_reduce가 못 돎 → **단일 스레드 overlap
  원천 불가**. 유일한 길은 런타임/컴파일 레벨에서 그 glue를 줄이는 것(prepare_inputs/address-patch 경량화).
- **(나) device drain 대기(GIL 놓음)** → 그래프 중간 host op이 device 결과를 기다리며 blocking →
  그 drain 지점을 없애거나 뒤로 미루면 **단일 스레드로도 all_reduce(N+1)를 그 사이에 낼 수 있음**.

### 판정 방법 (STEP 1, ~15분: 컴파일 + 생성)
```bash
cd ~/codebase/vllm-executor && source .venv/bin/activate
export VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error VLLM_RBLN_AUTO_PORT=1 \
  RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1 \
  SPDLOG_LEVEL=warning VLLM_LOGGING_LEVEL=INFO RBLN_DEVICES=0,1,2,3
# 캐시 재사용 위해 --cache-ignore 빼기(첫 컴파일만 오래 걸리고 이후 즉시). GIL_PROBE는 inline 경로에서 동작.
VLLM_RBLN_OPTIMISTIC_SCHED=1 VLLM_RBLN_GIL_PROBE=1 \
python3 -m vllm_rbln_exec.parity_runner --task r --model gpt-oss-120b --ep --dp 4 --rsd 1 \
  --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 --batch 1 \
  --num-hidden-layers 18 --max-num-blocks 129 --max-tokens 16 --num-prompts 4 2>&1 | tee /tmp/gil.log
grep "GIL_PROBE" /tmp/gil.log | tail -20   # gil_free_ratio 확인
```
`GIL_PROBE forward_ms=.. gil_free_ratio=X` 로그(rbln_model_runner.py:4005 블록):
- **ratio ~1.0 (GIL free)** → (나). forward가 device drain을 기다리며 GIL 놓는다 →
  그 drain 지점을 찾아(아래 §3) 제거/지연하면 단일 스레드 overlap 가능. STEP 2로.
- **ratio ~0 (GIL held)** → (가). Python glue가 host를 잡는다 → 단일 스레드 불가.
  결론: vllm 레벨로는 못 풀고 rebel_compiler 런타임에서 submission glue 경량화가 필요.
  (이 경우 유저와 방향 재논의: 런타임 최적화 vs threading 재검토.)

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

## 6. 현재 코드 상태 (branch `async-overlap-prototype`, 워킹트리 uncommitted)
- `vllm_rbln/forward_context.py`: DP all_reduce에 `VLLM_RBLN_SPAN_LOG` host span 로그(`num_tokens_across_dp`). **유지(측정용)**.
- `vllm_rbln/v1/worker/rbln_model_runner.py`: `_run_forward`의 SPAN fwd 로그. c9 record_function 마커는 제거함.
  `_DeviceForwardExecutor`/`_fast_defer`/`_sample_task` 등 **구 threading 코드가 아직 남아있음** —
  (나)로 단일 스레드 확정되면 정리 대상.
- `vllm_rbln/v1/worker/rbln_worker.py`: `profile_all_threads=True`. 단일 스레드론 불필요하나 무해.
- 게이트 플래그: `VLLM_RBLN_OPTIMISTIC_SCHED`(낙관 스케줄러 batch_queue depth2, overlap 전제),
  `VLLM_RBLN_ASYNC_FORWARD`(구 threading — 쓰지 말 것), `VLLM_RBLN_DISABLE_ASYNC`(sync baseline),
  `VLLM_RBLN_SPAN_LOG`, `VLLM_RBLN_GIL_PROBE`.

## 7. 핵심 파일:라인
- forward + SPAN: `rbln_model_runner.py` `_run_forward()`(~3910), `execute_model`(~3509), `sample_tokens`(~4182).
- immediate AsyncOutput(단일 스레드 D2H 지연): `AsyncRBLNModelRunnerOutput`(270-326, `sample_future=None` 경로).
- DP all_reduce: `forward_context.py` `num_tokens_across_dp`.
- GIL 프로브: `rbln_model_runner.py:4005` (`VLLM_RBLN_GIL_PROBE`).
- rebel 드라이버 큐/drain: `~/codebase/rebel_compiler/rebel/src/runtime/core/runtime_instance.cc`
  `Run()`(1163), `EnsureAllTasksCompleted()`(1046), 제출 `sync_runtime.py:204-274`.
- torch_rbln: device-wide `synchronize()`만 있고 per-op event 없음(`device/device.py:87`).
