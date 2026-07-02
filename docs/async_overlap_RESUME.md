# async overlap — RESUME (다음 세션 단일 진입점)

> 2026-07-02 갱신 (STEP 1 판정 **정정됨**). 이 파일 하나로 다음 세션에서 바로 이어서 진행한다.
> **방향 확정: 단일 host 스레드. multi-threading(구 `_DeviceForwardExecutor`) 방식은 기각(코드 제거 완료).**
>
> **★★ STEP 1 정정 (중요): 이전의 "(가) forward가 GIL 68% 점유" 판정은 GIL_PROBE의 spin 스레드가
> forward를 18배 왜곡해 만든 아티팩트였다. 트레이스(무왜곡) 재분석 결과 실제는 (나):**
> **decode forward host-walk ~6ms 중 ~4.1ms가 `.run()`의 device-완료 대기(`EnsureAllTasksCompleted`,
> `runtime_instance.cc:1187/1197`)이고 이 구간은 GIL을 놓는다(`compiled_model.cc:50` `gil_scoped_release`).
> 즉 forward는 ~73% GIL-free(device drain). GIL-잡는 Python glue는 ~1.5ms(~25%)뿐.**
>
> **→ overlap이 안 되는 진짜 이유: async scheduling(이미 ON)이 문제가 아니라, RBLN forward `.run()`이
> worker 스레드를 4.1ms 블로킹해서 execute_model(N)이 forward 도중 리턴을 못 함 → execute_model(N+1)이
> forward(N)과 못 겹침. GPU는 forward가 async라 worker가 비어서 저절로 겹침.**
> **→ non-blocking forward는 조사 결과 "쉬운 flag skip" 불가(§3): `.run()` drain은 decode 그래프의
> mid-graph host-op/const-buffer가 강제하는 필수 sync라, 없애려면 컴파일러 레벨 그래프 재구성 필요
> = 정합성-안전 최소변경 아님 → 미구현.**
> **→ 실현 가능한 경로 = all_reduce 프리페치(§3 하단, `VLLM_RBLN_OVERLAP_PROBE` 실증): forward의
> GIL-free drain 창에 gloo가 all_reduce(N+1)을 돌림. forward_context만 수정, vLLM 구조 불변.**
>
> 측정 도구: `docs/async_overlap_scripts/`(overlap_from_spans, span_to_perfetto, selftime_breakdown,
> step_breakdown, analyze_profile_trace, distill_profile_trace, span_overlap_viz).
> GPU vs RBLN 방향성 Q&A: `docs/async_overlap_gpu_vs_rbln_QA.md` (단, 그 문서의 "host glue가 병목"
> 결론은 이 정정으로 폐기 — 병목은 blocking forward).
> 실증: `VLLM_RBLN_OVERLAP_PROBE=1`(forward_context.py) — all_reduce(N+1)을 async 프리페치해 forward(N)과
> 겹침을 증명(collect wait 0.004ms, all_reduce ~100% 숨김). 단 shadow(추가 collective)라 지연은 안 줄임.

## 0. 목표
gpt-oss-120b EP+DP4 decode에서 매 step 도는 DP `gloo:all_reduce`(N+1)를 NPU forward(N) 실행 구간에
**겹쳐** step latency를 줄인다. 정합성(parity) 유지. 그리고 그 overlap이 **torch profile 결과물에서
GPU처럼(후처리 없이) 보이게** 한다. **반드시 단일 스레드**로 — 별도 device 스레드 방식은 기각됨.

## 1. 지금까지의 핵심 발견 (측정 근거 포함)

### (F1) ⚠️ 정정됨 — RBLN 제출은 non-blocking이 아니다 (§2 참조)
[구 주장, 폐기] "임시 프로브에서 `DynamoRuntime.run()` 중앙값 0.01ms → run()은 즉시 리턴, run_async 불필요."
→ **틀림.** 무왜곡 트레이스 self-time에서 `<built-in method run of PyCapsule>`(= `_runtime_handle.run()`,
`sync_runtime.py:268`)는 **~4.1ms/step**이고, 이는 `RuntimeInstance::Run`의 `EnsureAllTasksCompleted()`
device drain(`runtime_instance.cc:1187/1197`) 때문이다. 구 프로브가 µs를 본 건 측정 범위 오류로 추정.
**즉 forward는 blocking이고, 이걸 non-blocking으로 만드는 게 핵심 작업(§2).** (`rblnSubmitJob` 자체는 async지만
Run()이 출력 복사 전에 drain해서 결과적으로 blocking.)

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

## 2. ★ STEP 1 판정 정정 (2026-07-02) — **(나) 확정**: forward는 GIL-free device drain

**이전 판정 (가)는 폐기.** GIL_PROBE로 `gil_free_ratio≈0.32`(GIL 68% 점유)를 얻었으나, 이는 프로브의
spin 스레드가 forward를 3.5ms→~65ms(18배)로 왜곡한 **측정 아티팩트**였다. 무왜곡 트레이스(--profile)를
op별 self-time으로 재분석한 결과가 진실:

**decode forward host-walk ~5.99ms/step 분해 (baseline 트레이스, `selftime_breakdown.py`/`fwd_breakdown`):**
| 구간 | 시간 | GIL |
|---|---|---|
| `.run()` — NPU forward 완료 대기(`EnsureAllTasksCompleted`) | ~4.1ms (64%) | **놓음** (`compiled_model.cc:50` gil_scoped_release) |
| prepare_inputs/outputs (address-patch, CS DMA) | ~0.6ms (9%) | **놓음** (:80/:82) |
| torch.compile guards + aten + vllm-rbln python glue | ~1.5ms (25%) | 잡음 |

→ **forward는 ~73% GIL-free** (실제 gil_free_ratio≈0.73). `.run()`의 4.1ms는 host가 GIL 놓고 NPU가
forward 도는 걸 기다리는 **device drain**(`runtime_instance.cc:1187 force_sync / :1197 multi-output copy 전`).
= 원래 RESUME이 (나)로 적어둔 그 케이스.

**overlap이 안 되는 진짜 원인 (async scheduling 무관)**: async scheduling은 이미 ON이고 정상. 문제는
**RBLN forward `.run()`이 worker 스레드를 4.1ms 블로킹**해서 `execute_model(N)`이 forward 도중 리턴을
못 하는 것. 그래서 `execute_model(N+1)`(그 안의 `_prepare_inputs`→`get_dp_padding`→DP all_reduce)이
forward(N) 도중 시작을 못 함 → 직렬. GPU는 forward가 async(커널 launch 후 즉시 리턴)라 worker가 비어서
기존 async scheduling이 execute_model(N+1) prepare를 forward(N)과 저절로 겹침.

**해야 할 일 (기존 vLLM 틀 유지)**: RBLN forward를 **non-blocking**으로 — `.run()`의 output drain/copy를
뒤로 미뤄 device-tensor 핸들만 리턴하고, 실제 device sync는 소비 시점(`sample_tokens`)으로 이동. 그러면
worker가 forward 도중 비어서 vLLM async scheduling이 execute_model(N+1)을 forward(N)과 겹친다.
(콜 체인: `execute_model`:3337 → `_prepare_inputs`:1247 → `get_dp_padding`:1916 → `num_tokens_across_dp`
→ `dist.all_reduce` `forward_context.py:100`. 전부 worker.)

**실증 완료**: `VLLM_RBLN_OVERLAP_PROBE=1`(forward_context.py)로 all_reduce(N+1)을 async 프리페치 →
forward(N)의 GIL-free device 대기 동안 gloo runloop이 완료 → collect wait 0.004ms(vs blocking 0.6~2.4ms),
all_reduce ~100% 숨김, 데드락 없음. 단 이건 shadow(추가 collective)라 지연 자체는 안 줄임(가능성 증명용).
진짜 해법은 위 "non-blocking forward".

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
# [주의] GIL_PROBE ratio는 spin 스레드가 forward를 왜곡하므로 신뢰 불가(§2 정정). 대신 --profile
# 트레이스 self-time으로 forward 분해(selftime_breakdown.py / fwd_breakdown)가 신뢰 소스.
```

## 3. ★ non-blocking forward 실현가능성 조사 결과 (2026-07-02) — "쉬운 flag skip"은 불가

`.run()`의 ~4.1ms drain(`EnsureAllTasksCompleted`→`rblnWaitJob`, GIL 놓음)이 어디서 나는지 조사:
- **`:1197` multi-output copy 전 drain은 발동 안 함** (debug 로그에 "Copy slot to output_idx" 0건).
  device-tensor 모드에선 이 drain이 헛수고라 스킵 가능했겠지만, 애초에 안 탐 → 쉬운 fix 배제.
- **decode 그래프에 host op이 존재** (로그 "host ops compilation takes N ms" ×4 rank) + **const-buffer
  op이 step당 ~2개**(debug "const buffer" 128건 = 2×64step). → `:781`(host op 실행 전)과 `:871`
  (const-buffer op 전)의 `EnsureAllTasksCompleted`가 **의미상 필수 drain**. host op은 device 출력을
  읽어야 하므로 그 전에 device를 비워야 함 = 그냥 스킵하면 정합성 깨짐.
- 즉 4.1ms drain의 정체는 **그래프 중간 host-op/const-buffer가 강제하는 mid-graph device sync**.
  이걸 없애려면 **컴파일러 레벨에서 decode 그래프의 host op을 밖으로 빼는 재구성**이 필요 —
  런타임 flag로 스킵 불가, 정합성 보장 안 됨. **→ 최소·정합성-안전 변경이 아니므로 (유저 조건상) 미구현.**
- (device 순서 자체는 stream FIFO(seq chaining, `:1169/:1191`)로 보장돼 non-blocking이 *원리상*
  안전하나, 위 host-op drain 때문에 실제로 forward가 async가 안 됨. rebel `PyRblnAsyncRuntime`
  (`runtime.cc:323`, 스레드풀)은 존재하나 Dynamo 경로가 안 씀.)
- **런타임 trace 확정은 exclusive 박스 필요**(현재 yw.kim DP4 점유로 대기). trace로 `Wait job success.
  seq=` 발생 지점/횟수 확인 시 위 추정 검증됨.

### → 실제로 가능한 overlap 경로 = all_reduce 프리페치 (§2 실증, `VLLM_RBLN_OVERLAP_PROBE`)
forward를 non-blocking으로 못 만들어도, forward의 `.run()`이 drain 동안 **GIL을 놓으므로** 별도 스레드
(gloo `pt_gloo_runloop`)가 그 창을 쓸 수 있음. all_reduce(N+1)을 forward(N) 전에 async로 발사하면
gloo가 drain 창에 완료 → 실증됨(collect 0.004ms, all_reduce ~100% 숨김). **이게 RBLN에서 mid-graph
drain을 못 없애는 한 유일하게 실현 가능한 방향.** 프로덕션화 과제: 실제 all_reduce를 프리페치로 이동
(현재는 shadow) + decode 상수 가정의 prefill/ragged 전이 안전 처리(scheduler depth-2 lookahead 또는
전이 시 blocking fallback). vllm-rbln 내부(forward_context)만 수정, vLLM 구조 불변.

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
