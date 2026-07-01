# RBLN device-tensor 경로에서 DP `gloo:all_reduce` 를 forward 와 겹치기(async scheduling) — 분석·수정 보고서

- 작성 2026-06-30 · 최종 갱신 2026-07-01
- 대상: gpt-oss-120b, EP + DP4, device-tensor 경로 (`VLLM_RBLN_USE_DEVICE_TENSOR=1`). CI build #34 재현 환경.
- 리포지토리: `~/codebase/vllm-rbln` (branch `dev`), 런타임 `~/codebase/rebel_compiler`, 실행 환경 `~/codebase/vllm-executor/.venv`.
- 이 문서는 **자기완결적**이다. 사전 맥락 없이 이 파일만 읽고 이어서 작업할 수 있도록 배경·근본 원인·수정·실측·재현법·남은 과제를 모두 담았다.

---

## 0. TL;DR

- **목표**: 매 decode step 마다 도는 DP 좌표용 `gloo:all_reduce`(호스트 CPU collective, ~2.5ms/step)를 NPU `forward` 뒤에 숨겨(overlap) step latency 를 줄인다.
- **한 일 & 결과**:
  1. **낙관적 스케줄러(`RBLNAsyncScheduler`)를 wiring** 했다. 이건 vLLM 파이프라이닝(batch_queue depth 2)의 전제조건이다.
  2. 켜니 **출력이 깨졌다**. 원인은 **독립된 버그 2개**로 규명·수정했고 → **6-layer A/B 에서 sync baseline 과 0 mismatch(완전 동일) 달성 ✅.** (정합성 문제는 완결.)
  3. 정합성을 고친 뒤 처음으로 깨끗한 트레이스를 떠서 실측하니 — **all_reduce ↔ forward overlap 은 여전히 0%.**
- **overlap 이 0% 인 진짜 이유 (핵심)**: 스케줄러가 아니라 **런타임**이다. RBLN forward(`DynamoRuntime.run`)가 **호스트 워커 스레드를 ~5ms 동기 블록**한다. 호스트가 forward(N) 안에서 막혀 있으니 all_reduce(N+1) 을 forward(N) *도중* 에 발행할 수 없다 → 무조건 직렬. 낙관적 스케줄러는 필요조건이었을 뿐 **충분조건이 아니었다.**
- **torch stream/Event 로는 못 고친다**: `torch.rbln` 은 Stream 백엔드가 없고(`torch.Stream(device="rbln")` → *Backend doesn't support*), 무엇보다 블록이 C++ `run()` *안*, torch 레이어 *아래* 에서 일어난다. stream 은 비동기 호출을 정렬할 뿐 동기 호출을 논블로킹으로 바꾸지 못한다. (§6)
- **실제 overlap 을 내는 길**: (a) rebel 런타임의 비동기 submission 경로(`non_blocking_mode`/`AsyncRuntime`) 를 device-tensor 에 붙이거나, (b) blocking forward 를 별도 호스트 스레드에서 실행. 둘 다 vllm-rbln 스케줄러 밖의 작업. (§7)
- **현재 코드 상태**: 낙관적 경로는 `VLLM_RBLN_OPTIMISTIC_SCHED=1` 로 gating(기본 off). 기본 동작(plain `RBLNScheduler`)은 영향 없음. 정합성은 완결, overlap 은 런타임 비동기화가 남은 과제. (§8)

---

## 1. 무엇을 겹치려는가 — `gloo:all_reduce` 의 정체

torch 프로파일/스택에서 매 step 반복되는 `torch/distributed/distributed_c10d.py:all_reduce` 는 **GPU NCCL 이 아니라 RBLN 의 DP 좌표용 호스트 CPU gloo collective** 다.

- 위치: `vllm_rbln/forward_context.py:84` — `RBLNDPMetadata.num_tokens_across_dp()` 안의
  `dist.all_reduce(num_tokens_tensor, group=get_dp_group().cpu_group)`.
- 데이터 흐름: 각 DP rank 가 자기 `(is_prefill, num_reqs, num_tokens)` 를 **int32 하나로 bit-pack**(`num_tokens_and_reqs_across_dp`, `forward_context.py:87-148`) → CPU gloo all_reduce → 모든 rank 가 per-rank 벡터를 알게 됨 → MAX 로 공통 `(batch, query_len)` 결정.
- **왜 필요한가**: MoE + EP 는 모든 DP rank 가 **같은 입력 shape** 로 step 해야 한다. expert dispatch 의 cross-rank all-to-all/all-reduce 가 shape 불일치 시 hang 또는 hot-path recompile. 그래서 forward 직전에 shape 합의가 필요.
- **핵심(overlap 가능성의 근거)**: all_reduce 가 합의하는 값(`num_tokens`/`num_reqs`)은 **스케줄 결정**에서 나오지 forward 결과에서 나오지 않는다. 입력도 순수 host int(스칼라 packing)라 NPU→host readback 이 아니다. 따라서 "all_reduce 가 forward 결과를 기다린다"는 데이터 의존은 **없다** → 원리상 forward(N) 과 all_reduce(N+1) 은 겹칠 수 있다.

---

## 2. GPU 와 RBLN 의 근본 차이

| | GPU (NCCL) | RBLN (gloo) |
|---|---|---|
| collective 위치 | CUDA stream 위 device collective | **호스트 CPU** gloo (별도 `pt_gloo_runloop` 스레드) |
| forward submission | **비동기**: `model(x)` 가 커널을 stream 에 enqueue 하고 즉시 리턴 | **동기**: `DynamoRuntime.run` 이 호스트 스레드를 ~5ms 블록 (§5 실측) |
| overlap 메커니즘 | CUDA 기본 비동기 + async scheduler 파이프라이닝 | (이론) 스케줄러 파이프라이닝. (실제) 런타임이 동기라 막힘 |

**중요**: GPU 에서 forward(N)·all_reduce(N+1) 이 겹치는 건 stream/Event 를 직접 조작해서가 아니라 **CUDA submission 이 기본적으로 비동기**이기 때문이다. 호스트가 앞서 달려나가며 작업을 큐에 넣고 device 가 순서대로 소비한다. vLLM async scheduling 은 이 성질 위에 올라탄 것이다. RBLN 에는 이 성질이 없다는 게 이 보고서의 결론(§5)이다.

---

## 3. vLLM async scheduling 이 하려는 것 (배경)

- `async_scheduling=True` + pp1 → `MultiprocExecutor.max_concurrent_batches = 2`.
- `step_with_batch_queue` 가 batch_queue 를 깊이 2 로 운용: step(N) 의 출력을 보기 **전에** step(N+1) 을 schedule + execute 제출.
- 이게 되려면 스케줄러가 step(N) 결과(샘플 토큰 수, 종료 여부)를 **낙관적으로 가정**(`num_output_placeholders`)하고 미리 진행해야 한다 — vLLM `AsyncScheduler` 의 역할.
- 기존 `RBLNScheduler` 는 이 낙관적 동작이 **없어서** batch_queue 가 깊이 1 로만 돌았다(사실상 직렬). 게다가 platform.py 가 device-tensor 에서 async 를 강제로 끄고 있어 낙관적 스케줄러가 한 번도 wiring 된 적이 없었다. → 그래서 `AsyncScheduler` 믹스인이 필요했다(§4, §8).

---

## 4. 정합성이 깨진 근본 원인과 수정 (완결 ✅)

낙관적 스케줄러(`RBLNAsyncScheduler`)를 켜자 batch_queue 는 깊이 2 로 찼지만 출력이 퇴화했다:

```
sync  : ' blind topical legit vac banner pl rec Cour diplom heads projectile ...'
async : ' blind Sorry opt mix mix mix mix mix mix mix mix mix mix mix mix mix'   ← 16/16 mismatch
```

패턴(token[0] 정확 → token[1]부터 발산/반복)은 **이전 step 의 샘플 토큰이 다음 step 입력에 안 들어감**을 뜻한다. GPU 참조 구현(`gpu_model_runner.py`)과 한 줄씩 대조해 **독립된 버그 2개**를 찾았다.

### 4-1. 버그 ① — 매-step 리셋 부재 (freeze)
async 토큰 피드백은 `_bookkeeping_sync` 의 guard 로 캐시된다:
```python
if self.input_batch.prev_sampled_token_ids is None:      # rbln_model_runner.py:2970
    self.input_batch.prev_sampled_token_ids = sampled_token_ids
```
GPU 는 매 step `sample_tokens` 에서 **bookkeeping 직전에** `prev_sampled_token_ids = None`(`gpu_model_runner.py:4384`) 으로 리셋해 이 guard 가 매번 이번 step 토큰을 저장하게 한다. RBLN 에는 이 리셋이 **없어서** guard 가 **step 0 에서 딱 한 번만** 발동하고, 이후 `prev_sampled_token_ids` 가 step 0 토큰에 **얼어붙었다**.

**수정**: `sample_tokens` 의 `_bookkeeping_sync` 호출 직전에
```python
if self.use_async_scheduling:
    self.input_batch.prev_sampled_token_ids = None
```

### 4-2. 버그 ② — 버퍼 불일치: scatter 는 `.gpu`, 모델은 `.cpu` 소비 (진짜 블로커)
리셋만으로는 여전히 16/16 mismatch 였다. device-tensor 특유의 버퍼 경로가 원인이었다.

- `_prepare_input_ids` 의 async 분기는 이전 토큰을 **`self.input_ids.gpu`** 에 scatter 한다(`rbln_model_runner.py:1116/1131/1156`).
- 그런데 device-tensor 경로는 모델 입력을 **`self.input_ids.cpu`** 에서 만든다:
  ```python
  ids_buf = self.input_ids.cpu if use_dt else self.input_ids.gpu   # rbln_model_runner.py:2007
  input_ids = ids_buf[:num_input_tokens]                            # 이후 :3469 에서 .to(device)
  ```
- `input_ids.cpu` 의 decode 위치에는 **낙관적 스케줄러가 넣은 placeholder 토큰**이 채워져 있고(`token_ids_cpu` 에서 index_select, `:1273`), `.gpu` 로의 scatter 는 이 경로에서 **한 번도 읽히지 않는다.**
- 결론: **이전 step 토큰 피드백이 모델에 전혀 도달하지 못하고**, 모델은 매 decode step placeholder 로 forward → 발산/퇴화.

**수정**: `execute_model` 에서 `input_ids` 를 device 로 옮긴 직후(`rbln_model_runner.py:3469` 이후), decode 경로에서 이전 step 샘플 토큰을 **device-side 로** 실제 소비되는 텐서에 재적용:
```python
if (envs.VLLM_RBLN_USE_DEVICE_TENSOR and self.use_async_scheduling
        and input_ids is not None and not is_prefill_phase
        and self.input_batch.prev_sampled_token_ids is not None):
    prev = self.input_batch.prev_sampled_token_ids
    prev_map = self.input_batch.prev_req_id_to_index
    req_ids = self.input_batch.req_ids
    if prev_map is not None and all(r in prev_map for r in req_ids):
        perm = torch.tensor([prev_map[r] for r in req_ids], dtype=torch.long,
                            device=input_ids.device)
        n = len(req_ids)
        input_ids[:n, 0] = prev[perm, 0].to(input_ids.dtype)
```
`perm` 은 `prev_req_id_to_index`(직전 배치의 req_id→index)로 현재 배치 순서를 직전 배치 인덱스에 매핑한다. device→host sync 가 없다. **decode-only fast path** 이며 prefill/mixed/new-req 배치는 fall back(placeholder 유지) — §8 한계 참조.

### 4-3. 검증
6-layer A/B (gpt-oss-120b EP/DP4, `--max-tokens 16 --num-prompts 4`), `RBLNAsyncScheduler` 활성 로그 확인 후:
```
fix=16 sync=16 mismatches=0
RESULT: IDENTICAL ✅
```
즉 낙관적(overlap-capable) 스케줄러 출력이 sync baseline 과 **완전히 동일**. 위 두 버그가 정합성 깨짐의 전부였다.

---

## 5. overlap 실측 — 0%, 진짜 블로커는 런타임의 동기 forward (핵심)

정합성을 고친 뒤 처음으로 낙관적 경로의 깨끗한 full-layer torch.profile 을 떴다. `~/profile_overlap/{async_opt,sync}/*/*.pt.trace.json.gz` (DP rank 0~3).

**트레이스 구조 (dp0)**:
- `gloo:all_reduce` → 별도 스레드 `pt_gloo_runloop` 에서 실제 collective 수행.
- `torch/.../distributed_c10d.py all_reduce` → 메인 워커 스레드가 gloo 스레드를 **기다리는** 호스트 호출.
- `rebel/sync_run...`(= `sync_runtime.py` 의 device forward) → **메인 워커 스레드**.

**시간순 타임라인 (steady-state decode, 메인 스레드)**:
```
DEVICE_forward(sync_run)   5.1ms   ← forward 가 호스트 스레드를 블록
DEVICE_forward(sync_run)   0.8ms   ← 작은 graph(logits/sampler)
c10d.all_reduce            2.5ms   ← 이 대기 동안에만 pt_gloo_runloop 이 동시 실행
DEVICE_forward(sync_run)   5.2ms   ← 다음 step forward
...
```

**정량 (dp0)**:

| | optimistic async | sync |
|---|---|---|
| gloo:all_reduce ∩ device forward | **0.0ms (0%)** | **0.0ms (0%)** |
| step cadence(중앙값, c10d start-to-start) | ~20.2ms | ~24.0ms |

**결론**: gloo runloop 은 메인 스레드의 `c10d.all_reduce` 대기 구간에만 겹치고 **device forward 와는 절대 안 겹친다.** 이유는 `DynamoRuntime.run`(`rebel/python/rebel/sync_runtime.py:204`)이 `self._runtime_handle.run()`(`:268`)에서 **호스트 스레드를 ~5ms 동기 블록**하기 때문. (출력은 device 텐서로 할당(`:258`)되므로 D2H 복사가 아니라 submit/compute 자체가 블록.) forward(N) 이 호스트를 막으니 스케줄러가 아무리 다음 배치를 미리 잡아도 all_reduce(N+1) 은 forward(N) 이 끝난 뒤에야 발행된다.

즉 **overlap 을 막는 근본 원인은 비낙관적 스케줄러(1차 가설)가 아니라, 런타임의 동기(호스트-블로킹) device forward** 다. 스케줄러 수정은 파이프라이닝의 전제였을 뿐 실제 overlap 을 만들지 못한다.

> 참고(폐기된 초기 주장): 이전에 readback probe 로 "forward 는 async 제출"이라 판단했으나, 정합성 수정 후의 트레이스가 이를 뒤집는다 — 메인 스레드는 forward 호출 안에서 ~forward 시간만큼 실제로 멈춰 있다. cadence 가 async 에서 다소 빠른 건(24→20ms) all_reduce 은닉이 아니라 호스트측 스케줄링 오버헤드 감소로 보이며, all_reduce 이벤트 수가 달라 직접 비교는 주의.

---

## 6. "torch stream / torch.Event 로 하면 되지 않나?" — 안 된다

GPU vLLM 이 stream/Event 를 쓰니 RBLN 도 torch 레벨에서 같게 하면 되지 않냐는 질문. **순수 torch stream/Event 만으로는 불가능하다.**

1. **GPU 의 overlap 은 stream/Event 덕이 아니다** — CUDA submission 이 기본 비동기라 `model(x)` 가 즉시 리턴하는 덕이다(§2). stream/Event 는 *이미 비동기인* 작업의 순서/의존성을 맞추는 도구일 뿐(GPU 러너에서 명시적 사용처는 `_copy_valid_sampled_token_count` 의 D2H 복사 오버랩 정도).
2. **`torch.rbln` 에는 Stream 백엔드가 없다** (실측):
   - `torch.Stream(device="rbln")` → `RuntimeError: Backend doesn't support create a new Stream`.
   - `torch.rbln.Stream / current_stream / set_stream` 부재 (`synchronize` 만 존재).
   - `torch.Event(device="rbln")` 는 객체는 생기나 뒤에 stream 머신이 없어 no-op 껍데기.
3. **블록이 torch 레이어 아래에 있다** — forward 블록은 C++ `_runtime_handle.run()`(`sync_runtime.py:268`) *안* 에서 발생. **stream/Event 로 감싼다고 동기 호출이 비동기가 되지 않는다.** stream 은 비동기 호출을 정렬할 뿐 동기 호출을 논블로킹으로 바꾸지 못한다.

핵심: **"forward 가 비동기냐" 는 백엔드 submission 경로의 성질**이지 torch 추상화로 추가할 수 있는 게 아니다.

---

## 7. 실제로 overlap 을 내는 길 (남은 과제)

- **(a) rebel 런타임의 비동기 submission 경로** — `rebel/python/rebel/compiled_model.py` 에 `non_blocking_mode` 인자와 `create_async_runtime`, `async_runtime.py` 의 `async_run`, `rds.py` 의 `Stream` 클래스가 **이미 존재**한다. 이게 CUDA async 제출에 해당하는 RBLN 판. **관건**: device-tensor / Dynamo(torch.compile backend) 경로에 이 non-blocking 경로를 붙일 수 있는가. (과거 "AsyncRuntime 는 precompiled `.rbln` 전용, device-tensor 엔 못 씀" 이라 판단했으나, `non_blocking_mode` 인자 존재를 근거로 재검토 필요.) — **rebel 쪽 변경.**
- **(b) blocking forward 를 별도 호스트 스레드에서 실행** — 파이썬 스레드로 forward 를 돌리고 메인 스레드가 all_reduce 를 동시 발행. `run()`(pybind C++)이 GIL 을 놓아준다면 rebel 을 안 건드리고 vllm-rbln executor 레벨에서 가능. 단 stream 이 아니라 커스텀 executor 변경이며, 토큰 피드백/DP step 순서 의존성 관리가 필요. — **먼저 `run()` 이 GIL 을 놓는지 확인**이 값싼 첫 단계.

즉 남은 작업은 **torch stream/Event 가 아니라 (a) 런타임 비동기화 또는 (b) 스레딩** 이다.

---

## 8. 현재 코드 상태 · 변경 내역 · 정책

### 변경 파일 (`~/codebase/vllm-rbln`, branch `dev`, HEAD 대비)
- `vllm_rbln/v1/core/rbln_scheduler.py`
  - `from vllm.v1.core.sched.async_scheduler import AsyncScheduler` 추가.
  - `class RBLNAsyncScheduler(RBLNScheduler, AsyncScheduler)` 추가(docstring-only). MRO: `schedule`→`RBLNScheduler`(query backfill 등 RBLN 로직 유지), `_update_after_schedule`/`_update_request_with_output`→`AsyncScheduler`(낙관적 placeholder), `update_from_output`→`RBLNScheduler`.
- `vllm_rbln/platform.py`
  - device-tensor 의 async 강제 비활성화 제거(async 자동 enable).
  - 스케줄러 선택: `async_scheduling AND VLLM_RBLN_OPTIMISTIC_SCHED==1` 일 때만 `RBLNAsyncScheduler`, 그 외 `RBLNScheduler`. **기본 off.**
  - A/B 용 `VLLM_RBLN_DISABLE_ASYNC=1` 토글(async 끄고 sync).
- `vllm_rbln/v1/worker/rbln_model_runner.py`
  - `AsyncRBLNModelRunnerOutput`(`device_tensor_rebased_async` branch 유래): `get_output()` 이 `torch.rbln.synchronize(device_index)` 후 `.tolist()`.
  - **정합성 수정 1 (§4-1)**: `sample_tokens` bookkeeping 직전 `prev_sampled_token_ids = None` 리셋.
  - **정합성 수정 2 (§4-2)**: `execute_model` 의 `input_ids .to(device)` 직후 device-side 토큰 피드백.

### env 게이트

| env | 효과 | 정확성 | overlap |
|---|---|---|---|
| (기본) | async + plain `RBLNScheduler` | ✅ | ❌ |
| `VLLM_RBLN_OPTIMISTIC_SCHED=1` | async + `RBLNAsyncScheduler` | ✅ (0 mismatch, §4) | ❌ (0%, §5 — 런타임 블로커) |
| `VLLM_RBLN_DISABLE_ASYNC=1` | sync | ✅ | ❌ |

### 정책 / 한계
- 낙관적 경로는 gating 유지(기본 off): 정합성은 확인됐으나 overlap 이득이 아직 없고(§5), device-side 피드백이 **decode-only fast path** 라 mixed prefill+decode·spec decode 는 fall back 한다. 그런 워크로드를 켜기 전 해당 케이스의 피드백/`_prepare_input_ids` draft scatter 대응 필요.

---

## 9. 재현 방법

환경: `~/codebase/vllm-executor/.venv`. **device 점유 주의** — `rbln-stat` 으로 빈 NPU 확인 후 `RBLN_DEVICES` 지정(예: 다른 사용자가 0~3 쓰면 `RBLN_DEVICES=4,5,6,7`). 죽인 런의 좀비 `VLLM::EngineCore/Worker` 프로세스가 device 를 붙들 수 있으니 `kill -9` (프로세스 타이틀이 `VLLM::...` 라 `pkill -f` 로는 안 잡힘).

공통 env:
```bash
export VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error VLLM_RBLN_AUTO_PORT=1
export RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1
export RBLN_VERBOSE=2 SPDLOG_LEVEL=warning VLLM_LOGGING_LEVEL=INFO
export RBLN_DEVICES=4,5,6,7            # 빈 device 로
```

**정합성 A/B (6-layer, 빠름)**: 낙관적 경로에 `VLLM_RBLN_OPTIMISTIC_SCHED=1` 추가, sync baseline 에 `VLLM_RBLN_DISABLE_ASYNC=1`.
```bash
python3 -m vllm_rbln_exec.parity_runner --task r --model gpt-oss-120b --ep --dp 4 --rsd 1 \
  --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 --batch 1 \
  --num-hidden-layers 6 --max-num-blocks 129 --cache-results --cache-ignore \
  --max-tokens 16 --num-prompts 4
# 결과 json(캐시): ~/.cache/vllm-rbln-exec/rbln_results*L6_T16*Hb9304337*.json — 두 경로 text/token_ids 비교
```

**overlap 프로파일 (full-layer)**: 위 명령에서 `--num-hidden-layers` 빼고 `--profile` 추가. 트레이스는 `--profile` 이 켠 디렉터리(`profile/rbln_*_dp{r}/*.pt.trace.json.gz`)에 생성.
- 참고: vLLM 0.22 는 `VLLM_TORCH_PROFILER_DIR` 만으로 안 켜진다. `parity_runner.py` (~line 585) 에서 `ProfilerConfig` 를 주입하는 패치가 이미 들어가 있다.

**트레이스 overlap 정량** (perfetto 없이):
```python
import gzip, json, glob
def load(p):
    with gzip.open(p,'rt') as f: d=json.load(f)
    return d.get("traceEvents", d)
def iv(ev, sub):  # ph=="X", name 에 sub 포함
    return sorted((e["ts"], e["ts"]+e["dur"]) for e in ev
                  if e.get("ph")=="X" and e.get("dur",0)>0 and sub in e.get("name",""))
# gloo = iv(ev, "gloo:all_reduce"); dev = iv(ev, "rebel/sync_run")
# 두 인터벌 리스트의 교집합 시간 계산 → 현재 0
```
perfetto UI(https://ui.perfetto.dev)에 `.pt.trace.json.gz` 를 그대로 드롭. `pt_gloo_runloop` 스레드(all_reduce)와 메인 스레드의 `rebel/sync_run`(forward)를 비교.

---

## 부록 A — 핵심 file:line
- DP gloo all_reduce: `vllm_rbln/forward_context.py:84`, bit-pack `:87-148`.
- 스케줄러 믹스인: `vllm_rbln/v1/core/rbln_scheduler.py` (`RBLNAsyncScheduler`).
- 스케줄러 선택/게이트: `vllm_rbln/platform.py` (`VLLM_RBLN_OPTIMISTIC_SCHED`, `VLLM_RBLN_DISABLE_ASYNC`).
- 정합성 수정: `vllm_rbln/v1/worker/rbln_model_runner.py` — 리셋(`sample_tokens`, bookkeeping 직전), device-side 피드백(`execute_model`, `:3469` 직후), guard `:2970`, 버퍼 선택 `:2007`, cpu 채움 `:1273`, .gpu scatter `:1116/1131/1156`.
- 동기 forward: `rebel/python/rebel/sync_runtime.py:204`(`DynamoRuntime.run`), `:268`(`_runtime_handle.run()`).
- 비동기 경로 후보: `rebel/python/rebel/compiled_model.py`(`non_blocking_mode`, `create_async_runtime`), `async_runtime.py`(`async_run`), `rds.py`(`Stream`).
- 참고 branch: `origin/device_tensor_rebased_async` (plain `RBLNScheduler` + `device_index` get_output; 낙관적 스케줄러 없음 → 그 branch 도 overlap 안 남).

## 부록 B — 폐기된 가설(직접 측정으로 반증)
- ❌ "forward 는 async 제출" (readback probe) → 트레이스상 메인 스레드가 forward 안에서 ~5ms 블록(§5).
- ❌ "sampler 의 device→host 동기가 직렬화 원인" → sampler on-device, 별도 drain 없음.
- ❌ "async_scheduling=True 플래그만으로 overlap" → 낙관적 스케줄러 없으면 batch_queue 안 참(§3).
- ❌ "낙관적 스케줄러만 켜면 overlap" → 켜도 0%. 진짜 블로커는 런타임 동기 forward(§5).
- ❌ "prev_sampled 리셋만으로 정합성 해결" → 필요하지만 불충분. 버퍼 불일치(§4-2)까지 고쳐야 0 mismatch.
- ❌ "torch stream/Event 로 overlap" → `torch.rbln` Stream 백엔드 부재 + 블록이 torch 아래(§6).
- ❌ NCCL `.item()` GPU→host 와 gloo 경로 혼동 → gloo 는 순수 host scalar(§1).
