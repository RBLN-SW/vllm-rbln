# Deferred device-thread pipeline — design for actual all_reduce↔forward overlap

- 작성 2026-07-01. `async-overlap-prototype` 브랜치.
- 선행: `docs/async_overlap_report.md` (근본 원인·정합성 수정), C7 커밋(device-forward executor scaffold, 검증 완료: 6-layer A/B 0 mismatch).

## 목표
decode step 마다 도는 DP `gloo:all_reduce`(N+1) 를 forward(N) 의 NPU 실행 구간에 겹친다. C7 로 forward 를 device 스레드에서 돌려도 정합성이 유지됨(0 mismatch)을 확인했다. 이제 실제 overlap 을 내려면 **메인 워커 스레드가 forward(N) 완료 전에 EM(N+1) 로 진행**해야 한다.

## 왜 "deferral" 이 필요한가 (C7 로는 overlap 0%)
워커는 엔진코어의 RPC 를 직렬 처리한다: `EM(N) → ST(N) → EM(N+1)`. device 순서는 `forward(N) → logits(N) → sampler(N) → all_reduce(N+1)[EM(N+1) 의 _prepare_inputs] → forward(N+1)`. all_reduce(N+1) 가 forward(N) 뒤에 오므로, **EM(N)/ST(N) 가 device 를 기다리면 overlap 이 없다.** overlap 은 오직 **EM(N)·ST(N) 가 device 를 기다리지 않고 즉시 리턴**해서 메인 스레드가 EM(N+1) 의 host prep + all_reduce 를 device 작업과 동시에 돌릴 때만 생긴다.

## 아키텍처 — device 스레드 = FIFO 파이프라인
device 는 직렬이므로 단일 스레드가 모든 device 작업을 FIFO 로 소유한다(C7 의 `_DeviceForwardExecutor`). step 당 두 task 를 enqueue:

- `EM(N)` 이 enqueue: **fwd_task(N)** = [ 토큰 피드백(N) → `set_forward_context` 진입 → forward(N) → decode postprocess(compute_logits) ] → returns `logits(N)` + hidden/aux
- `ST(N)` 이 enqueue: **sample_task(N)** = [ sampler(N) → `prev_sampled_token_ids=None` 리셋 → bookkeeping(N) (→ `prev_sampled_token_ids(N)` 세팅) ] → returns ModelRunnerOutput 필드

FIFO 실행: `fwd(N) → sample(N) → fwd(N+1) → sample(N+1) …`

메인 스레드: `EM(N)` enqueue 후 즉시 `return None`; `ST(N)` enqueue 후 즉시 `AsyncRBLNModelRunnerOutput` 반환(device wait 없음). → `EM(N+1)` 의 `_prepare_inputs`(all_reduce 포함) 가 device 스레드의 fwd(N)/sample(N) 와 **동시 실행**. ✅

## 순서 의존성 증명 (정합성)
- **토큰 피드백(N)** 은 `prev_sampled_token_ids(N-1)` (device 텐서) 를 읽는다. 이는 sample_task(N-1) 이 세팅. FIFO 상 `sample(N-1) → fwd(N)` 이므로 fwd_task(N) 안의 피드백이 읽을 때 이미 준비됨. **그래서 피드백을 메인 스레드(현행 §4-2 위치)가 아니라 fwd_task 안(device 스레드)에 두어야 한다.**
- **all_reduce(N+1)** 은 스케줄 결정에서 나오며 forward(N) 결과에 의존하지 않음(리포트 §1). 낙관적 스케줄러가 placeholder 로 N+1 을 미리 확정하므로 forward(N) 완료 전에 발행 가능.
- **get_output(N)**: 엔진이 step N 결과를 실제로 읽는 시점(core.py `future.result()`)은 EM(N+1) 디스패치 이후. `AsyncRBLNModelRunnerOutput` 이 sample_task(N) future 를 join → tokens 반환.

## 반드시 해결할 하위 문제
1. **forward_context 를 task 안으로**: `_forward_context` 는 module-global. 현재 `with set_forward_context(...)` 가 execute_model(메인)에서 forward+postprocess 를 감싼다. deferral 시 이 with 가 forward 실행 전에 exit → context 소실. **fwd_task 가 자체적으로 `set_forward_context` 진입**해야 한다. 필요한 인자(attn_metadata, num_tokens_across_dp, num_padded_tokens, kv forward kwargs)를 task 에 캡처.
   - 동시성 주의: 메인 스레드 EM(N+1) 이 `set_forward_context(N+1)` 를 부르면 안 됨(global 덮어씀). → **메인 스레드는 forward_context 를 만지지 않는다**; 오직 fwd_task 만 진입/복원. `_prepare_inputs` 의 all_reduce 는 forward_context 없이 도는지 확인 필요(현재 all_reduce 는 `RBLNDPMetadata.num_tokens_across_dp` 정적 호출이라 무관할 가능성 큼 — 검증 항목).
2. **capture_reports / KV connector context**: 마찬가지로 fwd_task 안에서 진입. decode-only fast path 에선 KV connector 가 대개 no-op.
3. **sampler + bookkeeping 를 device 스레드로**: `_sample`, `_bookkeeping_sync` 가 sample_task 안에서 돌고 결과 dict 를 반환. `_bookkeeping_sync` 의 D2H(.tolist)/async output 처리와 충돌 없는지 확인.
4. **input_batch thread-safety**: 메인 스레드 `_update_states(N+1)` 이 device 스레드의 fwd(N)/sample(N) 와 동시에 `input_batch` 를 건드림. task 는 필요한 값(req_ids, prev_map, perm 등 host 측)을 **enqueue 시점에 스냅샷**하고, device 텐서만 task 안에서 접근.

## 스코프 (위험 한정)
초기 구현은 **decode-only + no-spec + no-KV-connector fast path** 에만 deferral 적용. prefill/mixed/spec/신규-req 배치는 C7 즉시-join 경로로 fallback. (§8 의 토큰 피드백 fast-path 철학과 동일.) gating: `VLLM_RBLN_ASYNC_FORWARD=1` 且 위 조건.

## 검증
1. 각 단계 6-layer A/B: `baseline_sync.json`(sync) 대비 0 mismatch.
2. 최종 full-layer `--profile` 트레이스로 `gloo:all_reduce ∩ forward` > 0 확인(리포트 §5 방법·정량 스크립트).

## 커밋 계획 (이 문서 위)
- C8 ✅: set_forward_context 를 `_run_forward` 안으로 (즉시 join). 6-layer A/B 0 mismatch 검증됨(커밋 `526854a1`).
- C9: sample_task 추출 + EM/ST deferral + 토큰 피드백 이동 + AsyncOutput join, decode-only 스코프 — overlap 개방.
- C10: full-layer 프로파일 정량 + 필요시 fallback/race 보강.

## 2026-07-01 추가 검증 — C9 의 진짜 난관 (코드로 확인)

### (확인 ✅) 워커 인프라가 overlap 을 지원한다
`multiproc_executor.py`:
- `worker_busy_loop`(메인 워커 스레드, :944)는 RPC(EM/ST)를 처리하고 `handle_output` 호출.
- `handle_output`(:916)은 **async scheduling 이면 output 을 `async_output_queue` 에 넣고 즉시 리턴**(:921-922).
- `get_output()`(device sync/materialize)은 **별도 `async_output_busy_loop` 스레드**(:926-942)에서 `enqueue_output`(:906-907) 을 통해 실행.

→ ST(N) 이 device 를 기다리지 않고 `AsyncRBLNModelRunnerOutput` 를 반환하면, 워커 메인 스레드는 즉시 EM(N+1) 로 진행한다. overlap 의 필요 인프라는 이미 있다.

### (진짜 난관) bookkeeping 의 sampler-output 핸들 확보 (RBLN vs GPU 차이)
GPU 가 overlap 되는 실제 이유: forward/sampler 가 **async 제출**이라, 메인 스레드는 sampler output **device 텐서 핸들**만 들고(값은 `AsyncGPUModelRunnerOutput.get_output` 에서 나중에 D2H) `_bookkeeping_sync` 를 **메인 스레드에서 순차** 처리한다. 그래서 device 는 앞서 달리고, 메인 host 작업(EM(N)→ST(N)/bookkeeping→EM(N+1))은 순차라 **input_batch race 가 없다**. `_bookkeeping_sync` 는 sampled_token_ids 의 **값이 아니라 핸들**만 있으면 된다(`prev_sampled_token_ids = sampled_token_ids` 는 no-sync; req 매핑/invalid 인덱스는 host).

RBLN 문제: `DynamoRuntime.run` 이 **동기**라 sampler 출력 텐서 핸들이 device 스레드가 run() 을 끝낼 때까지 나오지 않는다(출력 텐서가 run() **안**에서 `torch.empty` 로 할당됨, sync_runtime.py). 따라서:
- sampler 를 device 스레드로 보내면 → 메인 스레드 bookkeeping 이 핸들을 못 얻어 막힌다.
- bookkeeping 을 busy_loop/device 스레드로 옮기면 → EM(N+1) 의 `_update_states`(execute_model 맨 앞, :3164)와 `input_batch` 를 **동시** 접근 → race. 게다가 `_update_states(N+1)` 가 bookkeeping(N) 완료를 기다리면 all_reduce(N+1) 가 다시 forward(N) 뒤로 밀려 overlap 소멸.

### C9 로 가는 두 갈래
- **(가) rebel 출력 텐서 pre-alloc (eager_out)**: `sync_runtime.py` 의 `eager_execution_helper().out_tensors` 경로로 출력 텐서를 **미리 할당해 핸들을 메인 스레드가 선확보** → GPU 처럼 bookkeeping 을 메인 스레드에서 no-wait 순차 처리, device 스레드는 그 텐서를 채우기만. input_batch race 없음. **가장 GPU 에 충실**하나 rebel eager_out 경로 이해 필요.
- **(나) bookkeeping 을 device 스레드로 + input_batch 동시성 처리**: 필요한 host 상태를 enqueue 시점 스냅샷, `input_batch` 쓰기를 fwd/sample task 안으로 국한, `_update_states(N+1)` 와 disjoint 함을 보장(또는 lock). 더 침습적·위험.

권장: (가) 를 우선 조사. 안 되면 (나).

### (확인 ✅) path (가) 실현 가능 — rebel eager_out 메커니즘 존재
`rebel/python/rebel/core/torch_eager.py`: `EagerExecutionHelper.set_out_tensor(tensors)` 로 출력 텐서를 등록하면 `sync_runtime.py:247-256` 의 `run()` 이 새로 할당하는 대신 그 텐서에 채운다. → **호출자(메인 스레드)가 출력 텐서를 미리 할당해 핸들을 선확보** 가능. bookkeeping 이 device 완료를 안 기다리고 핸들만으로 진행(GPU 방식).
- 주의: `eager_execution_helper()` 는 `@lru_cache` **프로세스 전역 싱글톤**. 모든 `run()` 을 device 스레드로만 라우팅하면 싱글톤을 device 스레드만 만져 cross-thread race 없음. 메인은 텐서 pre-alloc 만 하고 task 에 전달.
- 남은 통합 작업(C9): (1) forward/sampler 출력 텐서 shape/dtype 를 output_profile 로 메인에서 pre-alloc, (2) 모든 device run() 을 device 스레드 경유, (3) task 가 set_out_tensor+run, (4) bookkeeping 은 메인 스레드에서 핸들로 진행(no-wait), (5) get_output 은 device 스레드 완료 대기 후 D2H. decode-only 스코프.

### 2026-07-01 실측 — EAGEROUT 프로브 판정: `alias=False` (naive (가) 불가)
exclusive box에서 `VLLM_RBLN_EAGEROUT_PROBE=1` 실행(프로브 dtype 버그 `b9ce798` 수정 후):
- `EAGEROUT_PROBE alias=False pre_shape=(1,) sampled_shape=(1,1) sampled_dtype=torch.int32` (전 rank·전 step 일관).
- **eager_out 자체는 동작**: 컴파일된 greedy_sample op(`rbln::argmax`, 출력 `(B,) int64`)에 대해 메인이 pre-alloc한 텐서를 `set_out_tensor`로 넘기면 op이 거기 채움(맞는 dtype/shape면 crash 없음). ← path (가)의 전제(메인이 device 출력 핸들 선확보)는 **op 단위로는 성립**.
- **그러나 최종 핸들은 못 잡음(구조적)**: `rbln_sampler.py forward()`가 컴파일 op 뒤에서 **무조건** `sampled = sampled.to(torch.int32)`(:313) 후 `sampled.unsqueeze(-1)`(:320)로 `sampled_token_ids`를 만든다. `.to(int32)`는 int64→int32 캐스트라 **항상 새 버퍼**를 할당하고, 이 캐스트는 eager_out 영역 **밖**(순수 eager). 그래서 `_bookkeeping_sync`가 잡는 `sampled_token_ids` 핸들은 eager_out pre-alloc과 **절대 alias 안 됨**.

**결론 / C9 방향 재정의**:
- **naive (가)**(`self.sampler()`를 eager_out으로 감싸 최종 핸들 선확보)는 **불가**.
- **(가′) 수정된 (가)**: sampler를 고쳐 최종 int32 (B,1) 출력을 **메인이 pre-alloc한 버퍼에 쓰게** 한다 — 예: `forward()`의 `.to(int32)`를 `torch.empty((B,1),int32)` pre-alloc + `copy_`/`out=`로 대체하고 그 버퍼를 eager_out/handshake로 메인이 선확보. sampler에 침습적이나 (나)보다 race 표면이 작음.
- **(나)** bookkeeping을 device 스레드로 + input_batch 동시성 처리(스냅샷/lock). 침습적·위험하나 sampler 안 건드림.
- 권장: **(가′) 우선 조사** — `.to(int32)` 한 곳만 pre-alloc 버퍼로 돌리면 되는지 PoC. 막히면 (나).

## C9 (가′) 구현 계획 — 코드 정독으로 확정한 상세 (2026-07-01)

### 결정적 단순화 (fast path에서 확인)
- **decode fast path**(=`not is_prefill` ∧ `speculative_config is None` ∧ `VLLM_RBLN_USE_DEVICE_TENSOR` ∧ no-LoRA(`use_wrapped_compute_logits()==True`) ∧ PP world_size==1 ∧ no-grammar)에서는:
  - `logits`가 `model_output`에서 **그대로** 나온다 — compute_logits는 wrapped(모델 내부), decode/no-spec/device_tensor 분기(rbln_model_runner.py:3950-3964)는 **slicing 없음**. 즉 execute_model의 post-forward tail(3869-3990)이 fast path에선 사실상 **unpack뿐**.
  - `_bookkeeping_sync`의 async 분기(3199-3244)는 **값이 아니라 핸들만** 필요: `prev_sampled_token_ids = sampled_token_ids`(D2H 없음), `sampled_ids=[-1]` placeholder로 host bookkeeping. `_get_nans_in_logits`는 기본 off, `_get_prompt_logprobs_dict`는 decode에서 no-op. → **메인 스레드가 device 대기 없이 bookkeeping 가능**(핸들만 있으면).

### (가′)의 실제 메커니즘 = copy_ into 메인-pre-alloc 버퍼 (eager_out 아님)
- eager_out은 컴파일 op 출력만 잡고 `.to(int32)` 뒤 최종 핸들은 못 잡음(alias=False, 위). 그래서 (가′)은 eager_out 대신:
  - 메인: `out_buf = torch.empty((num_reqs,1), int32, device=rbln)` 로 **핸들 선확보**(device alloc만, compute 없음). `prev_sampled_token_ids = out_buf` 로 bookkeeping은 이 핸들 사용.
  - device 스레드(sample_task): sampler 실행 → 최종 `sampled (B,1) int32` → `out_buf.copy_(sampled)`(in-place, device op). copy_는 값을 나중에 채우지만 **핸들은 불변** → 메인은 안 기다림.

### 2-커밋 분할 (각 6-layer A/B 0 mismatch 검증)
- **C9a (plumbing, overlap 0 — 격리 검증)**: fast path에서 execute_model이 forward closure를 device executor에 submit하되 **join 안 함**, future를 ExecuteModelState에 stash, `return None`. `sample_tokens` 시작부에서 `future.result()`로 model_output 받아 기존 흐름 그대로. → "logits가 device-thread closure 경유 + future 핸드오프"가 정합성 유지하는지만 검증(타이밍 이득 없음, worker 순서상 join이 ST(N) 안에서 일어나 all_reduce(N+1) 전).
  - **주의(반드시)**: forward를 감싸는 컨텍스트를 closure 안으로. `set_forward_context`+`capture_ctx`는 이미 `_run_forward` 안(C8). 하지만 **`maybe_get_kv_connector_output`(:3678)이 forward를 밖에서 감싼다** — defer 시 이 `with`가 forward(async) 전에 exit. fast path에선 KV connector no-op이라 무해할 **가능성**이나, closure 안으로 옮기거나 no-op임을 명시 검증할 것. `self.kv_connector_output` 대입/`_pending_model_report`(reports 처리)도 join 후 sample_tokens에서 재현.
- **C9b (overlap 개방)**: `sample_tokens`가 (1) `out_buf` 메인 pre-alloc, (2) sample_task=[fwd_future.result()→_sample→out_buf.copy_(sampled)]를 **동일 device executor에 enqueue(FIFO)**, (3) 메인은 out_buf 핸들로 bookkeeping(no-wait) + `prev_sampled_token_ids=out_buf`, (4) `AsyncRBLNModelRunnerOutput`이 sample_task future를 join 후 out_buf D2H. execute_model은 forward만 submit하고 **return None**(join 제거) → 메인이 EM(N+1) host prep+all_reduce를 fwd(N)/sample(N)와 동시 실행.
  - FIFO 보장: fwd(N)→sample(N)→fwd(N+1). 토큰 피드백(N+1)은 fwd_task(N+1) 안에서 `prev_sampled_token_ids(N)=out_buf(N)`를 읽고, sample(N)이 이미 채움. **토큰 피드백을 fwd_task 안(device)으로 이동** 필수(현재 execute_model 메인 3723-3740).
  - **out_buf 수명**: step마다 새 `out_buf`(N+1이 N을 덮어쓰지 않게). AsyncOutput이 N의 out_buf 참조 보유 → get_output까지 생존.
- **C10**: full-layer `--profile`로 `gloo:all_reduce ∩ forward` > 0 정량.

### 스코프/폴백
- 위 fast-path 조건 하나라도 불충족(prefill/mixed/spec/grammar/LoRA/PP) → **C7 즉시-join 경로로 fallback**(현행 `submit(...).result()`). 게이트는 execute_model 진입부 조건 + sample_tokens의 grammar_output 체크.

### C9a 착지(커밋 `96acef3`) + C9b의 남은 난관 (코드 정독으로 확정, 2026-07-01)
**C9a 완료·검증**(16 prompts 0 mismatch): execute_model이 fast path에서 forward를 submit만 하고 future stash+return None, sample_tokens 시작부에서 `fwd_future.result()`로 unpack. overlap은 아직 0(join이 ST(N) 안, EM(N+1) 전). gate: `_async_forward ∧ ¬warmup ∧ decode ∧ spec None ∧ async_sched ∧ device_tensor ∧ no-LoRA ∧ ¬has_kv_transfer_group ∧ pp==1 ∧ ¬pooling`.

**C9b(overlap 개방)에서 반드시 풀 것 — 셋 다 코드로 확인됨**:
1. **bookkeeping이 device 값 참조 회피**: `_bookkeeping_sync` async 분기(3199-3254)는 대부분 host지만 끝에서 `_get_prompt_logprobs_dict(hidden_states[:n], …)`(:3251)로 hidden_states를 슬라이스한다. 메인이 overlap하려면 `fwd_future.result()`를 **안 해야** 하고 → 메인엔 hidden_states가 없다. decode fast path엔 prompt-logprobs가 없으니 **handle-only bookkeeping 변형**(hidden_states/logits 불참조, `prev_sampled_token_ids=out_buf` + token_ids_cpu에 `[-1]` placeholder + num_tokens 갱신만) 필요. `_get_nans_in_logits`는 기본 off라 무관.
2. **sampling_metadata 스냅샷(race)**: sample_task가 device 스레드에서 `self.input_batch.sampling_metadata`를 읽는데, 메인의 EM(N+1) `_update_states`(:3402)가 **동시에** input_batch를 mutate → race. enqueue 시점(메인)에 sampling_metadata(및 sampler가 읽는 input_batch 파생값)를 **스냅샷**해 task에 넘길 것. (bookkeeping은 메인 ST(N)에서 EM(N+1) 전에 순차 실행되므로 bookkeeping↔update_states race는 없음 — sampler만 문제.)
3. **토큰 피드백을 fwd_task 안(device)로 이동**: 현재 execute_model 메인(3711-3740)에서 `prev_sampled_token_ids`를 input_ids에 scatter. overlap 시 fwd(N+1) 피드백은 sample(N)이 채운 prev(N)을 읽어야 하는데, 메인 EM(N+1)에 있으면 sample(N) **완료 전**에 실행돼 stale. FIFO상 fwd_task(N+1) 안에 두면 sample(N) 뒤라 안전. 피드백이 읽는 host 상태(prev_req_id_to_index, req_ids, perm)도 enqueue 스냅샷.

**out_buf 수명/더블버퍼**: step마다 새 `out_buf=(num_reqs,1)int32`. `AsyncRBLNModelRunnerOutput`이 out_buf 참조 보유 → get_output(sample_task future join 후 D2H)까지 생존. N+1이 N을 덮지 않음.

### C9b 착지(커밋 `d55098b`) — 정합성 검증 완료(16 prompts 0 mismatch)
위 3난관 전부 구현·검증. 구현 중 드러난 **동시성 버그 3개**(각 A/B로 확인):
1. **inplace on inference tensor (thread-local)**: `execute_model`/`sample_tokens`는 `@inference_mode`지만 thread-local이라 device 스레드/`async_output_busy_loop` 스레드엔 안 걸림. main에서 만든 inference tensor(out_buf, `_sampled_token_ids_cpu`)에 그 스레드들이 inplace(`copy_`, token-feedback scatter)하면 금지됨. → `_DeviceForwardExecutor._run`이 각 task를 `torch.inference_mode()`로 감싸고, `get_output`의 D2H도 감쌈.
2. **SYS_ENODEV (좀비)**: 크래시한 run이 device 메모리를 안 풀어 다음 run이 device 등록 실패. 코드 아님 — 크래시 후 반드시 내 `python3`/`VLLM::` 프로세스 kill + `/dev/shm` 정리.
3. **sampling_metadata 버퍼 재사용 race (1/16 mismatch)**: `_make_sampling_metadata`가 `temperature`/`top_k`/`top_p`를 persistent 버퍼의 슬라이스/`copy_`로 반환 → EM(N+1)이 in-place로 덮는데 device sample_task(N)가 동시에 읽음. GPU는 single-stream 순서로 안전, RBLN device 스레드는 아님. → enqueue 시점(main)에 metadata의 **모든 tensor 필드를 `.clone()`**(`dataclasses.replace`)해 device sampler가 private 버퍼를 읽게. **token 0,1은 맞고 2부터 발산**했던 게 실제 overlap 시작 시점과 일치 → race 확증.

**남은 것(C10)**: full-layer `--profile`로 `gloo:all_reduce ∩ forward`(rebel/sync_run) > 0 정량 + cadence(c10d start-to-start) 감소 확인. 6-layer 정합성은 끝났고, overlap **정량**만 남음.

### 2026-07-01 overlap 정량 시도 — deferral engage 확인 + full-layer는 toolchain 블로커(async 무관)
- **deferral engage 확인**: 실제 decode(warmup=False)에서 `C9_DIAG ... fast_defer=True` (전 rank). 즉 C9b 경로가 실제로 탄다. 정합성 0 mismatch와 합쳐 **기능적으로 정확·활성**.
- **6-layer overlap = 0.0% (측정 부적합, 실패 아님)**: trace(`profile/rbln_*_dp{r}/*.pt.trace.json.gz`)에서 `gloo:all_reduce ∩ rebel/sync_run` = 0. 이유는 6-layer forward가 **~1.35ms**로 메인 스레드의 host prep(EM(N+1)의 _update_states+_prepare_inputs)보다 짧아 **overlap 창 자체가 없음**. 원 report가 full-layer에서만 측정한 이유. (또 trace상 forward가 메인 tid에 붙어 나오는 kineto attribution 관찰 — full-layer면 forward가 지배적이라 명확해질 것.)
- **profiler 활성화 방법(중요)**: 이 vllm-executor(dev) 버전은 `VLLM_TORCH_PROFILER_DIR`를 config로 안 읽어 `llm.start_profile()`이 "Profiler is not enabled"로 죽는다. parity_runner에 `--profile`일 때 `llm_args["profiler_config"]=ProfilerConfig(profiler="torch", torch_profiler_dir=...)` 주입 패치 필요(vllm_rbln이 dir을 per-rank `profile/rbln_*_dp{r}`로 재작성). **이 패치는 vllm-executor working tree에만 있고 미커밋**(harness repo).
- **full-layer 블로커(async·C9 완전 무관 — 격리 증명 완료)**: `--num-hidden-layers` 빼면 warmup forward(`sync_runtime.py:268 run`, `_execute_dummy_requests→execute_model→_run_forward`)가 device에서 **`(System) code=504 SYS_TASK_ABORTED ... WaitForCompletion, seq=1815`**로 죽는다(처음 1회는 502 SYS_BUSY였으나 이후 일관되게 504). **매번 동일한 device job seq=1815**에서 abort. 격리 테스트 3종 전부 동일 재현:
  1. 내 branch + `ASYNC_FORWARD=1` → seq=1815 abort
  2. 내 branch + `DISABLE_ASYNC=1`(sync) → seq=1815 abort
  3. **plain `dev` vllm-rbln(async 커밋 0개) + sync → seq=1815 abort**
  → async / C9 / C8 / profiler 전부 배제. **full-layer gpt-oss forward 컴파일 그래프가 이 로컬 환경에서 실행 불가**. `--profile` 유무와도 무관(6-layer는 profile 포함 정상). DP collective라 seq=1815에서 전 rank 동시 abort(로그상 "first rank"는 순서일 뿐).
  - **원인 후보**: 로컬 **rebel `dev519`(b5d77d38)** vs CI가 PASS하는 **`dev508`(452723e9)** — 11 커밋 차 regression 가능성. CI(obedients `schedule-rebel-rebel-compiler-ci` build 36, job gpt-oss-120b-ep-dp4-b1-mml131072-bks1024)는 **정확히 같은 MODEL_CMD로 PASS**(THRESHOLD 0.99, `run-model-test.sh`, docker image, 8-NPU udc-08). 즉 config·모델은 정상, 로컬 toolchain/box 문제.
- **다음 액션(overlap 정량)**: full-layer가 되는 환경 필요. **deferral engage·6-layer 정합성은 이미 확인**됐으니 남은 건 full-layer overlap 수치뿐.

### 2026-07-01 full-layer 근본원인 최종 확정 + device wedge 경고
**근본원인 = device 타임아웃(async 무관, rebel 버전 무관)**:
- rebel **dev508**(CI 버전)로 재빌드해도 full-layer는 `SYS_TASK_ABORTED`(504)/`SYS_BUSY`(502)로 동일하게 죽음 → **rebel dev519 regression 아님**.
- dmesg: `queue_timedout: command-queue-0 TDR 1` 이 전 rank 동시 → warmup forward가 **kernel TDR 타임아웃(`timeout_s`=**6초**)** 초과. full-layer warmup은 `RBLN_WEIGHT_FREE=1`로 120B weight를 on-device transform하느라 6초 초과(6-layer는 통과). `rbln-smi tdr/timeout --group 0 --value N`(sudo)로 커널 타임아웃 조정 가능.
- 두 번째 타임아웃: optimum `RBLNModelConfig.timeout` **기본 60초**(`modeling.py:273` `rebel.Runtime(timeout=rbln_config.timeout)`) → 502 "user-defined timeout". vllm-rbln은 이 값을 안 건드림(env 없음). full-layer는 이것도 초과 가능. 조정하려면 `rebel.Runtime(timeout=...)` 또는 `ContextRblnConfig(timeout=...)`로 크게. **CI의 `COMPILE_INFERENCE_TIMEOUT_MIN=60`은 무관** — 그건 `run-model-test.sh`의 GNU `timeout -k 60s 60m` wrapper(전체 프로세스 kill 안전망)일 뿐, device 타임아웃 아님. → **CI 박스(udc-08)에선 full-layer forward가 기본 타임아웃(6s/60s) 안에 끝난다는 뜻**. 이 4-NPU 박스는 더 느려서(또는 marginal) 초과.
- **⚠️ device wedge (내 테스트 부작용, admin 조치 필요)**: 반복 실험 중 `kill -9`을 15+ 회 하여 **"killed" 상태 context가 NPU마다 23GiB/ASID를 안 놓고 누수** → `No free ASID (rc -16)` → 이후 모든 run이 `SYS_ENODEV`(device 등록 실패). `rbln-smi sort`(rebind)·tdr/timeout 리셋으로 회수 안 됨. 모듈 refcount=1이라 `rmmod` clean 불가. **박스 reboot(또는 KMD 강제 reload)이 필요** — irqs-disabled 경고가 떠서 강행은 안 함(박스 크래시 위험). reboot 후 killed context 사라지면 재개 가능.
- **재개 레시피(reboot 후)**: (1) `sudo rbln-smi timeout --group 0 --value 600` + `tdr 600`(커널), (2) optimum `modeling.py:273`을 `timeout=rbln_config.timeout or 600`로 패치(앱 502 방지), (3) full-layer sync로 baseline → async → `--profile`(ProfilerConfig 주입 패치) → `scratchpad/overlap_quant.py`. **kill -9 남발 금지**(ASID 누수) — teardown은 정상 종료 유도하고, 죽이면 즉시 context 상태 확인.
