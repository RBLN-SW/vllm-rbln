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
