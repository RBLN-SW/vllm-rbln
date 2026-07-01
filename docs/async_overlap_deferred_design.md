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
- C8: fwd_task 로 forward+postprocess+forward_context 추출 (즉시 join, 동작 동일) — context-on-device-thread 검증.
- C9: sample_task 추출 + EM/ST deferral + 토큰 피드백 이동 + AsyncOutput join, decode-only 스코프 — overlap 개방.
- C10: full-layer 프로파일 정량 + 필요시 fallback/race 보강.
