# async overlap — GPU vs RBLN 방향성 Q&A (근거 정리)

> 2026-07-02. STEP 1 판정((가) 확정, `async_overlap_RESUME.md` §2) 직후 나눈 방향성 논의를 정리.
> 목적: "GPU에선 되는 overlap이 RBLN에선 왜 안 되나, GIL/stream/Event가 각각 어떤 역할인가"를
> 코드 근거와 함께 못박아, (A) 런타임 glue 경량화로 가는 전제를 명확히 한다.
> 결론 먼저: **병목은 stream/Event 부재가 아니라 forward 제출 사이의 두꺼운 host glue(디코드 forward의
> ~68%를 GIL 잡은 채 도는 Python dispatch/prepare_inputs/address-patch)다.** threading(별도 host
> 스레드) 방향은 유저가 완전 배제. 남은 길은 (A) 그 glue를 런타임 레벨에서 줄이는 것.

---

## Q1. GPU에도 GIL이 있는데 거기선 overlap이 왜 되나? torch stream 때문인가?

**A. GPU도 GIL은 똑같이 있다. overlap은 GIL을 우회해서가 아니라, forward의 wall-time이 host가
아니라 device에 있어서 얻는 것이다.**

- **GPU**: 모든 CUDA 커널이 stream에 **async launch**된다(host는 launch당 ~µs만 GIL 점유, 즉시 리턴).
  실제 ms 단위 연산은 device가 비동기로 돈다. transformer forward의 eager dispatch는 이 싸구려
  async launch의 얇은 나열이라, **host는 forward 대부분의 시간 동안 비어 있다(GIL free).**
- **RBLN**: 개별 제출(`.run()`)은 **이미 GPU처럼 async**다(RESUME F1: run()은 device 완료를 안
  기다리고 드라이버 큐에 넣고 즉시 리턴, µs 단위). **여기까진 GPU와 같다.** 문제는 컴파일 그래프를
  걸어가며 제출들 **사이에** 무겁고 blocking인 host Python glue(subgraph dispatch,
  prepare_inputs, address-patch, 중간 host op)가 끼고, 이게 3.5ms forward의 **~68%를 GIL 잡은
  채** 돈다는 것(STEP 1 실측: `gil_free_ratio` 중앙값 0.32, 2회 실행 재현).

→ 핵심 차이는 "host가 있냐 없냐"가 아니라 **"host glue가 얇고 async냐(GPU) vs 두껍고 blocking이냐
(RBLN)"**. GPU는 all_reduce가 끼어들 빈 host 창이 forward 거의 전체지만, RBLN은 그 창이 ~32%뿐이라
연속된 빈 구간이 없다.

## Q2-1. GPU와 RBLN의 차이가 (1) host 바쁨 (2) stream/Event 부재, 두 개인가?

**A. 근본 차이는 1개(host glue의 두께)다. stream/Event 부재는 관련된 곁가지이지 이 문제의 병목이 아니다.**

- (1)이 근본 원인: RBLN host는 "얇은 async 디스패처"가 아니라 **바쁜 직렬화기**(68% GIL). Q1 참조.
- (2)는 실재하는 API 갭이지만 병목이 아님(Q2-2 참조).

## Q2-2. CUDA stream/Event가 RBLN엔 없다? stream/Event와 GIL은 관련이 있나?

**A. torch.rbln엔 CUDA식 per-op stream/Event가 없는 게 맞다**(device-wide `synchronize()`만,
`torch_rbln .../device/device.py:87`). **하지만 stream/Event와 GIL은 메커니즘상 별개(직교)다.**

- **GIL** = host/Python 동시성 락. **stream/Event** = device 쪽 순서·비동기 메커니즘(커널 async
  enqueue, 두 스트림 device에서 동시 실행, Event로 스트림 간 동기/완료 질의).
- **간접 연결만 있다**: CUDA는 stream 모델 덕에 launch가 async라서 host가 op당 GIL을 잠깐만 잡는다.
  즉 "async 제출"이 host를 GIL-light하게 만들고, 그 async 제출을 stream 모델이 제공한다.
- **그런데 RBLN은 stream 추상 없이도 제출이 이미 async다**(F1). 따라서 overlap이 안 되는 건
  "stream/Event가 없어서"가 아니라 **제출 사이 host glue가 두꺼워서**다. stream/Event를 torch.rbln에
  넣어도 그 glue는 안 줄어든다. (stream/Event가 결정적이었을 경우 = host가 device 완료를 기다리며
  blocking하는 case (나)였을 텐데, STEP 1은 case (가)로 판명.)

## Q3. GPU async scheduling에서 forward와 gloo all_reduce가 겹치는 건 stream 때문 아닌가?

**A. 아니다. 그 all_reduce가 어떤 종류냐가 결정적이다.**

- **device 콜렉티브(NCCL) all_reduce**(예: TP activation all_reduce): 예, GPU 위 별도 comm
  스트림에서 돌고 compute 스트림과 **device 하드웨어가 겹친다**. Event로 스트림 동기. "stream/Event로
  overlap"의 정석 사례.
- **여기서 겹치려는 DP num_tokens all_reduce**: **host gloo**다 — CPU 텐서를 `cpu_group`에서
  all_reduce (`vllm_rbln/forward_context.py:86`: `dist.all_reduce(num_tokens_tensor,
  group=get_dp_group().cpu_group)`). CUDA 스트림에 안 올라간다. GPU에서 이게 forward와 겹치는 이유는
  **스트림이 아니라, host가 async device forward 동안 비어서** gloo를 그 빈 host 시간에 돌리기
  때문이다.

→ **우리 타깃 all_reduce는 host op**라, GPU에서조차 "torch stream으로 겹치는" 게 아니라 **한가한
host 덕에** 겹친다. 그래서 RBLN에 필요한 것도 스트림이 아니라 **한가한 host** → (A) glue 경량화.

## Q4. 그러면 async scheduling이 실제로 쓰는 stream/torch.Event는 무슨 일을 하나? (output D2H 지연 한 곳?)

**A. "forward를 async로 만드는 것"이 아니다(그건 sync에서도 동일). async scheduling의 stream/Event는
출력 readback을 device-side ordering + 지연된 host sync로 미뤄서, step N의 GPU 꼬리(sampling→D2H)가
도는 동안 CPU가 step N+1을 앞당겨 준비하게 하는 것이다.**

`AsyncGPUModelRunnerOutput` (`vllm/v1/worker/gpu_model_runner.py:238-292`):
```python
default_stream = torch.cuda.current_stream()
with torch.cuda.stream(async_output_copy_stream):        # (1) 전용 copy stream
    async_output_copy_stream.wait_stream(default_stream) # (2) device-side ordering: sampling 완료 대기 (host 안 막힘)
    self.sampled_token_ids_cpu = sampled_token_ids.to("cpu", non_blocking=True)  # (3) async D2H 발행
    self.async_copy_ready_event.record()                 # (4) copy 완료 지점에 Event 기록
def get_output(self):
    self.async_copy_ready_event.synchronize()            # (5) host는 "값이 실제 필요할 때"(N+1 시점) 처음 막힘
```
- (1) 전용 copy stream → 출력 D2H가 compute 스트림/다음 forward와 직렬화 안 됨.
- (2) `wait_stream` → **device-side** 순서 보장(host 블록 아님).
- (3) `non_blocking=True` → copy async 발행, host 즉시 리턴.
- (4)(5) `torch.Event` record/synchronize → coarse full-device sync 대신 **딱 그 copy만** 지연 동기.
  host는 다음 step 처리에서 값이 필요할 때 비로소 막힘 → 그 사이 CPU가 N+1 스케줄/prep.
- (추가) `prepare_inputs_event`(:688) — 겹치는 두 step이 재사용 CPU 입력 버퍼를 덮어쓰지 않게 보호.

**겹치는 대상은 "GPU 꼬리 ↔ CPU의 다음 step 스케줄링"이지 gloo all_reduce가 아니다.** 그리고 이
패턴을 **RBLN은 이미 미러링**한다 — `AsyncRBLNModelRunnerOutput`(RESUME F3: D2H를
`torch.rbln.synchronize`로 get_output까지 지연). 즉 async-scheduling의 stream/Event 역할은 RBLN에
이미 등가물이 있다.

---

## 종합 결론 (→ (A)의 근거)

1. GPU overlap은 GIL 우회가 아니라 **host가 async device forward 동안 비어 있음** 덕분.
2. RBLN은 제출은 async지만(F1), **제출 사이 host glue가 forward의 ~68%를 GIL 잡은 채** 돌아 host가 안 빔(STEP 1).
3. stream/Event는 device-ordering 메커니즘으로 GIL과 직교. RBLN이 못 하는 건 stream 부재가 아님.
4. 타깃 DP all_reduce는 host gloo(`cpu_group`)라, GPU에서도 스트림이 아니라 **한가한 host**로 겹침.
5. async scheduling의 stream/Event는 **출력 readback 지연**용이고 RBLN에 이미 등가물 있음(AsyncRBLNModelRunnerOutput).

→ **유일하게 남은 지렛대: forward 제출 사이의 host glue(GIL 68%)를 줄여 host를 GPU처럼 비우는 것 = (A).**
threading(B, 별도 host 스레드로 all_reduce)은 유저 배제.

## 근거 출처
- STEP 1 실측·판정: `async_overlap_RESUME.md` §2 (gil_free_ratio~0.32, forward GIL ~68% held).
- RBLN 제출 async(F1)/forward host 3.5ms(F2)/inline D2H 지연(F3): 같은 문서 §1.
- DP all_reduce = host gloo: `vllm_rbln/forward_context.py:86` (`get_dp_group().cpu_group`).
- GPU async output stream/Event: `vllm/v1/worker/gpu_model_runner.py:238-292`, copy stream/event 생성 :682-688.
- RBLN 미러: `AsyncRBLNModelRunnerOutput` (`rbln_model_runner.py:270-326`).
- torch.rbln per-op event 부재: `torch_rbln .../device/device.py:87` (device-wide synchronize만).
