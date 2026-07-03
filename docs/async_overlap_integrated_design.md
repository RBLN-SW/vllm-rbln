# DP all_reduce 전체 hiding — compiler+runtime+vllm-rbln 통합 근본 해법

> 2026-07-03. 목표: gpt-oss-120b EP+DP4 decode에서 per-step DP `gloo:all_reduce(N+1)`
> **전체**를 NPU `forward(N)` 뒤로 완전히 숨긴다. 부분(intra-step) hiding이 아니라 100%.
> 전제: 이 브랜치의 async scheduling(RBLNAsyncScheduler, batch_queue depth-2)은 이미 있음.
> 유저는 vllm-rbln + rebel_compiler 양쪽 개발자 → 스택 통합 수정 가능.

---

## 0. 결론 (three-layer 매핑 후, 2026-07-03 정정판)

**근본 원인은 "worker가 serial + forward가 blocking". forward가 blocking인 이유는 device drain 자체가
아니라, 그래프 walk(device submit + mid-graph host op마다의 drain+CPU 실행)가 통째로 Python worker
스레드를 잡고 인라인으로 돌기 때문이다.**

> ⚠️ 초판의 "컴파일러로 host op을 제거" 방향은 폐기. host op은 device 불가(예: Cast-scalar, ArgMax,
> gather 등)라 반드시 존재해야 하고, host op 앞 drain을 빼면 **결과 mismatch**(host op이 아직 안 끝난
> device 출력을 읽음). host op은 dataflow상 device→host→device 직렬점이라 drain은 정합성상 필수다.

**올바른 해법 = "walk 전체를 Python 스레드에서 떼어내 런타임 소유 exec 스레드에서 async 실행"**
(host op은 그대로 두되, drain·실행을 exec 스레드에서 수행 → 정합성 100% 동일, Python worker만 즉시 풀림).
그러면 all_reduce는 worker에 그대로(실제값·dummy·prefill·spec 전부 기존 로직으로 정확) 두고도, 기존 async
scheduling이 execute_model(N+1)의 host 부분(=all_reduce 전체)을 forward(N) exec-스레드 실행과 overlap.
엔진-side 이전(H1~H4, all-or-nothing)·speculation·barrier 문제 전부 소멸. GPU와 동일 구조.

**왜 성립하는가 (코드 확인):**
- host op = 컴파일된 host function(C++, `tvm_host_op`/`getHostFunctions()`), **Python 아님** → exec
  스레드에서 GIL 없이 실행 가능.
- device job은 이미 driver-async(`rblnSubmitJob`, `RblnJobDepDesc` 의존성) → exec 스레드가 submit만.
- driver에 host-op 콜백 enqueue API **없음**(`rblnSubmitJob`/`rblnWaitJob`뿐) → host op은 CPU 스레드
  필수, 즉 "driver stream에 다 넣기"는 불가하고 "runtime exec 스레드"가 정답.
- `Run()`(`runtime_instance.cc:1172`)은 `executions_` walk를 caller 스레드에서 인라인 실행 →
  이 walk를 exec 스레드로 옮기는 것이 변경의 핵심.

---

## 1. 계층별 근거 (코드 확인 완료)

### (L1) 런타임: device-only forward는 이미 async. 범인은 per-host-op drain.
`rebel/src/runtime/core/runtime_instance.cc`:
- device op은 `RunAsync()`→`CommandDispatcher::Dispatch()`→`rblnSubmitJob()`으로 **enqueue 후 즉시 리턴**
  (`command_dispatcher.cc:46-66`). `SeqState{in,in_deps,out}`로 의존성 체인만 기록(`seq_state.h`).
- `Run()`은 기본적으로 drain 안 함: force_sync(`:1187`, env `RBLN_RUNTIME_FORCE_SYNC`, 기본 off),
  multi-output copy(`:1197`, decode에서 보통 미발생)뿐.
- **진짜 blocking = `:781` — 모든 host op 실행 직전 `EnsureAllTasksCompleted()` (직전까지 device work
  전부 대기).** 즉 forward가 device-op(async)→host-op(**여기서 drain**+CPU 실행)→device-op(async)→
  host-op(drain)… 로 걸으며 매 host op마다 device를 기다려 전체 직렬화.
- `:871` const-buffer drain은 `has_const_buffer`일 때만(대개 init 1회).
- **이미 있는 async 인프라**: `Stream{last_seq_, Drain()}`(`base/stream.h`),
  `CommandDispatcher::{Dispatch(async), WaitForCompletion(seq)(deferred)}`. submit/wait 분리 존재.

### (L2) 컴파일러: host op 중 상당수는 hard constraint가 아니라 heuristic pruning.
`rebel/src/mlir/.../Transforms/Partition/Partition.cpp`:
- `isHostPreferredOp()`(`:63-69`): Where, DynamicTake, **Cast, Concatenate, ExpandDims, Reshape,
  Split, Squeeze, StridedSlice, Take, Pad, Quantize/Dequantize/Requantize** (14종).
- 배치 결정은 **혼합**:
  - **hard constraint**(`RblnTensorOpsInterfaces.cpp`, `isDevRunnable()==FAIL`): Cast-scalar(64-align
    불가, `:785-800`), ArgMax/ArgMin(`:1504`), Pad-circular(`:994`) 등 — device 불가.
  - **heuristic pruning**(`pruneSubGraph`, `:137-187`): subgraph가 host-preferred op만 있거나,
    trivial op <13개 & `devFuncCount>1`이면 host로. **임계값(maxCount=13, devFuncCount>1) 튜닝 가능.**
- `mergeSubgroups()`(`:480-571`)는 taint 분석으로 device subgroup 사이에 host op이 끼면 **merge 거부**
  (=mid-graph host op 형성을 일부 예방)하지만, **이미 생긴 mid-graph host op을 제거·hoist·batch하는 pass는
  없음**(확인). PostPartition에도 없음.

### (L3) vllm-rbln/torch_rbln: output은 이미 deferred, 하지만 forward walk 자체가 막음.
- `.run()`은 GIL 해제(`pyrbln/compiled_model.cc:50` `gil_scoped_release`) — 블로킹은 Python 아니라
  device drain(L1 `:781`).
- `AsyncRBLNModelRunnerOutput`(`rbln_model_runner.py:206-249`)은 sampled token D2H를 non_blocking
  copy + `get_output()`에서 `torch.rbln.synchronize()`로 지연 — **GPU AsyncGPUModelRunnerOutput 미러 이미
  존재**. 하지만 이건 output readback만 숨김. forward `model_executable()`(`:3761`)이 그래프를 걸으며
  `:781`에서 막히므로 execute_model(N)이 일찍 리턴 못 함 → N+1 못 겹침.
- torch_rbln엔 per-op stream/event 없음, device-wide `synchronize`만(`torch_rbln .../device/device.py:87`).
  단 output-deferred 구조상 device-wide로도 충분.

---

## 2. 통합 해법 — ★ 기존 async runtime 활용 + 빠진 기능 이식 (주경로)

> 2026-07-03 발견: 런타임에 **이미 async runtime이 프로덕션 존재**한다. exec 스레드를 새로 만들 필요 없음.
> 작업 = "동시성 구축"이 아니라 "async runtime에 vLLM decode용 기능 이식" + torch_rbln 배선.

### 이미 존재하는 것 (그대로 씀)
`rebel/src/runtime/core/async_runtime.cc`, `rebel/src/pyrbln/compiled_model.cc:127-144`:
- **worker 스레드 풀**(`num_thread_ = context_->GetNumBuf()`, 보통 1~2)이 각자 `RuntimeInstance`를 소유하고
  **`RuntimeInstance::Run()` 전체 walk(device submit + host op `:781` drain + host op 실행)를 백그라운드에서 실행.**
- **submit→rid→await API**: `PyRblnAsyncRuntime.run(inputs,outputs)->uint64_t rid`, `await_task(rid,timeout)`
  (`AsyncTaskQueue.EnqueueTask/DequeueTask` + `AsyncResultMap.NotifyFinished/WaitForCompletion`).
- **Context는 worker들이 공유(thread-safe)** — driver context 스레드 소유 문제 이미 해결돼 있음(cross-thread submit
  검증된 셈). ★ 유저 걱정("context를 exec 스레드가 소유")은 이미 이렇게 구현됨.
- 테스트·프로덕션 사용: `test_async_runtime.py`(parallel 1/2 = 더블버퍼), `bucketing_runtime.py`.
- **decode엔 `num_thread_=1`**: 직렬 walk = 우리가 설계한 "exec 스레드 1개" 그 자체(N+1 KV 의존성·seq 체이닝
  보존). 더블버퍼가 필요하면 2로.

### 진짜 작업 = sync에는 있고 async엔 없는 기능 이식 (vLLM decode 필수)
async runtime이 아직 지원 안 하는 것(= 착수 시 이식 대상, 스코프 명확):
| 기능 | vLLM decode에 필요한 이유 | 우선순위 |
|---|---|---|
| `CopyKvCache/FetchKvCache/UpdateKvCache` | paged KV cache — decode 필수 | ★블로커 |
| `UpdateInputAddr/UpdateOutputAddr` | block 주소 재배치(paged attn I/O) | ★블로커 |
| `Begin/EndIOPatchBatch` | I/O patch DMA coalescing (vllm-rbln 사용중) | ★블로커 |
| `ApplyExternTransforms/ApplyHostParamsOnly` | weight streaming/host param patch | 모델따라 |
| Profiler | **overlap 검증(perfetto)** — 현재 async는 명시적 금지 | 검증에 필요 |
| `EnableExecuteHostOnly` | 디버그 | 낮음 |

### torch_rbln 배선
- `DynamoRuntime`(torch_rbln, 현재 `PyRblnSyncRuntime.{PrepareInputs,PrepareOutputs,Run}` 직접 호출)를
  **async runtime 경로로 전환**: forward에서 `run()`→rid 즉시 리턴, `get_output()`/`synchronize`에서 `await_task(rid)`.
- 경로: `vLLM model_executable() → torch.compile → torch_rbln DynamoRuntime → PyRblnAsyncRuntime`.

### vllm-rbln
- 입력 스테이징 더블버퍼(N+1 prepare가 N in-flight 입력 안 덮게). async runtime `num_thread_=2`(2 instance=2 buffer)로
  런타임 레벨에서 흡수될 수도 있으니 이식 후 확인. output은 `AsyncRBLNModelRunnerOutput`로 이미 deferred.

### 결과
forward walk가 exec 스레드로 → Python worker 즉시 리턴 → **기존 async scheduling이 execute_model(N+1)
전체(all_reduce 포함)를 forward(N) walk와 overlap.** all_reduce 100% hidden. host op 그대로(정합성 동일),
컴파일러 무변경, vLLM 엔진 무변경(all_reduce worker 유지), speculation·barrier 문제 전무. 보너스: N+1 input
prepare 전체가 숨음(GPU와 동일 이득 구조).

### 리스크 (검증 필요, 우선순위 순)
1. **★ async runtime 기능 이식 범위/정확성**: KV cache·IO addr patch·io_patch_batch를 async worker 경로에서
   정확히 동작시키기(sync `RuntimeInstance`엔 있으나 async 경로 검증 안 됨). decode 정합성의 핵심 — 이식 후
   0-mismatch 확인 필수.
   ~~driver context thread-affinity~~ → **해소**: async runtime이 이미 공유 Context를 worker 스레드들에서
   thread-safe하게 씀(cross-thread submit 프로덕션 검증됨).
2. **입력 버퍼 수명/더블버퍼 정확성**: overlap 중 버퍼 재사용 레이스. `num_thread_=2`(2 instance=2 buffer)로
   런타임이 흡수하는지, 아니면 vllm-rbln 스테이징 더블버퍼 필요한지 이식 후 확인.
3. **profiler 부재**: async runtime은 현재 profiler 금지 → 초기 overlap 검증은 host-side span 계측(기존 tooling)
   으로, 이후 async profiler 지원 추가해 perfetto 확인.
4. **decode 순서/ordering**: `num_thread_=1`로 직렬(autoregressive KV 의존성 보존). 2 이상 시 N+1이 N보다 먼저
   돌지 않게 의존성 보장 필요.
3. host op이 slot을 device op과 공유 → exec 스레드 내 drain 순서가 정합성 보장(현행 그대로라 유지됨).
4. profiling: 적용 후 perfetto에서 all_reduce가 forward walk와 실제 겹치는지 확인.

### (보조·선택) 컴파일러/런타임 미세 최적화 — 주경로 아님
- host op 앞 drain을 `EnsureAllTasksCompleted()`(전체) → 해당 host op 입력 seq만(`SeqState.in_deps`)으로
  축소하면 walk 자체의 wall-time도 단축(단 Python worker는 이미 exec 스레드로 풀렸으므로 overlap엔 불필요,
  forward 절대시간만 개선).
- heuristic pruning으로 host에 간 op을 device로 되돌리는 건 forward 절대시간엔 도움되나, overlap 목표엔
  주경로(exec 스레드)만으로 충분하므로 후순위.

---

## 3. 기능 gap 판정 결과 (2026-07-03, 코드 확인 완료) + 갈림길

async 인프라는 **이미 ~70% 존재**: async runtime(C++ `async_runtime.cc`, worker 풀+공유 thread-safe Context)
+ Python 래퍼(`async_runtime.py`: prepare→`run(inputs,outputs)`→rid→`await_task`) + `non_blocking_mode`
플러밍(`compiled_model.py:302-360`) + `create_async_runtime`.

**빠진 것 = 배선뿐**: torch.compile 백엔드(`core/torch_compile.py:431-492`)가 현재 **sync만**
(`create_runtime`→`DynamoRuntime`(sync_runtime.py)). async용 DynamoRuntime + torch_compile 분기 + vLLM await 필요.

**gap 난이도 (에이전트 조사)**: KV cache(Copy/Fetch/Update/GetSize) = 로직이 공유 `RuntimeBase`에 있어 async
pybind 노출만 하면 **EASY**. UpdateInput/OutputAddr = async `run(inputs,outputs)`가 매 호출 포인터 전달로
**subsume(대체) 가능성 큼**. io_patch_batch = 단일스레드 batching 가정이라 HARD지만 addr patch 안 쓰면 **불필요**.
profiler = 구조적 제약(정합성 아닌 검증용) → host-span으로 우회.

### ★ 결정적 갈림길 (실험으로만 판정)
sync forward = `begin_io_patch_batch → prepare_inputs(device_inputs,cpu_inputs)[주소 CS-buffer patch] →
prepare_outputs → end_io_patch_batch → run()`. async forward = `prepare_inputs → run(inputs,outputs 포인터)`
(**io_patch/device-side prepare 없음**). KV block 주소는 입력 텐서 `data_ptr()`.

**Q: async의 포인터 전달이 sync의 io_patch 주소 patch를 대체하는가?** (컴파일 그래프가 KV를 넘긴 포인터에서
직접 읽나 vs CS-buffer patch된 주소로 읽나)
- **(a) 대체됨 → EASY**: 순수 배선(torch_compile async 분기 + async DynamoRuntime + vLLM await). C++ 거의 불필요.
- **(b) 안 됨 → HARD**: io_patch/UpdateInputAddr을 멀티스레드 안전하게 async 경로로 이식.

소스만으론 판정 불가. **실험 = 구현 첫 조각**(async 인프라가 이미 있으니).

## 3.5 ★ 실험 결과 (2026-07-03, 온디바이스 실측) — 갈림길 = (b), 원인 정밀 특정

배선 완료 후 소형 baked-weight 모델(Linear stack)로 sync vs async(`RBLN_DYNAMO_ASYNC=1`, 즉시 await) 출력
비교(`scratchpad/test_async_dynamo.py`). **결과: MISMATCH (max_abs_diff ~0.5–0.85, fp16 noise 훨씬 초과).**

**원인 (코드로 특정, airtight)**: rbln 텐서의 `data_ptr()`는 torch_rbln **vmem 가상주소(vaddr)**다. sync
`DynamoRuntime`는 `PrepareInputs`(pyrbln_impl/runtime.cc:146)에서 device 입력마다 vmem 변환을 한다:
`set_device_alloc → ensure_synced_on_device(vaddr) → get_device_addrs(vaddr) → runtime_.UpdateInputAddr(idx, device_addrs)`
(device-op 입력) 또는 host physical view ptr(host-op 입력). 즉 vaddr→**물리 device 주소**로 변환 + 그래프
CS-buffer에 relocation. 그런데 async 경로는 worker가 `RuntimeInstance::SetInput(i, inputs[i])`
(runtime_instance.cc:1115)로 **vaddr를 변환 없이 그대로** 바인딩(또는 host memcpy) → 그래프가 미변환 주소로
실행 → 틀린 결과. **fork 원인은 io_patch DMA coalescing이 아니라 vmem view 변환 + UpdateInputAddr 누락.**

**Python-only 불가 확인**: vmem primitive는 `rebel._C.vmem`에 노출됨(`ensure_synced_on_device`,
`get_device_addrs`, `get_pv_host_ptr`, `set_device_alloc`). 하지만 device-op 입력에 필수인
`UpdateInputAddr`(CS-buffer relocation)는 C++ 전용이고 `PyRblnAsyncRuntime`에 없음. SetInput은 device-op
입력을 memcpy로 처리(`IsSafeToUseOuterPtrForInput`가 device-op면 false)해 물리주소를 줘도 부정확. → **C++ 수정 필수.**

## 3.6 ★ 다음 스텝 = C++ (빌드: `source ~/.venv/bin/activate && cd ~/rebel_compiler && ./rebel_install.sh -a -n`)
> Python은 editable(즉시 반영)이나 C++ `_C.so`는 prebuilt → C++ 변경은 위 스크립트로 재빌드. skill: `rebel-compiler-build`.

**추가 제약(코드 확인)**: async worker(async_runtime.cc:16)는 `nin = executor_->inputs().size()`(고정)로
`for i<nin: instance->SetInput(i, task.inputs[i])` → **task 리스트를 비워 "run-prepared"하는 꼼수는 OOB crash**.
그리고 device-op 입력은 `SetInput`이 host memcpy 처리(`IsSafeToUseOuterPtrForInput`가 false)라 물리주소를 줘도
부정확 → **worker가 device 입력엔 `UpdateAndRelocateIOAddress`를 써야** 함.

**따라서 채택안 = (ii) worker 확장** (여러 파일 조율):
1. **Python `AsyncDynamoRuntime`**: 각 rbln 입력의 vaddr를 `rebel._C.vmem`으로 물리 device_addrs 해석
   (`set_device_alloc?`→`ensure_synced_on_device(vaddr)`→`get_device_addrs(vaddr)`), host 입력은 host ptr.
   이 (idx→device_addrs / idx→host_ptr / 출력 동일) 정보를 새 async run에 전달.
2. **AsyncTask/EnqueueTask/AsyncRuntime::Run**: `auxiliary_data_`(이미 존재) 등으로 per-input device_addrs +
   device/host 플래그를 실어 전달.
3. **worker 루프(async_runtime.cc)**: 각 입력에 대해 device면 `instance->UpdateAndRelocateIOAddress(i, addrs, true)`,
   host면 `instance->SetInput(i, ptr)`. 출력도 동일 분기. 그 뒤 `instance->Run()`.
4. **pybind(compiled_model.cc:127-144)** + **PyRblnAsyncRuntime::Run 시그니처** 확장.
5. **config**: sync는 `SetDeviceAllocConfiguration(vaddr, cached_config)`를 매번 호출(cached_input_configs_는
   ctor에서 executor input config로 채움). async도 동일 캐싱 필요할 수 있음 — vmem set_device_alloc 경로 확인.
- 권장 진행: **num_thread_=1 고정**(LLM 기본 parallel=1)으로 시작 → decode 0-mismatch(즉시 await) → `defer`로 overlap.
- 검증: `scratchpad/test_async_dynamo.py`(소형 모델 sync=async 0-diff)부터 통과시키고 vLLM decode로 확대.

**현재 상태**: C++ 미변경(빌드·기본 경로 안전). Python 배선(AsyncDynamoRuntime + torch_compile 분기)은 게이트
off로 존재하나, 위 worker 확장 전엔 정합성 X(실험이 그걸 입증). worker 확장이 들어가면 이 Python 배선이 그대로 맞물림.

## 3.7 ★ C++ 구현 완료 (2026-07-03) — 빌드 플러밍만 남음
async 경로에 sync의 vmem+relocation을 이식 완료. worker가 device 입력은 `UpdateAndRelocateIOAddress`,
host 입력은 `SetInput`을 하도록 확장하고, vmem 해석(vaddr→물리주소)은 PyRblnAsyncRuntime::RunIO에서 sync
PrepareInputs/PrepareOutputs 로직을 그대로 재사용해 수행 후 태스크에 실어 보냄.

**변경 파일(모두 gated·additive, 기존 경로 무변경)**:
- `rebel/include/rebel/runtime/async/async_task_queue.h` + `src/.../async_task_queue.cc`: AsyncTask에
  per-index `device_input_addrs_`/`device_output_addrs_` + EnqueueTask 오버로드.
- `rebel/include/rebel/runtime/core/async_runtime.h` + `src/.../async_runtime.cc`: Run 오버로드(host ptr +
  device addrs) + worker 루프 분기(device→UpdateAndRelocateIOAddress, host→SetInput, nullptr skip, io_patch batch).
- `rebel/include/rebel/pyrbln_impl/compiled_model.h` + `src/pyrbln_impl/runtime.cc`: PyRblnAsyncRuntime에
  executor_·cached configs·`RunIO(device_inputs,cpu_inputs,device_outputs,cpu_outputs)`(vmem 해석).
- `rebel/src/pyrbln/compiled_model.cc`: pybind `run_io` 노출.
- `rebel/python/rebel/sync_runtime.py` `AsyncDynamoRuntime`: device/cpu 맵 만들어 `run_io` 호출(+즉시/defer await).
- `rebel/python/rebel/core/torch_compile.py`: `_build_dynamo_runtime`(env `RBLN_DYNAMO_ASYNC`).

**빌드 (확정)**: librbln.so는 `./rebel_install.sh -a -n`(~/.venv)로 빌드. pybind `_C`는 vllm venv에서
**`source ~/codebase/vllm-executor/.venv/bin/activate && uv pip install -e ~/codebase/rebel_compiler/python`**
(26s). rebel_install.sh의 _C 자동재빌드는 게이트(`pip show rebel-compiler`)로 스킵되므로 이 uv 명령이 정석.

**★ 검증 결과 (2026-07-03) — 정합성 PROVEN**: `scratchpad/test_async_dynamo.py`로 소형 baked-weight 모델
sync vs async(`RBLN_DYNAMO_ASYNC=1`, 즉시 await): **전 스텝 max_abs_diff=0.000e+00, PASS.** worker의
`UpdateAndRelocateIOAddress` + `RunIO` vmem 해석이 정확 → **async I/O 경로가 device I/O를 정확히 전달**(fork 해소).

**다음**: (1) `defer` 모드 overlap — naive 테스트는 output 즉시 읽어 defer-unsafe라, vLLM
`AsyncRBLNModelRunnerOutput`(output 이미 deferred)와 결합해 검증. (2) vLLM decode 통합→0-mismatch→host-span으로
all_reduce↔forward overlap 확인. (3) weight-free(gpt-oss-120b) 미지원 → weight streaming을 async에 이식 필요.

## 4. 구현 순서 (참고 — 실험으로 (b) 판정되어 아래 1은 완료, 2가 C++ 의존)
1. **실험겸 배선 (Python, C++ 빌드 불필요)**: `core/torch_compile.py`에 async 분기(env gate, 예 `RBLN_DYNAMO_ASYNC=1`)
   + async용 DynamoRuntime(`run()`: prepare inputs → output 텐서 alloc → `rid=handle.run(in_ptrs,out_ptrs)` →
   rid stash 후 output 텐서 즉시 반환). decode 소규모 run으로 **0-mismatch 확인** → 갈림길 판정.
2. (a)면: vLLM `AsyncRBLNModelRunnerOutput.get_output`에 `await_task(rid)` 배선 + 입력 더블버퍼(또는 `num_thread_=2`
   흡수 확인) → host-span으로 all_reduce↔forward overlap 확인.
   (b)면: async 경로 io_patch/addr-patch 이식(C++, thread-safety) 후 1로 복귀.
3. KV pybind(async 노출)는 forward 경로엔 불필요(forward가 copy_kv_cache 호출 안 함); block 관리용으로 vLLM이
   별도 호출하면 그때 노출(EASY).

## 근거 파일
- L1: `runtime_instance.cc:{781,871,1039,1187,1197}`, `command_dispatcher.cc:{46,68}`, `base/stream.h`, `seq_state.h`
- L2: `Partition/Partition.cpp:{63-69,137-187,480-571}`, `RblnTensorOpsInterfaces.cpp:{785,994,1504}`, `PostPartition.cpp:245`
- L3: `pyrbln/compiled_model.cc:50`, `rbln_model_runner.py:{206-249,3761}`, `torch_rbln .../device/device.py:87`
