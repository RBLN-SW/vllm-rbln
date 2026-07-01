# async overlap 작업 — 서버 이주용 HANDOFF (fresh agent 바로 착수용)

> 이 파일 하나로 **다른 서버에서 처음 접하는 agent가 바로 이어서 작업**할 수 있도록 정리했다.
> 배경/설계 심화는 같은 폴더의 `async_overlap_report.md`(근본원인·정합성), `async_overlap_deferred_design.md`(deferral 설계)를 참조. 이 파일은 **현재 상태 + 다음 액션 + 검증 레시피**에 집중한다.
> 최종 갱신 2026-07-01.

---

## 0. 목표 한 줄
gpt-oss-120b EP+DP4, device-tensor 경로에서 매 decode step 도는 DP `gloo:all_reduce`(~2.5ms, 호스트 CPU)를 NPU `forward`(~5ms) 실행 구간에 **겹쳐(overlap)** step latency 를 줄인다.

## 1. 현재 상태 (branch `async-overlap-prototype`, origin push됨)

리포지토리: `git@github.com:RBLN-SW/vllm-rbln.git` branch **`async-overlap-prototype`** (base `dev`=`fb6b687b`).

| 커밋 | 내용 | 상태 |
|---|---|---|
| `3e388a07` | docs: overlap 분석 리포트 | — |
| `5d5dc98` | feat: 낙관적 `RBLNAsyncScheduler` wiring (batch_queue depth 2) | 검증됨(과거) |
| `9222f46` | feat: device-tensor `AsyncRBLNModelRunnerOutput` (`torch.rbln.synchronize`) | 검증됨 |
| `793a94b` | fix: `prev_sampled_token_ids` 매-step 리셋 (§4-1) | 검증됨 |
| `fa7b2c2` | fix: device-side 토큰 피드백 (§4-2) | 검증됨 |
| `e48c8cd` | chore: GIL probe (`VLLM_RBLN_GIL_PROBE=1`) | — |
| `87b8c0f` | feat: **C7** `_DeviceForwardExecutor` (forward를 device 스레드에서 실행, 즉시 join) | **✅ 6-layer A/B 0 mismatch** |
| `929be43`,`6caa1ef` | docs: deferral 설계 + C9 난관/eager_out 가능성 | — |
| `526854a` | refactor: **C8** `set_forward_context`를 `_run_forward` 안으로 | **✅ 6-layer A/B 0 mismatch** |
| `c80de05` | chore: **C9 eager_out 프로브** (`VLLM_RBLN_EAGEROUT_PROBE=1`) | — |
| `8cd86dd` | fix: executor 스레드 `torch.rbln.set_device` (device context) | ✅ 검증됨 (2026-07-01, 아래) |
| `4a5e961` | fix: **warmup 중 executor 우회**(`is_warmup_active` 게이트) | ✅ 검증됨 |
| `9561d02` | fix: device-forward executor **lazy 생성**(post-warmup) | ✅ 검증됨 |
| `b9ce798` | fix: **probe pre-alloc dtype**(int32 (B,1)→int64 (B,)) — argmax op에 맞춤 | ✅ (프로브 crash 해소) |
| `96acef3` | feat: **C9a** forward deferral (fast path, join in sample_tokens) | ✅ 6-layer 0 mismatch |
| `d55098b` | feat: **C9b** sampler를 device 스레드로 → all_reduce↔forward overlap | ✅ 6-layer 0 mismatch |

**정합성·vmem fix·C9 deferral 모두 검증 완료**(전부 0 mismatch). 남은 건 **overlap 정량**(full-layer `--profile`, C10) 뿐 — deferred_design.md 하단 참조.

> **✅ 최신 상태(2026-07-01, exclusive box에서 실측 완료)**: 아래 세 가지 전부 확인됨 —
> 1. **vmem fix 3단계 검증**: `VLLM_RBLN_ASYNC_FORWARD=1`(프로브 없이)로 warmup+generation 완주, `verify.cc:77` 크래시 없음. → `8cd86dd`+`4a5e961`+`9561d02` 유효.
> 2. **ASYNC_FORWARD 정합성**: sync baseline 대비 **16 prompts 0 mismatch**.
> 3. **EAGEROUT alias 판정 = `alias=False`** (구조적). 원 프로브가 `verify.cc:77`로 크래시했던 건 async가 아니라 **프로브 pre-alloc dtype 버그**(int32 (B,1) vs argmax의 int64 (B,))였음 — `b9ce798`로 수정. 수정 후: eager_out은 컴파일된 greedy_sample op(=`rbln::argmax`, (B,) int64) 출력은 pre-alloc 가능하나, **최종 `sampled_token_ids`는 `forward()`가 op 뒤에서 항상 `sampled.to(int32).unsqueeze(-1)`로 새 버퍼를 만들어** 절대 alias되지 않음(rbln_sampler.py:313,320). → **naive path (가) 불가**, §3/deferred_design 참조.
>
> **주의**: 이 실측은 vllm-rbln `async-overlap-prototype`에 **dev merge된 상태**(`828d06b`)에서 됨. dev merge 전 HEAD(`9561d02`)에서는 gpt-oss-120b MoE MXFP4가 `rtosa.add_n` shape verify로 컴파일 크래시했음(rebel_compiler dev와의 궁합) — **dev를 merge/rebase한 최신 브랜치를 쓸 것.** 또한 venv에 `torch_rbln`(rbln device backend 등록)이 설치돼 있어야 함.

## 2. 지금 당장 할 일 (다른 서버에서)

### (A) 환경 전제
- 다른 RBLN dev 박스 (8×RBLN-CR03 급), `~/codebase/vllm-executor` (parity 하네스 + venv), gpt-oss-120b 접근.
- **박스가 exclusive여야 함** (§5 참조 — 동시 DP4/EP RCCL 배타). `ps -eo user,args|grep VLLM::` 로 남의 잡 없고 `rbln-stat` free NPU≥4 확인.
- 코드 반영: 이 브랜치는 `vllm-rbln` editable install 이므로 `git checkout async-overlap-prototype` 후 바로 반영됨(재설치 불필요). 확인: `python3 -c "import vllm_rbln"` 경로가 이 repo인지.

### (B) 공통 env
```bash
cd ~/codebase/vllm-executor && source .venv/bin/activate
export VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error VLLM_RBLN_AUTO_PORT=1
export RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1
export RBLN_VERBOSE=2 SPDLOG_LEVEL=warning VLLM_LOGGING_LEVEL=INFO
export RBLN_DEVICES=0,1,2,3        # 빈 NPU 4개로
```

### (C) STEP 1 — sync baseline 생성 (golden, ~15분 컴파일 포함)
```bash
rm -f ~/.cache/vllm-rbln-exec/rbln_results_*L6_T16*P4_*
VLLM_RBLN_DISABLE_ASYNC=1 python3 -m vllm_rbln_exec.parity_runner --task r --model gpt-oss-120b \
  --ep --dp 4 --rsd 1 --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 --batch 1 \
  --num-hidden-layers 6 --max-num-blocks 129 --cache-results --cache-ignore --max-tokens 16 --num-prompts 4
cp ~/.cache/vllm-rbln-exec/rbln_results_*L6_T16*P4_*.json /tmp/baseline_sync.json
```
주의: 파일명 glob `*L6_T16*P4_*` 는 `DP4_` 로 매칭됨(실제 P16 파일). parity `exit=1` 은 **양성**(생성 후 teardown). "Generation complete"/"Cached RBLN results" 로그 확인.

### (D) STEP 2 — **C9 eager_out 프로브 실행** (다음 액션의 핵심)
```bash
rm -f ~/.cache/vllm-rbln-exec/rbln_results_*L6_T16*P4_*
VLLM_RBLN_OPTIMISTIC_SCHED=1 VLLM_RBLN_ASYNC_FORWARD=1 VLLM_RBLN_EAGEROUT_PROBE=1 \
python3 -m vllm_rbln_exec.parity_runner --task r --model gpt-oss-120b --ep --dp 4 --rsd 1 \
  --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 --batch 1 \
  --num-hidden-layers 6 --max-num-blocks 129 --cache-results --cache-ignore --max-tokens 16 --num-prompts 4 \
  2>&1 | tee /tmp/c9probe.log
grep "EAGEROUT_PROBE alias" /tmp/c9probe.log | head
```
이 한 번의 실행이 **두 가지를 동시에** 준다: (1) `ASYNC_FORWARD=1` 경로가 **warmup vmem 크래시 없이 generation 완료** → §5 fix 2개 검증, (2) `EAGEROUT_PROBE alias` 값 → C9 방향. 만약 `vmem_size ... verify.cc:77`가 또 뜨면 fix가 불완전한 것이니 §5 재검토(스레드 device context / warmup 경계).

**판정** (프로브는 `_sample`에서 sampler 출력을 `EagerExecutionHelper.set_out_tensor`로 pre-alloc 후 alias 여부 로깅):
- `alias=True` → **rebel eager_out가 sampler 출력을 pre-alloc 텐서로 alias함** → 메인 스레드가 device 실행 전 핸들 선확보 가능 → **path (가) 확정, C9 구현 진행**(§3).
- `alias=False` → sampler 출력이 eager_out로 안 잡힘 → §3 (나)안(bookkeeping을 device 스레드로 + input_batch 동시성) 검토.
- crash(shape/count) → pre_shape 로그 보고 pre-alloc shape 조정(현재 `(logits.shape[0],1)` int32).
그리고 이 프로브 실행의 token_ids를 baseline과 비교해 pre-alloc이 정합성 안 깨는지도 확인(STEP 4 compare).

### (E) STEP 3 — C9 구현 (deferral, overlap 개방)
`async_overlap_deferred_design.md` 의 "C9로 가는 두 갈래" + eager_out 확정 결과대로. 요지:
- device 스레드(=`_DeviceForwardExecutor`, 이미 있음)가 step당 FIFO로 **fwd_task**[토큰피드백+forward+logits] → **sample_task**[sampler+bookkeeping] 실행.
- `execute_model`/`sample_tokens`는 device 안 기다리고 **즉시 리턴**(future/AsyncOutput) → 메인 스레드가 EM(N+1)의 host prep+all_reduce를 forward(N)와 **동시** 실행.
- sampler 출력은 **메인에서 pre-alloc**(eager_out)해 핸들 확보 → `_bookkeeping_sync`는 핸들만 쓰므로(async 모드에서 값 안 읽음, `[-1]` placeholder) 메인 스레드 순차 처리 → **input_batch race 없음**. D2H 실제 복사는 `AsyncRBLNModelRunnerOutput.get_output`(별도 `async_output_busy_loop` 스레드)로 지연.
- **forward_context**: 이미 C8에서 `_run_forward` 안으로 옮김. deferral 시 메인은 forward_context 안 건드림(§ deferred_design 하위문제 1).
- 토큰 피드백을 fwd_task 안(device 스레드)으로 이동(현재는 `execute_model` 메인, `rbln_model_runner.py` "Async-scheduling token feedback" 블록) → FIFO상 prev sampler 뒤라 순서 보장.
- **스코프**: decode-only + no-spec + no-grammar + no-KV-connector fast path만 deferral, 나머지는 C7 즉시-join fallback.

### (F) STEP 4 — 검증 (매 커밋)
정합성 A/B (C-경로 1회 + baseline 비교):
```bash
python3 - /tmp/baseline_sync.json /tmp/c9probe.json <<'PY'
import json,sys,os
b=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2]))
def t(d):
 it=d if isinstance(d,list) else d.get("outputs",d)
 return [list(o.get("token_ids") or []) if isinstance(o,dict) else [] for o in it]
tb,tc=t(b),t(c); n=min(len(tb),len(tc)); mm=sum(1 for i in range(n) if tb[i]!=tc[i])
print(f"prompts={n} mismatches={mm}", "IDENTICAL ✅" if mm==0 and n>0 else "DIFF ❌")
PY
```
(c9probe.json = 위 STEP2/C-경로 실행이 캐시에 쓴 `rbln_results_*L6_T16*P4_*.json` 복사본)

최종 overlap 정량 (full-layer, `--num-hidden-layers` 빼고 `--profile` 추가):
- 트레이스 `profile/rbln_*_dp{r}/*.pt.trace.json.gz` 에서 `gloo:all_reduce` ∩ `rebel/sync_run`(forward) 교집합 시간 > 0 확인. 정량 스크립트는 `async_overlap_report.md` §9.
- 목표: 현재 0% → >0%. cadence(c10d start-to-start 중앙값) 감소도 확인.

## 3. 핵심 파일:라인 (이 브랜치 기준)
- forward executor + `_run_forward`: `vllm_rbln/v1/worker/rbln_model_runner.py` — `_DeviceForwardExecutor`(class), `__init__`의 `self._device_executor`, `execute_model`의 `_run_forward()`/executor 분기.
- eager_out 프로브: 같은 파일 `_sample()` 의 `VLLM_RBLN_EAGEROUT_PROBE` 블록.
- 토큰 피드백(deferral 시 이동 대상): `execute_model`의 "Async-scheduling token feedback" 블록.
- bookkeeping(핸들만 사용 확인): `_bookkeeping_sync()` — async 분기에서 `prev_sampled_token_ids = sampled_token_ids`(no-sync), `sampled_ids=[-1]` placeholder.
- AsyncOutput: `AsyncRBLNModelRunnerOutput` (get_output = `torch.rbln.synchronize`; deferral 시 device-thread future 대기로 변경).
- 스케줄러/게이트: `vllm_rbln/platform.py`(`VLLM_RBLN_OPTIMISTIC_SCHED`, `VLLM_RBLN_DISABLE_ASYNC`), `vllm_rbln/v1/core/rbln_scheduler.py`(`RBLNAsyncScheduler`).
- rebel eager_out: `~/codebase/rebel_compiler/python/rebel/core/torch_eager.py`(`EagerExecutionHelper.set_out_tensor`), `sync_runtime.py:247-256`(`use_eager_out`), `run()`이 GIL 놓음: `rebel/src/pyrbln/compiled_model.cc:50`.

## 4. env 플래그 레퍼런스
| env | 효과 |
|---|---|
| `VLLM_RBLN_OPTIMISTIC_SCHED=1` | `RBLNAsyncScheduler` (batch_queue depth 2). overlap 전제. |
| `VLLM_RBLN_ASYNC_FORWARD=1` | forward를 `_DeviceForwardExecutor`(device 스레드)로. 현재 즉시-join(C7/C8). C9에서 deferral. |
| `VLLM_RBLN_DISABLE_ASYNC=1` | sync baseline (A/B용). |
| `VLLM_RBLN_GIL_PROBE=1` | forward 중 GIL 여유 측정(디버그). |
| `VLLM_RBLN_EAGEROUT_PROBE=1` | sampler 출력 eager_out pre-alloc alias 확인(C9 de-risk). |

## 5. 이미 잡은 실제 코드 버그 + 환경 경합 (반드시 숙지)

### (해결됨) C7 executor 스레드 device context 버그 → vmem 실패
- 증상: `VLLM_RBLN_ASYNC_FORWARD=1` 일 때 init/warmup 에서 `RUN_INTERNAL (vmem_size >= transform.op_replay()[0].src_elem_count()... verify.cc:77)` / `register device ID: 0`.
- bisect 로 확정: plain·OPTIMISTIC-only 는 정상 generation, **ASYNC_FORWARD(=`_DeviceForwardExecutor`)만** vmem 실패 → **내 코드 버그**(환경 아님, device 번호 무관).
- 원인: executor **스레드가 rbln device context 를 안 물려받아** device 0 으로 폴백 → weight-free transform vmem 을 엉뚱한 device 에 할당.
- 수정(커밋됨): `_DeviceForwardExecutor._run` 에서 **첫 task 실행 시 lazy 하게** `torch.rbln.set_device(worker_device_index)` 호출. (스레드 생성 시점엔 device 미등록이라 `SYS_ENODEV` → lazy 필수. `current_platform.set_device` 는 RBLN no-op 이라 `torch.rbln` API 직접 사용.) vLLM `async_output_busy_loop`(multiproc_executor.py:929-938)가 같은 이유로 스레드에 set_device 하는 것과 동일 패턴.
- **교훈: device 스레드에서 device 작업 하려면 반드시 그 스레드에 device context 설정.** C9 에서 sampler/bookkeeping 을 device 스레드로 옮길 때도 동일하게 필요. (검증은 exclusive box 대기 중이었음 — 새 박스에서 STEP 2 로 확인.)

### 환경 경합
- 이 박스류는 **공유**. **동시에 두 DP4/EP 잡 불가** — 두 번째의 RCCL init 이 `[Rccl] fail rcclCommInitRank ret=-12`(collective 배타 자원). 남이 DP4/EP 돌리면 어느 device 잡아도 무의미 → **박스 exclusive 필요**(그래서 서버 이주).
- free window 도 모델로드~register ~12s 사이 `SYS_ENODEV` TOCTOU 로 뺏길 수 있음. `ps -eo user,args|grep VLLM::` 로 남의 잡 0 확인 후 실행.
- 내 좀비/leak 정리: `kill -9` **내** `VLLM::` 프로세스만(`pkill -f` 는 `VLLM::` 타이틀 못 잡음), `find /dev/shm -maxdepth 1 -user $USER -delete`. 남의 프로세스 절대 금지.
- offload 캐시(`~/.cache/rbln_cache/offload`) 오염은 vmem 원인 **아님**(지워도 재발했음). 위 device-context 가 진짜 원인이었다.

## 5.5 병렬 작업(다른 서버) 시 주의
- **코드/문서는 이 브랜치(`async-overlap-prototype`)에 전부 push됨** — 다른 서버에서 `git fetch && git checkout async-overlap-prototype` 하면 그대로 쓸 수 있다. 별도 준비물 없음(이 3개 docs + 코드).
- 단 **repo 밖 전제**는 새 서버가 갖춰야 함: `~/codebase/vllm-executor`(parity 하네스 + editable venv), gpt-oss-120b 접근, **exclusive한 RBLN 8-NPU 박스**(§5 RCCL 배타). scratchpad의 baseline/스크립트는 세션-로컬이라 안 옮겨짐 → STEP 1~4 명령이 문서에 embed돼 있으니 그걸로 재생성.
- **브랜치 충돌 방지**: 다른 서버 agent가 *검증만* 하면(STEP 1~2 실행, alias 읽기) commit 불필요 → 충돌 없음. *C9 구현까지* 하면 이 브랜치에 push하게 되므로, **원 세션과 동시 push 금지** — 둘 중 하나만 구현하거나, 새 서버는 `async-overlap-prototype-2` 같은 분기 브랜치에서 작업 후 나중에 머지. (fetch로 최신 받고 시작할 것.)

## 6. 요약: fresh agent 체크리스트
1. `git checkout async-overlap-prototype` (vllm-rbln), editable install 확인.
2. 박스 exclusive 확인(§5).
3. STEP 1: sync baseline 생성 → `/tmp/baseline_sync.json`.
4. STEP 2: `VLLM_RBLN_EAGEROUT_PROBE=1` 실행 → `alias=True/False` 판정.
5. STEP 3: 판정대로 C9 deferral 구현(§2E, deferred_design.md), decode-only 스코프, 커밋 단위로.
6. STEP 4: 매 커밋 6-layer A/B 0 mismatch → 최종 full-layer `--profile`로 overlap>0.
