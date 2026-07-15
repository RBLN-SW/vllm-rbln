# Async-overlap — open work

`RBLN_DYNAMO_ASYNC=defer` overlaps the per-step DP `num_tokens` `all_reduce` with the
previous step's NPU forward (greedy argmax deferred one step). Gate off ⇒ dev sync behavior.

Functionality is validated by the **REBEL DevTensor CI**; the implementation is
self-describing in code, so this doc keeps only the **open work** and how to run it. Key files:
- `vllm_rbln/v1/worker/rbln_model_runner.py` — defer gate, `_drain_prev_output`, `sample_tokens`
- `vllm_rbln/torch_compile_backend.py` — `_assert_warmcache_async_safe`
- `rebel_compiler/python/rebel/sync_runtime.py` — `AsyncDynamoRuntime`, `_PENDING_ASYNC`, `force_sync`

---

## How to run

```bash
cd ~/codebase/vllm-executor && source .venv/bin/activate
# Pick an idle device set (rbln-smi shows 0.0B). BETWEEN RUNS, clean up or the next run flakes:
#   find /dev/shm -maxdepth 1 -uid $(id -u) -delete   # + confirm rbln-smi back to 0.0B
export VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error \
  VLLM_RBLN_AUTO_PORT=1 RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 \
  VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1 \
  SPDLOG_LEVEL=warning RBLN_VERBOSE=warning VLLM_LOGGING_LEVEL=WARNING RBLN_DEVICES=4,5,6,7
ARGS="--task r --model gpt-oss-120b --ep --dp 4 --rsd 1 --trust-remote-code \
  --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 \
  --batch 1 --max-num-blocks 129 --max-tokens 128 --num-prompts 4 \
  --use-cached-models --cache-results --cache-ignore --repro-run 3 --skip-validation --mode 0"

# sync reference
python3 -m vllm_rbln_exec.parity_runner $ARGS
# async overlap
RBLN_DYNAMO_ASYNC=defer VLLM_RBLN_OPTIMISTIC_SCHED=1 python3 -m vllm_rbln_exec.parity_runner $ARGS
```

Flags that matter here:
- `--use-cached-models` → `.rbln` compile cache on (`VLLM_CACHE_ROOT=~/.cache/vllm-rbln-exec/compiled_results`).
  **Omit it to disable the compile cache** (recompile every launch) — needed for TODO 3.
- `--cache-ignore` → ignore the cached *result* JSON, actually re-run inference.
- `--repro-run N` → after the main run, re-generate N times in-process and assert identical (determinism check).
- A `.rbln` cache **hit vs recompile** is read from the log, not the file count: a cold compile emits
  the big `progress total size mismatch total=19461, counter=71074` lines (one per graph); a cache
  hit has **zero** of them (only a small `total=310, counter=948` warmup graph, built every launch).

---

## TODO 1 — async aborts frequently; sync never does  [P0]

**Observed (full gpt-oss-120b, DP4+EP, batch=1, tok128, logprobs=16, devices 4-7):**
async `defer` **aborted 4 of 6 launches**; sync (same config, `RBLN_DYNAMO_ASYNC` unset)
was **3/3 clean, cross-launch bit-identical, repro-identical**.

**Failure signature (always rank 2):**
```
code=504 SYS_TASK_ABORTED "Wait job task aborted (code 3)" / "[stream] Drain failed, seq=…"
  then rid N..M all: code=201 INIT_INTERNAL "BeginBatch called while a batch is already active"
  surfaced on the main thread at sample_tokens -> drain_pending_async -> await_task(rid,0) -> 504
```

**Root cause — CONFIRMED by isolation (2026-07-15, full model, same config, one variable changed):**

| config | overlap (defer) | cross-rank collective | result |
|---|:---:|:---:|---|
| sync DP4 | ✗ | ✓ | 3/3 clean |
| async=1 DP4 (inline await) | ✗ | ✓ | 3/3 clean |
| defer DP4 | ✓ | ✓ | **4/6 abort** (504→BeginBatch, rank2) |
| defer DP1 | ✓ | ✗ | 0/3 abort (2 clean + 1 unrelated `SYS_ENODEV` init flake) |

**The abort occurs iff (overlap AND cross-rank) are both present.** `async=1` (same async worker,
run_io, and cross-rank collectives, but awaits each submit inline) is clean → the async worker /
batch machinery itself is fine. `defer DP1` (full overlap, no cross-rank collective) is clean →
it is not a local batch-lifecycle bug. So: the defer overlap drops the tight per-step DP lockstep
(`AsyncDynamoRuntime.run` submits `run_io` non-blocking and returns without awaiting; drains late),
ranks drift, and a **cross-rank device collective in the MoE/EP forward fails to rendezvous on a
lagging rank → device aborts that task (504 SYS_TASK_ABORTED / "Wait job task aborted") → the
aborted task leaves its batch open → later `run_io` on that handle cascade "BeginBatch already
active" → the deferred drain (`await_task`) surfaces the 504.** Intermittent because it rides
inter-rank timing jitter (logprobs' extra async graphs widen the window). Sync/async=1 keep every
rank in per-step lockstep (each batch begins→ends→awaits before the next), so the collective always
rendezvous.

**Fix direction:** restore cross-rank alignment under the overlap without collapsing it — e.g. a
per-step cross-rank barrier bounding how far ranks may drift before the shared-communicator
collective, or align the deferred drain across ranks. Must keep the forward↔all_reduce overlap.

**Runtime-architecture context (likely the same underlying weakness):** the RBLN async runtime
is a **single-FIFO-worker + full-drain, single-buffered** model, not GPU's multi-stream +
per-op-event model. All submits funnel through one `SharedAsyncWorker` onto
`context_->default_stream()` (`async_runtime.cc:63-101,203-231`), always the same instance
`instances_.front()` (`async_runtime.cc:216`) writing fixed output slots (`runtime_instance.cc:630,1237`);
there is **no per-submission completion event** — only the coarse `EnsureAllTasksCompleted()` /
`WaitForDeviceCompletion()` / `WaitForCompletion(wait_seq_id_)` (`runtime_instance.h:79`,
`runtime_instance.cc:1066-1071`, `command_dispatcher.cc:68-84`). "BeginBatch already active"
after a task abort is a batch-lifecycle failure in exactly this shared-instance/single-worker
model. The token-0 fix (rebel_compiler `44f3614427`) already had to bolt `WaitForDeviceCompletion()`
after `Run()` for the same reason.

**Deeper mechanism (2026-07-15, graph-trace verified):** the forward (compute + compute_logits)
is **one fused async submission** — `dev_ops=37` under DP4+EP with the MoE **CCL op(s) as nodes
in the MIDDLE of that single graph** (DP1: `dev_ops=1`, no CCL). So the CCL cannot be isolated at
the submission level. MoE compute is data-dependent, so ranks reach the mid-graph CCL at slightly
different times; under defer (forward left in flight + variable host work between forwards:
deferred argmax, prep, all_reduce) that skew intermittently exceeds the FW watchdog → the CCL job
is device-aborted. `=1` (inline-await each forward) stays clean because each forward's CCL acts as
a natural cross-rank barrier and runahead is blocked.

**A host barrier before the forward submit does NOT work** (verified: identical abort, and it hurt
perf) — the CCL is mid-graph, so aligning the graph's *submit* doesn't align the *CCL node's*
device execution.

**Fix altitude (conclusion):** an overlap-preserving fix is NOT achievable from vllm-rbln/Python —
the only Python levers are inline-await the forward (= sync, no overlap) or barrier the submit
(proven not to align the mid-graph CCL). It must live in **rebel_compiler runtime**: a cross-rank
alignment immediately before the `CCL_OP` node executes on the worker (dedicated process group for
main/worker gloo thread-safety), keeping the forward async so all_reduce↔forward overlap survives;
**or in SSW/platform**: a graceful/longer cross-rank timeout for the on-device collective so
intermittent MoE skew doesn't hard-abort + reset. This blocks TODO 2.

**Pinpointed to the collective (2026-07-15):** the MoE dispatch is `rcclAllToAllX`
(`ccl_runtime_op.cc:493`, flag `RBLN_CCL_ALLTOALLX_NB` default false = the barrier variant, which
does an *initial cross-rank readiness barrier*). Under defer, ranks reach it host-jitter-skewed; the
late-peer wait hits the FW device watchdog → abort (`-EIO`, EVENT-IRQ path, verified via dmesg +
kmd/umd experts). `=1` (inline-await) is clean because lockstep keeps the wait short.

**Ruled out as fixes (all verified on full DP4 defer):**
- vllm-rbln **host barrier before the forward submit** — identical abort (CCL is mid-graph, not at
  submit) + hurt perf. Reverted.
- **`MKL_NUM_THREADS=1`** (reduce host jitter) — reduced frequency but STILL aborted (MK3/5). Not a fix.
- **`RBLN_CCL_ALLTOALLX_NB=1`** (skip the readiness barrier) — STILL aborted (NBD2 `SYS_TASK_ABORTED`).
  So the abort is NOT the readiness barrier; it's the cross-rank **data exchange itself** stalling on a
  skewed peer, independent of the barrier.

**Remaining viable directions (device-level RCCL/rbln only — no host gloo):** (a) bound cross-rank
skew so the collective wait is always short (async-pipeline redesign: pre-queue/device-chain the next
forward, or use the collectives themselves as device-level sync points as `=1` implicitly does); or
(b) FW/SSW: make the collective tolerate bounded skew (watchdog/timeout).

## TODO 2 — remove the logprobs `force_sync`

The deferred sampler is wrapped in `force_sync()` when logprobs are on
(`rbln_model_runner.py:~4467-4487`; `rebel/sync_runtime.py` `force_sync`), because the extra
full-vocab logprobs graphs (`compute_logprobs`=`log_softmax`, `gather_logprobs`,
`rbln_sampler.py:273-311`) submitted onto the single-buffered worker race the shared logits/output
slot the next forward reuses. In theory the deferred path should be correct without `force_sync`.
Remove it and cross-check parity; **if it then mismatches, root-cause that in the async runtime**
rather than re-adding the shim. Assumes TODO 1 is fixed first (stable async path).

Two fix directions (trace the device-level mechanism first — unconfirmed):
- **Targeted (cheaper, maybe Python-only):** give the logprobs graphs a **private device logits
  buffer** in `rbln_sampler.py` so the next forward can overwrite the shared slot without
  corrupting an in-flight logprobs read; drop `force_sync` once proven. May also need a private
  *output* slot (then it converges on the runtime fix).
- **Proper (rebel_compiler C++):** give the async runtime GPU-parity ordering — a
  **per-submission completion event** (`rid`→device seq, so `Await(rid)` waits on *that* seq, not
  the instance-wide `wait_seq_id_`) + **rotating/double output buffers**. This also lets the
  deferred path finally **return logprobs under overlap** (today it returns `logprobs=None` via
  `_bookkeeping_sync` with `sampler_output=None`).

**Acceptance:** batch=1 DP4 defer + logprobs, `--repro-run 4` → **4/4 identical WITHOUT force_sync**,
16/16 bit-identical to sync, and throughput ≥ the force_sync path.

## TODO 3 — MKL_NUM_THREADS=1 determinism sweep

With `MKL_NUM_THREADS=1` and the **compile cache disabled** (omit `--use-cached-models`), run the
same command **10+ times** and check the output is identical across launches — for **both sync and
async**. Purpose: confirm host-thread nondeterminism is the (only) variable and that pinning it
gives run-to-run identical output.
