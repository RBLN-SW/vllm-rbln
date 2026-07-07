# Async-scheduling forward / all_reduce overlap (gpt-oss EP+DP decode)

This branch hides the per-step **DP `num_tokens` `all_reduce`** (a host-side gloo
collective) behind the NPU **forward**, cutting decode step latency while staying
**token-identical to dev's synchronous path** (greedy). Measured on gpt-oss-120b
(L18, DP4, batch=1, greedy): **2.43 s/it vs dev-sync 3.01 s/it — ~19% faster**, with
`tok64` output bit-exact to sync.

The feature is gated behind `RBLN_DYNAMO_ASYNC=defer` (plus RBLN device sampler +
async scheduling). With the gate off, behavior is identical to dev.

---

## 1. What differs from dev's sync path

In **dev**, even with async scheduling on (`VLLM_RBLN_OPTIMISTIC_SCHED=1`), each RBLN
decode step runs serially on the worker's main thread:

```
step N:   [ DP all_reduce(N) ] → [ forward(N) (blocking) ] → [ sampler(N) ]
```

The `num_tokens` `all_reduce` sits on the critical path immediately before the forward.

On **this branch** (`RBLN_DYNAMO_ASYNC=defer`), the forward/compute_logits are dispatched
**non-blocking** and left in flight, and the greedy **argmax is deferred** to the next
step. So while step N+1 fires its `all_reduce`, forward(N) is still running on the device:

```
step N:   dispatch forward(N) (non-blocking), defer its argmax
step N+1: [ all_reduce(N+1) ‖ forward(N) still in flight ]
          → await forward(N) → argmax(N) → dispatch forward(N+1) → ...
```

The argmax must be deferred (not run at the end of step N) precisely so forward(N) stays
in flight past step N+1's `all_reduce`; running it eagerly would force an await of
forward(N) and collapse the overlap.

**Net effect vs dev:** identical tokens, forward↔all_reduce overlapped. The forward is the
only thing made async; the sampler still runs on the device with all its logits processors
(no host-side sampling). Off (`RBLN_DYNAMO_ASYNC` unset) ⟹ dev behavior.

---

## 2. How to run / verify

Prereqs: **rebel_compiler has C++ changes → build & install it**, then vllm-rbln editable:

```bash
# build rebel_compiler (see rebel-compiler-build), then:
uv pip install -e ~/codebase/rebel_compiler/python   # in the vllm-executor venv
```

Recipe (gpt-oss-120b truncated to L18 for a fast loop; DP4, greedy, EP):

```bash
cd ~/codebase/vllm-executor && source .venv/bin/activate
# Pick a device set that is idle (rbln-smi shows 0.0B / no context). Poll if busy.
export VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error \
  VLLM_RBLN_AUTO_PORT=1 RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 \
  VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1 \
  SPDLOG_LEVEL=warning RBLN_VERBOSE=warning VLLM_LOGGING_LEVEL=WARNING RBLN_DEVICES=4,5,6,7
ARGS="--task r --model gpt-oss-120b --ep --dp 4 --rsd 1 --max-model-len 131072 \
  --block-size 1024 --max-num-batched-tokens 512 --batch 1 --num-hidden-layers 18 \
  --max-num-blocks 129 --max-tokens 64 --num-prompts 1 --cache-results"
CJ=~/.cache/vllm-rbln-exec/rbln_results_openai_gpt-oss-120b_L18_T64_*DP4*.json

# (a) sync reference (async scheduling on, sync forward = dev behavior)
rm -f $CJ; VLLM_RBLN_OPTIMISTIC_SCHED=1 python3 -m vllm_rbln_exec.parity_runner $ARGS
python3 -c "import json,glob;print([p['token_ids'] for p in json.load(open(glob.glob('$CJ')[0]))])"

# (b) async overlap (this feature)
rm -f $CJ; RBLN_DYNAMO_ASYNC=defer VLLM_RBLN_OPTIMISTIC_SCHED=1 python3 -m vllm_rbln_exec.parity_runner $ARGS
python3 -c "import json,glob;print([p['token_ids'] for p in json.load(open(glob.glob('$CJ')[0]))])"
```

- **Parity**: (b) token_ids must equal (a) per prompt (greedy). Expect 4/4 exact.
- **Perf**: read the `Processed prompts … s/it` line (per prompt = prefill + decode). Use a
  longer `--max-tokens` (e.g. 128) for a steadier decode-latency signal; defer should be
  clearly below sync (~19% on L18 DP4 batch=1).

Notes:
- **Device flake**: warmup occasionally fails `RBLNRuntimeError: Inference failed: 501/503`
  or `RCCL InitWithUniqueId failed` at init — a device-level flake, not the deferred path.
  It is badly aggravated by rapid re-runs and `kill -9` (which can wedge device state and
  cause the *next* run to hang at init). Between runs: kill leftover `VLLM::` procs **by
  PID**, `find /dev/shm -maxdepth 1 -uid $(id -u) -delete`, confirm `rbln-smi` shows the
  devices back to `0.0B`, then relaunch (retry on flake).
- **Numerical note (batch>1 on L18)**: the truncated 18-layer model's MoE/EP reduce-scatter
  is numerically order-sensitive and its argmax is full of near-ties, so *sync itself* is
  sort-order-sensitive at batch>1 (`VLLM_RBLN_SORT_BATCH=0` vs `1` differ). Token parity at
  batch>1 is therefore only meaningful on a full (numerically robust) model; batch=1 is
  bit-exact and is the reliable correctness check on L18.

---

## 3. Code changes for the `dev ← this branch` merge

Two repos change. **rebel_compiler has C++ changes (needs a build)**; vllm-rbln is Python.

### rebel_compiler — async runtime foundation

- **`rebel/src/runtime/core/async_runtime.cc`, `async/async_task_queue.cc`,
  `async/async_result_map.cc`** (+ headers): a shared single-worker FIFO async runtime.
  Submissions run in order on one worker thread; `Await(rid, timeout)` blocks on a result
  map until that submission completes (`timeout==0` ⇒ 1 h, i.e. effectively blocking).
- **`rebel/src/pyrbln_impl/runtime.cc`, `pyrbln/compiled_model.cc`** (+ header): the
  `PyRblnAsyncRuntime` binding — `run_io(device_inputs, cpu_inputs, device_outputs,
  cpu_outputs)` submits a graph non-blocking and returns a request id; `await_task(rid,
  timeout)` awaits it. Same vmem/device-address resolution as the sync runtime, just
  submitted to the FIFO worker.
- **`rebel/python/rebel/core/torch_compile.py`**: route compiled graphs to the async
  runtime path.
- **`rebel/python/rebel/sync_runtime.py`** (main Python entry):
  - `AsyncDynamoRuntime.run()`: builds the same inputs/outputs as the sync `DynamoRuntime`,
    calls `run_io`, and — when `self._defer` (`RBLN_DYNAMO_ASYNC=="defer"`, read at init) —
    **registers** `(handle, rid, keepalive)` in the module-global `_PENDING_ASYNC` instead
    of awaiting. The keepalive `(inputs, outputs)` keeps the tensors alive until drained.
  - `_PENDING_ASYNC` + `_PENDING_ASYNC_LOCK` + `register_pending_async()` /
    `consume_pending_async()` (claim-and-clear a per-step snapshot) / `await_pending()` /
    `drain_pending_async()`: the in-flight-submission bookkeeping the runner uses to settle
    forward/compute_logits/argmax at the right point. Thread-safe because the runner's main
    thread and vLLM's async-output-copy thread both touch it.
  - `force_sync()`: a thread-local context that makes `run()` take the **blocking** branch
    (`await_task` inline, output materialized before return) even under defer. Used only on
    the rare eager/terminal sampler paths (see below); the hot deferred path stays async.

### vllm-rbln

- **`vllm_rbln/torch_compile_backend.py`**:
  - `is_warmup_active()` — exposes the warmup flag (used to gate deferral off during warmup).
  - `_maybe_disable_warmcache_for_async()` — under `RBLN_DYNAMO_ASYNC`, disables
    `torch_rbln`'s warm-cache shim, which hardcodes a `PyRblnSyncRuntime*` layout and would
    mis-cast the async handle and segfault. Disabling it routes execution back through
    `AsyncDynamoRuntime.run`.

- **`vllm_rbln/v1/sample/rbln_sampler.py`**:
  - `RBLNSampler.forward(..., skip_int32_cast=False)` — when True, skips the final
    `sampled.to(torch.int32)` and returns the int64 argmax. Needed by the deferred path:
    that cast, run inside the sampler, would read the argmax output **before the async
    submission is materialized** (→ token 0). The caller casts to int32 after awaiting.

- **`vllm_rbln/v1/worker/rbln_model_runner.py`** (the bulk of the feature):
  - `AsyncRBLNModelRunnerOutput` — carries a lazy deferred output: `sampled_token_ids=None`
    until `_ensure_sampled()` runs the deferred argmax thunk once (lock-guarded);
    `_done_event`/`capture_host()` copy the tokens to host on the main thread;
    `get_output()` (runs on vLLM's async-copy thread) **waits** on `_done_event` rather than
    computing there. `_final_step_capture()` is the last-resort path for a deferred output
    with no following step (uses `force_sync()`; reached only by discarded EOS trailing
    steps).
  - `sample_tokens()` — the `_defer_sampler` gate: `RBLN_DYNAMO_ASYNC==defer` + async
    scheduling + RBLN device sampler + device tensor + no spec-decode + logits present +
    not prefill + not warmup, **and not a terminal step** (below). When set: skip the eager
    `_sample`, snapshot `sampling_metadata`/`num_reqs`, and build a deferred output whose
    thunk calls `_sample(..., skip_int32_cast=True)`. Each step claims its own
    forward/compute_logits submissions via `consume_pending_async()` (per-step scoped — no
    cross-step accumulation).
  - **Terminal-step detection**: a step where every scheduled request hits `max_tokens`
    (mirrors the async scheduler's guard: `num_computed_tokens + n + look_ahead - 1 >=
    num_prompt_tokens + max_tokens`; look-ahead = `max_concurrent_batches`). The scheduler
    schedules no next step, so there is nowhere to run the deferred argmax on the main
    thread — such a step is **not deferred** and samples eagerly instead (no overlap is lost
    because there is no next `all_reduce`). EOS/stop stops aren't detectable ahead but
    self-heal (async look-ahead already dispatched a trailing step that captures the token).
  - **`_prepare_inputs` deferred drain (the hot path)**: `await_pending(prev step's
    forward)` → run the deferred argmax **async** (`_ensure_sampled`, `skip_int32_cast`) →
    `await_pending(consume_pending_async())` to materialize it → **cast int64→int32 on the
    now-settled result** → publish `prev_sampled_token_ids` (device feedback for the next
    step) → `capture_host()`. Keeping the argmax async is what preserves the ~19% speedup.
  - `_bookkeeping_sync()` tolerates `sampler_output is None` (the deferred phase stores the
    feedback/tokens later).
  - Eager/terminal `_sample` and `_final_step_capture` wrap the argmax in `force_sync()`
    (same int32-cast race, but these are once-per-generation so blocking costs nothing).
