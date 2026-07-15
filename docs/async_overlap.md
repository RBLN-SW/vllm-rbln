# Async-scheduling forward / all_reduce overlap (gpt-oss EP+DP decode)

This branch hides the per-step **DP `num_tokens` `all_reduce`** (a host-side gloo
collective) behind the NPU **forward**, cutting decode step latency while staying
**token-identical to dev's synchronous path** (greedy). Measured on gpt-oss-120b
(L18, DP4, batch=1, greedy): **2.43 s/it vs dev-sync 3.01 s/it — ~19% faster**, with
`tok64` output bit-exact to sync. Both fixed-length and **EOS/stop workloads** (early,
variable-length termination across DP ranks) are bit-exact to sync and deterministic.

**Full-model batch=1 verified (2026-07-07):** on the full (all-layers) gpt-oss-120b DP4+EP,
batch=1 greedy is **bit-exact to sync + deterministic** (run-to-run), and tok128 decode
throughput beats sync **rank-for-rank** (+5%..+49%). The token-0 fix (§3) preserves this.

**batch>1 (2026-07-08): root-caused + bit-exact with `VLLM_RBLN_MOE_REDUCE_SCATTER=0`.** With the
default `VLLM_RBLN_MOE_REDUCE_SCATTER=1`, batch>1 defer is nondeterministic vs sync (near-tie flips)
— NOT a deferral bug: the MoE EP **reduce_scatter reduces cross-rank in data-arrival order**, and the
async worker thread's wakeup jitter varies that order (sync stays deterministic via tight main-thread
DP-barrier lockstep). Reproduces with `RBLN_DYNAMO_ASYNC=1` (no overlap) and is NOT fixed by
`RBLN_RUNTIME_FORCE_SYNC=1` (per-rank op drain). With **`VLLM_RBLN_MOE_REDUCE_SCATTER=0`** the defer
path is **16/16 bit-exact to sync + deterministic** on full gpt-oss-120b DP4 batch=4. Proper fix =
deterministic (fixed rank-order) reduce_scatter in `librbln-ccl.so` (RBLN CCL team); until then use
RS=0 for batch>1. See `async_overlap_TODO.md` TODO 1.

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
  - **token-0 fix (`async_runtime.cc` `ProcessAsyncIO` + `runtime_instance.h`
    `WaitForDeviceCompletion()`)**: `RuntimeInstance::Run()` only *dispatches* onto the
    context's default stream; a later reader sees the output only if it issues on the SAME
    stream. Sync's D2H readback is on the caller thread (naturally ordered after Run()), but
    the async worker runs Run() on its own thread while the main thread reads the output back
    on a different stream — so `Await(rid)` returned *before* the output DMA completed, and a
    batched output could be read mid-DMA → spurious **token-0** (batch=1's tiny output finished
    within the timing slop and hid it; batch>1 exposed it). Fix: the worker calls
    `WaitForDeviceCompletion()` (thin wrapper over the existing `EnsureAllTasksCompleted()`)
    after `Run()`, so `Await` now implies full materialization. Async-path only; sync `Run()`
    is unchanged. Overlap is preserved (the drain is on the worker thread; the main thread keeps
    doing its DP all_reduce). rebel_compiler `async-overlap` 4c57f496.
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
  - `_assert_warmcache_async_safe()` — under `RBLN_DYNAMO_ASYNC`, verifies the installed
    `torch_rbln` has the warm-cache type gate (`warm_cache._is_expected_runtime_handle`,
    torch_rbln >= 0.3.0rc0) and raises otherwise. With the gate, `install_pending` refuses
    to cache the `PyRblnAsyncRuntime` handle, so the warm-cache fast path (which hardcodes a
    `PyRblnSyncRuntime*` layout) never fires on the async handle — the cache simply always
    misses on the async path and falls back to `AsyncDynamoRuntime.run`. On an older
    torch_rbln the fast path would mis-cast the async handle and segfault; the guard fails
    fast instead. (Superseded `_maybe_disable_warmcache_for_async`, which bluntly disabled
    the whole warm-cache before the type gate existed.)

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
    computing there. `_final_step_capture()` is a placeholder safety net (emits a host
    zero-tensor, **no device work**): the async-copy thread is main-thread-affine for RBLN
    device execution, so running the argmax there would race the main thread's next forward
    and corrupt its tokens. With the drain below it is no longer reached in practice.
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
    because there is no next `all_reduce`). EOS/stop stops aren't detectable ahead; they are
    handled by the drain below (every deferred output is captured on the main thread).
  - **`_drain_prev_output()` — main-thread capture of the previous step's deferred output**:
    `await_pending(prev step's forward)` → run the deferred argmax **async**
    (`_ensure_sampled`, `skip_int32_cast`) → `await_pending(consume_pending_async())` to
    materialize it → **cast int64→int32 on the now-settled result** → publish
    `prev_sampled_token_ids` (device feedback for the next step) → `capture_host()`. Keeping
    the argmax async is what preserves the ~19% speedup. Called from **two** main-thread
    sites: (a) the start of each `_prepare_inputs` (the hot decode path), and (b) **before
    `execute_model`'s empty-step (`num_scheduled_tokens == 0`) early return**. Site (b) is the
    EOS/stop correctness fix: when one DP rank finishes early, the still-generating ranks run
    a **DP dummy step** that has no scheduled tokens and returns `EMPTY_MODEL_RUNNER_OUTPUT`
    before `_prepare_inputs` — without the drain there, the previous real step's deferred
    output is orphaned (never captured on the main thread), `get_output` falls to the
    copy-thread `_final_step_capture`, and that output token is lost (racy garbage /
    placeholder 0) even though the argmax still runs for feedback. Draining at (b) keeps every
    token captured on the main thread. **Verified**: EOS/stop workload (DP4, `stop_token_ids`,
    variable lengths) is now bit-exact to sync and deterministic across runs.
  - `_bookkeeping_sync()` tolerates `sampler_output is None` (the deferred phase stores the
    feedback/tokens later).
  - Eager/terminal `_sample` wraps the argmax in `force_sync()` (same int32-cast race, but
    once-per-generation so blocking costs nothing).
