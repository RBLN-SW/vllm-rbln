# async-overlap batch=1 nondeterminism — root cause + fix

Model `openai/gpt-oss-120b`, EP+DP4, `RBLN_DYNAMO_ASYNC=defer` + `VLLM_RBLN_OPTIMISTIC_SCHED=1`.
This documents the confirmed root cause of the batch=1 run-to-run token flip, why the
runtime-mutex idea turned out to be moot, and the fix that shipped.

## ★ Root cause (confirmed)

Under `defer`, the **deferred sampler** (the async-overlap thunk that runs in the next step's
`_prepare_inputs`/`_drain_prev_output`) submits, for each step, the compiled **argmax** plus —
when logprobs are requested — the **logprobs graphs** (`compute_logprobs` = full-vocab
`log_softmax`, `gather_logprobs`). All of these are submitted to the async runtime's **single
FIFO worker thread** (`SharedAsyncWorker`) and, under `defer`, are **non-blocking**: the main
thread returns from the thunk and races ahead to the next step while the worker is still
draining the sampler's submissions.

With the extra (large, full-vocab) logprobs graphs in flight, that overlap corrupts near-tie
tokens **run-to-run at batch=1**. It is a device-runtime interaction, **not** wrong math:

- the device argmax equals the host argmax at every step (the token the sampler computes is
  faithful to the logits it read);
- logprobs **off** → deterministic; logprobs **on** → flips;
- `async1` (`RBLN_DYNAMO_ASYNC=1`, same worker but **inline await**, i.e. no deferral) →
  deterministic even with logprobs on.

The decisive proof: `force_sync()` around the deferred sampler (so its worker submissions
complete inline before the main thread proceeds) makes **batch=1 DP4 defer + logprobs
deterministic**.

### Ruled out (verified previously; do NOT re-investigate)
device argmax wrong / MoE-combine padding / reduce_scatter-specific / get_output copy-thread
D2H / sampler op reorder / per-op `RBLN_RUNTIME_FORCE_SYNC` / caching-pool block reuse
(`RBLN_DISABLE_EAGER_CACHE_ALLOC`) / free-during-inflight block reuse (seq-quarantine, engaged
368× but still flipped) / main-thread D2H-H2D racing the worker (`CAND2`, 0 during busy).

## ★ Why the runtime-mutex idea (old "option B") was dropped — it is moot

The idea was a process-global mutex around `RuntimeInstance::Run()` to serialize a main-thread
(eager) submit against the async worker's submit. Investigation (see the call-path map below)
showed that premise is false:

- Eager on-device aten ops (`log_softmax`, `gather`, `abs`, `ne`, `eq`, …) do reach the same
  `RuntimeInstance::Run()` — but via `torch_rbln` JIT-recompiling each op through
  `torch.compile(backend="rbln")` → `_build_dynamo_runtime`.
- `_build_dynamo_runtime` chooses async vs sync **purely** from `RBLN_DYNAMO_ASYNC`
  (`rebel/python/rebel/core/torch_compile.py`), with no other gate. So with `defer` set
  process-wide, **every** `backend="rbln"` compile — the forward, the argmax, and every eager
  single-op graph — builds `AsyncDynamoRuntime` and runs on the **one** `SharedAsyncWorker`
  thread. (Confirmed empirically with a compile-time probe: all graphs, including the 1-op
  `argmax`/`abs`/`ne`/`eq` graphs, report `async=True`.)

Because all device submits already funnel through a single FIFO worker thread, they are
**already serialized** — there is no concurrent main-thread `RuntimeInstance::Run()` to
serialize, so a mutex there guards a single already-serialized thread and changes nothing. The
remaining nondeterminism is not two threads racing inside `Run()`; it is the **deferral
itself** (the main thread proceeding while the worker still has the sampler's submissions in
flight). The precise device-level mechanism is a runtime-team question (device-memory
tracing); it is not fixable by a `Run()` mutex.

## ★ Fix that shipped (application-level, Python-only, no rebuild)

In the deferred thunk (`rbln_model_runner.py`, `sample_tokens`): **keep logprobs computed** (do
NOT strip `max_num_logprobs`), and when logprobs are requested run `self._sample(...)` under
`rebel.sync_runtime.force_sync()` so the sampler's worker submissions complete inline before
the main thread races ahead. This removes the concurrency directly.

- Keeps logprobs computed (does not adopt the strip commit `c38f37b4`, which dodged the
  trigger by dropping the discarded logprobs).
- Keeps the forward↔all_reduce overlap: that overlap already happened in the *prior* step and
  was drained before the deferred sampler runs, so making only the (small) sampler synchronous
  does not collapse it.
- Keeps the device sampler (argmax on device).
- Gated on logprobs-present (`max_num_logprobs is not None`): with no logprobs there are no
  extra graphs and the path is already deterministic, so no `force_sync` cost is paid.

NOTE: the deferred path still returns `logprobs=None` (`_bookkeeping_sync` with
`sampler_output=None`). Keeping the computation makes returning logprobs *under* overlap a
straightforward follow-up now that the concurrency is safe.

## Call-path map (for future reference)

- Compiled-graph submit: `AsyncDynamoRuntime.run` (`sync_runtime.py`) → `run_io` →
  `AsyncRuntime::Run` (`async_runtime.cc`) → `SharedAsyncWorker::Enqueue` → worker thread →
  `ProcessAsyncIO` → `RuntimeInstance::Run()` (`runtime_instance.cc:1156`).
- Eager on-device aten op: `torch_rbln` op IMPL → `compile_and_run_view_aware` →
  `compile_rbln_cached` → `torch.compile(backend="rbln")` → `rbln_backend` →
  `_build_dynamo_runtime` (async under `defer`) → same `RuntimeInstance::Run()` on the worker.
- Shared per-device state: `RuntimeInstance::Run()` chains dependencies on
  `context_->default_stream()` (`Stream::last_seq()`/`Record()`), shared across
  RuntimeInstances on the same device. `Stream` has no internal lock, but with all submits on
  the one worker thread it is not concurrently accessed. `EndIOPatchBatch` (IO-patch flush)
  does not touch the stream seq; init-time `Context::Submit()` is not on the hot path.

## Verification (option A, clean rebel `30d55c5`)

Criterion: per-input parity (keyed on prompt_token_ids, never batch index); confident tokens
match sync, low-entropy near-tie tail divergence is accepted variation. Run-to-run determinism
via `--repro-run`.

gpt-oss-120b, EP+DP4, L18, tok64/tok128.

- **batch=1 (DP4, np=4)** — core fix:
  - determinism: cache ON → `--repro-run 4` **4/4 identical**; cache OFF → **4/4 identical**.
  - parity: defer+A vs plain sync → **16/16 inputs bit-identical** (composition is fixed at
    batch=1: 1 request/rank, so no co-batching variance).
- **batch=8 (DP4, np=8, tok128)** — no regression:
  - runs without crash (one `SYS_TASK_ABORTED`+gloo env-flake retry; clean on rerun).
  - per-input vs sync diverges early, but that is the **known composition MAJOR**, not the
    fix: the control **optsync (sync fwd) vs none (sync fwd), different scheduler = 1/31** —
    the *same* early divergence as defer-vs-none, proving it is scheduler-driven composition
    (optimistic admits a different request set → different `num_tokens_across_dp` → EP-DP is
    non-invariant → different logits), independent of async/defer/the fix. batch=1 (fixed
    composition) is bit-identical, isolating the forward+sampler as correct.
- **perf (tok128 b8 DP4, per-rank est. tok/s, no-profile)**: defer+A **≈290** vs pure sync
  **≈159** (**~1.8×**). force_sync cost is within noise: defer+logprobs-on(force_sync)
  **≈290** ≈ defer+logprobs-off(no force_sync) **≈290**. Overlap win dominates; force_sync
  barely registers (and is skipped entirely when no logprobs are requested).
- **spec-decode**: excluded from the deferred path by construction (`_defer_sampler` requires
  `spec_decode_metadata is None`, `rbln_model_runner.py:4135`) → spec uses the eager path,
  untouched by the fix. (ngram/suffix are config-incompatible with async_scheduling upstream;
  eagle-family needs a draft model, not set up.)
- **EOS**: batch=1 defer + EOS enabled → deterministic (`--repro-run` 2/2), no crash.
- **non-greedy** (temp/top-p/top-k): generates correctly through the deferred+force_sync
  path, no crash. Exact determinism is not testable (the RBLN device sampler ignores the
  per-request seed, so both sync and defer are non-deterministic for random sampling) — a
  device limitation, unrelated to overlap.
- **env note**: `gloo "Connection closed by peer"` / `SYS_TASK_ABORTED` / `Failed to
  allocate memory` appeared on rapid back-to-back runs and cleared on retry after
  `ps`-kill + `/dev/shm` cleanup + device-0.0B confirm + ~20s wait — device/env flakes, not
  the fix.

## Repro / build

- `runone.sh <tag> <mode> <RS> <batch> <np> <maxtok> <nrepro>`; modes `defer`/`none`(pure
  sync)/`optsync`/`syncref`/`async1`. Sets `--cache-results` so per-input compares use a fresh
  cache.
- Devices via `DEVS=`. Between runs: kill leftovers by `ps` (not `pgrep -f`), `find /dev/shm
  -maxdepth 1 -uid $(id -u) -delete`, wait ~20s, confirm `rbln-smi` shows 0.0B. gloo/RCCL init
  failures are env flakes → retry.
- C++ build (rebel): `~/.venv`; `./rebel_install.sh -a -n`; then in the vllm-executor venv
  `uv pip install -e ~/codebase/rebel_compiler/python --no-deps` (ABI-match `_C`). (The
  triton_rbln sub-build may fail on a missing generated `.inc`; that is unrelated to the
  runtime and does not affect `librbln.so`.)
