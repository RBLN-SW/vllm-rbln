# Async-overlap — status & analysis

The async-overlap prototype aims to hide the per-step **host DP `num_tokens`
`all_reduce`** (`get_dp_padding`, a gloo collective run every decode step) behind
the previous step's NPU forward, so decode throughput should go up.

**Current status (2026-07-21):**
- **504 abort (`SYS_TASK_ABORTED`) — ✅ FIXED.** 0 aborts across the full batch
  sweep. See §3.
- **Determinism — ✅ FIXED for per-rank batch ≤ 8; ⚠️ batch 16 still diverges.**
  The divergence is a *token-scheduling* effect, not a functional bug. See §4.
- **Performance — ✅ MEASURED, and the result is negative.** The prototype is a
  throughput *loss* vs sync at every batch, worsening with batch. See §5.

Reproduction scripts and a server-migration guide live in
[`async_overlap_tests/`](async_overlap_tests/README.md) — this doc is the
analysis/status; that dir is how to run it.

---

## 1. Two independent async axes (code-verified)

Two **separate** switches. Confusing them makes A/B numbers meaningless.

### Axis A — scheduling: `RBLNScheduler` (sync) vs `RBLNAsyncScheduler` (optimistic)
- Selected in `vllm_rbln/platform.py`:
  `scheduler_config.async_scheduling AND VLLM_RBLN_OPTIMISTIC_SCHED==1`
  → `RBLNAsyncScheduler`, else `RBLNScheduler`.
- `async_scheduling` defaults auto-True for a generation model on a supported
  executor; force OFF with **`VLLM_RBLN_DISABLE_ASYNC=1`**.
- "Optimistic" = schedule-ahead: `_update_after_schedule` bumps
  `num_output_placeholders += 1` per running decode right after scheduling step N,
  so step N+1 can be scheduled **before** output(N) lands. Scheduler-only; does
  **not** make the forward non-blocking.

### Axis B — runtime: `SyncRuntime` (blocking) vs `AsyncDynamoRuntime` (deferred forward)
- `rebel_compiler/.../sync_runtime.py`: `RBLN_DYNAMO_ASYNC=defer` ⇒
  `AsyncDynamoRuntime` (forward submitted non-blocking, awaited at the next step's
  `_drain_prev_output`). Unset ⇒ blocking `SyncRuntime`.
- This is what actually lets `all_reduce(N+1)` overlap `forward(N)`.

**The throughput win needs BOTH** (schedule-ahead + deferred forward). Optimistic
scheduling alone overlaps nothing (a `SyncRuntime` forward blocks `execute_model`).

---

## 2. The three configs

| # | scheduling | runtime | env | meaning |
|---|---|---|---|---|
| #1 | `RBLNScheduler` | `SyncRuntime` | `VLLM_RBLN_DISABLE_ASYNC=1` | true sync baseline |
| #2 | `RBLNAsyncScheduler` | `SyncRuntime` | `VLLM_RBLN_OPTIMISTIC_SCHED=1` | schedule-ahead only |
| #3 | `RBLNAsyncScheduler` | `AsyncDynamoRuntime` | `VLLM_RBLN_OPTIMISTIC_SCHED=1 RBLN_DYNAMO_ASYNC=defer` | full async |

Always verify the config actually took (grep the log, don't assume from env):
`Using custom scheduler class ... RBLNAsyncScheduler` and `Asynchronous
scheduling is enabled` for #2/#3. Run commands + scripts:
[`async_overlap_tests/README.md`](async_overlap_tests/README.md).

---

## 3. 504 abort (functionality) — ✅ FIXED

**Symptom:** under #3 at DP4+EP, decode intermittently aborts with device
`SYS_TASK_ABORTED` (504); FW `hw_status 0x10007 = ERR_PAGE_FAULT` (NPU MMU page
fault), confirmed by an `oh-my-debugger` fw-expert coredump decode.

**Root cause:** an async in-flight collective (`rcclAllToAllX`) walks the device
page table while a **non-worker thread commits a page-table change**
(`Context::Submit` → `rblnSubmitContext`) for a device-memory alloc/free — the
commit transiently invalidates a PTE the collective is mid-walk on → page fault →
job abort. Two distinct buffers triggered it: the per-step forward **output**
buffer, and the DP MoE **combine** buffer (all_gather/reduce_scatter, shape
changes with padding so it isn't caught by the output ring).

**Fix (rebel_compiler `async_rebase`, complementary — all three required):**
- `9463c95766` — reuse a ring of persistent device **output** buffers in
  `AsyncDynamoRuntime.run` (env `RBLN_ASYNC_OUTPUT_RING`, default 8) instead of a
  fresh `torch.empty` per step → no per-step alloc/Submit for the output buffer.
- `41a2a9869e` — **stream-ordered device memory**: defer `Context::Submit` while
  an async collective is in flight on a non-worker thread (commit-deferral, not a
  fence — flushed by the next worker/idle Submit, so overlap is preserved and the
  old fence deadlock is avoided) + free-side quarantine of caching-allocator
  blocks freed mid-flight. Covers the combine buffer.
- `4ff56e8742` — make `Stream` (`last_seq_`/`has_pending_`) thread-safe
  (hardening; a real multi-thread data race, though not the 504 trigger).

> The earlier "504 fixed via approach A (the ring) alone" claim (Jul 16) was
> **incomplete** — the ring only covers the output buffer; without `41a2a9869e`
> the combine buffer still faults. Keep the ring *and* the stream-ordered dealloc.

**Abandoned (do NOT revisit):** a CP-level fence blocking `rblnSubmitContext`
until the collective drains **deadlocks** (one MoE forward dispatches ~66
collectives; cross-rank lock-step makes a per-rank pause stall the others'
rendezvous). `WaitIdle` in the caching allocator and a seq-drain fence
(`rblnWaitJob` doesn't track `has_dep=false` collectives) also failed. Removing /
deferring the `Submit` sidesteps the overlap entirely.

**Validated:** DP4+EP gpt-oss-120b, decode 405504-byte `AllocBlock`s 17→0,
`SYS_TASK_ABORTED` 0 across the whole batch sweep, `repro PASS` at batch ≤ 8.

---

## 4. Determinism (batch>1) — ✅ FIXED for batch ≤ 8; batch 16 open (scheduling)

**Symptom:** greedy (temp=0, seed 42) with a fixed compiled graph, full async at
batch>1 produced **different output text run-to-run** (per-input comparison:
`parity_runner` checks `outputs[i].text`, aligned to input order). Sync is always
deterministic.

**Root cause:** optimistic scheduling steps non-blocking, so it begins the first
step before all in-flight requests have been ingested from the EngineCore input
queue. The number admitted before the first prefill varies run-to-run with IPC
timing → the per-step DP batch composition (`num_tokens_across_dp`) varies → and
**the EP+DP MoE forward is not batch-composition-invariant** (cross-rank
dispatch/combine mixes a rank's tokens with its co-batched neighbours) → logits
shift → near-tie argmax tokens flip → divergence. Sync avoids this only
incidentally, via its blocking first prefill.

**Fix (vllm-rbln `async-overlap-prototype`, entirely in-scheduler):**
- `ca061d54` — cold-start **quiesce gate** in `RBLNAsyncScheduler`: hold the first
  step (schedule nothing, via the existing `token_budget=0` path) until the
  waiting set is stable for 3 checks, so the initial DP batch composition is
  fixed run-to-run. Re-arms on idle; inert for sync; always-on (no env flag).
- `f6b3cf4c` — revert the misattributed unconditional force_sync of the deferred
  sampler (`def1bc9b`); determinism is owned by the scheduler, not the sampler.
- Regression test: `tests/torch_compile/unit/v1/core/test_async_determinism_gate.py`.

> ⚠️ **Upstream vLLM must stay pristine.** An earlier `RBLN_DETERMINISTIC_ADMIT`
> env that patched `vllm/v1/engine/core.py._process_input_queue` was a diagnostic
> hack; it has been **removed**. The fix lives only in vllm-rbln.

**Batch 16 still diverges** — a **token-scheduling** limitation, not a functional
bug: sync batch 16 is fully deterministic (same test), so the model/kernel/
allocator are correct at b16. Prefill batch size is limited to 1, so with 16
prompts many prefill steps interleave with decodes; async timing varies *that*
interleaving (not just the first step), so later sequences' composition still
shifts. The quiesce gate fixes only the first-step composition — enough for ≤8. A
general fix would need deterministic prefill/decode interleaving, or accept that
strict bit-reproducibility at high batch under async requires sync scheduling
(GPU vLLM has the same batching non-determinism). The `eraase_count <= 0` runtime
warning seen in async batch>1 is a pre-existing, non-fatal `TryMergeBlocks` check
(rebel `1d3f8eaced`, not ours) and is uncorrelated with the divergence.

---

## 5. Performance — ✅ MEASURED (async is a throughput loss)

Metric: **total_output_tokens / generate wall-clock, warm runs only** (the
cold/compile run is dropped), summed across DP ranks. DP4+EP, gpt-oss-120b,
max_tokens=128, per-rank batch B. (See `async_overlap_tests/perf_metric.sh`.)

| per-rank batch | #3 async tok/s | #1 sync tok/s | async/sync |
|---:|---:|---:|---:|
| 1  | 145.4 | 167.1  | 0.87x |
| 4  | 360.7 | 513.4  | 0.70x |
| 8  | 612.6 | 1246.0 | 0.49x |
| 16 | 984.4 | 1947.9 | 0.51x |

**The async-overlap prototype is slower than sync at every batch, worsening with
batch (~2x slower at b8/b16)** — the opposite of the overlap premise. Overlap
should help *more* when there is more device work to hide behind, so the fact that
it worsens with batch points at a systematic overhead: the single process-wide
FIFO async worker (serializes all async runtimes' submissions) + per-step
deferred-forward `_drain_prev_output` (await(forward)) cost, which dominates as
the batch's device work grows.

> Earlier median-of-per-step-rate numbers (and the "#3 should be fastest"
> expectation) are **superseded** — that metric was contaminated by startup-0
> samples and was unreliable.

**Implication for the rewrite:** do **not** assume async helps at scale. The
overlap *mechanism itself* (FIFO worker serialization + drain overhead) needs
redesign, or the approach reconsidered. Historical batch=1 instrumentation (still
valid): the deferred forward *does* overlap (a 15 ms post-submit sleep cut the
next step's `await(forward)` 18.5→3.5 ms), and the drain is dominated by
`await(forward)` — consistent with the forward/CCL path, not argmax/D2H, being the
cost. A likely contributor is DP-rank desync: the EP all-to-all (a barrier) inside
the forward absorbs the skew and inflates `forward(N)`, cancelling the hidden
all_reduce. Per-rank forward/CCL timing on a quiet machine would confirm.

---

## 6. Key files & commits

**vllm-rbln** (`async-overlap-prototype`):
- `vllm_rbln/platform.py` — scheduler selection.
- `vllm_rbln/v1/core/rbln_scheduler.py` — `RBLNScheduler` / `RBLNAsyncScheduler`
  (+ the determinism quiesce gate).
- `vllm_rbln/v1/worker/rbln_model_runner.py` — defer gate, `_drain_prev_output`,
  deferred sampler.
- commits: `ca061d54` (determinism gate), `f6b3cf4c` (force_sync revert),
  `6ad78624` (test handoff docs), `ab959d9e` (gate regression test).

**rebel_compiler** (`async_rebase`):
- `python/rebel/sync_runtime.py` — `AsyncDynamoRuntime`, output-buffer ring.
- `rebel/src/runtime/base/{context,caching_allocator,stream}.*`,
  `core/async_runtime.cc` — stream-ordered dealloc / commit-deferral / quarantine.
- commits: `9463c95766` (output ring), `41a2a9869e` (stream-ordered dealloc),
  `4ff56e8742` (Stream thread-safe).

**upstream vLLM** — must be pristine (no `RBLN_DETERMINISTIC_ADMIT`).

See [`async_overlap_tests/README.md`](async_overlap_tests/README.md) for the exact
required code state, build steps, and how to reproduce on another server.
