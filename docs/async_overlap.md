# Async-overlap — status (504 abort ✅ fixed · performance open)

The async-overlap prototype hides the per-step **CPU DP `num_tokens` `all_reduce`**
(`get_dp_padding`, a host gloo collective run every decode step) behind the previous
step's NPU forward, so decode throughput should go **up**.

**Current status:**
- **504 abort — ✅ FIXED** via *approach A* (reuse a ring of device output buffers instead of
  allocating one per forward). One small change in `AsyncDynamoRuntime`
  (`rebel_compiler/.../sync_runtime.py`, ~30 lines). Validated: decode `AllocBlock`s 17→0,
  `SYS_TASK_ABORTED` 0, `repro ✅ PASS`. Details in §2.
- **performance — OPEN**: with the 504 gone #3 finally runs clean, so a fair #1-vs-#3
  comparison is possible; confirming #3 is actually faster is the remaining work (§3).

---

## 0. Two independent async axes (code-verified)

There are **two separate** async switches. Confusing them makes A/B numbers meaningless.

### Axis A — scheduling: `RBLNScheduler` (sync) vs `RBLNAsyncScheduler` (optimistic)
- Selected in `vllm_rbln/platform.py:276`:
  `scheduler_config.async_scheduling AND os.environ["VLLM_RBLN_OPTIMISTIC_SCHED"]=="1"`
  → `RBLNAsyncScheduler`, else `RBLNScheduler`.
- `async_scheduling` **defaults to auto-True** for a generation model on a supported
  executor (`vllm/config/vllm.py:935-975`, field default `None` → enabled). Force it OFF
  with **`VLLM_RBLN_DISABLE_ASYNC=1`** (`platform.py:235`).
- What "optimistic" actually does (`vllm/v1/core/sched/async_scheduler.py`, real code):
  - `_update_after_schedule` (`:32`): right after scheduling step N, for each running
    decode request `request.num_output_placeholders += 1` (optimistically assume it emits
    1 token this step, before output(N) is back).
  - `_update_request_with_output` (`:53`): on output arrival `num_output_placeholders -=
    len(new_token_ids)` (reconcile).
  - Base schedulable-token formula (`vllm/v1/core/sched/scheduler.py:385`):
    `num_new_tokens = num_tokens_with_spec + num_output_placeholders − num_computed_tokens`.
    Without the placeholder this is ≤0 for step N+1 until output(N) lands (→ serial); the
    +1 lets step N+1 be **scheduled before output(N)** = *schedule-ahead*.
- **This is a scheduler-only change.** It does NOT make the forward non-blocking.

### Axis B — runtime: `SyncRuntime` (blocking) vs `AsyncDynamoRuntime` (deferred forward)
- `rebel_compiler/python/rebel/sync_runtime.py:407`: `self._defer = os.environ.get(
  "RBLN_DYNAMO_ASYNC") == "defer"`. Unset ⇒ blocking `SyncRuntime`; `defer` ⇒
  `AsyncDynamoRuntime` (forward submitted non-blocking, awaited at the next step's
  `_drain_prev_output`).
- **This is what actually lets `all_reduce(N+1)` overlap `forward(N)`** — the forward must
  be in flight while the next step's host prep (the DP all_reduce) runs.

**The throughput win needs BOTH:** schedule-ahead (Axis A) *and* deferred forward (Axis B).
Optimistic scheduling alone cannot overlap anything, because a `SyncRuntime` forward blocks
`execute_model` — so schedule-ahead has nothing to run into.

---

## 1. The three configs + how to run

Common setup (pick an idle device set; `rbln-smi` shows `0.0B`). **Between runs**:
`find /dev/shm -maxdepth 1 -uid $(id -u) -delete` and confirm `rbln-smi` back to `0.0B`,
else the next launch RCCL-init-flakes.

```bash
cd ~/codebase/vllm-executor && source .venv/bin/activate
COMMON="VLLM_RBLN_USE_DEVICE_TENSOR=1 TORCH_RBLN_DISABLE_FALLBACK=compile_error \
  VLLM_RBLN_AUTO_PORT=1 RBLN_WEIGHT_FREE=1 VLLM_RBLN_BATCH_ATTN_OPT=1 \
  VLLM_RBLN_SORT_BATCH=1 VLLM_RBLN_MOE_REDUCE_SCATTER=1 \
  SPDLOG_LEVEL=warning RBLN_VERBOSE=warning VLLM_LOGGING_LEVEL=WARNING RBLN_DEVICES=4,5,6,7"
ARGS="--task r --model gpt-oss-120b --ep --dp 4 --rsd 1 --trust-remote-code \
  --max-model-len 131072 --block-size 1024 --max-num-batched-tokens 512 \
  --batch 1 --max-num-blocks 129 --max-tokens 256 --num-prompts 2 \
  --logprobs 0 --no-inspect-logits --repro-run 0 \
  --use-cached-models --cache-ignore --skip-validation --mode 0"

# #1  sync_runtime + sync_scheduling   (RBLNScheduler + SyncRuntime)   = TRUE sync baseline
env $COMMON VLLM_RBLN_DISABLE_ASYNC=1 \
  python3 -m vllm_rbln_exec.parity_runner $ARGS

# #2  sync_runtime + async_scheduling  (RBLNAsyncScheduler + SyncRuntime)
env $COMMON VLLM_RBLN_OPTIMISTIC_SCHED=1 \
  python3 -m vllm_rbln_exec.parity_runner $ARGS

# #3  async_runtime + async_scheduling (RBLNAsyncScheduler + AsyncDynamoRuntime) = full async
env $COMMON VLLM_RBLN_OPTIMISTIC_SCHED=1 RBLN_DYNAMO_ASYNC=defer \
  python3 -m vllm_rbln_exec.parity_runner $ARGS
```

**Always verify the config actually took** (grep the log — do not assume from env):
- `scheduler.py:181` → `Using custom scheduler class ... RBLNAsyncScheduler` (#2/#3) or
  `RBLNScheduler` (#1).
- `vllm/config/vllm.py:977` → `Asynchronous scheduling is enabled` (#2/#3) / `disabled` (#1).
- throughput: `output: <N> toks/s` (tqdm est. speed; take the median across prompts).

**Expected:** `#1 ≈ #2` (optimistic scheduling can't overlap a blocking forward) and **`#3`
the fastest** (it hides the DP all_reduce). Any deviation from this is a bug.

---

## 2. Issue A — 504 abort (functionality)  [✅ RESOLVED — approach A]

**Symptom:** under #3 (defer) at DP4+EP, decode intermittently aborts with a device
`SYS_TASK_ABORTED` (504); KMD/FW: `hw_status 0x10007 = ERR_PAGE_FAULT` (confirmed by an
`oh-my-debugger` fw-expert coredump decode).

**Root cause (confirmed by measurement):** every deferred forward allocates a *fresh* device
output tensor (`AsyncDynamoRuntime.run` → `torch.empty` per step, unlike sync `DynamoRuntime`
which has `_use_static_output`). Because `defer` keeps the previous step's output alive
(keepalive) across the in-flight window, the caching pool has no free block at the next step's
alloc, so it calls `DeviceCachingAllocator::AllocBlock` → `context_.Submit()` — a
**whole-context device page-table commit** — *while the async worker's `rcclAllToAllX` is
walking the device page table*. The commit transiently invalidates a PTE the collective is
mid-walk on → page fault → job abort. Instrumenting `AllocBlock` measured **~17 such
Submits during decode, all `req_size=405504 → 2 MiB kSmallBuffer`** (one per step, never
reused).

**Fix (approach A — no fence, no reservation, no workaround):** reuse a small **ring of
persistent device output buffers** in `AsyncDynamoRuntime.run` instead of `torch.empty` per
step (`rebel_compiler/.../sync_runtime.py`, env `RBLN_ASYNC_OUTPUT_RING`, default 8). Device
memory is allocated once (during warm-up / prefill) and reused every decode step, so there is
**no per-step `AllocBlock`/`Submit` → nothing for the collective to overlap → no 504**. The
ring depth must exceed the max undrained in-flight outputs (async scheduler runs ~1 step
ahead → in-flight ~2; each slot re-allocates only on a shape/dtype change, so steady-state
decode never re-allocates).

**Validated** (DP4+EP, gpt-oss-120b, devices 0-3): decode-phase 405504 `AllocBlock`s **17 → 0**,
`SYS_TASK_ABORTED` **0**, `repro ✅ PASS` (bit-identical across 3 repro runs → the ring reuse
introduces no corruption), full run completes end-to-end.

**Why the fence approach was abandoned (do not repeat):** a CP-level fence that blocks
`rblnSubmitContext` until the collective drains **deadlocks** — a single MoE forward dispatches
~66 collectives, the worker blocks mid-run at the next collective's inc while the main thread's
`Submit` waits for `inflight==0`, and the batched Run-end decrement can never fire. Fencing at
the DP level is also unworkable because collectives are lock-step across ranks: pausing one
rank for a `Submit` stalls the others' rendezvous (cross-rank deadlock). Eliminating the
`Submit` (approach A) sidesteps the overlap entirely. Earlier failed attempts also include:
`WaitIdle` in the caching allocator (deadlocks the main thread vs the DP host collective); a
seq-drain fence (`rblnWaitJob(seq)` does not reliably track the raw-librccl collective —
compounded by `has_dep=false` collectives never recording `seq_out`).

---

## 3. Issue B — performance  [OPEN — #3 must be fastest]

**Goal:** #3 must be the fastest (that is the entire point — hide the CPU DP all_reduce).

**Status:** with Issue A fixed (approach A), #3 now runs clean end-to-end, so a fair #1 vs #3
comparison is finally possible. Earlier historical numbers (below) predate the fix and the
per-step `Submit` churn — treat them as stale.

**Historical (pre-fix, stale)** — #3 looked *slowest*; kept only as a warning not to trust
old runs: #2 ~53–55 tok/s, #3 ~38–40 tok/s (batch=1). The old "defer 26% slower than sync"
report mislabeled **#2** as sync; true sync **#1** was never measured (setting
`VLLM_RBLN_OPTIMISTIC_SCHED=1` auto-enables `async_scheduling`, silently making it #2).

**How to measure (reproducible):**
- Devices: use a **free** NPU set. This is a shared box; a neighbour's DP job pushed 1-min
  load to ~150 and starved decode to ~10 tok/s (was ~32) — perf is meaningless under load.
  Check `uptime` / `/proc/loadavg`; wait for load1 < ~40 before trusting numbers.
- Amortize init: model weights are streamed (`RBLN_WEIGHT_FREE=1`) on every process launch
  (~several min; compilation itself IS cached by `--use-cached-models`). Do NOT relaunch per
  sample — use **`--repro-run N`** so one weight-load yields N+1 back-to-back generation
  samples (each repro run reports its own `output: X toks/s`). Alternate configs across a few
  loads to also capture cross-load DVFS drift.
- Config flags (code-verified, see §1):
  - #1 sync:  `VLLM_RBLN_DISABLE_ASYNC=1` (no `RBLN_DYNAMO_ASYNC`, no `OPTIMISTIC_SCHED`).
  - #3 async: `RBLN_DYNAMO_ASYNC=defer VLLM_RBLN_OPTIMISTIC_SCHED=1`.
- `--cache-ignore` only bypasses the parity **result** cache (forces a real run) — it does
  NOT force recompile; leave it on for perf so generation actually executes.

**Instrumentation already established (batch=1, still valid):** the deferred forward *does*
overlap (a 15 ms sleep after forward-submit cut the next step's `await(forward)` 18.5→3.5 ms);
the drain cost is dominated by `await(forward)` (~18.5 ms b1 / ~36 ms b8), not the argmax
(~0.9 ms) or D2H (~0.05 ms).

**Hypothesis to (re)confirm now that #3 is clean:** the async pipeline may desync the DP ranks
so the EP all-to-all (a barrier) inside the forward absorbs the skew and inflates `forward(N)`,
cancelling the hidden all_reduce. Needs per-rank forward/CCL timing on a *quiet* machine.

---

## 4. Priorities

1. **Perf [now unblocked]:** confirm #3 ≥ #1 on a quiet machine (`--repro-run` sampling). If
   #3 is not faster, get per-rank forward/CCL timing to see whether DP-skew inflates the
   forward and cancels the hidden all_reduce.
2. ~~**504**~~ **DONE** — approach A (output-buffer ring reuse) removed the per-step device
   `Submit`; no fence/reservation/workaround. See §2.

## Key files
- `vllm_rbln/platform.py` — scheduler selection (`async_scheduling`, `OPTIMISTIC_SCHED`, `DISABLE_ASYNC`)
- `vllm_rbln/v1/core/rbln_scheduler.py` — `RBLNScheduler` / `RBLNAsyncScheduler`
- `vllm/v1/core/sched/async_scheduler.py` — optimistic `num_output_placeholders` mechanism
- `vllm_rbln/v1/worker/rbln_model_runner.py` — defer gate, `_drain_prev_output`, `sample_tokens`
- `rebel_compiler/python/rebel/sync_runtime.py` — `SyncRuntime` / `AsyncDynamoRuntime`, `_PENDING_ASYNC`, `force_sync`.
  **The approach-A 504 fix lives here**: `AsyncDynamoRuntime.__init__` (`_out_ring*`) +
  `_acquire_output_buffer` + the `run()` output loop (ring reuse instead of `torch.empty`).
  (Note: the git-tracked source is `rebel_compiler/rebel/python/rebel/sync_runtime.py`,
  hardlinked to the loaded `.../python/rebel/sync_runtime.py`.)
- `rebel_compiler/.../caching_allocator.cc` (`AllocBlock`→`context_.Submit`), `.../ops/ccl_runtime_op.cc`
  (`RcclInvoke`; `has_dep=false`⇒no `seq_out`), `.../distributed/rbln_rccl.cc` — 504 page-table / CCL path.

## How to continue (perf)
- The clean #1-vs-#3 perf comparison is the only open item. Run each config with
  `--repro-run 12` on a **quiet** machine (see §3) and compare the `output: X toks/s` samples.
- A perf watcher was used during development: `~/perf_when_free.sh` (waits for load1<40 then
  runs `~/perf10_run.sh`, which does sync/async/sync/async with `--repro-run 12`, devices 0-3,
  writing `p10_*.log`). These are throwaway helper scripts, not committed — re-create as needed.
