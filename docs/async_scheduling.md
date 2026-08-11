# Async scheduling on RBLN — how to run and measure it

Switches, reproduction procedure, measurement instruments and environment for the
`async-scheduling` branch.

**Scope: no analysis, no measured numbers.** Earlier records of what async scheduling
does to performance were removed on 2026-07-31 to restart the investigation from
scratch; nothing in this file interprets a result. Recover the removed records from git
if you want to see what was previously claimed — and treat all of it as unverified:

```bash
git log --diff-filter=D --name-only -- 'docs/async_scheduling*'
```

Line numbers are hints — symbols are the stable reference.

---

## 1. Switches

| want | how |
|---|---|
| async (schedule-ahead) | nothing — vLLM turns `scheduler_config.async_scheduling` on by default for generation models |
| sync | `VLLM_RBLN_DISABLE_ASYNC=1` |

`scheduler_config.async_scheduling` is the single field that decides everything: the
scheduler class, the engine's batch queue, and the runner's output path.
`vllm_rbln/platform.py:256-258` is the kill switch; `:303-317` picks the class from the
same field. vLLM's own `--async-scheduling` sets that field too, so the two compose. The
optimum path forces it off (`platform.py:376-380`).

`VLLM_RBLN_ASYNC_SCHED` was removed; setting it logs a warning and is ignored
(`platform.py:258-264`).

### Why the class must follow the field

`RBLNAsyncScheduler` is not optional. Plain `RBLNScheduler` sizes a running decode
request as `num_tokens_with_spec + num_output_placeholders - num_computed_tokens`, which
is `<= 0` until `update_from_output(N)` lands — so the engine's batch queue never fills
and async scheduling would be on while the engine ran serially. Only
`AsyncScheduler._update_after_schedule` bumps `num_output_placeholders` at schedule time
(`vllm/v1/core/sched/async_scheduler.py:18-36`), and `RBLNAsyncScheduler` composes it in
via MRO (`vllm_rbln/v1/core/rbln_scheduler.py:1046-1062`).

---

## 2. Running it

Scripts live in [`async_scheduling/`](async_scheduling/). All default to
`RBLN_DEVICES=4,5,6,7`; CR13 needs `RBLN_DISABLE_AUTO_RDMA_IP=1` (already set). Both
scripts install a `cleanup` trap, but verify afterwards anyway.

**Check the devices are free and yours before and after every run.** This box is shared;
a co-tenant job has twice invalidated measurements here, once by degrading them silently
and once by taking every device so runs failed at startup with
`this process sees 0 RBLN device(s)`. `rbln-stat` shows both the per-NPU table and the
context table — record it either side of a run.

**Free the devices when done.** `pkill -f vllm_rbln_executor.cli` misses the `VLLM::Worker_DP` /
`VLLM::EngineCore` children, which keep holding tens of GiB. Check `rbln-smi` and kill
only PIDs that `ps -u "$(id -un)"` confirms are yours.

### 2.1 Throughput A/B

```bash
OUTDIR=./_ab BATCH=8 MAXTOK=1024 REPRO=10 ROUNDS=2 \
  CONFIGS="async sync" docs/async_scheduling/ab_throughput.sh

python3 docs/async_scheduling/agg_tokps.py \
  --pool async=_ab/async_*.log --pool sync=_ab/sync_*.log
```

Env knobs: `OUTDIR`, `BATCH`(8), `MAXTOK`(1024), `REPRO`(2), `ROUNDS`(1), `DP`(4),
`CONFIGS`, `EXTRA_ENV`, `RBLN_DEVICES`, `VENV`.

Why this shape:

- **b8, `max_tokens=1024`** — below ~256 tokens, prefill and wave transients dominate the
  average.
- **`REPRO=10`** — each generate call is an independent sample and costs ~65 s; a process
  restart costs a full recompile. Raise `REPRO` before `ROUNDS`.
- **`ROUNDS=2`** — two process starts per config puts cross-process variance in the range
  too. Configs alternate round by round, so machine drift hits both.
- **Anything you want to compare must be a `CONFIGS` arm**, not a separate invocation, so
  the arms alternate in time. `BATCH` is overridable per config (`CFG_BATCH`, used by the
  `b1`/`b8` arms) for exactly this reason.
- `agg_tokps.py --pool` takes **relative globs only** (`Path().glob` raises on absolute
  patterns).

`EXTRA_ENV` passes anything extra into every arm; use it for whatever instrumentation
you have added (there is none committed — §3).

To A/B the DP reduce form, add `VLLM_RBLN_DP_ALL_REDUCE_ASYNC=0`.

### 2.2 Perfetto trace

```bash
BATCH=8 MAXTOK=32 docs/async_scheduling/profile.sh async
BATCH=8 MAXTOK=32 docs/async_scheduling/profile.sh sync
mv _prof ~/async_perfetto_$(date +%Y%m%d)      # keep it out of the repo
```

Writes `_prof/<cfg>/rbln_*_dp{0..3}/*.pt.trace.json.gz`; upload the `.gz` straight to
ui.perfetto.dev. Keep `MAXTOK` small — profiling roughly halves throughput and an 8 s run
already yields millions of events per rank. `torch_profiler_with_stack=True` is hardcoded
in the executor, and profiling covers the whole generate call, so `MAXTOK` is the only
lever on trace size.

The trace is **CPU-only**: `TorchProfilerActivityMap` has no RBLN entry, so there is no
NPU track. Cross-thread structure *is* visible (gloo on `pt_gloo_runloop`, `get_output` on
the async output thread).

```bash
python3 docs/async_scheduling/trace_analyze.py <trace.json.gz>            # spans by thread
python3 docs/async_scheduling/trace_analyze.py <trace.json.gz> --step     # one decode step
python3 docs/async_scheduling/trace_analyze.py <trace.json.gz> --overlap  # gloo ∩ main thread
python3 docs/async_scheduling/trace_analyze.py <trace.json.gz> --window T_MS
```

There is no prefill/decode marker; distinguish by forward duration and by
`Torch-Compiled Region: 2/0` vs `2/1`.

### 2.3 Unit tests

Scheduler-level behaviour, no device needed:

```bash
pytest tests/v1/core/test_async_scheduler.py -v
```

`test_schedule_alloc_block`, `test_running_queue`, `test_preempt` — all build a scheduler
through `tests/v1/core/utils.py:create_scheduler(..., async_scheduling=True)`, which is
what exercises the `RBLNAsyncScheduler` MRO of §1.

---

## 3. Measurement instruments

No host-side timing instrumentation is committed on this branch — it was all removed
on 2026-07-31 along with the analysis records, so the tree carries no `_mark`,
no `STEP_TIMING`, no overlap trace and no rebel-side timers. What follows is what
survives that removal, plus the notes worth having before adding timers back.

### 3.1 `proc_probe.sh` — host-bound or device-bound, from `/proc`

```bash
docs/async_scheduling/proc_probe.sh [seconds] [step_ms]
```

Needs no build, no code change and no restart; run it against a live worker. Reports, for
the worker's main thread: utime+stime vs wall (how much of the step is CPU at all),
`wchan` samples (what it sleeps in — `dma_fence_*` = device, `futex_*` = lock), and
voluntary vs involuntary context switches (chose to sleep vs CPU starved).

**Run this before adding any timer.** It costs nothing and it decides whether host-side
timing is worth instrumenting at all.

### 3.2 If you add timers back

Nothing here endorses a particular design; these are the mistakes that cost real
conclusions on this branch.

- **Wall clock cannot tell work from waiting.** A region that measures 5 ms may be 0.4 ms
  of work and 4.6 ms of blocking, and the two call for opposite fixes. Pair every
  `perf_counter` with `time.thread_time()` from the start.
- **Per call is not per step.** There is more than one compiled-graph run per decode step
  (the model forward and the sampler graph are separate), so a cumulative counter must be
  divided by a *measured* step count over the same window, never by its own call count.
- **Cumulative dumps must be diffed mid-run.** Diffing the last two lines spans process
  shutdown and yields negative rates.
- **Emit the rank.** A `fprintf` from C++ carries no worker prefix, so it cannot be
  attributed to a DP rank; correlating one rank's Python marks with all four ranks' C++
  lines produces confident nonsense.
- **Durations cannot answer overlap questions.** Whether two things ran at the same time
  needs absolute timestamps from both threads, compared offline as interval intersection.
- **A probe that drains perturbs everything after it.** `torch.rbln.synchronize` is
  transfer-only, and `Stream::Drain` resets the stream's pending state, so a probe placed
  after any drain reads ~0 regardless of what the device did. One probe site per run,
  compared across arms.

### 3.3 tok/s

vLLM's own tqdm `est. speed output`, i.e. `total_out_toks / elapsed` — pure wall clock,
and fair across configs because `ignore_eos=True` (a suite default) pins the
token count. Read it with `agg_tokps.py`, never by eye.

---

## 4. Measurement pitfalls

These cost several wrong conclusions on this branch.

1. **Run-to-run spread is 1–3 %, within a run too.** The 4 DP ranks are in lockstep and
   agree to 4 significant digits, which makes a single reading look precise; it is not.
   **Nothing below ~3 % is decidable** from tok/s. Anything smaller needs a per-step
   measurement over hundreds of steps, not a throughput reading.
2. **Never read the last tok/s line.** tqdm rewrites it continuously; only the
   100 %-completion reading is meaningful. `agg_tokps.py` matches exactly that (`DONE`
   regex) and aggregates every warm sample.
3. **Drop the first generate call.** It is cold and lands ~20 % low. `agg_tokps.py` drops
   it by a relative threshold (`WARM_FRACTION = 0.8`), not by position — the ranks
   interleave in the log and position is unreliable.
4. **Never derive throughput from device or dispatch timers.** `metrics_v2` DECODE E2E
   times the host dispatch, not device completion, so it under-reports. Wall clock (§3.3)
   is the only throughput metric.
5. **A shrinking region is not a shrinking step.** Any claim of improvement has to be an
   end-to-end tok/s claim read with `agg_tokps.py`, never a region delta.
6. **Wall clock cannot tell work from waiting.** Pair every region timer with
   `thread_time`, or establish the split with §3.1 first.
7. **`torch.rbln.synchronize` is transfer-only.** With no pending transfer it returns
   immediately and tells you nothing about compute. It is meaningful only after a
   `non_blocking` copy whose sequence chains behind the compute.
8. **Do not use perfetto for overlap.** Profiling roughly halves throughput and inflates
   regions dense in small Python calls the most — exactly the host residue usually under
   investigation. Treat the trace as *structural* only.
9. **When reading a trace, do not filter spans before concluding "nothing runs here".** A
   `dur > 30 µs` filter and a non-overlapping top-level selection both produced phantom
   gaps on this branch. Count *all* spans starting in the window, and remember enclosing
   frames make naive union-coverage 100 %.
10. **One change per run**, and **never build while a run is measuring** — a rebuild
    mid-run silently invalidates it.
11. **`fprintf` diagnostics from C++ carry no worker prefix**, so they cannot be attributed
    to a DP rank (§3.2).

---

## 5. Environment

### This box

| item | note |
|---|---|
| executor config | the scripts scope it per run via `VLLM_RBLN_EXEC_CONFIG_HOME=<OUTDIR>/.config/vllm-rbln-exec` and set `local_cache` / `remote_cache` / `hf_home` there, so the global `~/.config/vllm-rbln-exec` is never overwritten. `hf_home` must point at the shared cache or offline load fails |
| compile cache | the compiled model is read from `local_cache` by cache key, or from `--compiled-model-path` (`COMPILED_MODEL_PATH` in the scripts). Check it hits before starting; it is the difference between a few minutes and a lot of them |
| `--cache-ignore` | bypasses cached *inference results* only, not compiled models |
| shared devices | other users run here. `rbln-stat` before and after every run; a co-tenant has both silently degraded a run and taken every device (`this process sees 0 RBLN device(s)` at startup) |
| `perf` | `perf_event_paranoid=4` here → unprivileged `perf record` denied |
| `gdb` / `py-spy` attach | `ptrace_scope=1` → no attach to a running pid; descendants of the tracer are fine, so `py-spy record -- <cmd>` works |
| thread states without ptrace | sample `/proc/<pid>/task/<tid>/{stat,wchan}` — see §3.1. Match the worker by `/proc/<pid>/comm` (truncated to `VLLM::Worker_DP`), never by cmdline, which self-matches; gate the window on the run log showing `Processed prompts:` rather than a fixed sleep, or it lands in compile; abort on state `Z`, or a run ending mid-window silently deflates every percentage |

### Another machine

Three checkouts, all imported from source, so a stale one is silently used rather than
erroring:

| repo | note |
|---|---|
| `vllm-rbln` | branch `async-scheduling`, installed editable into the venv |
| `rebel_compiler` | must be **built** from a branch carrying the `Stream` mutex (`grep -q LockForDispatch rebel/include/rebel/runtime/base/stream.h`). Without it the off-main-thread D2H races the compute dispatch — it does not crash, it perturbs logits and flips near-tie argmax tokens, so a wrong build looks fine and produces quietly wrong comparisons. Installed **editable**, and `rebel_compiler/python/rebel` is a symlink to `rebel_compiler/rebel/python/rebel`, so Python edits take effect immediately while C++ edits need `./rebel_install.sh -a -n` |
| `vllm-rbln-executor` | provides the `vllm_rbln_executor.cli` driver and the venv. Invocation is `cli vllm-decoderonly gpt-oss <opts> rbln-run`; the flags used here are `--run-iter`, `--torch-profile`, `--profile-dir`, `--compiled-model-path`, `--cache-ignore`, `-o` |
| native golden | `remote_cache` copy at `/mnt/shared_data/users/yunseong.kim/nas_data/vllm-rbln-exec-golden`. `native_results_cache_key` excludes dp/ep/block_size/pcs/nblk, so the key is model_id + max_num_seqs + mt: `b8` + `--max-tokens 256` selects the 64-prompt golden. Off-key values silently compare against nothing |

```bash
python -c "import rebel, os; print(os.path.dirname(rebel.__file__))"   # from that checkout?
```

Model in the local HF cache (`HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`), and free RBLN
devices — confirm with `rbln-smi` and set `RBLN_DEVICES`. Scripts default to
`VENV=$HOME/codebase/vllm-rbln-executor/.venv`; override per invocation:

```bash
VENV=/path/to/.venv RBLN_DEVICES=0,1,2,3 OUTDIR=/path/for/logs ... ab_throughput.sh
```
