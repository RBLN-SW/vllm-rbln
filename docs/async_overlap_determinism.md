# Async-overlap: determinism analysis & the get_output race fix

Status 2026-07-09. Model `openai/gpt-oss-120b`, EP + DP4, L18. Async-overlap =
`RBLN_DYNAMO_ASYNC=defer` + `VLLM_RBLN_OPTIMISTIC_SCHED=1`.

## Validation criterion: per-input parity (NOT bit-exact)

Compare **per input** (input token sequence → output tokens), keyed on input CONTENT,
never on batch position/`global_prompt_index` — the optimistic scheduler reorders and
reindexes prompts, so an index-based diff shows spurious swaps (e.g. prompt 18's output
appearing at prompt 19). The bar is: **each input's output matches the sync reference on
the decisive tokens; near-tie flips in low-entropy tails (whitespace/repetition) are
accepted variation** (the known L18 numerical fragility). Bit-exact run-to-run
reproducibility is NOT required.

Under this criterion, async matches sync per input: divergence, when present, starts only
in low-entropy tails (e.g. `" preliminary whose"` vs `" preliminary! whose"`), never on the
confident early tokens.

## Two nondeterminism layers found

### MAJOR — optimistic scheduler batch-composition variance (upstream vLLM, known)
The optimistic scheduler steps non-blocking and can begin the first step before all
requests are ingested from the EngineCore IPC queue, so the number of requests admitted
before the first prefill varies run-to-run → prefill/decode interleaving varies → per-step
`num_tokens_across_dp` varies. On EP+DP the MoE forward is not invariant to batch
composition, so a different composition yields different logits → near-tie argmax flips.
This is a property of upstream `async_scheduling` (GPU vLLM batches non-deterministically
too), documented in the `RBLNAsyncScheduler` docstring. **Left as-is (known upstream); it
only causes accepted near-tie variation under per-input parity.**

### MINOR — get_output copy-thread D2H race (RBLN) — FIXED
`AsyncRBLNModelRunnerOutput.get_output()` ran on vLLM's `WorkerAsyncOutputCopy` thread and,
for non-deferred defer outputs, did `torch.rbln.synchronize` + `.to("cpu")` (device D2H)
there — concurrently with the worker-thread forward, on the shared device stream
(`Stream::last_seq_/has_pending_` are non-atomic). That perturbed the forward's logits and
increased near-tie flips. **Fix:** get_output now waits for the main-thread `capture_host`
(like the deferred path); it only falls back to a copy-thread D2H when no next step captures
it (final step → no concurrent forward → safe). Python-only; no rebel_compiler change; the
`RBLNAsyncScheduler` docstring documents MAJOR.

## Validation results (batch=8, DP4, L18)
- async forward == sync forward at identical batch composition: exact (32/32 prompts).
- per-input parity vs sync: matches on decisive tokens; only low-entropy-tail near-tie
  variation.
- perf (clean, no profiler): defer 124 tok/s vs plain sync 87.5 (+42%). Note: `--profile`
  massively distorts async (thread-heavy) — measure perf only without the profiler.
- perfetto traces (same L18/b8/tok16 spec): `perfetto_v2/{async,sync}_dp{0-3}.pt.trace.json.gz`.

## batch=1 (DP4): FIXED — force_sync the deferred sampler when logprobs are on

Earlier notes attributed batch=1 defer nondeterminism to accepted near-tie variation from
MoE-combine low-token/padding sensitivity. **That was wrong.** The real cause is the
**deferred sampler's logprobs graphs being in flight during the overlap window**: under
`defer` the sampler's compiled argmax and its logprobs graphs (`compute_logprobs` =
full-vocab `log_softmax`, `gather_logprobs`) are submitted non-blocking, so the main thread
races ahead to the next step while the async worker is still draining them. With the extra
logprobs graphs in flight, that overlap flips near-tie tokens run-to-run at batch=1. See
`async_overlap_batch1_rootcause.md` for the full analysis.

Evidence (batch=1 DP4 L18, `--repro-run` determinism):
- **Clean toggle**: defer + logprobs=16 → nondeterministic; defer + `--logprobs 0` →
  deterministic.
- **Isolated to the overlap/deferral**: `async1` (same worker, inline await, NO deferral)
  is deterministic *with logprobs on*. So logprobs compute alone is fine; overlap alone
  (no logprobs) is fine; only overlap × deferred-logprobs flips.
- **Sampler is faithful**: device argmax == host argmax at every step — a device-runtime
  interaction between the deferral and the interposed logprobs graphs, not wrong math.
- **Refuted hypotheses** (each still flipped): MoE-combine padding (re-zeroing the pad via
  `get_tokens_mask`), reduce_scatter-specific, the get_output
  copy-thread fix (pre-existing), compute_logprobs-corrupts-argmax (sampler reorder,
  argmax-first), per-op `RBLN_RUNTIME_FORCE_SYNC`, and disabling the caching allocator
  (`RBLN_DISABLE_EAGER_CACHE_ALLOC`).
- **A `RuntimeInstance::Run()` mutex is moot**: under `defer` every `backend="rbln"` compile
  — forward, argmax, and each eager single-op graph (torch_rbln recompiles each) — builds
  the async runtime and runs on the *one* `SharedAsyncWorker` FIFO thread (confirmed by a
  compile-time probe). All device submits are already serialized on that single thread, so
  there is no concurrent main-thread `Run()` to serialize. The nondeterminism is the
  deferral itself, not two threads racing inside `Run()`.

**Fix** (`rbln_model_runner.py`, deferred thunk): **keep logprobs computed** (do NOT strip
`max_num_logprobs`), and when logprobs are requested run the deferred `self._sample(...)`
under `rebel.sync_runtime.force_sync()` so the sampler's worker submissions complete inline
before the main thread races ahead — removing the concurrency. Only the (small) sampler is
made synchronous; the forward↔all_reduce overlap already happened in the prior step and was
drained before the deferred sampler runs, so overlap + device sampler are preserved. Gated
on logprobs-present, so no-logprobs greedy pays nothing. **Result (clean rebel 30d55c5):
batch=1 DP4 defer + logprobs is deterministic (4/4 repro identical) and 16/16 inputs
bit-identical to plain sync.** This supersedes the strip approach (commit `c38f37b4`, which
dropped the discarded logprobs); the strip is faster but forecloses returning logprobs.

**Open (rebel-runtime follow-up)**: returning logprobs *under* overlap. The deferred path
still returns `logprobs=None` (`_bookkeeping_sync` with `sampler_output=None`); the sampler
now computes them safely, so wiring the return is a follow-up.

## Reproduce
Scratchpad `runone.sh <tag> <mode> <RS> <batch> <np> <maxtok> <nrepro>` — modes `defer` /
`optsync` (unset async env + optimistic sched) / `syncref` (`RBLN_DYNAMO_ASYNC=0` +
optimistic = sync fwd) / `async1` (worker fwd + inline await) / `none` (plain sync). Add
`--logprobs 0` to disable logprobs. Env flakes are common on rapid re-runs (compiler MLIR
`setWeightHash`/`completeFuncInitGenResult` nodeID assertion, RCCL `ret=-12`, gloo
"Connection closed by peer") → between runs kill zombies via `ps` (not `pgrep -f`),
`find /dev/shm -maxdepth 1 -uid $(id -u) -delete`, confirm devices 0.0B, wait ~20s.
