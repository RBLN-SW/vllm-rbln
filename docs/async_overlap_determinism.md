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

## batch=1 (DP4): FIXED — the deferred sampler's logprobs was the trigger

Earlier notes attributed batch=1 defer nondeterminism to accepted near-tie variation from
MoE-combine low-token/padding sensitivity. **That was wrong.** The real cause is computing
**logprobs inside the deferred sampler** during the overlap window; it is now fixed by not
computing those (discarded) logprobs there.

Evidence (batch=1 DP4 L18, `--repro-run` determinism):
- **Clean toggle**: defer + logprobs=16 → nondeterministic; defer + `--logprobs 0` →
  deterministic.
- **Isolated to the overlap**: `async1` (worker-thread forward, inline await, NO deferral)
  is deterministic *with logprobs on*. So logprobs compute alone is fine; overlap alone
  (no logprobs) is fine; only overlap × deferred-logprobs flips.
- **Sampler is faithful**: device argmax == host argmax at every step — so it is a
  device-runtime interaction between the overlap and the interposed logprobs graphs, not
  wrong sampling math.
- **Refuted hypotheses** (each still flipped): MoE-combine padding (re-zeroing the pad via
  `get_tokens_mask`), reduce_scatter-specific (RS=0 all_reduce flips too), the get_output
  copy-thread fix (pre-existing), compute_logprobs-corrupts-argmax (sampler reorder,
  argmax-first), per-op `RBLN_RUNTIME_FORCE_SYNC`, and disabling the caching allocator
  (`RBLN_DISABLE_EAGER_CACHE_ALLOC`). The exact device-runtime cause is still open.

**Fix** (`rbln_model_runner.py`, deferred thunk): drop `max_num_logprobs` for the deferred
sampler. The deferred path already returns `logprobs=None` (`_bookkeeping_sync` with
`sampler_output=None`), so those logprobs are discarded — dropping them removes both the
dead work and the nondeterminism trigger. Greedy argmax is unaffected — **token-neutral**:
batch=1 with-logprobs-stripped == with-logprobs-off, 4/4 bit-identical. **Result: batch=1
DP4 defer is now deterministic** (repro pass on clean code). Overlap + device sampler
preserved; the large full-vocab logprobs graphs are no longer run in the deferred window,
so this is also faster.

**Open (rebel-runtime follow-up)**: returning logprobs *under* overlap. That needs the
device-runtime interaction understood/fixed so the deferred sampler can compute logprobs
without perturbing the overlapped forward. Until then, defer returns `logprobs=None`
(unchanged pre-existing behavior).

## Reproduce
Scratchpad `runone.sh <tag> <mode> <RS> <batch> <np> <maxtok> <nrepro>` — modes `defer` /
`optsync` (unset async env + optimistic sched) / `syncref` (`RBLN_DYNAMO_ASYNC=0` +
optimistic = sync fwd) / `async1` (worker fwd + inline await) / `none` (plain sync). Add
`--logprobs 0` to disable logprobs. Env flakes are common on rapid re-runs (compiler MLIR
`setWeightHash`/`completeFuncInitGenResult` nodeID assertion, RCCL `ret=-12`, gloo
"Connection closed by peer") → between runs kill zombies via `ps` (not `pgrep -f`),
`find /dev/shm -maxdepth 1 -uid $(id -u) -delete`, confirm devices 0.0B, wait ~20s.
