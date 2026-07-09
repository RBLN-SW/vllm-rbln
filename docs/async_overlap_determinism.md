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

## batch=1 (DP4): near-tie variation, within the per-input-parity bar

batch=1 flips near-tie tail tokens run-to-run more often than batch=8. Investigation
(session 2026-07-09) established this is **pre-existing (reproduces on the ORIGINAL committed
code with the session's fix git-stashed) and NOT caused by the get_output fix**. It is also:
NOT batch composition (identical across gens with the ingestion pinned), NOT cross-thread
device concurrency (thread timeline is main-thread-serial; get_output takes the wait path),
NOT prompt transitions (`--num-prompts 1` also flips). It is DP4-collective + low-token
specific (DP1 deterministic, batch=8 deterministic, only batch=1 DP4 flips): at batch=1
decode each rank contributes 1 token to the MoE combine (1 real token + 63 padding per
64-block), suggesting low-count/padding sensitivity in the collective (CCL/kernel domain;
RBLN CCL reduction is stated order-independent). **Under per-input parity this is accepted
near-tie variation, not a correctness bug.** If bit-exact batch=1 is ever required, start
from the MoE combine padding hypothesis (§ next-session below) — but any change MUST be
re-verified at batch=8 (10-run, cache on/off) to prove no regression.

Diagnostic note: an `RBLN_DETERMINISTIC_ADMIT` env harness (patches upstream vLLM
`EngineCore._process_input_queue` to pin ingestion before the first step) was used during
investigation to isolate MAJOR from MINOR. It is NOT a shipping requirement and is NOT
needed for per-input-parity validation.

## Reproduce
Scratchpad `repro.sh <mode> <RS> <batch> <np> <maxtok> <nrepro>` — modes `defer` / `optsync`
(sync fwd + optimistic sched) / `async1` (worker fwd + inline await) / `none` (plain sync);
env `MB=<bucket>` for a single decode bucket, `NL` for `--num-hidden-layers`. For per-input
parity use the parity_runner golden comparison (compares each prompt's output to its sync
reference by prompt identity). Env flakes are common on rapid re-runs (compiler MLIR
`setWeightHash`/`completeFuncInitGenResult` nodeID assertion on fresh MB=1/MB=4 compiles,
RCCL `ret=-12`, gloo "Connection closed by peer") → between runs kill zombies via `ps` (not
`pgrep -f`), `find /dev/shm -maxdepth 1 -uid $(id -u) -delete`, confirm devices 0.0B, wait.

## Next-session (only if bit-exact batch=1 is wanted)
1. Device-dump the MoE combine input/output at batch=1 across gens — is the INPUT identical
   but OUTPUT different? Is the padding region non-zero / varying?
2. Try zeroing the `fused_moe` send/recv buffers (`fused_moe/layer.py`, `all2all.py`); if it
   is in the compiled graph, re-verify no recompile break AND re-verify batch=8.
3. Compare reduce_scatter (RS=1) / all_reduce (RS=0) / all2all combine at batch=1.
4. If it bottoms out in the CCL binary, escalate to the RBLN CCL/runtime team.
