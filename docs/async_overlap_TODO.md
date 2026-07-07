# Async-overlap — remaining functional-equivalence TODO

Companion to `async_overlap.md`. That doc describes the feature + what is **already
verified**; this file lists what is **not yet proven equivalent to dev's sync path**,
so a future session can close the gaps. Work items are ordered by priority.

## Verified so far (do NOT redo)
- **greedy**, batch=1, DP4+EP, L18 (truncated): sync-identical, deterministic, ~19% faster.
- Both **fixed-length** (`ignore_eos`) and **EOS/stop** (early, variable-length termination
  across DP ranks) → bit-exact to sync + deterministic.
- Feature gated behind `RBLN_DYNAMO_ASYNC=defer`; gate off ⇒ dev behavior.

## Standard test harness
Reuse the recipe in `async_overlap.md` §2 (env block + `parity_runner` ARGS). Parity check:
defer `token_ids` must equal sync `token_ids` per `global_prompt_index` (greedy). Between
runs: kill leftover procs **by PID** (avoid `pgrep -f` matching your own shell — it self-kills),
`find /dev/shm -maxdepth 1 -uid $(id -u) -delete`, confirm `rbln-smi` devices back to `0.0B`.
EOS/stop is exercised by adding `stop_token_ids` to the `SamplingParams` in `parity_runner`
(a `RBLN_TEST_STOP_TOKEN` env hook was used during development and then reverted — re-add
temporarily if needed).

---

## TODO 1 — batch>1 parity on a FULL (non-truncated) model  [INVESTIGATED 2026-07-07]
**Outcome: batch>1 defer is NOT bit-exact to sync, and the cause is NOT the deferral
feature — it is the async runtime's execution of the multi-token forward.** Details:

- **Sync IS deterministic** run-to-run on the full model at batch>1 (2 runs, 16/16 identical),
  unlike L18. So the reference is stable and divergence is meaningful.
- Full model DP4+EP `--batch 4 --num-prompts 4` (16 prompts), tok64, greedy: **defer diverges
  from sync on ~11/16 prompts** and is **nondeterministic run-to-run** (near-tie token flips,
  scattered positions). This is *not* the token-0 materialization race (that is fixed — 0 zeros).
- **Root cause isolated with `RBLN_DYNAMO_ASYNC=1`** (async I/O on the worker thread, but
  *blocking* await inline — no overlap, no deferred sampler): it *also* diverges from sync AND
  is nondeterministic at batch>1. So the divergence comes from running the multi-token forward
  **on the async worker thread** (vs sync's main thread), independent of the overlap/deferral.
  Likely the cross-rank MoE/EP reduce-scatter losing the tight main-thread lockstep (the per-step
  DP `num_tokens` all_reduce barrier) that keeps sync's FP reduction order deterministic.
- **Cannot be fixed by retiming the sampler.** A real fix would need the async worker's forward
  to reproduce sync's cross-rank ordering — out of scope for this feature and likely a
  runtime-level concern. `RBLN_DYNAMO_ASYNC=1` is the codebase's intended "functionally-sync
  validation" mode, so its batch>1 nondeterminism is arguably a runtime defect worth a separate fix.

**Conclusion:** batch=1 is the bit-exact-safe config (see TODO 3 — DONE). batch>1 with the async
forward produces valid-but-nondeterministic near-tie-different tokens. Recommend gating the
feature to batch=1 until the async-runtime multi-token nondeterminism is resolved.

**batch>1 EOS/stop:** not run (moot until batch>1 numerics are deterministic).

## TODO 2 — non-greedy sampling (temperature>0 / top-p / top-k)
**Why open:** the deferred thunk runs the **same** device sampler with all logits processors,
so it *should* be equivalent, but it is untested, and random sampling is non-deterministic by
nature (can't use "bit-exact" directly).
**Do:** either (a) fix the RNG seed and compare defer vs sync token-by-token, or (b) compare
output distributions over many samples. Confirm the `_defer_sampler` gate actually engages for
non-greedy (check whether anything implicitly assumes greedy/argmax), and that `skip_int32_cast`
+ deferred cast is correct for the sampled (not just argmax) output.
**Pass:** defer == sync under a fixed seed (greedy-equivalent determinism), or matched
distribution.

## TODO 3 — full-model parity + perf (greedy, batch=1)  [DONE 2026-07-07]
**Verified on full (all-layers) gpt-oss-120b DP4+EP, batch=1:**
- **Parity:** defer == sync, 4/4 bit-exact, and **deterministic** (run1==run2==sync).
- **Perf (tok128):** defer decode throughput beats sync **rank-for-rank** (output toks/s
  27.1→36.7, 28.1→37.0, 29.2→43.5, 46.3→48.5; +5%..+49%). Overlap win holds on the full model.
- This also confirms the token-0 C++ fix (see async_overlap.md §3) preserves batch=1
  bit-exactness and perf.
**Pass:** ✅ defer bit-exact to sync + faster than sync on the full model at batch=1.

## TODO 4 — spec-decode interaction
**Why open:** the `_defer_sampler` gate **excludes** spec-decode (deferral is skipped), so spec
runs fall back to the normal path. Confirm that fallback is correct (no perf regression, no
partial-deferral state leak), and decide whether deferral *should* support spec-decode.
**Pass:** spec-decode + `RBLN_DYNAMO_ASYNC=defer` behaves exactly as spec-decode without it.

## TODO 5 — auto-enable `RBLN_DYNAMO_ASYNC` (remove the manual toggle)
**Why open:** the feature is currently opt-in via env. Auto-enabling under async scheduling +
device sampler is cross-layer (rebel_compiler reads `RBLN_DYNAMO_ASYNC` at init) and needs
re-verification that the gate conditions are complete.
**Pass:** feature turns on automatically for the supported configs; all above parity checks
still pass; gate-off path unchanged.

## TODO 6 — operational: device flake / wedge (not a correctness item)
Warmup `Inference failed: 501/503`, `RCCL InitWithUniqueId failed`, engine-core init hangs,
"progress total size mismatch (IR changes during compilation)". Aggravated by rapid re-runs +
`kill -9`. Mitigation is in `async_overlap.md` §2 notes. Track only if it blocks the above.
