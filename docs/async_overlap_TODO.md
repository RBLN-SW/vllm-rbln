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

## TODO 1 — batch>1 parity on a FULL (non-truncated) model  [highest priority]
**Why open:** L18-truncated model's MoE/EP reduce-scatter is numerically order-sensitive and
its argmax is full of near-ties, so **sync itself** differs between `VLLM_RBLN_SORT_BATCH=0`
and `1` at batch>1 — token parity on L18 is meaningless. The deferred argmax was shown reliable
(device-vs-host mismatch = 0), so this is a *validation* gap, not a known bug.
**Do:** run a full (all-layers) gpt-oss on DP4+EP with `--batch N>1`, sync vs defer, compare
tokens per prompt. Include an **EOS/stop batch>1** case (the dummy-step fix should generalize,
but batch>1 EOS is untested).
**Pass:** defer == sync per prompt (greedy) at batch>1 on the full model.

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

## TODO 3 — full-model parity + perf (greedy, batch=1)
**Why open:** only L18-truncated measured. Confirm the ~19% overlap win and bit-exact parity
hold on the full model (more layers ⇒ longer forward ⇒ potentially larger overlap benefit, but
also different all_reduce/forward ratio).
**Pass:** defer bit-exact to sync + step latency ≤ sync on the full model.

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
