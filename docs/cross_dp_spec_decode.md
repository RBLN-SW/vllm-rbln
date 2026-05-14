# Cross-DP Synchronization for MoE + Speculative Decoding

## Why this exists

MoE models running with data parallelism need every DP rank to step
forward with the **same model_wrapper input shape**. The MoE layers
issue cross-rank all-reduce / all-to-all for expert dispatch, and
those collectives require matching tensor shapes on every
participating rank. Any rank-local shape divergence will hang the
all-reduce or, at best, force a hot-path model_wrapper recompile.

Speculative decoding makes per-rank **local decisions** diverge
naturally:

- A DP rank whose batch happens to have **no useful ngram drafts**
  this step locally chooses **no-spec** (`query_len = 1` per req —
  just the base position).
- A DP rank whose batch **does** have drafts locally chooses
  **full-spec** (`query_len = num_spec_tokens + 1` per req — base +
  drafts to verify).

Local decisions alone are not enough. If one rank runs no-spec while
another runs full-spec at the same step, their `(batch, query_len)`
model inputs disagree and the MoE collectives in that step break (or
the disagreement triggers a hot-path model_wrapper recompile).

The cross-DP collective here exists to **lift every rank's local
decision into a single consistent global decision** for that step,
so the MoE model_wrapper compiles only against the shapes the DP
world will actually drive together — never against a rank-local
disagreement.

## Building block: bit-packed cross-DP all-reduce

`RBLNDPMetadata.num_tokens_across_dp` (`vllm_rbln/forward_context.py`)
packs three per-rank scalars into a single `int32` and runs **one**
all-reduce on the existing gloo cpu group:

```
bit 30           : is_prefill flag
bits 16..29 (14) : num_reqs
bits 0..15  (16) : num_tokens
```

After one reduce, each rank has the per-DP-rank vectors for
`num_tokens` and `num_reqs`, and the global is_prefill flag — enough
to decide a common bucket and padding without further communication.

This same channel carries the cross-DP signals used by the
spec-decode path below.

## Fixed-length DP-aware speculative decoding

Commit `907f5090` constrains the runtime query length to **exactly
two values**: `1` (no spec) or `num_spec_tokens + 1` (full spec).
Intermediate values used to be possible (any draft count from
`scheduled_spec_decode_tokens`) and forced model_wrapper recompiles
because `(input_ids[1], max_pads)` for MoE multicast must be uniform
and divisible.

Each step's shape is determined in two phases:

1. **Local proposal** (`execute_model`): per-rank
   `spec_decode_max_query_len` is `num_spec_tokens + 1` when this
   rank has any draft tokens this step (full-spec), otherwise `1`
   (no-spec). This is what the rank would do if it could run alone.
2. **Cross-DP collective MAX** (`get_dp_padding`): the local
   proposals are MAX-reduced across DP. If **any** peer proposed
   full-spec, every rank's per-req length is lifted to
   `num_spec_tokens + 1` so the DP world ends the step with a
   uniform compile shape. The lifted (originally no-spec) ranks pad
   their `(batch, 1)` input out to `(batch, num_spec_tokens + 1)`
   using lookahead-allocated slots; pad-position outputs are
   discarded by the rejection sampler.

Batch dimension goes through the same collective: bucket lookup
uses `max(num_reqs)` across DP so the MoE `max_pads_across_dp`
buffer always fits. Warmup compiles cover prefill plus the
(q=1, q=num_spec_tokens+1) decode shapes — five model_wrapper
slots total — exactly the shapes the collective can pick.

Constraints baked in at init time:

- `(num_speculative_tokens + 1)` must be a power of two so the MoE
  multicast tensor dimensions stay divisible.
- Sampler shape is independent of model_wrapper shape, so it's still
  warmed up for every batch size in `[1, max_decode_batch]`.

## The block boundary problem

A decode request whose `num_computed_tokens % block_size` is close to
`block_size` has fewer than `num_spec_tokens + 1` slots left in its
current KV block. Stepping it with the full spec window would write
KV past the block boundary into a slot that hasn't been allocated to
this request — corruption.

Two designs have been used:

### Legacy: cross-DP collective fallback (kept inactive)

The scheduler sets `RBLNSchedulerOutput.step_no_spec_required = True`
whenever **any** local decode req would have to drop drafts due to a
boundary. The runner OR-reduces that flag across DP using the bit-
packed channel above. If **any** rank votes True, **every** rank
scrubs its drafts (`scheduled_spec_decode_tokens.clear()`, all
`num_scheduled_tokens` → 1) before `_prepare_inputs` runs. The whole
batch falls back to `query_len = 1` that step.

Why collective and not per-req: under the fixed-length design above,
cross-DP MAX still lifts everyone to `num_spec_tokens + 1` if even
one rank stays in spec mode. Padding the boundary-hit rank up to
that length writes pad-position KV past its block — exactly what we
were trying to avoid. The only safe options under the OG constraint
were "everyone full spec" or "everyone no spec," hence the OR-reduce.

This path is still wired up but never fires now — the scheduler does
not set `step_no_spec_required = True` under the sliding-window
design. Kept around for backward compatibility with any code reading
`step_no_spec_required`; removable in a later cleanup.

### Current: per-request sliding window

Instead of dropping drafts when a request nears the boundary, the
scheduler slides the **query window backward** by `slide_distance`
past positions whose KV is already in the current block. The runner
re-feeds those past tokens (idempotent KV re-write) and excludes
their logits from sampling. Each request keeps a full
`num_spec_tokens + 1` query window without ever writing KV past its
block boundary, so cross-DP MAX is happy. No collective is needed.

See `docs/sliding_window.md` for the trace-level walkthrough and the
empirical verification.

## Files touched

| Concern | File |
|---|---|
| Bit-packed cross-DP signal | `vllm_rbln/forward_context.py` (`RBLNDPMetadata`) |
| Fixed-length query / DP padding decision / boundary handling | `vllm_rbln/v1/worker/rbln_model_runner.py` (`execute_model`, `_prepare_inputs`) |
| Per-req boundary decision (sliding) and legacy `step_no_spec_required` | `vllm_rbln/v1/core/rbln_scheduler.py` (`RBLNScheduler.schedule`) |
| Unit tests | `tests/torch_compile/unit/test_dp_padding.py`, `tests/torch_compile/unit/test_sliding_window.py` |

## Quick reference

- One int32 per step, one all-reduce, no extra collectives.
- Decode query length at runtime is always 1 or `num_spec_tokens + 1`.
- Warmup compiles 5 model_wrapper slots (prefill + 4 decode variants).
- Boundary handling is per-request via sliding window; the legacy
  cross-DP collective fallback is dead code retained for back-compat.
