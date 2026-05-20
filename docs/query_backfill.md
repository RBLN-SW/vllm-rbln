# Spec-Decode Query Backfill

> **Naming note**: this mechanism was previously referred to as the
> **"sliding window"** approach. The two terms are equivalent. Code-level
> identifiers (variables, fields, helper functions such as
> `slide_distance`, `spec_decode_slide_distance`, `_run_sliding_math`)
> keep the `slide`/`sliding` naming for stability across modules;
> user-facing docs, log prefixes, and the env var
> `VLLM_RBLN_BACKFILL_TRACE_REQS` use the **backfill** naming.

## 1. Problem

Speculative decode on RBLN previously had to support **two query shapes**
per decode step:

- `query_len = num_spec_tokens + 1` for the full-spec path, and
- `query_len = 1` for the boundary-induced no-spec fallback (when
  `remaining_in_block` or `remaining_in_maxlen` cannot fit the full
  window, or when a variable-length proposer like ngram returns fewer
  than `num_spec_tokens` drafts).

This forced two compile variants of the decode graph, a cross-DP
`step_no_spec_required` OR-reduce to agree on which shape to drive each
step, and runtime branching that the `specialized_moe_decode` path could
not reconcile in a single graph.

## 2. Key idea

The scheduler unconditionally keeps every decode step's query window at
`num_spec_tokens + 1` by **back-filling the deficit with past positions
whose KV is already in the current block**:

```
slide_distance = max_spec_decode_len − min(old_n, effective_remaining)
              ↑ 0 when the proposer fills the cap off the boundary,
                up to num_spec_tokens at the tightest boundary squeeze.
```

The runner prepends `slide_distance` past tokens to `input_ids` /
`positions`, the model re-runs them through attention (an **idempotent
KV re-write**), and the past positions' logits are pruned out of
`logits_indices` so the rejection sampler never sees them. The decision
is **per-request and purely local**: no cross-DP collective is needed
because every rank arrives at the same shape independently.

Result: the runtime decode graph is a single `(batch, num_spec_tokens +
1)` shape — the `no_spec` variant and the `step_no_spec_required`
collective are unused.

## 3. Examples (running trace)

All three examples are real `vllm bench serve` traces on MiniMax-M2.5
(DP=4, `block_size=1024`, `num_speculative_tokens=3`, ngram proposer,
`random-output-len=1500` so every request crosses block 4 / 5).

### Case A — ngram miss, three consecutive backfill steps

Request near the 1024 boundary; ngram returns no drafts at any of the
three steps. Backfill still pads the window to length 4 so the runtime
shape stays uniform:

```
step1   num_computed=1021  slide=1  flat=[0,2)   positions=[1020,1021]
        input_ids=[367, 85794]                    num_draft_tokens=0
step2   num_computed=1022  slide=2  flat=[0,3)   positions=[1020,1021,1022]
        input_ids=[367, 85794, 9344]              num_draft_tokens=0
step3   num_computed=1023  slide=3  flat=[0,4)   positions=[1020,1021,1022,1023]
        input_ids=[367, 85794, 9344, 106349]      num_draft_tokens=0
```

Past tokens are exactly the ones the previous step sampled and committed
(`85794` at step1's base reappears at step2 position 1021, etc.) —
idempotent re-feed. `slot_blocks=[4]` every step → no write crosses
into block 5.

### Case B — ngram hit, drafts kept

```
slide=1  flat=[4,8)  positions=[1020,1021,1022,1023]
         input_ids=[128365, 170513, 103462, 147932]
         logits_in_req=[5,6,7] (relative=[1,2,3])  num_draft_tokens=2
```

`input_ids[0]` is the past at position 1020; the other three are base +
two ngram drafts. The past relative index `0` is excluded from
`logits_indices`. `num_draft_tokens=2` matches `kept_drafts=2`.

### Case C — ngram hit, drafts dropped by backfill (the central payoff)

`slide=3` at the tightest boundary squeeze drops all 3 proposed drafts:

```
scheduler:  num_computed=1023  slide=3  advance=1  proposed_drafts=3  kept_drafts=0
runner:     positions=[1020,1021,1022,1023]
            input_ids=[761, 761, 761, 0]
            logits_in_req=[3] (relative=[3])  num_draft_tokens=0
```

Only the base position sees the sampler. The 4-token window is 3 past +
1 base. `num_draft_tokens=0` correctly reflects post-trim count, so the
rejection sampler verifies zero drafts — no spurious work.

---

## Appendix

### A1. Diagnostic logging

Two log prefixes, both at DEBUG level (default-off; enable via standard
logging config). The runner trace is additionally gated by env var so it
stays off in production unless explicitly requested.

**Scheduler** (`rbln_scheduler.py`):
```
spec-decode backfill: req=<id>
  num_computed=<N> remaining_in_block=<R> remaining_in_maxlen=<M>
  slide=<S> advance=<A> proposed_drafts=<P> kept_drafts=<K>
```

**Runner** (`rbln_model_runner.py`, gated by
`VLLM_RBLN_BACKFILL_TRACE_REQS=N` for the first `N` reqs per worker):
```
backfill-trace req=<id> slide=<S> flat_range=[<s>,<e>)
  positions=<list> input_ids=<list> slot_blocks=<list>
  logits_indices_in_req=<absolute> (relative_to_req=<offset>)
  num_draft_tokens=<actual>
```

Field semantics:
- `slide`: how many past positions are prepended.
- `advance` = `min(old_n, effective_remaining)`: new
  `num_scheduled_tokens` cap (1 base + up to `advance-1` drafts).
- `proposed_drafts` = `old_n-1`: drafts the proposer returned **before**
  any trim.
- `kept_drafts` = `max(advance-1, 0)`: scheduler-side cap.
- `num_draft_tokens` (runner): **actual** drafts that reach the
  rejection sampler — matches `kept_drafts` when the proposer fills the
  cap, less otherwise.
- `slot_blocks`: distinct KV blocks the slot mapping touches. **Always a
  singleton** under backfill — no write crosses the boundary.

### A2. Drop-rate observed in a 16-prompt bench

`output_len=1500` so every prompt crosses block 4 / 5 at least once.
Over 35 backfill events:

| Slide | `kept_drafts` cap | Cases with `proposed_drafts=3` | Drops per case |
|---|---|---|---|
| 1 | 2 | 7 | 1 |
| 2 | 1 | 6 | 2 |
| 3 | 0 | 6 | 3 |

Of 19 events with a non-empty ngram proposal, **20 drafts dropped
total**, exactly matching `Σ (proposed_drafts − kept_drafts)`.

### A3. Cross-case invariants confirmed (16-prompt bench)

| Invariant | Evidence |
|---|---|
| No KV crosses boundary | `slot_blocks` singleton in 27/27 traced events; 0 errors across 41 scheduler events |
| Past tokens re-fed idempotently | Step-N `input_ids` = step-(N-1) extended by exactly one new tail token |
| Past positions never reach logits | `relative_to_req` always starts at `slide`, never 0 |
| Effective drafts surface to sampler | `num_draft_tokens` matches `kept_drafts`-trimmed count |
| Per-step monotonic advance | `num_computed` increases monotonically across actual model forwards |
| No regression | 16/16 (small) and 256/256 (stress) bench succeed, zero `IndexError` / `Worker failed` |

### A4. How to reproduce

```bash
export VLLM_RBLN_BACKFILL_TRACE_REQS=3
vllm serve MiniMaxAI/MiniMax-M2.5 \
  --data-parallel-size 4 --enable-expert-parallel \
  --max-model-len 196608 --block-size 1024 \
  --max-num-seqs 8 --enable-chunked-prefill \
  --speculative-config '{"method":"ngram","num_speculative_tokens":3,
                         "prompt_lookup_max":5,"prompt_lookup_min":2}'

vllm bench serve --model MiniMaxAI/MiniMax-M2.5 \
  --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
  --dataset-name random --random-input-len 512 --random-output-len 1500 \
  --num-prompts 16 --max-concurrency 16 --request-rate 4

grep "spec-decode backfill\|backfill-trace" server.log
```

### A5. Notes

- The trace flag defaults to `0` so production incurs no logging
  overhead. Instrumentation lives in `_prepare_inputs` (CPU host code)
  outside any compiled graph — safe to keep checked in.
- `kept_drafts` (scheduler) is a cap; `num_draft_tokens` (runner) is
  the post-trim actual count. They agree only when the proposer fills
  the cap.
- Scheduler log entries occasionally repeat the same `num_computed`
  (~0.6% in our bench). This is vLLM v1's scheduler running its
  decision logic more than once per engine step; per-step
  instrumentation confirmed every backfill step's sampler committed at
  least one token, no backward moves, bench 256/256 succeeded — a
  measurement artifact, not a correctness issue.
