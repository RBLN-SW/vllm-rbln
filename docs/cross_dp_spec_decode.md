# Cross-DP Synchronization for MoE + Speculative Decoding

> **Naming note**: "sliding window" and "query backfill" refer to the
> same mechanism (see `docs/query_backfill.md`). Code-level identifiers
> keep `slide`/`sliding` naming; user-facing docs use `backfill`.

## 1. Problem

MoE models with data parallelism need every DP rank to step forward with
**the same `(batch, query_len)` model input shape**. The MoE layers fire
cross-rank all-reduce / all-to-all for expert dispatch, which requires
matching tensor shapes on every rank. Any divergence either hangs the
collective or triggers a hot-path `model_wrapper` recompile.

Speculative decoding makes per-rank local decisions diverge naturally:

- A rank with no useful ngram drafts this step locally wants
  `query_len = 1` (no-spec).
- A rank with drafts locally wants `query_len = num_spec_tokens + 1`
  (full-spec).
- A rank with a request near its KV block boundary used to want to drop
  drafts entirely to avoid KV writes crossing the block.

Local decisions alone disagree; the DP world cannot drive a single
compiled graph.

## 2. Key idea

Two mechanisms combine to keep every step driving a **single uniform
compile shape**:

1. **Bit-packed cross-DP all-reduce.** `RBLNDPMetadata.num_tokens_across_dp`
   packs `(is_prefill, num_reqs, num_tokens)` into one int32 and runs a
   single gloo all-reduce on the existing CPU group. After one reduce,
   every rank knows the per-rank vectors and lifts shape decisions via
   MAX — no extra collectives.

2. **Query backfill makes the lift trivial.** Every decode req's query
   window is unconditionally `num_spec_tokens + 1` (the
   boundary-affected req is padded by backfilling past tokens — see
   `docs/query_backfill.md`). With every rank already at the same
   per-req shape locally, the cross-DP MAX is a no-op invariant rather
   than a conflict resolver: there is only one shape to agree on.

Result: the runtime decode graph is a single
`(batch_bucket, num_spec_tokens + 1)` slot, plus prefill. The legacy
`step_no_spec_required` collective (used when drops were the only
option) is dead code, retained only for backward compatibility.

Constraints baked in at init:

- `(num_speculative_tokens + 1)` must be a power of two so MoE
  multicast tensor dimensions stay divisible.
- Sampler shape is independent of `model_wrapper`'s, so it is still
  warmed up separately for every batch size in `[1, max_decode_batch]`.

## 3. Example

DP=4, `num_speculative_tokens=3` → per-req `query_len = 4` everywhere.
Step `t`, rank-local proposer state diverges:

| Rank | Local drafts | Local proposal (old design) |
|---|---|---|
| DP0 | 3 ngram drafts | full-spec, q=4 |
| DP1 | 0 drafts (miss) | no-spec, q=1 (would diverge) |
| DP2 | 2 drafts | full-spec, q=4 |
| DP3 | req near boundary | drop-all (would diverge) |

Under query backfill, the scheduler reshapes each rank locally **before**
any cross-DP communication:

```
DP0:  base + 3 drafts                    → q=4  (full draft fit)
DP1:  3 past + base                      → q=4  (length-only pad)
DP2:  1 past + base + 2 drafts           → q=4  (length-only pad)
DP3:  3 past + base                      → q=4  (boundary squeeze)
```

All four ranks already arrive at the same per-req shape — the
bit-packed all-reduce then resolves only the **batch** dimension (max
`num_reqs` across DP, bucket lookup). No query-length lift is needed at
all; the cross-DP MAX returns `q=4` unconditionally.

The model_wrapper runs against the single warmup-compiled
`(batch_bucket, 4)` slot for every step — no recompile, no per-step
shape branching, no collective hangs.

---

## Appendix

### A1. Bit-packed cross-DP signal layout

`RBLNDPMetadata.num_tokens_across_dp` (`vllm_rbln/forward_context.py`):

```
bit 30           : is_prefill flag
bits 16..29 (14) : num_reqs       (max 16383)
bits 0..15  (16) : num_tokens     (max 65535)
```

One int32 per step, one all-reduce on the existing gloo CPU group.

### A2. Step shape decision (two phases)

1. **Local proposal** in `execute_model`: per-rank
   `spec_decode_max_query_len = num_spec_tokens + 1` whenever spec
   decode is configured (always, under always-full-spec backfill).
2. **Cross-DP collective MAX** in `get_dp_padding`: max over DP. With
   backfill, every rank already votes `num_spec_tokens + 1`, so the MAX
   is structurally a no-op for the query dimension; the batch
   dimension still goes through `max(num_reqs)` for bucket lookup.

DP-idle ranks (no real reqs this step) call `dummy_run` which reports
the same `num_spec_tokens + 1` to keep the MAX consistent — see the
runner's `dummy_run` implementation.

### A3. Legacy: cross-DP collective fallback (inactive)

`RBLNSchedulerOutput.step_no_spec_required` was the old boundary-trigger:
when any rank's req would have to drop drafts due to a block boundary,
all ranks OR-reduce the flag and collectively fall back to `q=1`. With
backfill the scheduler never sets this flag — drafts trim individually
without forcing every rank into no-spec. Kept around as dead code for
back-compat with any caller reading `step_no_spec_required`; removable
in a later cleanup.

### A4. Files touched

| Concern | File |
|---|---|
| Bit-packed cross-DP signal | `vllm_rbln/forward_context.py` (`RBLNDPMetadata`) |
| Fixed-length query + DP padding decision | `vllm_rbln/v1/worker/rbln_model_runner.py` (`execute_model`, `_prepare_inputs`, `dummy_run`) |
| Per-req backfill + legacy `step_no_spec_required` | `vllm_rbln/v1/core/rbln_scheduler.py` (`RBLNScheduler.schedule`) |
| Unit tests | `tests/torch_compile/unit/test_dp_padding.py`, `tests/torch_compile/unit/test_query_backfill.py` |

### A5. Quick reference

- One int32 per step, one all-reduce, no extra collectives.
- Decode query length at runtime is always `num_spec_tokens + 1` under
  backfill. (`q=1` is the no-spec compile slot, kept warm but never
  driven when spec decode is configured.)
- Warmup compiles prefill + one decode shape per `batch_bucket`.
- Boundary handling is per-request via query backfill; the legacy
  cross-DP collective fallback is dead code retained for back-compat.
