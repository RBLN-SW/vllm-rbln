# Sub-Block Prefix Caching

Upstream vLLM prefix caching works only at the granularity of fully filled blocks.
However, RBLN uses large KV cache blocks (1k–4k tokens),
which can cause substantial prefix cache misses in the last, partially filled block.

To address this,
sub-block prefix caching adds a finer-grained (e.g., 128-token) caching layer
on top of the upstream `KVCacheManager`.
It first applies the upstream full-block prefix caching,
then attempts to *extend* the cache hit at sub-block granularity.
The matched sub-blocks are copied into a new block,
since the block containing the partial matches cannot be reused directly.

Sub-block prefix caching is automatically enabled when
`enable_prefix_caching=True` and `VLLM_RBLN_SUB_BLOCK_CACHE=true` (default)
and all KV cache groups have an *eligible spec type* with
`block_size > sub_block_size` and `block_size % sub_block_size == 0`.

A KV cache spec is *eligible* if it stores per-token KV data, i.e.,
if the KV cache tensor has a token dimension that can be sliced for partial copying.
* Eligible: `FullAttentionSpec`, `SlidingWindowSpec`, `ChunkedLocalAttentionSpec`
* Ineligible
    * `MambaSpec`: It stores one accumulated SSM state per block
      (the checkpoint after processing `block_size` tokens).
      This is not per-token data.
    * `CrossAttentionSpec`: Prefix caching is entirely disabled for this.


## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `VLLM_RBLN_SUB_BLOCK_CACHE` | `true` | Enable sub-block prefix caching. |

The `sub_block_size` is automatically set to prefill chunk size (`max_num_batched_tokens`)
so that each prefill does not span multiple blocks.

## Key components

```
┌─────────────────────────────────────────────────────────┐
│  RBLNScheduler                                          │
│  ┌───────────────────────────────────────────────────┐  │
│  │  RBLNKVCacheManager (extends KVCacheManager)      │  │
│  │  ┌───────────────┐  ┌──────────────────────────┐  │  │
│  │  │ SubBlockHasher│  │ Per-group SubBlockIndex  │  │  │
│  │  │ (chained hash │  │ (hash → containing)      │  │  │
│  │  │  per sub-blk) │  │         block_ids        │  │  │
│  │  └───────────────┘  └──────────────────────────┘  │  │
│  └───────────────────────────────────────────────────┘  │
│          ↕ arbitration (connector vs sub-block)         │
│  ┌───────────────────────────────────────────────────┐  │
│  │  KV Connector (optional)                          │  │
│  └───────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
              │  RBLNSchedulerOutput (kv_cache_copy_ops)
              ▼
┌─────────────────────────────────────────────────────────┐
│  RBLNModelRunner                                        │
│  execute_model() processes copy ops (memcpy sub-block   │
│  KV data between physical blocks) before forward pass   │
└─────────────────────────────────────────────────────────┘
```

* `vllm_rbln.v1.core.rbln_kv_cache_manager`
   * `SubBlockHasher`:
     Computes chained hashes at sub-block granularity.
     Uses the same `hash_block_tokens` as upstream.
   * `SubBlockIndex`:
     Maps sub-block hashes to sets of physical block IDs containing the sub-block.
     Supports `update`, `pop`, and `longest_match`.
   * `RBLNKVCacheManager`: Extends upstream `KVCacheManager`.
        * Overrides
            * `allocate_slots` queues the request for sub-block indexing work
              to be processed by `do_pending_indexing`.
            * `free` indexes all blocks (full + partial) for the finishing request
              and assigns a synthetic `block_hash` to the partial block.
            * `reset_prefix_cache` clears sub-block indices and pending indexing.
        * New methods
            * `get_computed_blocks_sub_block(request, num_computed_tokens)`
              discovers sub-block matches and returns a `SubBlockMatch` handle.
            * `apply_sub_block_match` / `release_sub_block_match` consume or discard the handle.
            * `drain_pending_copy_ops()` retrieves the KV cache copy ops accumulated in the current scheduling step.
            * `release_copy_ops()` releases the source-block references after the model runner finishes copying.
            * `do_pending_indexing()` indexes sub-blocks for requests for which
              `allocate_slots` was called in the current scheduling step.
              Must be called after `super().update_from_output()`.
            * `can_use_sub_block_caching()` checks eligibility.
   * `KVCacheCopyOp`: Dataclass describing a sub-block KV data copy:
     `(group_id, src_block_id, dst_block_id, num_tokens)`.
* `vllm_rbln.v1.core.rbln_scheduler`
   * `RBLNSchedulerOutput`: Extends `SchedulerOutput` with `kv_cache_copy_ops` field.
   * `RBLNScheduler.__init__`: Creates `RBLNKVCacheManager` when prefix caching is
     enabled and `can_use_sub_block_caching()` passes.
   * `RBLNScheduler.schedule`: Returns `RBLNSchedulerOutput`, draining copy ops from the manager.
   * `RBLNScheduler.update_from_output`: Calls `super().update_from_output()`
     (updates `num_computed_tokens`, `free()`s finished requests),
     then `do_pending_indexing` and releases source-block refs from copy ops.
* `vllm_rbln.v1.worker.rbln_model_runner`
   * `_process_kv_cache_copy_ops`: Copies KV data between blocks before the forward pass.

## How it works

### Multi-group support

Sub-block caching supports both single-group (`UnitaryKVCacheCoordinator`) and
multi-group (`HybridKVCacheCoordinator`) setups. Each KV cache group
independently maintains its own `SubBlockIndex`.

Since `num_computed_tokens` must agree across all groups,
sub-block caching either works for all groups or is disabled entirely, and
the extension of `num_computed_tokens` by the sub-block match is the
**minimum** match length across all groups.

### Step 1: Sub-block hash computation

`SubBlockHasher` splits a request's token sequence into fixed-size sub-blocks
and computes a chained hash for each, similarly to upstream's block hashing.

Hashes are computed incrementally and cached per-request in
`RBLNKVCacheManager._req_sub_hashes`.

Note that we do not exploit upstream's `hash_block_size`,
because `UnitaryKVCacheCoordinator` asserts `hash_block_size == block_size`.

### Step 2: Index maintenance

Sub-block indexing is **deferred** until after the forward pass writes KV data.
This ensures that concurrent prefills in the same scheduling step cannot match
sub-blocks whose KV data has not yet been computed and thus should not be copied.

During `allocate_slots`, requests are queued for deferred indexing.
`RBLNScheduler.update_from_output` first calls `super().update_from_output()`
(which updates `num_computed_tokens` and `free()`s finished requests),
then calls `do_pending_indexing` for the remaining running requests.

`free()` consumes its own pending-indexing entry, indexes all blocks
(full + partial), and assigns a synthetic `block_hash` to the partial block
so upstream's LRU preserves it for potential reuse.

`do_pending_indexing` processes each remaining running request:
1. Indexes newly cached full blocks' sub-block hashes into the per-group `SubBlockIndex`.
   Each hash at depth *k* maps to the set of physical blocks whose first *k* sub-blocks match.
2. Indexes complete sub-blocks within the request's current partial block,
   making them visible for matching by new requests.

### Step 3: Partial-match lookup

`get_computed_blocks_sub_block(request, num_computed_tokens)`
1.  Compute sub-block hashes for the request
2.  For each group:
    1. Starting after the last full-block boundary,
       query the group's index with up to `sub_blocks_per_block − 1` hashes
    2. Record the match length
3.  Extension = min match length across all groups
4.  If match found → return a `SubBlockMatch` handle
    (with per-group `_GroupPartialMatch`),
    bump `src_block`s' ref counts to prevent eviction
5.  Return the sub-block extension in tokens (0 if no match),
    capped at `request.num_tokens − 1` to preserve the upstream
    "must recompute last token" invariant.

The query is limited to `sub_blocks_per_block - 1` because a match of all
sub-blocks would be a full-block match, which upstream handles already. The
cap at `num_tokens - 1` ensures the scheduler always schedules at least one
token for computation (upstream requires the last token to be recomputed to
produce logits).

### Step 4: Copy op scheduling

After `allocate_slots` succeeds, the scheduler calls
`apply_sub_block_match(match, request)` which, for each group:
1.  Looks up the destination block (newly allocated at the match boundary)
2.  Appends `KVCacheCopyOp(group_id, src_block_id, dst_block_id, num_tokens)`

(`allocate_slots` itself only queues deferred sub-block indexing.)

### Step 5: Copy execution (model runner)

The scheduler returns `RBLNSchedulerOutput` containing `kv_cache_copy_ops`.
Before the forward pass, the model runner copies sub-block KV data:

```python
# For each copy op targeting a specific group's layers:
# kv_cache tensor (shape: 2, num_blocks, H, 1, block_size, D):
kv_cache[:, dst_block_id, :, :, :num_tokens, :] = \
    kv_cache[:, src_block_id, :, :, :num_tokens, :]
```

### Block lifecycle

- **Indexing running requests**:
  Scheduled by `allocate_slots`, then executed by `do_pending_indexing`
  (called after `super().update_from_output()`).
  Indexes both full blocks and complete sub-blocks within partial blocks.
- **Indexing finished requests**: `free()` consumes the pending-indexing
  entry, indexes all blocks, and assigns a synthetic `block_hash` to the
  partial block so the upstream LRU preserves it.
- **Eviction**: A monkey-patched eviction hook calls
  `SubBlockIndex.pop()` whenever the upstream evicts a cached block.
- **Reset**: `reset_prefix_cache()` clears all sub-block indices and pending
  indexing queue.

## Interaction with KV Connectors

Sub-block prefix caching and KV connectors are conflicting in that
both attempt to extend the full-block prefix cache hit with additional tokens.
Ideally, they should cooperate:
sub-block match first extends the cache hit,
then the connector extends the match further from there.
However, sub-block caching is RBLN custom extension,
so we cannot assume that all KV connectors will work properly
when the starting position of their match is not full-block-aligned.

**Compromise: mutual-exclusion arbitration.**
The scheduler picks whichever offers more tokens.
On ties, sub-block wins because a local memcpy is cheaper than a remote transfer.
The worst case performance loss due to this compromise is the diff between
a remote transfer of `block_size - sub_block_size + 1` tokens (current) vs.
a local memcpy of `block_size - sub_block_size` tokens + a remote transfer of 1 token (ideal).

In the future, we can allow-list KV connectors
that are verified to be compatible with sub-block caching,
or patch existing connectors as needed.

## Limitations

- KV cache copy implementation (`_copy_kv_cache`) limitations
    - **Per-tensor copy not yet implemented.**
      Currently it copies for all layers, so it is not applicable to multi-group setups.
    - **Synchronous.**
      Currently it is blocking.
