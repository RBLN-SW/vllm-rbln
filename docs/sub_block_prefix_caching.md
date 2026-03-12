# Sub-Block Prefix Caching

Upstream vLLM prefix caching operates at the granularity of fully filled blocks.
However, RBLN uses large KV cache blocks (1k–4k tokens),
leading to large missed prefix cache hits at the last (partially filled) block.

Sub-block prefix caching adds a finer-grained (e.g. 128-token) caching layer
on top of the upstream `KVCacheManager`.
It detects when a request's prefix matches one or more *sub-blocks*
of a previously cached block and copies the matching KV data
into the new request's block.

Sub-block prefix caching is automatically enabled when
`enable_prefix_caching=True` and all KV cache groups have an *eligible spec
type* with `block_size > VLLM_RBLN_SUB_BLOCK_SIZE > 0` and `block_size` evenly
divisible by `VLLM_RBLN_SUB_BLOCK_SIZE`.

A KV cache spec is *eligible* if it stores per-token KV data, meaning the KV
cache tensor has a token dimension that can be sliced for a partial copy.
* Eligible: `FullAttentionSpec`, `SlidingWindowSpec`, `ChunkedLocalAttentionSpec`
* Ineligible
    * `MambaSpec`: It stores one accumulated SSM state per block
      (the checkpoint after processing `block_size` tokens).
      This is not per-token data.
    * `CrossAttentionSpec`: Prefix caching is entirely disabled for this.


## Configuration

| Environment variable | Default | Description |
|---|---|---|
| `VLLM_RBLN_SUB_BLOCK_SIZE` | `128` | Sub-block size in tokens. Must evenly divide every group's `block_size`. 0 to disable sub-block caching. |

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
     Supports `insert`, `pop`, and `longest_match`.
   * `RBLNKVCacheManager`: Extends upstream `KVCacheManager`. Overrides
     `get_computed_blocks` (adds sub-block matching after full-block matching),
     `allocate_slots` (schedules copy ops), and
     `free` (caches partial blocks).
     The sub-block machinery is hidden in the manager for compatibility with the original interface,
     except that the scheduler should `drain_pending_copy_ops()` to retrieve
     the KV cache copy ops accumulated in the current scheduling step.
     Additionally, the manager provides `can_use_sub_block_caching()` for checking eligibility.
   * `KVCacheCopyOp`: Dataclass describing a sub-block KV data copy:
     `(group_id, src_block_id, dst_block_id, num_tokens)`.
* `vllm_rbln.v1.core.rbln_scheduler`
   * `RBLNSchedulerOutput`: Extends `SchedulerOutput` with `kv_cache_copy_ops` field.
   * `RBLNScheduler.__init__`: Creates `RBLNKVCacheManager` when prefix caching is
     enabled and `can_use_sub_block_caching()` passes.
   * `RBLNScheduler.schedule`: Returns `RBLNSchedulerOutput`, draining copy ops from the manager.
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

When the upstream `allocate_slots` caches a full block (assigns it a
`block_hash`), `RBLNKVCacheManager` also inserts that block's sub-block hashes
into the per-group `SubBlockIndex`. Each hash at depth *k* maps to the set of
physical blocks whose first *k* sub-blocks match.

The last partial block of a request (the one that doesn't fill a full block)
is indexed during `free()` per group and given a synthetic `block_hash` so
upstream's LRU preserves it for potential reuse.

### Step 3: Partial-match lookup (get_computed_blocks)

`get_computed_blocks(request)`
1.  Call upstream → full-block matches, `num_computed_tokens`
2.  Compute sub-block hashes for the request
3.  For each group:
    1. Starting after the last full-block boundary,
       query the group's index with up to `sub_blocks_per_block − 1` hashes
    2. Record the match length
4.  Extension = min match length across all groups
5.  If match found → record `_PendingPartialMatch`
    (with per-group `_GroupPartialMatch`),
    bump `src_block`s' ref counts to prevent eviction
6.  Return `num_computed_tokens` + matched sub-block tokens
    (capped at `request.num_tokens − 1` to preserve the upstream
    "must recompute last token" invariant)

The query is limited to `sub_blocks_per_block - 1` because a match of all
sub-blocks would be a full-block match, which upstream handles already. The
cap at `num_tokens - 1` ensures the scheduler always schedules at least one
token for computation (upstream requires the last token to be recomputed to
produce logits).

### Step 4: Copy op scheduling (allocate_slots)

allocate_slots(request, ...)
1.  Snapshot per-group cached block counts
2.  Call upstream → allocates blocks
3.  Index any newly cached full blocks for each group
4.  If `_PendingPartialMatch` exists for this request,
    for each group:
    1. Look up the destination block (newly allocated at the match boundary)
    2. Append `KVCacheCopyOp(group_id, src_block_id, dst_block_id, num_tokens)`

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

- **Caching full blocks**: Indexed during `allocate_slots` when
  upstream marks them as cached.
- **Caching partial blocks**: Indexed during `free()`. A synthetic
  `block_hash` is assigned so the block stays in the upstream LRU rather than
  being immediately reused.
- **Eviction**: A monkey-patched eviction hook calls
  `SubBlockIndex.pop()` whenever the upstream evicts a cached block.
- **Reset**: `reset_prefix_cache()` clears all sub-block indices.

## Limitations

- To simplify `SubBlockIndex` maintenance,
  **the last partial block of a running request is not indexed until `free()`.**
  Eagerly indexing partial blocks of running requests would require
  updating/replacing index entries on each sub-block boundary.
  The reduction in cache hit rate caused by this is negligible.
- KV cache copy implementation (`_copy_kv_cache`) limitations
    - **Per-tensor copy not yet implemented.**
      Currently it copies for all layers, so it is not applicable to multi-group setups.
    - **Synchronous.**
      Currently it is blocking.
- **Incompatible with KV connector.**
  Currently, the sub-blocks are not exposed outside of the manager,
  but KV connector works at the block level.
  To fix this, the sub-blocks should be exposed as the "canonical" blocks to
  the rest of the system, and `RBLNKVCacheManager` should ensure that
  consecutive sub-blocks are contiguous in memory.
  This requires a complete redesign.
