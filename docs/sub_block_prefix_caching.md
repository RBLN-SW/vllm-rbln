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

For cross-engine prefix-aware routing (e.g., llm-d):
enable KV events via `--kv-events-config` in vLLM and
set the router's token processing block size to vLLM `--max-num-batched-tokens`
(**not** `--block-size`; see [Using with llm-d](#using-with-llm-d)).

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
              when `delay_cache_blocks=False`.
              When `delay_cache_blocks=True`, the caller must call
              `schedule_sub_block_indexing()` after `cache_blocks()`.
            * `free` indexes all blocks (full + partial) for the finishing request
              and assigns a synthetic `block_hash` to the partial block.
            * `reset_prefix_cache` clears sub-block indices and pending indexing.
        * New methods
            * `get_computed_blocks_sub_block(request, num_computed_tokens)`
              discovers sub-block matches and returns a `SubBlockMatch` handle.
            * `apply_sub_block_match` / `release_sub_block_match` consume or discard the handle.
            * `drain_pending_copy_ops()` retrieves the KV cache copy ops accumulated in the current scheduling step.
            * `release_copy_ops()` releases the source-block references after the model runner finishes copying.
            * `schedule_sub_block_indexing(request)` records that a request
              needs sub-block indexing.
            * `do_pending_indexing()` executes the scheduled indexing work.
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

When a scheduler schedules a request, it should schedule sub-block indexing for that request.
This is done automatically when `allocate_slots` is called with `delay_cache_blocks=False`.
If `delay_cache_blocks=True`, the user must call `schedule_sub_block_indexing()` after upstream `cache_blocks()`.
The current implementation of `RBLNScheduler` uses the latter approach,
because its complex scheduling logic requires manual control over full block caching.

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
`apply_sub_block_match(match)` which, for each group:
1.  Looks up the destination block (newly allocated at the match boundary)
2.  Appends `KVCacheCopyOp(group_id, src_block_id, dst_block_id, num_tokens)`

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
  Scheduled by `allocate_slots` or by user, then executed by `do_pending_indexing`
  (called after `super().update_from_output()`).
  Indexes both full blocks and complete sub-blocks within partial blocks.
- **Indexing finished requests**: `free()` consumes the pending-indexing
  entry, indexes all blocks, and assigns a synthetic `block_hash` to the
  partial block so the upstream LRU preserves it.
- **Eviction**: A monkey-patched eviction hook calls
  `SubBlockIndex.pop()` whenever the upstream evicts a cached block.
- **Reset**: `reset_prefix_cache()` clears all sub-block indices and pending
  indexing queue.

## KV cache events

Sub-block caching so far is an intra-engine optimisation:
one instance serving multiple requests that share a prefix reuses KV locally.
Prefix-cache-aware routers (e.g. llm-d) extend that reuse *across* engines by
subscribing to KV events and routing each request to an engine that already
caches its prefix.

Upstream vLLM emits those events at block granularity.
For RBLN's 1k–4k token big blocks, that means a partial big block never gets advertised.
So routers cannot utilize the sub-block prefix information.
Therefore, we adapt the KV cache manager to emit events at sub-block granularity.

### Contract

When sub-block caching is active, `RBLNKVCacheManager` replaces upstream's
block events with sub-block-granular ones:

- `BlockStored(parent, [h_0, ..., h_n])`: a chained run of sub-block hashes
  (each `h_i = hash(h_{i-1}, tokens_i, extras_i)`, with `h_{-1} = parent`)
  became *newly* cached on this engine.
- `BlockRemoved([h_0, ..., h_n])`: the *last* big block holding each hash
  was evicted.
- `AllBlocksCleared`: all sub-block state cleared.

This is stricter than upstream.
Upstream emits per-physical-block, so two blocks that happen to hold the same
hash produce duplicate Store and premature Remove.
Our Store/Remove fire only on 0↔1 transitions in the per-hash refcount
(`SubBlockIndex._hash_to_blocks` bucket size), so the stream is a clean
set-membership view.

#### Why dedup matters here

llm-d's `kvblock.Index` as of 2026-04 treats events as **set-membership**, not
multiplicity: duplicate Store collapses to one entry, Remove for a missing key
is a no-op.
So a single Remove can cause the router to drop a hash the pod still caches in
another block, and re-route away from a still-warm pod.
Upstream GPU deployments see this rarely because block-level hash collisions are rare.

Sub-block caching makes the collision path the *common* one.
Partial matches are serviced by an intra-engine sub-block copy:
the source big block and the freshly-allocated destination big block both carry
the same prefix sub-block hashes from the moment of the copy.
Every partial match mints a handful of duplicates.
Without dedup, the first eviction of either block would incorrectly signal
"gone" to llm-d.

Dedup'ing at the hash-refcount level in the engine gives routers exactly the
set-membership stream they already model, and sidesteps the multiplicity
mismatch entirely.

#### Dedup is chain-safe

We say that a `BlockStored` event is *chain-safe* if it carries a contiguous,
reconstructible hash chain:
using the event's `parent_block_hash` as the seed, re-hashing `token_ids`
sub-block by sub-block reproduces `block_hashes` entry-by-entry.
A consumer that re-hashes the prompt to key its index depends on this.

We need dedup to preserve chain-safety.
If dedup dropped `h_1` out of `[h_0, h_1, h_2]` and emitted `[h_0, h_2]`,
the consumer would compute `hash(h_0, tokens_h_2)` and miss `h_2`'s real value
(whose parent is `h_1`, not `h_0`).

Here, dedup never creates such gaps.
Within one `SubBlockIndex.update` call,
the "already indexed elsewhere" hashes form a contiguous prefix of the argument list,
and the fresh hashes form a contiguous suffix.
This is a consequence of position-chained hashing: if some other block holds
`h_k`, that block also holds `h_0..h_{k-1}`.
So we emit one event covering the fresh suffix.

### Single-group only (for now)

Events are gated to single-group configs, matching upstream's current
restriction (`need_disable_hybrid_kv_cache_manager` in `vllm/config/vllm.py`
trips when events are enabled).  Our per-group dedup would otherwise emit
the same bare hash from each group and consumers couldn't disambiguate.

Upstream vLLM is lifting this: `vllm-project/vllm@cc07dad789` (#37688) drops
the gate and adds `group_idx` to `BlockStored` / `BlockRemoved`.  Once that
feature is released, we can lift our assert and pass `group_idx` in our
emissions.

### No in-process consumer

`Scheduler.take_events` feeds events directly to the external
`KVEventPublisher` (ZMQ); nothing in-engine reads them, and KV connectors
publish their own events rather than subscribing to the manager's.
This allows us to change emission granularity unilaterally.

### Using with llm-d

llm-d's `precise-prefix-cache-scorer` chunks incoming prompts at its
configured block size, looks those keys up in the index it builds from our
events, and scores pods by how many consecutive blocks match.  Routing
works iff the scorer's `blockSize` equals the engine's emitted
`block_size` — our sub-block size, which is the prefill chunk size
(`--max-num-batched-tokens`).

> **Not** vLLM's `--block-size`.  For stock vLLM those two happen to be
> the same knob, so generic guides tell you to align `--block-size` with
> `blockSize`.  Here, `--block-size` is the big-block size (used for
> attention-kernel layout) while the scorer sees sub-block events.  Align
> `blockSize` with `--max-num-batched-tokens`, not `--block-size`.

For example, with prefill chunk size 128:

- vllm-rbln:
  ```
  vllm serve ... \
      --enable-prefix-caching \
      --max-num-batched-tokens=128 \
      --kv-events-config='{"enable_kv_cache_events": true, "publisher": "zmq", ...}'
  ```
- EPP `prefix-cache-scorer` plugin:
  ```
  tokenProcessorConfig:
    blockSize: 128
  ```

llm-d doesn't need to know anything about the big-block layout.
Other event fields (`token_ids`, `parent_block_hash`, `extra_keys`,
`lora_name`) are forwarded as-is from upstream vLLM.

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
