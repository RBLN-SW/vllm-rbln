# Scratch+Stable SWA Prefix Caching — Code Walkthrough

**NOTE:** I created this document with Claude and I looked over everything it said and it seems pretty accurate (this is completely Claude generated however so please verify yourself or ask me if I'm still at the company before you make any serious design changes - there isn't a lot of code though so it should be fine). I created this document to help anyone understand all the features I added for my scratch stable SWA prefix caching implementation. While I was playing with the profiler, I also found a metadata performance optimization so I asked Claude to include some description on that as well. Hopefully this doc is helpful with helping understanding everything - stephen gwon
---

## 0. The mental model (read this first)

Standard sliding-window attention on RBLN keeps **one** rolling block per request: new K/V are
inserted at the head and the window slides, so old tokens fall off. That rolling block can never
be prefix-cached, because it is mutated in place and its contents depend on history.

Scratch+stable keeps that rolling block (now called the **scratch block**) AND adds a parallel
**stable block** copy:

- **Scratch block** — the mutable rolling SWA window. It is the *only* block the attention
  kernel reads. It is allocated outside the prefix-cache machinery and is never hashed or shared.
  Exactly today's naive-SWA behavior.
- **Stable blocks** — an *append-only* copy of the full sequence, stored in `W`-sized blocks
  (`block_size == window == W`). They are managed by the normal vLLM block pool, so they are
  hashed, frozen, evicted, and **prefix-cached for free**, exactly like full-attention blocks.
  The kernel never *reads* a stable block; it only *appends* the new K/V into it each step.

On a prefix-cache **hit**, the cached stable block(s) are **D2D-copied into the request's fresh
scratch block** (a "seed") before the first forward pass, so the rolling kernel resumes with the
cached window in place — no prefill recompute. This sidesteps the SHM-stride bug that killed the
earlier two-block approach: attention still touches only one block (scratch), so there is no
two-block join in the kernel.

**Data flow each step (decode):**

```
scheduler                          worker / runner                     kernel
---------                          ---------------                     ------
manager reserves scratch block     reads swa_scratch_block_ids,        scratch attention
(off req_to_blocks)                builds per-request scratch tensor    (rolling window,
        │                                  │                            unchanged)
        ├─ scratch block id ───────────────┤                                +
        │   (swa_scratch_block_ids)        ├─ scratch_stable_index_tensors  append-only write
        │                                  │  -> ss_* metadata              of new K/V into
        └─ on a HIT: seed copy ops ────────┴─ _process_kv_cache_copy_ops    the stable block
            (stable -> scratch)               (D2D stable->scratch)         at local_seq_idx
```

**Invariants worth remembering:**
- `block_size == sliding_window == W` (one window per block).
- An append never straddles a block boundary: `local_seq_idx + query_len <= W`.
- The scratch block is reserved **outside** `req_to_blocks` so the cache logic never touches it,
  which is why both its *id* and its *seed copies* need dedicated side-channels to the worker.

## 1. File map & suggested reading order

| # | File | What it adds |
|---|---|---|
| §2 | `rbln_envs.py`, `platform.py` | the on/off flags and the prefix-caching gate |
| §3 | `v1/kv_cache.py` | the KV-cache spec + manager (scratch reservation, free-on-fill, seeds) |
| §4 | `triton_kernels/sliding_window_attention.py` | the scratch+stable decode/prefill kernels |
| §5 | `v1/attention/backends/flash_attention.py` | index math, metadata, builder, optimizations |
| §6 | `v1/attention/backends/flash_attention.py` | the attention dispatch (compiled-only) |
| §7 | `v1/core/rbln_kv_cache_manager.py` | the cache-hit seed copy ops (Case 1 / Case 2) |
| §8 | `v1/core/rbln_scheduler.py` | plumbing the scratch id + seed ops into the scheduler output |
| §9 | `v1/worker/rbln_model_runner.py` | scratch-id channel, copy-op execution, guards, build cache |

---

## 2. Flags & gating (rbln_envs.py, platform.py)

### `rbln_envs.py` — flag declarations and defaults

The scratch+stable feature is governed by environment flags declared in `vllm_rbln/rbln_envs.py`. The flags first appear as typed defaults on the env-vars dataclass:

```python
# vllm_rbln/rbln_envs.py:50-52
    VLLM_RBLN_SUB_BLOCK_CACHE: bool = True
    VLLM_RBLN_USE_DEVICE_TENSOR: bool = False
    VLLM_RBLN_SWA_SCRATCH_STABLE: bool = False
```

`VLLM_RBLN_SWA_SCRATCH_STABLE` is the master switch for the entire scratch+stable SWA path, and it defaults to off — the feature is strictly opt-in, so existing sliding-window deployments behave exactly as before unless an operator explicitly turns it on. `VLLM_RBLN_SUB_BLOCK_CACHE` defaults on and controls the more general sub-block prefix-caching mechanism (caching at a granularity finer than a full KV block). `VLLM_RBLN_USE_DEVICE_TENSOR` is an orthogonal device-placement opt-in, included here only for context.

The actual runtime values are produced by the environment-lookup table further down the file, which reads the OS environment variables lazily at access time:

```python
# vllm_rbln/rbln_envs.py:271-282
    "VLLM_RBLN_SUB_BLOCK_CACHE": lambda: (
        os.environ.get("VLLM_RBLN_SUB_BLOCK_CACHE", "True").lower() in ("true", "1")
    ),
    "VLLM_RBLN_SWA_SCRATCH_STABLE": lambda: (
        os.environ.get("VLLM_RBLN_SWA_SCRATCH_STABLE", "False").lower() in ("true", "1")
    ),
```

Each entry is a zero-argument lambda so the environment is read lazily on first access rather than frozen at import time. Truthy values are `"true"`/`"1"` case-insensitively. Turning on `VLLM_RBLN_SWA_SCRATCH_STABLE` selects the scratch+stable path: a dedicated KV-cache spec (`RBLNScratchStableSWASpec`), a matching manager, and the scratch+stable attention kernels. The hard precondition is that block size must equal the sliding-window length.

### `platform.py` — not disabling prefix caching for SWA models

By default vLLM-RBLN strips prefix caching from sliding-window models, because a rolling window normally cannot be prefix-cached. The scratch+stable path is the exception, handled in `disable_unsupported_prefix_caching`:

```python
# vllm_rbln/platform.py:377-385
    def disable_unsupported_prefix_caching(cls, vllm_config: VllmConfig) -> None:
        if not vllm_config.cache_config.enable_prefix_caching:
            return

        hf_config = vllm_config.model_config.hf_config

        # The scratch+stable (VLLM_RBLN_SWA_SCRATCH_STABLE) SWA path makes
        # sliding-window blocks prefix-cacheable, so do NOT disable caching for it.
        swa_caching_supported = envs.VLLM_RBLN_SWA_SCRATCH_STABLE
```

The method short-circuits if caching is already disabled, then computes a single boolean `swa_caching_supported` directly from the flag. This is the bridge between the env flag and the gating logic: when scratch+stable is on, the platform considers SWA caching supported.

```python
# vllm_rbln/platform.py:387-393
        if envs.VLLM_RBLN_USE_VLLM_MODEL:
            if (
                not swa_caching_supported
                and getattr(hf_config, "sliding_window", None) is not None
                and getattr(hf_config, "use_sliding_window", True)
            ):
                cls._disable_prefix_caching(vllm_config, "sliding window models")
```

Caching is disabled for a sliding-window model only when all three conditions hold: scratch+stable is NOT enabled, the model declares a sliding window, and the window is in use. The `not swa_caching_supported` term is the key addition — when the operator sets `VLLM_RBLN_SWA_SCRATCH_STABLE`, this term is false, the whole condition is false, and caching is left enabled. The same guard is duplicated in the non-upstream (optimum-RBLN) branch (lines 395-411) so the behavior is consistent across both model-loading paths. The helper `_disable_prefix_caching` (lines 367-374) just logs a warning and sets `enable_prefix_caching = False`; with `swa_caching_supported` true it is never reached for SWA models.

---

## 3. KV-cache spec & manager (v1/kv_cache.py)

The core idea: a request's sliding window is held in a single **mutable SCRATCH block** that the attention kernel actually reads. In parallel, the manager also keeps conventional append-only **STABLE blocks** that vLLM's block pool already knows how to hash, freeze, evict and prefix-cache. The stable blocks exist purely so prefix caching works for free; the scratch block is the live window and is never cached. On a prefix-cache hit, the matched stable block is D2D-copied into the fresh scratch block before the first forward pass.

### `RBLNScratchStableSWASpec` — the spec

```python
# lines 98-116
@dataclass(frozen=True)
class RBLNScratchStableSWASpec(SlidingWindowSpec):
    """Scratch + stable SWA spec ..."""

    def __post_init__(self):
        assert self.block_size % self.sliding_window == 0

    def max_memory_usage_bytes(self, vllm_config: VllmConfig) -> int:
        # Real multi-block SWA footprint (the stable blocks) + 1 page for the per-request
        # scratch block, so the pool is sized to fit one scratch per concurrent request.
        return super().max_memory_usage_bytes(vllm_config) + self.page_size_bytes
```

A KV-cache spec is the declarative description of one cache group — block size, KV heads, head dim, dtype, sliding-window length — that vLLM consults to size memory and pick the right manager. By subclassing `SlidingWindowSpec`, this spec reuses all of that and changes only two things. `__post_init__` asserts `block_size % sliding_window == 0` (used as `block_size == sliding_window`): one window per block, the marker that distinguishes a scratch+stable spec. `max_memory_usage_bytes` adds exactly one page on top of the normal multi-block SWA footprint, so the pool is sized to hold one scratch block per concurrent request; without this `+1 page` the pool could be exhausted by stable blocks alone and the scratch reservation would fail.

The class is registered to its manager:

```python
# lines 251-256
single_type_kv_cache_manager.spec_manager_map.update(
    {
        RBLNSlidingWindowSpec: RBLNSlidingWindowManager,
        RBLNScratchStableSWASpec: RBLNScratchStableSWAManager,
    }
)
```

`spec_manager_map` is vLLM's lookup from a spec class to its manager class; this entry makes vLLM instantiate `RBLNScratchStableSWAManager` whenever it sees a `RBLNScratchStableSWASpec`.

### `RBLNScratchStableSWAManager` — the manager

```python
# lines 119-141
class RBLNScratchStableSWAManager(SlidingWindowManager):
    def __init__(self, kv_cache_spec, **kwargs) -> None:
        super().__init__(kv_cache_spec, **kwargs)
        # req_id -> the request's mutable scratch block (separate from req_to_blocks).
        self.scratch_blocks: dict[str, KVCacheBlock] = {}
        # Pending scratch-seed copies (stable_block_id -> scratch_block_id) emitted on a
        # prefix-cache hit; drained each schedule step. (src_block_id, dst_scratch, num_tokens).
        self.pending_seed_copies: list[tuple[int, int, int]] = []
```

The manager subclasses the upstream `SlidingWindowManager`, so the entire normal lifecycle for `req_to_blocks` (the per-request list vLLM hashes, freezes, prefix-caches) is inherited unchanged — the stable blocks behave like ordinary full-attention blocks. The crucial constraint is that the scratch block must stay **out** of `req_to_blocks`, because `cache_blocks`, `find_longest_cache_hit`, and `remove_skipped_blocks` all index `req_to_blocks` positionally; a scratch block in that list would be wrongly hashed/freed as cacheable.

The constructor adds two pieces of state: `scratch_blocks` maps a request id to its single mutable scratch block (the live window, deliberately separate from `req_to_blocks`); `pending_seed_copies` queues D2D seed operations `(src, dst, num_tokens)` recorded on a hit, drained by the scheduler each step.

#### `get_num_blocks_to_allocate` — reserve +1 for scratch

```python
# lines 143-161
def get_num_blocks_to_allocate(self, request_id, num_tokens, new_computed_blocks,
                               total_computed_tokens, num_tokens_main_model) -> int:
    n = super().get_num_blocks_to_allocate(...)
    # Reserve one scratch block on the request's first allocation.
    if request_id not in self.scratch_blocks:
        n += 1
    return n
```

This tells the scheduler how many free blocks a request needs before admission. The parent computes the stable-block count; this override adds one more, but only on the request's first allocation (`request_id not in self.scratch_blocks`), since the scratch block is reserved once and reused for the request's lifetime. This keeps admission accounting honest so the later scratch allocation cannot fail for lack of a block.

#### `allocate_new_blocks` — allocate scratch + seed on a hit

```python
# lines 163-197
def allocate_new_blocks(self, request_id, num_tokens, num_tokens_main_model):
    if request_id not in self.scratch_blocks:
        # Reserve the scratch block OUTSIDE req_to_blocks (never hashed/freed).
        scratch = self.block_pool.get_new_blocks(1)[0]
        self.scratch_blocks[request_id] = scratch
        num_cached = self.num_cached_block.get(request_id, 0)
        if num_cached > 0:
            req_blocks = self.req_to_blocks.get(request_id, [])
            seed = next(
                (blk for blk in reversed(req_blocks[:num_cached])
                 if blk is not self._null_block and not blk.is_null),
                None,
            )
            if seed is not None:
                self.pending_seed_copies.append(
                    (seed.block_id, scratch.block_id, self.block_size)
                )
    return super().allocate_new_blocks(request_id, num_tokens, num_tokens_main_model)
```

This is where the scratch block is physically obtained and prefix-cache hits are seeded. The guard creates the scratch exactly once; `get_new_blocks(1)[0]` pulls one fresh block straight from the pool into `scratch_blocks` — never appending to `req_to_blocks`, so the cache machinery can't see it. The seed logic handles a hit: because `block_size == sliding_window`, a hit on a full window returns exactly one cached stable block, and by the time this runs `req_to_blocks` looks like `[null]*(m-1) + [cached]` (earlier windows freed under free-on-fill). `num_cached` is the match count; the code scans the cached prefix in reverse for the last non-null block — the cached window the next token must attend to — and queues `(seed.block_id, scratch.block_id, block_size)`: a whole-block, offset-0 D2D copy into the empty scratch. This must happen before the first forward, otherwise the kernel (reading only scratch) attends to garbage.

#### `cache_blocks` — strict free-on-fill

```python
# lines 199-218
def cache_blocks(self, request, num_tokens: int) -> None:
    super().cache_blocks(request, num_tokens)
    if not self.enable_caching:
        return
    num_full_blocks = num_tokens // self.block_size
    req_blocks = self.req_to_blocks[request.request_id]
    to_free: list[KVCacheBlock] = []
    for i in range(min(num_full_blocks, len(req_blocks))):
        blk = req_blocks[i]
        if blk is not self._null_block and not blk.is_null:
            to_free.append(blk)  # forward order -> oldest evicted first (LRU)
            req_blocks[i] = self._null_block
    if to_free:
        self.block_pool.free_blocks(to_free)
```

`cache_blocks` runs after a step to hash/freeze newly-full blocks. After the parent does the hashing, this applies **strict free-on-fill**. The key insight: a stable block is **write-only** — once full it is never read again by this request, because the scratch block holds the live window. So full stable blocks need not stay pinned. The loop walks the filled blocks oldest-first (LRU-correct), collects each non-null block, and nulls its slot. Nulling is essential: it records the slot no longer owns a live block so later `free()`/`remove_skipped_blocks` won't double-free. `free_blocks` drops the refcount to zero, but the block stays **hashed** in the pool, so it remains available for LRU prefix-cache reuse until evicted — the data isn't lost, just unpinned. Net: only the single in-progress stable block stays live per request.

#### `free` — release the scratch too

```python
# lines 220-224
def free(self, request_id: str) -> None:
    scratch = self.scratch_blocks.pop(request_id, None)
    if scratch is not None:
        self.block_pool.free_blocks([scratch])
    super().free(request_id)
```

Because the scratch block lives in the separate side-table, the parent `free` knows nothing about it, so this override must release it explicitly (pop from `scratch_blocks`, return to pool) before delegating to the parent for the stable blocks. Without this the scratch block would leak and the pool would slowly run dry.

#### `cancel_pending_seed` / `drain_pending_seed_copies` / `scratch_block_id`

```python
# lines 226-248
def cancel_pending_seed(self, request_id: str) -> None:
    """Drop the offset-0 full-block seed queued by allocate_new_blocks ... (Case 2 straddle)."""
    scratch = self.scratch_blocks.get(request_id)
    if scratch is None:
        return
    self.pending_seed_copies = [
        (s, d, n) for (s, d, n) in self.pending_seed_copies if d != scratch.block_id
    ]

def drain_pending_seed_copies(self) -> list[tuple[int, int, int]]:
    ops = self.pending_seed_copies
    self.pending_seed_copies = []
    return ops

def scratch_block_id(self, request_id: str) -> int:
    return self.scratch_blocks[request_id].block_id
```

`allocate_new_blocks` always queues a simple offset-0 whole-block seed on a hit, but that is only correct when the cached window is block-aligned. In a "Case 2" sub-block hit, where the rolling window straddles a block boundary, the caller emits a custom straddle seed instead; `cancel_pending_seed` lets it first retract the default seed (filtering out any entry whose destination is this request's scratch). `drain_pending_seed_copies` is called once per schedule step — it returns the queued copies and clears the queue atomically, so each is handed out exactly once. `scratch_block_id` is the accessor the metadata builder uses: since the scratch block is hidden from `req_to_blocks`, this is the only way its physical id leaves the manager.

---

## 4. The kernels (triton_kernels/sliding_window_attention.py)

This file contains two families of Triton kernels: the **naive** SWA kernels (`sliding_window_attention_naive_prefill` at line 24, `..._naive_decode` at line 218), which maintain a single rolling sliding-window block per sequence; and the **scratch+stable** kernels (`sliding_window_attention_scratch_stable_decode` at line 595, `..._scratch_stable_prefill` at line 968), which keep that rolling block (now the *scratch* block) and additionally maintain a second, append-only *stable* block so the sequence can be prefix-cached.

Mental model: the **scratch block** is the mutable rolling SWA window the attention math reads — new K/V are inserted at the right edge and the window slides, so old tokens fall off. The **stable block** is a separate copy of the sequence in `W`-sized chunks (`block_size == sliding_window == W`), written append-only and never overwritten in place. The kernel never *reads* the stable block; it only *writes* to it. The stable block exists so vLLM can treat it like a normal full-attention KV block — hash it, share it, reuse it — so a prefix-cache hit can D2D-copy it straight into a scratch block instead of recomputing prefill. **Everything that is not the stable append is byte-for-byte the same attention as the naive kernels.**

A hard invariant makes the stable write cheap and safe: an append must never straddle a block boundary, i.e. `local_seq_idx + seq_len <= W`. Because the chunk lands entirely inside one `W`-sized block, the kernel writes it as a *whole-block* insert+slice+store rather than a partial indexed store — which dodges the SHM-stride bug.

### Shared helpers: window_insert / window_slice / dynamic_store

All four kernels build their KV update from the same RBLN libdevice primitives (naive decode, lines 376-379):

```python
376	        k_base = rblib.dynamic_load(k_cache_base_ptr)
377	        k_insert = rblib.window_insert(k_base, k_state, WINDOW_AXIS, cache_start)
378	        k_slice = rblib.window_slice(k_insert, cache_end, WINDOW_AXIS, WINDOW_SIZE)
379	        rblib.dynamic_store(k_cache_base_ptr, k_slice)
```

- `dynamic_load(ptr)` reads the whole physical KV block into registers/SHM.
- `window_insert(base, new, axis, at)` overlays `new` onto `base` along the window axis starting at dynamic position `at`; the result is *longer* than a window (old window with new rows spliced in).
- `window_slice(buf, end, axis, W)` cuts a length-`W` window out ending at dynamic position `end`. This is what makes the window *roll*: as `end` advances past `W`, the oldest rows drop.
- `dynamic_store(ptr, value)` writes a whole block back. Storing a *whole* block (not a partial offset slice) is the SHM-safe path; a partial indexed store is what trips the SHM-stride bug.

`cache_start` (from `cache_seq_len`) is the rolling insert position, `cache_end` (from `cache_offset`) is the slice end, `WINDOW_AXIS = 4` is the sequence axis of the 6-D `(2, NUM_BLOCK, NUM_HEAD, 1, WINDOW_SIZE, HEAD_DIM)` cache (leading `2` selects K vs V; `block_number` from the block table picks the physical block). The scratch+stable kernels reuse these primitives twice: once for the scratch (rolling) write, once for the stable (append-only) write.

### What the scratch+stable decode kernel adds (line 595)

The signature gains four arguments (lines 604-607):

```python
604	    block_table_base,  # scratch block table (rolling SWA window)
605	    stable_block_table_base,  # NEW: append-only stable block (prefix-cacheable)
606	    local_seq_idx_base,  # NEW: write position in the stable block (= seq_idx % block_size)
607	    stable_end_base,  # NEW: stable window_slice end (= local_seq_idx + query_len, in (0, W])
```

- **`block_table_base`** — unchanged role, but now explicitly the *scratch* block table: the physical block holding the rolling window attention reads.
- **`stable_block_table_base`** — the parallel block table mapping this sequence to its *stable* physical block; allocated/tracked like an ordinary full-attention block, which makes it prefix-cacheable.
- **`local_seq_idx_base`** — the write offset *inside* the stable block (`seq_idx % block_size`). The stable analogue of `cache_start`, except it counts up from 0 within a fresh stable block instead of riding the rolling-window head.
- **`stable_end_base`** — the slice end for the stable write (`local_seq_idx + query_len`). By the no-straddle invariant this is always in `(0, W]`, so the slice keeps the whole accumulated block `[0, W)` and never rolls anything off.

These load alongside the existing indices (lines 679-682), then two extra block pointers `stable_k_cache_ptr` (742-756) and `stable_v_cache_ptr` (757-771) are built — structurally identical to the scratch pointers but indexed by `stable_block_number`, so they point at a *different physical block* in the same KV cache tensor.

**The attention computation is untouched.** Query load, scratch K insert/slice/store, pad/reshape/permute/broadcast, `tl.dot(q,k)`, scale, `window_softmax`, V path, final `tl.dot` / store are line-for-line the naive decode kernel. That is the whole design claim: attention reads only the scratch block, so there is no two-block join and no SHM bug in the math.

The only new work is the stable append, immediately after each scratch store (K, lines 818-825):

```python
818	        stable_k_base = rblib.dynamic_load(stable_k_cache_ptr)
819	        stable_k_insert = rblib.window_insert(
820	            stable_k_base, k_state, WINDOW_AXIS, local_seq_idx
821	        )
822	        stable_k_slice = rblib.window_slice(
823	            stable_k_insert, stable_end, WINDOW_AXIS, WINDOW_SIZE
824	        )
825	        rblib.dynamic_store(stable_k_cache_ptr, stable_k_slice)
```

In plain English: load the whole current stable block, splice the new K rows (`k_state`, the same tensor already inserted into scratch) at `local_seq_idx`, slice back to a `W`-wide window ending at `stable_end`, store the whole block back. The V append (850-857) is identical with V tensors.

Two deliberate subtleties (comment at 810-817):

1. **Why whole-block insert/slice/store instead of a direct partial write.** Writing `k_state` straight into `[local_seq_idx, local_seq_idx+query_len)` is a partial, offset store, which trips the SHM-stride bug. Mirroring the scratch path keeps every store a whole-block store (the SHM-safe pattern). The no-straddle invariant guarantees this whole-block round-trip leaves the already-written prefix intact and only fills the new rows.
2. **Why `stable_end` is passed in rather than reusing `cache_end`.** The stable slice end must be `local_seq_idx + query_len`. You cannot reuse `cache_start`/`cache_end`: on the first chunk those are `0`, and slicing a `W`-window ending at `0` produces a degenerate slice that would *zero out* the chunk just inserted. So `stable_end` is precomputed host-side (always in `(0, W]`) and passed explicitly. This is the single most error-prone detail and the reason `stable_end` exists as its own argument.

### What the scratch+stable prefill kernel adds (line 968)

The prefill variant is the exact analogue. Its `QUERY_LEN` is the chunk length `L` (not `1`) and `NUM_BATCH == 1`. As in naive prefill, the scratch and stable block tables are **1-D** (lines 1030-1046) rather than 2-D — the only structural difference in pointer setup versus decode. The four new args (977-980) are identical, with `local_seq_idx` now the *start* of the L-row chunk inside the stable block (invariant `local_seq_idx + L <= block_size`). The scratch attention path (1176-1237) is again line-for-line naive prefill, and the stable append (K at 1191-1198, V at 1221-1228) is identical in shape to decode — `dynamic_load`, `window_insert` the L new rows at `local_seq_idx`, `window_slice` to `W` ending at `stable_end`, `dynamic_store` — just L rows at once. The same two rationales apply verbatim.

### Summary of the delta

Relative to the naive kernels, the scratch+stable kernels add exactly one thing: after each rolling-window (scratch) K/V update, an append-only, whole-block write of the same new K/V into a separate stable block, addressed by `stable_block_number` at offset `local_seq_idx` and sliced to `stable_end`. Attention is unchanged because it only ever touches the scratch block. The whole-block insert/slice/store shape and the host-computed `stable_end` are the two details that keep the write correct (no SHM-stride bug, no first-chunk zeroing) under the no-straddle invariant.

---

## 5. Metadata, index math & builder (flash_attention.py)

This covers the per-request index math, the metadata fields that carry it to the kernel, the builder branch that wires it up, the reusable device buffers, and the two 2026-06-22 performance optimizations.

### 5.1 `scratch_stable_index_tensors(...)` — the per-request index math

```python
996	def scratch_stable_index_tensors(
997	    num_computed_tokens, query_lens, stable_block_tables, scratch_block_ids, window_size):
1031	    W = window_size
1032	    n = num_computed_tokens.reshape(-1, 1).to(torch.int64)
1033	    ql = query_lens.reshape(-1, 1).to(torch.int64)
1034	    max_blocks = stable_block_tables.size(1)
1036	    cache_seq_lens = torch.clamp(n, max=W)
1037	    cache_offsets = cache_seq_lens + ql
1038	    local_seq_idx = n % W
1044	    stable_end = (local_seq_idx + ql).clamp(min=1, max=W)
1045	    block_idx = (n // W).clamp(min=0, max=max_blocks - 1)
1046	    stable_block_table = torch.gather(stable_block_tables.to(torch.int64), 1, block_idx)
1049	    scratch_block_table = scratch_block_ids.reshape(-1, 1).to(torch.int64)
1051	    return (cache_seq_lens.to(int32), cache_offsets.to(int32), scratch_block_table.to(int32),
1055	            stable_block_table.to(int32), local_seq_idx.to(int32), stable_end.to(int32))
```

This is the single source of truth for every index the kernels consume. It is plain torch (no `@torch.library`), so it runs identically eager and compiled. It turns two per-request counters — `num_computed_tokens` (`n`, tokens already cached) and `query_lens` (`ql`, new tokens this step: 1 for decode, chunk size for prefill) — into six int32 tensors. Each, in plain terms:

- **`cache_seq_lens = clamp(n, W)`** — how many valid tokens are in the rolling scratch window. Saturates at `W` because the window only ever holds `W` tokens. Unchanged naive-SWA state.
- **`cache_offsets = cache_seq_lens + ql`** — the window occupancy after the new tokens append; where the kernel writes them in scratch.
- **`local_seq_idx = n % W`** — the append position *inside* the current stable block (stable blocks are `W` wide, so token `n` lands at row `n mod W`). The NEW datum the stable write needs.
- **`block_idx = clamp(n // W, 0, max_blocks-1)`** — which stable block to write into; the clamp guards against indexing past the allocated blocks.
- **`stable_block_table`** — the physical block id for that block, fetched with `gather`. Stable blocks are vLLM-managed and prefix-cacheable.
- **`scratch_block_table`** — the per-request scratch block id, reshaped to `[NB,1]`.
- **`stable_end = clamp(local_seq_idx + ql, 1, W)`** — the end index for the stable block's `window_slice`. The clamps are load-bearing: the upper `max=W` keeps it within one block (no-straddle), and the lower `min=1` avoids a degenerate `end=0` slice that would zero the chunk on the first row of a fresh block. (Note: the function returns six values; the docstring is slightly stale and lists five — it omits `stable_end`.)

### 5.2 `RBLNFlashAttentionMetadata` — the `ss_*` fields

```python
1098	    ss_scratch_block_table: torch.Tensor | None = None
1099	    ss_stable_block_table: torch.Tensor | None = None
1100	    ss_local_seq_idx: torch.Tensor | None = None
1102	    ss_stable_end: torch.Tensor | None = None
```

These four optional, on-device int32 fields carry the index tensors from the builder to the attention layer. `ss_scratch_block_table` is the scratch (live-window) block the kernel reads attention from; `ss_stable_block_table` is the append-only stable block it writes the new chunk into; `ss_local_seq_idx` is the write offset within that stable block; `ss_stable_end` is the slice end. They default to `None` so non-SWA groups leave them unset.

### 5.3 The builder `build()` scratch+stable branch

```python
1354	            if isinstance(self.kv_cache_spec, RBLNScratchStableSWASpec):
1355	                assert swa_scratch_block_ids is not None, (...)
1359	                bt2d = block_tables_tensor
1360	                if is_prefill:
1361	                    bt2d = bt2d.unsqueeze(0)  # [max_blocks] -> [1, max_blocks]
1362	                bt2d = bt2d[:num_reqs]
1363	                (_ss_csl, _ss_cofs, ss_scratch_bt, ss_stable_bt,
1369	                 ss_local_seq_idx, ss_stable_end) = scratch_stable_index_tensors(
1371	                    nct_src.reshape(-1, 1), query_lens.reshape(-1, 1),
1373	                    bt2d, swa_scratch_block_ids[:num_reqs], sliding_window)
```

This branch fires only for `RBLNScratchStableSWASpec` groups. It sits *inside* the broader `if sliding_window := ...` block, so `cache_seq_lens`/`cache_offsets` and the generic SWA mask are already computed and reused — the scratch+stable path only *adds* the stable-block write indices. It asserts the runner supplied `swa_scratch_block_ids` (the scratch block lives off `req_to_blocks`, so without them the kernel has nowhere to read the live window), normalizes the block table to 2-D (`unsqueeze` in prefill, where it arrives 1-D), slices to `[:num_reqs]`, and calls the index function — discarding the first two returns since `cache_seq_lens`/`cache_offsets` were already built.

```python
1377	                if not is_prefill:
1378	                    ss_scratch_bt = rbln_utils.pad(ss_scratch_bt, 0, batch_pad)
1379	                    ss_stable_bt = rbln_utils.pad(ss_stable_bt, 0, batch_pad)
1380	                    ss_local_seq_idx = rbln_utils.pad(ss_local_seq_idx, 0, batch_pad)
1381	                    ss_stable_end = rbln_utils.pad(ss_stable_end, 0, batch_pad)
1399	                else:
1402	                    ss_scratch_bt = ss_scratch_bt.reshape(-1)  # prefill: 1D block tables
1403	                    ss_stable_bt = ss_stable_bt.reshape(-1)
1405	                local_block_tables = ss_scratch_bt
```

Decode pads all four `ss_*` to the padded batch; prefill flattens the two block tables to 1-D (the prefill kernel convention) while `local_seq_idx`/`stable_end` stay 2-D. Crucially, `local_block_tables` is overwritten with `ss_scratch_bt` — this is the field the kernel reads KV from, so pointing it at the scratch block makes attention read the *live rolling window*.

### 5.4 Reusable device buffers + `_to_device_inplace`

```python
1150	        self._swa_ss_scratch_block_buf = None
1151	        self._swa_ss_stable_block_buf = None
1152	        self._swa_ss_local_seq_idx_buf = None
1153	        self._swa_ss_stable_end_buf = None
1156	        self._swa_ss_stack_buf = None   # decode coalesced [4,NB,1] buffer
1158	    def _to_device_inplace(self, cpu_tensor, attr_name):
1161	        buf = getattr(self, attr_name)
1162	        if buf is None or buf.shape != cpu_tensor.shape or buf.dtype != cpu_tensor.dtype:
1167	            buf = torch.empty(cpu_tensor.shape, dtype=cpu_tensor.dtype, device=self.device)
1170	            setattr(self, attr_name, buf)
1171	        buf.copy_(cpu_tensor)
1172	        return buf
```

`build()` runs every step, so naively each `ss_*` tensor would allocate a fresh device tensor and do an H2D copy each time. These persistent per-builder buffers avoid the churn. `_to_device_inplace` reuses the existing device buffer when shape and dtype match; only on a mismatch (e.g. padded batch changed) does it allocate a new `torch.empty` on device. Then a single in-place `buf.copy_(cpu_tensor)` does one H2D transfer. Across steps of a stable shape, allocation happens once and every step is just one copy.

### 5.5 Optimization (a) — coalesce the four `ss_*` transfers (2026-06-22)

```python
1388	                    ss_stack_cpu = torch.stack(
1389	                        [ss_scratch_bt, ss_stable_bt, ss_local_seq_idx, ss_stable_end], dim=0)
1392	                    ss_stack_dev = self._to_device_inplace(ss_stack_cpu, "_swa_ss_stack_buf")
1395	                    ss_scratch_dev = ss_stack_dev[0]
1396	                    ss_stable_dev = ss_stack_dev[1]
1397	                    ss_lsi_dev = ss_stack_dev[2]
1398	                    ss_end_dev = ss_stack_dev[3]
```

In the decode path all four padded `ss_*` are identical shape `[NB_pad,1]` int32. Instead of four separate H2D copies, they are `torch.stack`-ed into one `[4,NB_pad,1]` CPU tensor, pushed to device with **one** `_to_device_inplace`, then sliced into four dim-0 views. Indexing the leading dim of a contiguous stack yields contiguous sub-tensors, so each view is a valid standalone block table — no copy, no stride surprises. This eliminates 3 of 4 H2D transfers per SWA group per decode step; values are bit-for-bit identical, so kernel output is unchanged. Downstream, the metadata constructor (lines 1448-1475) prefers these pre-transferred views when present (`ss_*_dev is not None`) and falls back to the per-tensor `_to_device_inplace` path otherwise (prefill, where `ss_*_dev` stays `None`).

### 5.6 Optimization (b) — group-invariant hoist via `step_cache`, keyed by `block_size` (2026-06-22)

```python
1211	        num_computed_tokens = None
1212	        num_computed_tokens_cpu = None
1213	        seq_lens_cpu = None
1214	        cached = step_cache.get(self.block_size) if step_cache is not None else None
1215	        if cached is not None:
1216	            (seq_idx, seq_lens_tensor, num_computed_tokens,
1221	             num_computed_tokens_cpu, seq_lens_cpu) = cached
1223	        else:
1242	            partition_len = self.block_size
1243	            num_partition = max_seq_len // partition_len
1244	            cs = seq_idx.repeat(1, num_partition)            # <- hot op
1245	            pidx = torch.arange(num_partition, dtype=torch.int32)
1247	            seq_lens_tensor = torch.clamp(cs - pidx * partition_len, 0, partition_len).to(int16)
1250	            if step_cache is not None:
1251	                step_cache[self.block_size] = (seq_idx, seq_lens_tensor, num_computed_tokens,
1256	                    num_computed_tokens_cpu, seq_lens_cpu)
```

`build()` is called once **per KV-cache group** — 7× per step on a gemma-3 hybrid. But the bundle `seq_idx` (per-request sequence offset), the partitioned `seq_lens_tensor` (whose `seq_idx.repeat(1, num_partition)` is a hot Preprocess op), and the `num_computed`/`seq_lens` CPU mirrors depends only on the *batch* (`query_start_loc`, `seq_lens`, `positions`) and `self.block_size`, NOT on the per-group block table. So the runner passes a fresh per-step `step_cache` dict and `build()` memoizes the bundle in it. On a hit the whole bundle — including the expensive `repeat` — is reused; on a miss it is computed and stored.

**Why keyed by `block_size`:** the bundle's only per-group-varying dependency is `self.block_size`, which on a hybrid is per-group (`kv_cache_spec.block_size`). `num_partition` and the `seq_lens_tensor` shape both derive from it, so groups that *share* a block size safely share the bundle (the `repeat` runs once), while a group with a different block size correctly recomputes. **Why cross-group sharing is safe:** every cached value is a CPU intermediate consumed downstream only via `.to(self.device)` copies (which produce new device tensors, never writing back into the cached CPU tensor), and the only other operation, `rbln_utils.pad`, is non-mutating — so no group can corrupt the shared bundle for the next group. Effect (deterministic op-count drop, profiled): `repeat` 707->101, `arange` 3416->992, plus ~8500 fewer `to`/`_to_copy`/`copy_` ops inside `Preprocess`.

---

## 6. Attention dispatch — and why there is no eager path (flash_attention.py)

(The index math and metadata that feed this are in §5; this section covers only the dispatch branch in the attention forward.)

**Key finding:** there is **no eager scratch+stable implementation**. Unlike naive SWA — which swaps in plain-torch `sliding_window_attention_naive_{prefill,decode}_impl` functions when `VLLM_RBLN_COMPILE_MODEL` is off — the scratch+stable branch asserts the compiled kernels are present and has no fallback. The append-only stable write lives only inside the compiled Triton kernels (§4).

```python
1653	            if envs.VLLM_RBLN_SWA_SCRATCH_STABLE:
1659	                assert attn_metadata.ss_scratch_block_table is not None, (...)
1663	                assert (
1664	                    envs.VLLM_RBLN_COMPILE_MODEL and envs.VLLM_RBLN_USE_CUSTOM_KERNEL
1665	                ), (
1666	                    "scratch+stable SWA requires the compiled triton kernels "
1667	                    "(VLLM_RBLN_COMPILE_MODEL=1 RBLN_USE_CUSTOM_KERNEL=1); no eager "
1668	                    "impl is wired."
1669	                )
1670	                ss_ops = torch.ops.rbln_triton_ops
1671	                ss_args = [
1672	                    query, key, value, kv_cache,
1676	                    attn_metadata.cache_seq_lens,
1677	                    attn_metadata.cache_offsets,
1678	                    self.scale,
1679	                    attn_metadata.ss_scratch_block_table,
1680	                    attn_metadata.ss_stable_block_table,
1681	                    attn_metadata.ss_local_seq_idx,
1682	                    attn_metadata.ss_stable_end,
1683	                    self.scale,  # dummy
1684	                ]
1685	                if not attn_metadata.is_prefill:
1687	                    attn_output = ss_ops.sliding_window_attention_scratch_stable_decode(*ss_args)
1690	                else:
1691	                    attn_output = ss_ops.sliding_window_attention_scratch_stable_prefill(*ss_args)
```

Two guards run first: line 1659 asserts the builder actually produced `ss_scratch_block_table` (i.e. the spec really was `RBLNScratchStableSWASpec`); line 1663 is the decisive one — scratch+stable demands both compile flags and states outright that no eager impl is wired.

The `ss_args` list is built once for both variants. The first seven entries (`query`, `key`, `value`, `kv_cache`, `cache_seq_lens`, `cache_offsets`, `scale`) are exactly what the naive SWA op receives — that is the *scratch attention*, unchanged. The next four are the NEW append-only stable-write indices in order: `ss_scratch_block_table`, `ss_stable_block_table`, `ss_local_seq_idx`, `ss_stable_end`. A trailing `self.scale` is a positional dummy matching the naive op's signature. Then `is_prefill` dispatches to the decode or prefill compiled op (both in `triton_kernels/sliding_window_attention.py`).

**If an eager reference were ever needed,** the naive SWA eager impl `sliding_window_attention_naive_prefill_impl` (lines 681-787) is the template for the *scratch half* — it reads the scratch block, concatenates cached + new keys, slices the last `window_size` rows, writes the rolled window back, then does masked `q·kᵀ·scale → softmax → ·v`. A genuine eager `_scratch_stable_impl` would run that *and then* additionally append the new K/V into `ss_stable_block_table[req]` at rows `[ss_local_seq_idx : ss_stable_end]`. No such function exists today.

---

## 7. Cache-hit copy ops (v1/core/rbln_kv_cache_manager.py)

Hold this in mind: each SWA request keeps **two** physical blocks. A *stable* block stores the canonical, block-aligned KV the prefix cache hashes and reuses; a *scratch* block is what the attention kernel actually reads. The scratch is a **linear window buffer** — the kernel writes new tokens with `window_insert` at `cache_start` and reads with a `window_slice` ending at `cache_end`, so the live window must be laid out **oldest token first, newest last, contiguously from offset 0**. It is *not* a ring buffer. That linear-layout requirement is the entire reason the seed copies exist and why some need offsets.

### 7.1 The `KVCacheCopyOp` dataclass

```python
72	@dataclass
73	class KVCacheCopyOp:
76	    group_id: int
78	    src_block_id: int
79	    dst_block_id: int
81	    num_tokens: int
86	    is_scratch_seed: bool = False
91	    src_offset: int = 0
92	    dst_offset: int = 0
```

A copy op is "copy `num_tokens` tokens from `src_block_id` into `dst_block_id` for group `group_id`." The three fields added for scratch+stable:

- **`is_scratch_seed`** — marks this op as a *seed* of the scratch block (copying cached KV in so the kernel, which reads scratch, sees the cached prefix). It matters for lifetime: a seed's source is *already* ref-counted elsewhere (by `req_to_blocks` for a full-block seed, or by the sibling sub-block copy's `touch()`), so the op owns no fresh ref and must be excluded from the later free pass (§7.3).
- **`src_offset` / `dst_offset`** — token-axis offsets into the `block_size`-long window, default 0. Only the straddle seed (§7.2 Case 2) uses non-zero offsets; the native D2D primitive can only do offset-0, so an offset-bearing op is routed to the eager slice-assign path instead.

### 7.2 `apply_sub_block_match(...)` — emitting the scratch seed

```python
525	        for i, gm in enumerate(match.group_matches):
528	            dst_block = block_list[gm.dst_req_block_index]
529	            self.pending_copy_ops.append(KVCacheCopyOp(
531	                group_id=i, src_block_id=gm.src_block.block_id,
533	                dst_block_id=dst_block.block_id, num_tokens=match.num_tokens))
542	            mgr = self.coordinator.single_type_managers[i]
543	            if isinstance(mgr, RBLNScratchStableSWAManager):
544	                scratch_id = mgr.scratch_block_id(request.request_id)
545	                W = mgr.block_size
546	                m = gm.dst_req_block_index
547	                split = match.num_tokens  # extra tokens matched in block m (< W)
```

For each group, it first queues the ordinary copy that populates the *stable* block. Then, for a scratch+stable group, it must also seed the *scratch* block, because attention never reads the stable block. `m` is the destination block index within the request; `split` is the tokens matched inside block `m` (< W).

**Case 1 — hit lands in block 0 (`m == 0`):**

```python
548	                if m == 0:
551	                    self.pending_copy_ops.append(KVCacheCopyOp(
553	                        group_id=i, src_block_id=gm.src_block.block_id,
555	                        dst_block_id=scratch_id, num_tokens=split, is_scratch_seed=True))
```

The whole cached prefix is tokens `[0, split)`, which go to the front of scratch — `scratch[0, split)`. Already oldest-to-newest, starting at offset 0, so a single offset-0 seed suffices (fast native copy). Bytes past `split` may be stale, but the kernel's `cache_seq_len = split` masks them.

**Case 2 — straddle (`m > 0`):**

```python
560	                else:
567	                    mgr.cancel_pending_seed(request.request_id)
568	                    prev_block = block_list[m - 1]
572	                    self.pending_copy_ops.append(KVCacheCopyOp(   # older half
575	                        group_id=i, src_block_id=prev_block.block_id, dst_block_id=scratch_id,
577	                        num_tokens=W - split, src_offset=split, dst_offset=0, is_scratch_seed=True))
583	                    self.pending_copy_ops.append(KVCacheCopyOp(   # newer half
586	                        group_id=i, src_block_id=gm.src_block.block_id, dst_block_id=scratch_id,
588	                        num_tokens=split, src_offset=0, dst_offset=W - split, is_scratch_seed=True))
```

When the match lands past block 0, `num_computed = m*W + split` and the width-`W` window `[n-W, n)` crosses a boundary: the *tail* of block `m-1` (tokens `[split, W)`, the older half) followed by the *head* of block `m` (tokens `[0, split)`, the newer half). To lay scratch out linearly oldest-first, the seed splits into two offset copies: `scratch[0, W-split) <- block(m-1)[split, W)` and `scratch[W-split, W) <- block(m)[0, split)`. Concatenated, they reconstruct the exact window contiguously. Both carry non-zero offsets — which is why the offset fields exist. `cancel_pending_seed` first drops the optimistic offset-0 full-block seed that `allocate_new_blocks` had queued (a block-aligned seed would place the wrong tokens for a straddle).

### 7.3 `release_copy_ops(...)` — skip seeds to avoid double-free

```python
666	    def release_copy_ops(self, ops):
673	        to_release = [op for op in ops if not op.is_scratch_seed]
674	        if to_release:
675	            self.block_pool.free_blocks(
676	                [self.block_pool.blocks[op.src_block_id] for op in to_release])
```

After the runner executes the copies, the source-block refs taken to keep data alive must be released. An *ordinary* copy owns one such ref (from `touch()`), so it is freed. A *seed* never took its own ref — its source is kept alive by `req_to_blocks` or the sibling sub-block copy's `touch()` — so freeing it here would over-free and corrupt the pool's accounting. The filter `not op.is_scratch_seed` excludes seeds.

### 7.4 `get_computed_blocks_sub_block(...)` — the straddle gate

```python
424	        for gid, gi in enumerate(self._group_infos):
425	            num_full_blocks = num_computed_tokens // gi.block_size
441	            mgr = self.coordinator.single_type_managers[gid]
442	            if (isinstance(mgr, RBLNScratchStableSWAManager)
444	                and num_full_blocks > 0
445	                and not envs.VLLM_RBLN_USE_DEVICE_TENSOR):
447	                return None
```

This is the discovery-time gate. The Case 2 straddle seed is correct in principle but needs an offset-capable copy. In the default (non-device-tensor) mode no usable one exists (native D2D is offset-0; slice-assign misses the compiled buffers; host round-trip corrupts dlfloat16). So when a scratch+stable group's hit would straddle (`num_full_blocks > 0`) and we're not in device-tensor mode, it returns `None` — discarding the sub-block match for the request this step and falling back to plain full-block prefix caching (correct, just less reuse). The `and not envs.VLLM_RBLN_USE_DEVICE_TENSOR` clause is the escape hatch: in device-tensor mode the KV cache is a real `device='rbln'` tensor, so the offset slice-assign writes device memory directly and the straddle works for real — so the gate does not fire.

---

## 8. Scheduler plumbing (v1/core/rbln_scheduler.py)

The scratch block lives *outside* `req_to_blocks` so the prefix-cache machinery never hashes, shares, or reuses it. That isolation is what makes scratch+stable correct, but it also makes the block invisible to every normal path that carries block info from scheduler to worker. Two things therefore have to be hand-carried into `RBLNSchedulerOutput`: (1) the scratch block's **id**, so the metadata builder can point attention at it; and (2) the **seed-copy ops** that fill the scratch on a hit, so the runner copies the cached window in before the model runs.

### 8.1 `_drain_swa_seed_copy_ops(...)` — queued seeds -> tagged copy ops

```python
88	def _drain_swa_seed_copy_ops(kv_cache_manager) -> "list[KVCacheCopyOp]":
97	    coordinator = getattr(kv_cache_manager, "coordinator", None)
98	    if coordinator is None:
99	        return []
100	    ops: list[KVCacheCopyOp] = []
101	    for mgr in coordinator.single_type_managers:
102	        if isinstance(mgr, RBLNScratchStableSWAManager):
103	            for src, dst, num_tokens in mgr.drain_pending_seed_copies():
104	                ops.append(KVCacheCopyOp(
105	                    group_id=mgr.kv_cache_group_id, src_block_id=src,
108	                    dst_block_id=dst, num_tokens=num_tokens, is_scratch_seed=True))
113	    return ops
```

On a full-block prefix-cache hit the manager cannot just hand the cached block to the request (the cached data is a *stable* block that must stay immutable and shared, while the request needs a private, writable *scratch* block). So at allocation time it records an intent "copy this stable block into this request's scratch." This function harvests those intents at the end of scheduling: it reaches the coordinator (returning `[]` if absent), walks the per-type managers, and for each `RBLNScratchStableSWAManager` calls `drain_pending_seed_copies()` (which both returns and clears, so each copy is emitted exactly once). Each `(src, dst, num_tokens)` triple becomes a `KVCacheCopyOp` tagged `is_scratch_seed=True` (so the runner treats it as a whole-block scratch seed, not a sub-block merge) and `group_id` (hybrids have several SWA groups). Block ids are globally unique (one shared pool), so the runner can replay the copy across all layers of the group. Reusing the existing copy-op channel means no new execution path is needed on the runner.

### 8.2 `_collect_swa_scratch_block_ids(...)` — the side-channel for scratch ids

```python
60	def _collect_swa_scratch_block_ids(kv_cache_manager) -> dict[int, dict[str, int]]:
75	    coordinator = getattr(kv_cache_manager, "coordinator", None)
76	    if coordinator is None:
77	        return {}
78	    out: dict[int, dict[str, int]] = {}
79	    for gi, mgr in enumerate(coordinator.single_type_managers):
80	        if isinstance(mgr, RBLNScratchStableSWAManager):
81	            out[gi] = {req_id: blk.block_id for req_id, blk in mgr.scratch_blocks.items()}
85	    return out
```

This builds the map the metadata builder needs: keyed by **group index** `gi`, then **request id**, giving the scratch block id. It reads each manager's private `scratch_blocks` and converts each block to its bare int id. The outer key `gi` is the enumeration index of the manager in `single_type_managers`, which corresponds to `kv_cache_groups[i]` — the same index the runner iterates — so the runner can index straight in.

**Why the side-channel is needed:** a scratch block does not appear in `req_to_blocks`, and every normal route by which ids reach the runner derives from `req_to_blocks`. So the scratch id has no normal path; this reaches directly into the manager's bookkeeping. **Why keyed by group, not just request:** hybrids have one SWA group per SWA layer, each with its own scratch block. Flattening to `{req_id: id}` would let the last group overwrite the others for a request, collapsing all groups onto one block id and causing cross-layer KV corruption.

### 8.3 The `RBLNSchedulerOutput` field and where it's populated

```python
44	@dataclass
45	class RBLNSchedulerOutput(SchedulerOutput):
49	    kv_cache_copy_ops: list[KVCacheCopyOp] = field(default_factory=list)
53	    swa_scratch_block_ids: dict[int, dict[str, int]] = field(default_factory=dict)
```

`RBLNSchedulerOutput` extends the upstream output with the copy-ops list (shared by sub-block caching and now scratch seeds) and the per-group scratch-id map. Both default empty, so non-SWA runs behave exactly like upstream. Population happens in `schedule()`:

```python
945	        if isinstance(self.kv_cache_manager, RBLNKVCacheManager):
946	            scheduler_output.kv_cache_copy_ops = self.kv_cache_manager.drain_pending_copy_ops()
952	        seed_copy_ops = _drain_swa_seed_copy_ops(self.kv_cache_manager)
953	        if seed_copy_ops:
954	            scheduler_output.kv_cache_copy_ops = (
955	                scheduler_output.kv_cache_copy_ops + seed_copy_ops)
962	        scheduler_output.swa_scratch_block_ids = _collect_swa_scratch_block_ids(
963	            self.kv_cache_manager)
```

First, *only* when the RBLN sub-block manager is active, ordinary sub-block copy ops are drained. Next the scratch seeds are drained and **appended** to the same list, so seeds and sub-block copies travel together. Finally `swa_scratch_block_ids` is filled. Crucially, the seed drain and the scratch-id collection are **deliberately unguarded** (unlike the first drain): the scratch blocks live in `coordinator.single_type_managers`, which exists on the *upstream* `KVCacheManager` too. Scratch+stable is active on hybrid models where `VLLM_RBLN_SUB_BLOCK_CACHE=false`, i.e. exactly when `RBLNKVCacheManager` is NOT installed. Gating these on `RBLNKVCacheManager` would mean scratch ids never get collected on hybrid runs and every request would silently fall back to scratch block 0 — corrupting all requests' SWA state. (Source blocks referenced by these ops are kept alive until `update_from_output()` calls `release_copy_ops` after the runner finishes the copies — safe for async scheduling / pipeline parallelism.)

---

## 9. Worker / model-runner integration (v1/worker/rbln_model_runner.py)

Four places wire scratch+stable into the live forward pass: the scratch-id channel into the builder, the copy-op execution, the relaxed multi-group guard, and the group-invariant build cache. (Note: the `is_scratch_seed` *flag* is set in the scheduler/manager; here the offset-0 case is just labelled the "seed" copy in comments.)

### 9.1 The scratch-id channel into the builder

At the top of `execute_model` the runner stashes the scheduler's map onto itself:

```python
2902	            self._swa_scratch_block_ids = (
2903	                scheduler_output.swa_scratch_block_ids
2904	                if isinstance(scheduler_output, RBLNSchedulerOutput)
2905	                else {}
2906	            )
```

The scratch block isn't part of the normal block table, so it can't ride the usual tensor into the kernel — it needs its own side-channel. The scheduler hands it over as `swa_scratch_block_ids` (keyed by group then request); this stashes it for the per-group metadata loop. The `isinstance` guard degrades to `{}` on a vanilla scheduler output (warmup / non-SWA).

Then in the per-group metadata loop it becomes a batch-aligned tensor:

```python
1447	                    if isinstance(kv_cache_group_spec.kv_cache_spec, RBLNScratchStableSWASpec):
1450	                        scratch_map = getattr(self, "_swa_scratch_block_ids", {})
1451	                        group_scratch = scratch_map.get(kv_cache_group_id, {})
1452	                        req_ids = self.input_batch.req_ids
1453	                        scratch_ids = torch.tensor(
1454	                            [group_scratch.get(req_ids[i], 0) for i in range(num_reqs)],
1458	                            dtype=torch.int32)
1460	                        extra_attn_metadata_args["swa_scratch_block_ids"] = scratch_ids
```

This runs only for SWA groups. It narrows the stashed map to the current group, then — the key correctness concern — builds the scratch-id tensor in **block-table row order**: the kernel needs row `i` of the scratch tensor to be the same request as row `i` of the block table, which is ordered by `self.input_batch.req_ids`, so the comprehension walks `req_ids` in that exact order. The fallback `.get(req_ids[i], 0)` gives scratch id `0` to any request without an assigned scratch (benign — such a request won't read a scratch block). It is passed to `build()` via `extra_attn_metadata_args["swa_scratch_block_ids"]`; the builder moves it on-device.

### 9.2 `_process_kv_cache_copy_ops` — executing the copies

```python
2808	    def _process_kv_cache_copy_ops(self, copy_ops):
2812	        if not copy_ops:
2813	            return
2814	        eager = (envs.VLLM_RBLN_USE_DEVICE_TENSOR or self.model_config.enforce_eager
2817	                 or not envs.VLLM_RBLN_COMPILE_MODEL)
```

Each `KVCacheCopyOp` has src/dst block ids, `num_tokens`, and optional `src_offset`/`dst_offset`. `eager` is true for device-tensor mode, eager mode, or non-compiled mode; otherwise compiled.

**Compiled path** — the runtime owns the KV buffers, so use the native D2D copy, which is offset-0-only:

```python
2820	        if not eager:
2824	            runtime = self.runtime_holder[0]
2825	            for op in copy_ops:
2826	                if op.src_offset or op.dst_offset:
2827	                    raise NotImplementedError(
2828	                        "offset KV-cache copy (Case 2 straddle) is unsupported "
2829	                        "in compiled mode; needs an offset-capable D2D copy")
2831	                runtime._copy_kv_cache(op.src_block_id, op.dst_block_id, op.num_tokens)
2834	            return
```

It delegates whole-block copies to `runtime._copy_kv_cache(src, dst, n)`. Any op with a non-zero offset (a Case 2 straddle) cannot be expressed by that primitive, so rather than produce wrong KV it **raises** — a regression guard ensuring the upstream gate (§7.4) really keeps straddle ops off this path.

**Device-tensor path** — `self.kv_caches` are real device tensors, so a slice-assign writes device memory directly:

```python
2841	        for op in copy_ops:
2842	            sb, db = op.src_block_id, op.dst_block_id
2843	            s0, d0, n = op.src_offset, op.dst_offset, op.num_tokens
2844	            if s0 or d0:    # straddle: partial-window slice
2846	                for kv_cache in self.kv_caches:
2847	                    kv_cache[:, db : db + 1, :, :, d0 : d0 + n, :] = kv_cache[
2848	                        :, sb : sb + 1, :, :, s0 : s0 + n, :]
2850	            else:           # offset-0 seed: whole-block slice (mask hides the tail)
2852	                for kv_cache in self.kv_caches:
2853	                    kv_cache[:, db : db + 1] = kv_cache[:, sb : sb + 1]
```

The crucial detail (and the reason for the long in-code comment) is the use of a **contiguous single-block slice** `[:, db:db+1]` rather than **fancy indexing** `[:, idx]`: fancy indexing is a gather/scatter that host-bounces, while a `b:b+1` slice stays on the fast device-to-device path — same logical result, much faster. The straddle branch slices the token axis (`d0:d0+n` <- `s0:s0+n`) to copy a partial window; the offset-0 branch copies the whole block (the attention mask hides any tail beyond the valid window). Both apply to every per-layer cache in `self.kv_caches`.

### 9.3 `initialize_kv_cache` — the relaxed multi-group guard

```python
4738	        if (envs.VLLM_RBLN_SUB_BLOCK_CACHE
4740	            and len(kv_cache_config.kv_cache_groups) > 1
4741	            and not envs.VLLM_RBLN_SWA_SCRATCH_STABLE):
4749	            raise NotImplementedError(
4750	                "Sub-block prefix caching does not support multi-group KV caches yet. "
4752	                "Set VLLM_RBLN_SUB_BLOCK_CACHE=false to disable, "
4753	                "or enable VLLM_RBLN_SWA_SCRATCH_STABLE for the SWA hybrid.")
```

Historically sub-block caching was rejected outright for multi-group caches (untested). The scratch+stable work validated exactly one multi-group shape — the gemma hybrid (full-attn + per-layer SWA groups) — so the guard was relaxed with `and not envs.VLLM_RBLN_SWA_SCRATCH_STABLE`: the error now fires only when sub-block caching is on AND there are multiple groups AND scratch+stable is *not* enabled. With the flag set, the validated hybrid passes; every other untested multi-group + sub-block combination is still blocked. (KV-events + multi-group is rejected separately inside `RBLNKVCacheManager`.)

### 9.4 `swa_step_invariant_cache` — the group-invariant build hoist (2026-06-22)

```python
1360	        swa_step_invariant_cache: dict = {}
1361	        for kv_cache_group_id, kv_cache_group_spec in enumerate(
1362	            self.kv_cache_config.kv_cache_groups):
...
1437	                if isinstance(builder, RBLNFlashAttentionMetadataBuilder):
1441	                    extra_attn_metadata_args["step_cache"] = swa_step_invariant_cache
```

`build()` runs once per KV-cache group per step, and on a hybrid much of its work is identical across groups sharing a block size (§5.6). The fix creates a single empty dict **before** the per-group loop and passes the same dict into every group's `build()` as `step_cache`. The builder memoizes its group-invariant intermediates (notably the hot `repeat`) in that dict on the first group; later groups with the same block size read them back instead of recomputing. The dict is recreated fresh each step, so nothing leaks across steps — strictly step-scoped. It is added to `extra_attn_metadata_args` only for `RBLNFlashAttentionMetadataBuilder` (the only builder that uses it).

---

*End of walkthrough. See `scratch_stable_swa/scratch_stable_testing.md` for how to run the tests and the benchmarking / profiling / optimization results that motivated §5.5–§5.6 and §9.4.*
