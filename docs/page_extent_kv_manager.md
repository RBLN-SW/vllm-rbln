# Page + Extent KV Manager (Design)

Status: **partially implemented**, off by default behind `VLLM_RBLN_PAGE_EXTENT`.
See [§ Implementation status](#implementation-status).
Related: [sub_block_prefix_caching.md](./sub_block_prefix_caching.md) (current overlay implementation)
Upstream analogues: `CacheConfig.prefix_match_unit` / `hash_block_size`,
partial prefix-cache hits ([RFC #45702](https://github.com/vllm-project/vllm/issues/45702))

## Problem

RBLN KV transfers are DMA-bound: the attention kernel's effective bandwidth
scales with the size of the contiguous KV region it addresses, so the physical
KV unit has to stay large (typically 1k–8k tokens). On the optimum path the
size is also *baked in*: compilation sets `kvcache_partition_len` from the
block size and selects `attn_impl="flash_attn"`, and the resulting
`kvcache_block_size` reaches the runtime as
`additional_config["attn_block_size"]`.

On the **native path the operator supplies it** — nothing publishes
`attn_block_size` there, so it is passed explicitly:

```bash
--block-size 1024 --additional-config '{"attn_block_size": 8192}'
```

Both paths therefore read the same key, and it is the single source of truth.
The trade-off is accepted deliberately: the native path cannot cross-check the
value against the compiled artifact, so an operator who names an extent size
the kernel does not actually use gets no warning. Deriving it from the native
compile config would close that hole and is the natural follow-up.

Upstream vLLM treats `--block-size` as one unit for everything: scheduling,
hashing, prefix matching, and physical storage. With a 4096-token block, any
reusable prefix that ends inside a block is missed, so vllm-rbln added an
overlay ([sub-block prefix caching](./sub_block_prefix_caching.md)).

**Scope.** This design targets the **native vLLM-model path**
(`VLLM_RBLN_USE_VLLM_MODEL=True`), where `--block-size` is still the coarse
unit. The optimum path (the default) already runs a two-level scheme under
different names — `attn_block_size`/`ob_size` is the extent,
`cache_config.block_size`/`ib_size` is the page, `get_block_ratio()` is
`pages_per_extent` — implemented in `RBLNPrefixKVCacheManager`. That path is
prior art and is deliberately **out of scope**: this work ports its proven
structure rather than refactoring it.

The overlay works, but the contracts are awkward:

- Scheduler / hash / routers / LMCache want a fine unit (~prefill chunk, e.g. 512).
- The DMA path wants a coarse contiguous unit (e.g. 4096).
- Today `--block-size` exposes the coarse unit, so llm-d must be told to use
  `--max-num-batched-tokens` instead of `--block-size`.
- Upstream's fine-grained matching cannot be adopted as-is (see
  [§ vs upstream](#vs-upstream-partial-prefix-cache)).

## Mental model: a hybrid-mapping FTL

The two-unit problem is structurally identical to an SSD flash translation
layer, and the correspondence is exact enough to reuse its vocabulary and its
cost model rather than reinventing both.

| SSD / FTL | This design |
|---|---|
| program / read unit (NAND page) | **page** (e.g. 512) |
| erase unit (erase block) | **extent** (e.g. 4096) |
| LBA space exposed to the host | page-id space exposed to scheduler / connectors |
| physical address `(block, page offset)` | `(extent_id, page_offset)` |
| FTL mapping table | page-hash index (`page_hash → (extent_id, page_offset)`) |
| out-of-place update (no in-place overwrite) | copy-on-write on a shared prefix |
| open block + sequential write pointer | per-request open extent, append-only |
| multi-stream (one open block per stream) | one open extent per request |
| trim (invalidate) vs erase (reclaim) | request free vs prefix-cache eviction |
| write amplification | copy amplification |
| over-provisioning | reserved free extents |
| GC victim selection by valid-page count | extent compaction (future, not MVP) |

FTLs come in three mapping granularities: page-level (fine map, no merges,
huge table), block-level (small table, read-modify-write on every partial
update), and **hybrid** (fine map, coarse allocation, merge on partial update).
This design is the hybrid variant, so its dominant cost is the same one hybrid
FTLs pay: the **merge**, which here is the CoW copy. The mitigation is also the
same — arrange for *partial* merges only, never full merges
(see [§ Partial merge](#partial-merge-cow)).

**Scope of the analogy.** It applies to address translation, allocation
granularity, and the GC/write-amplification cost model. It does **not** apply
to endurance management — wear leveling, read disturb, P/E cycles, and bad
block handling have no counterpart here.

## Terms

| Name | Example | Role |
|---|---|---|
| **chunk** | 512 | Prefill step size (`--max-num-batched-tokens`) |
| **page** | 512 | Schedule, hash, prefix match, connector addressing |
| **extent** | 4096 | Allocation / reclaim / DMA unit; contiguous token storage |

```text
extent (physical, DMA-addressed)
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ page0 │ page1 │ page2 │ page3 │ page4 │ page5 │ page6 │ page7 │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
 ←──── one request's pages, written sequentially from offset 0 ────→
```

### Naming vs today's code

| This design | Current overlay (`RBLNKVCacheManager`) |
|---|---|
| page | sub-block (`sub_block_size`) |
| extent | physical / upstream block (`block_size`) |
| page hash chain | `SubBlockHasher` |
| page → extent map | `SubBlockIndex` |
| partial merge (CoW) | `KVCacheCopyOp` |
| full dedup | upstream full-block hash hit + collapse of the private copy |

Internal code should use `page` / `extent`, never upstream's
`kernel_block_size` (a different, opposite-direction concept — see
[§ Why not `kernel_block_size`](#why-not-kernel_block_size)).

## Spec

### Sizes

```text
chunk_size       = scheduler_config.max_num_batched_tokens
page_size        = cache_config.block_size          # --block-size
extent_size      = backend-derived (not user-visible)
pages_per_extent = extent_size // page_size

REQUIRED: page_size   % chunk_size == 0             # a prefill never spans two pages
REQUIRED: extent_size % page_size  == 0
REQUIRED: pages_per_extent >= 2                     # otherwise use plain upstream
```

`page_size == chunk_size` is the default (one chunk completes one page).
`page_size = k * chunk_size, k > 1` is allowed: the page becomes
hash-complete after `k` chunks. `chunk_size > page_size` is rejected.

### Configuration

User-visible:

| Knob | Role | Note |
|---|---|---|
| `--block-size` | **page** size | must be a multiple of the prefill chunk, must divide `extent_size` |
| `--max-num-batched-tokens` | prefill chunk | `page_size % chunk_size == 0` |
| `--enable-prefix-caching` | on/off | unchanged |
| `VLLM_RBLN_SUB_BLOCK_CACHE` | feature gate | keep as the enable switch; rename to `VLLM_RBLN_PAGE_EXTENT` with an alias once the overlay is retired |

Published by the model, not tuned per run:

| Knob | Role |
|---|---|
| `additional_config["attn_block_size"]` | `extent_size` — the DMA packing size the attention kernel was compiled for |
| `pages_per_extent` | derived: `extent_size // page_size` |

Do **not** add `VLLM_RBLN_KERNEL_BLOCK_SIZE`. A model that publishes nothing
gets a degenerate geometry (one page per extent) and the layer is a no-op, so
enabling the feature is always safe.

Worked example — page 1024, extent 8192, eight pages per extent:

```bash
VLLM_RBLN_PAGE_EXTENT=1 vllm serve <model> \
    --block-size 1024 \
    --max-num-batched-tokens 512 \
    --additional-config '{"attn_block_size": 8192}'
```

### Address space

Page ids are upstream's **logical** ids, not physical addresses. Translation is
an explicit table, not page-id arithmetic — which is what lets upstream's
`BlockPool` stay untouched.

Two maps, and conflating them is a correctness bug:

| Map | Role |
|---|---|
| `request → [extent_id]` | **the address** — the block table the worker uses |
| `page_id → {extent_id}` | **content locator** — used only to pick a CoW source |

The locator is *many-valued*: a copy-on-write gives a request its own physical
copy of a shared prefix, so one logical page legitimately has different
physical homes in different requests. Physical address is therefore per-request
and never per-page. Any holder of a page is an equally valid copy source — all
holders have identical bytes.

Within an extent, sequential writes (I3) make the slot a pure function of the
logical page index:

```text
slot(page_index) = page_index % pages_per_extent
```

**Page-id recycling.** Upstream reissues a freed page id immediately (unhashed
blocks go to the head of its free queue), so an older extent can still claim a
page id whose content has been replaced. Binding therefore distinguishes a
*fresh* page (upstream just handed the id out; every other claim on it is
stale and must be revoked) from a *copy* (content that remains valid
elsewhere). Revoking poisons the stale slot rather than removing it — removing
would shift later slots and break the positional addressing I3 rests on.

### Allocator

The free list holds extents; reclaim returns whole extents (I7). A slice of the
pool is withheld as over-provisioning and released only to copy-on-write
destinations, so a pool that is merely full still degrades gracefully.

## Invariants

| # | Invariant | Why |
|---|---|---|
| **I1** | Prefix-cache hits land only on page boundaries (`num_tokens % page_size == 0`). | Hash keys exist only there. |
| **I2** | A request's `num_computed_tokens` is page-aligned throughout prefill. | Keeps pages either complete or untouched. Holds because `page_size % chunk_size == 0`, at most one prefill runs per step, and every per-step token clamp is page-aligned (see [§ Scheduler](#scheduler-integration)). |
| **I3** | Within an extent, a request's pages are written sequentially from offset 0. | Sequential-write (ZNS-style) rule; makes `page_offset` derivable and the extent DMA-contiguous. |
| **I4** | Never append into an extent another request references. A partial match is resolved by copying into a private extent. | Out-of-place update: an in-place append would corrupt the sharer's prefix. |
| **I5** | Only **full** extents are attach targets. Partial extents may be **copy sources** but are never attached by reference. | A partial extent still has a live write pointer (I3, I4). Its completed pages are still indexed — that is the hit-rate win over upstream. |
| **I6** | Dedup happens only when an extent becomes full, keyed by its extent hash (the chained page hash at its last page boundary). | Partial extents have no stable identity. |
| **I7** | Reclaim is at extent granularity; no mid-extent holes. | Erase-unit asymmetry. Page-only eviction is forbidden. |
| **I8** | Refcounts live on the **extent**. The page-hash index is mapping metadata and owns no reference. | By I4/I5, sharing is always whole-extent, so per-page refcounts would be uniform across an extent by construction. |
| **I9** | The scheduler always leaves at least one token to recompute (`match ≤ num_tokens - 1`). | Upstream requires the last token's forward pass to produce logits. |
| **I10** | Eligible specs store per-token KV: `FullAttentionSpec` (MVP), later `SlidingWindowSpec` / `ChunkedLocalAttentionSpec`. `MambaSpec` / `CrossAttentionSpec` are ineligible. | Partial copying requires a sliceable token dimension. |

**Two distinct counters — do not conflate.**

- *Extent refcount* (I8) governs memory lifetime.
- *Hash multiplicity* — how many extents currently carry a given page hash —
  governs KV-event emission (Store/Remove fire on 0↔1 transitions). CoW
  duplicates page hashes across two extents, which is precisely why this
  counter exists.

## Lifecycle

```text
1. schedule / hash   : page unit (--block-size)
2. allocate          : extent unit, per-request open extent, append-only
3. partial match     : full extents attached by reference; the straddling
                       extent's matched pages copied into a private extent
4. extent fills      : if its extent hash already exists, collapse to canonical
5. request finishes  : extents become retained (cached, unreferenced)
6. eviction          : whole extent reclaimed, its page hashes unmapped
```

### Allocate

Append into the request's open extent; when it fills, allocate a new one.
Capacity is accounted in extents even though the scheduler counts pages
(see [§ Capacity](#capacity-and-cost-model)).

### Partial merge (CoW)

Split the matched prefix at extent boundaries:

```text
matched prefix = [ full extent ][ full extent ] ... [ k pages of extent E ]
                  └── attach by reference ──┘        └── copy into private F,
                                                          slots 0..k-1 ──┘
then continue appending into F at slot k.
```

Consequences:

- Copy cost per match is `< extent_size` tokens regardless of match length —
  hybrid-FTL *partial* merge only; a full merge never occurs.
- Source pages (which may live in a partial extent, per I5) are pinned until
  the worker's copy completes — today's `release_copy_ops` contract.
- Lookups may match interior pages of a full extent, not just prompt tails.
  This is deliberately denser than upstream and preserves today's hit rate.

### Full dedup

When a private extent fills, publish its extent hash. If the hash is already
mapped to a canonical extent, repoint the request at the canonical one and
release the private copy when its refcount allows. This bounds how long CoW
duplicates persist.

### Extent lifetime

Three states, mirroring trim-vs-erase:

| State | Meaning |
|---|---|
| **live** | refcount > 0; at least one request owns it |
| **retained** | refcount == 0, still reachable through the page-hash index; a prefix-cache hit can revive it |
| **reclaimed** | evicted; page hashes unmapped, extent returned to the free list |

The overlay's trick of assigning a synthetic `block_hash` to a finishing
request's partial block exists to make the *retained* state reachable. In this
design it is a first-class transition rather than a workaround.

## Capacity and cost model

Three quantities, all measurable:

**Copy amplification.**

```text
CA = copied tokens / newly computed tokens
per-match bound: < extent_size tokens (one partial merge)
```

Report it; treat a sustained rise as the signal that extents are too large or
over-provisioning is too low.

**Internal fragmentation.** One open extent per concurrent request (the
multi-stream cost):

```text
worst case = concurrent_requests * (extent_size - page_size) tokens
REQUIRED:  max_num_seqs * extent_size  <<  KV pool capacity
```

`RBLN_DEFAULT_MAX_NUM_SEQS` is 1, but serving deployments raise it; at
`max_num_seqs=64, extent_size=4096` this pins ~256K tokens in partial extents.
Validate this at startup rather than discovering it as preemption thrash.

**Over-provisioning.** The scheduler's page count must not expose the whole
physical pool: CoW needs a destination extent, and if free extents run dry the
system degrades sharply (the SSD write-cliff analogue). Reserve a fraction of
extents, and round each request's need **up to extent granularity** when
reporting free capacity — otherwise upstream's `memory / page_size_bytes`
accounting over-admits by exactly the fragmentation above.

## Scheduler integration

After the CLI flip, `block_size` means *page* everywhere it is read. Every
existing use must be classified; the ones that actually mean *extent* have to
be changed. Known sites in `vllm_rbln/v1/core/rbln_scheduler.py`:

| Site | Current meaning | After flip |
|---|---|---|
| `long_prefill_token_threshold` clamp (L221) | arbitrary token cap | must be floored to a page multiple, or it breaks **I2** — the only remaining misalignment path |
| spec-decode "contiguous KV window" clamp (L260-262), `self.block_size` | physical block | must use `extent_size`; using the page is safe but needlessly narrows the decode window |
| `_mamba_block_aligned_split` | mamba block alignment | unchanged; also the template for the page-aligned clamp above |

I2 does *not* need a per-request budget clamp for chunked prefill: prefill
batch size is fixed to 1 (`rbln_scheduler.py` L189-196, L407-412), so a prefill
request always gets the full chunk and alignment holds inductively.

**Hybrid models must be rejected, not merely unoptimized.** `--block-size` is
global, so flipping it to the page size also reshapes non-attention groups.
Gate on I10 at startup and refuse (or fall back to the overlay) when any group
is ineligible.

## Connectors and LMCache

The page-id space is what connectors already consume, so most of the shadow
config in `sub_block_paging` disappears: `--block-size` *is* the page size, so
`pages_per_block` collapses to 1 and the expand/split becomes the identity.

**But the device-side translation does not disappear** — the KV tensor stays
extent-major, so gather/scatter still needs `(extent_id, page_offset)`. Since
`pages_per_extent` can no longer be derived from vLLM config (both numbers are
now equal), the backend must expose `extent_size` to the connector through an
explicit channel. This is an open item; see [§ Open questions](#open-questions).

### Cache hierarchy policy

Device extents plus LMCache CPU/disk form a multi-level cache, and the policy
is currently unspecified. Because the eviction unit is a whole 4096-token
extent, these choices matter more here than on GPU vLLM:

- **Write-back vs write-through**: push a page to LMCache when it completes, or
  when its extent is evicted? Write-through spends steady bandwidth;
  write-back concentrates it into eviction stalls.
- **Inclusive vs exclusive**: may the CPU tier duplicate what is resident on
  device, or only hold what was evicted?
- **Admission control**: a long, never-reused prompt sweeps the cache (the
  sequential-scan pollution problem). Consider not admitting one-shot data, or
  a scan-resistant policy (segmented LRU / ARC) instead of plain LRU.

## KV events

Emit and index at **page** granularity — which, after the flip, is simply
upstream's own block granularity, so the custom emission path shrinks to the
dedup rule.

CoW makes two extents carry the same page hashes, so Store/Remove must fire on
0↔1 transitions of *hash multiplicity* (not extent refcount, per I8) to keep
llm-d's set-membership index correct. Rationale and the chain-safety argument
are unchanged from
[sub_block_prefix_caching.md § KV cache events](./sub_block_prefix_caching.md#kv-cache-events).

Routers configure their block size to `--block-size`. The "use
`--max-num-batched-tokens` instead" exception goes away.

## Comparison

### vs current overlay

| | Overlay today | Page + extent |
|---|---|---|
| CLI `--block-size` | extent (large) | **page** |
| Schedule / hash unit | extent + sub-block extension | **page** |
| Indexing | `SubBlockIndex` layered on extents | pages first-class; extent is packing |
| Partial hit | memcpy into a new block | partial merge into a private extent (same mechanism, named) |
| Interior page of a full block | indexed | kept — no regression to tail-only |
| Full dedup | implicit via upstream full hash | explicit end of the CoW lifetime |
| Contiguity | ensured by copying into a fresh block | allocator invariant (I3) |
| Connector arbitration | mutually exclusive with KV connectors | **cooperative** — see below |

**Connector cooperation is a direct win.** The overlay must arbitrate
(sub-block match *or* connector match, whichever is longer) because a connector
cannot be assumed to handle a match that starts off block boundary. When the
page is the native scheduling unit, every match already ends on a page
boundary, so a page match and a connector match can compose: page match
extends the hit, the connector extends further from there.

### vs upstream partial prefix cache

Upstream `prefix_match_unit` + `cache_partial_block` + CoW is the closest
concept, but cannot be adopted as-is:

- `UnitaryKVCacheCoordinator` asserts `hash_block_size == block_size`
  (`vllm/v1/core/kv_cache_coordinator.py:477`), so a single attention group with
  large blocks is excluded.
- Routing around it via a hybrid config does not help either:
  `HybridKVCacheCoordinator.enable_partial_hash_hits` is gated on the presence
  of a `MambaSpec` group in `align` mode (same file, L583-589). Attention-only
  deployments are excluded from fine-grained hits regardless of coordinator.
- Partial registration is **prompt-tail only** —
  `_cache_partial_tail_block` explicitly skips intermediate hash boundaries
  inside a block — so a shorter prefix inside a full block misses.
- Sliding-window fine-grained hits are unsupported.

Long-term convergence is desirable; short-term RBLN still needs unitary
fine-grained behavior with interior-page indexing.

### Why not `kernel_block_size`

Upstream's `kernel_block_size` looks related but runs in the opposite
direction: `prepare_kernel_block_sizes` → `select_common_block_size` **splits**
a manager block into smaller kernel blocks, so `kernel_block_size ≤ block_size`
is structural (`num_blocks_per_kv_block = block_size // kernel_block_size`).
An extent **groups** pages, which that axis cannot express.

It is currently inert in vllm-rbln: no RBLN backend overrides
`get_supported_kernel_block_sizes`, so `select_common_block_size` returns
`block_size` unchanged and the split factor is 1. Reusing the name for extents
would collide with a live upstream mechanism for no gain.

## Non-goals (MVP)

- Sliding-window / Mamba / hybrid fine-grained paths.
- Replacing upstream's hybrid partial-tail machinery for Mamba align mode.
- **Extent compaction (GC).** I7 forbids holes, so an extent whose early pages
  are hot and whose tail is dead cannot be partially reclaimed. The standard
  answer — victim selection by valid-page ratio (greedy or cost-benefit),
  followed by migrating the valid pages — is deferred. Record the trigger
  metric (per-extent valid-page ratio) now so the hook has a home.
- Shipping code in this document's PR.

## Open questions

1. ~~How is `extent_size` published?~~ **Resolved**: through
   `additional_config["attn_block_size"]` — set by the converter on the
   optimum path, given on the command line on the native path. No new
   environment variable. Follow-up: derive it from the native compile config
   so a mismatch with the compiled kernel is caught at startup.
2. Over-provisioning ratio: fixed fraction, or derived from `max_num_seqs`?
3. Cache hierarchy policy: write-back vs write-through, inclusive vs exclusive,
   admission control.
4. Should the number of concurrently open extents be capped (bounding
   fragmentation) at the cost of rejecting or downgrading some requests?
5. Measured `extent_size` vs DMA bandwidth curve — the number that justifies
   the whole design and sets the CoW budget.

## Risks

| Risk | Mitigation |
|---|---|
| vLLM assumes `block_size` == KV tensor token dim | Explicit page→extent table + extent-major tensors; audit every `block_size` read ([§ Scheduler](#scheduler-integration)) |
| Copy amplification | Partial merges only (`< extent_size` per match); dedup promptly on full; track CA as a metric |
| Free-extent exhaustion (write cliff) | Over-provisioning; extent-granular capacity accounting |
| Internal fragmentation | Startup check `max_num_seqs * extent_size << pool` |
| Allocator complexity | MVP: append-only open extent + CoW on partial match — close to today's always-copy behavior |
| Hybrid / SWA / Mamba | Reject at startup (I10), do not silently degrade |

## Implementation status

Enabled with `VLLM_RBLN_PAGE_EXTENT=1` (default off). A model that publishes no
extent size gets a degenerate geometry and the layer is a no-op, so the flag is
safe to set blindly.

Landed in `afd7142d`: `page_extent.py` (geometry, extent pool, page->extent
map, binding policy), `rbln_page_extent_kv_cache_manager.py`, scheduler wiring
and extent-id block tables, worker geometry restatement and slot-range copies.
409 unit tests.

### Measured 2026-08-13 (MiniMax-M2.5, DP4+EP, 1536-token shared prefix)

| config | match | physical | mean TTFT | mean TPOT | tok/s |
|---|---|---|---|---|---|
| sub-block off | 1024 | 1024 | 877.1 ms | 65.67 ms | 101.8 |
| sub-block on | 512 | 1024 | 493.2 ms | 62.72 ms | 113.9 |
| page/extent | 512 | 8192 | 519.2 ms | 51.87 ms | 134.2 |
| sub-block on | 512 | 8192 | 539.0 ms | 51.98 ms | 133.9 |
| sub-block off | 8192 | 8192 | 1330.8 ms | 69.77 ms | 88.8 |

The effect decomposes cleanly and neither half belongs to this design:
**match granularity** (512) buys TTFT (-44%), **physical block size** (8192)
buys TPOT (-17%) and throughput (+18%). Page/extent and the overlay are within
noise of each other once both run at 8192, so **page/extent carries no
performance advantage**. Its case is alignment: `--block-size` becomes the unit
the scheduler, routers and connectors share, and upstream's native hashing and
events replace the overlay's reimplementation.

Immediately actionable and independent of this work: the current default
deployment gains ~18% throughput from `--block-size 8192` alone.

### Verification

Wrong KV here is silent *and reads faster* -- fixed output length makes token
counts identical across configs, and every benchmark metric improved while the
model emitted garbage. Only a greedy probe diffed against the overlay catches
it. Do this after any addressing change.

Eviction paths are unit-tested but have never run on hardware: the benchmark
pool (222K tokens) never evicted. Force it with `--num-gpu-blocks-override`.

## Next steps

1. **Pin down what consumes `cache_config.block_size` in the worker.**
   Restating only the KV cache spec produced corrupt output; also overwriting
   `cache_config` fixed it, but the RBLN attention impls are *not* the
   consumer -- their `block_size` only selects a mode. This blocks (2).
2. **Per-group physical units, for SWA hybrids.** The SWA kernel asserts
   `sliding_window == kv_cache.size(-2)`, so an SWA group's physical unit is
   fixed by its window, not chosen. The page stays global (upstream shares
   `hash_block_size` and `num_computed_tokens` across groups) but the extent
   must become per-group, which the single global `cache_config.block_size`
   overwrite cannot express -- hence (1) first. Then drop the single-group
   restriction in `can_use_page_extent`.
3. **Rename with (2).** `extent` collides with two existing names for the same
   thing: the compiler's `kvcache_partition_len` / `attn_block_size` and
   upstream's `kernel_block_size`. Prefer `kernel_block`, which is already the
   vocabulary in the SWA assert message; note upstream's namesake *splits*
   where this *groups*. `ExtentGeometry` -> `PageLayout`. Not worth doing
   before (2) reshapes it.
4. **O(1) `bind()`.** It rewalks every extent of a request each step: 0.54 us
   at 4 pages, 20.66 us at 1024. That is 0.08% of TPOT at `max_model_len`, so
   it is scaling, not a present cost. Skip sealed extents.
5. **Decide whether to keep this path at all**, given it is performance-neutral
   and costs ~230 lines in the scheduler and worker plus a second KV stack to
   maintain.

## Migration

1. Agree the invariants and the spec (this doc).
2. Land the addressing layer behind the overlay, without flipping CLI
   semantics. **Done** — `vllm_rbln/v1/core/page_extent/` (geometry, allocator,
   page→extent table, binding policy) with unit tests.
3. Flip `--block-size` to the page; derive `extent_size` in the platform;
   update the `block_size` audit sites, tests, llm-d notes, and LMCache paging
   defaults.
4. Enable connector cooperation (drop the mutual-exclusion arbitration).
5. Retire overlay-only APIs (`get_computed_blocks_sub_block`, etc.).
6. Track upstream support for unitary `hash_block_size < block_size`.

## Decision summary

| Decision | Choice |
|---|---|
| Schedule / hash / match unit | page (`--block-size`), `page % chunk == 0` |
| Physical / DMA unit | extent (backend-derived, e.g. 4096) |
| Mapping granularity | hybrid: fine map, coarse allocation, merge on partial update |
| New env `VLLM_RBLN_KERNEL_BLOCK_SIZE` | **No** |
| Partial match | partial merge (CoW) into a private extent |
| Dedup | full extents only, by extent hash |
| Refcount granularity | extent (I8) |
| Interior page indexing | **Yes** (preserves today's hit rate) |
| Connector interaction | cooperative, not arbitrated |
| GC / compaction | out of MVP; trigger metric recorded |
| MVP specs | full attention only |
