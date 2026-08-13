# Page + Kernel Block KV Manager (Design)

Status: **partially implemented**, off by default behind `VLLM_RBLN_PAGE_LAYOUT`.
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
value against the compiled artifact, so an operator who names a kernel block size
the kernel does not actually use gets no warning. Deriving it from the native
compile config would close that hole and is the natural follow-up.

Upstream vLLM treats `--block-size` as one unit for everything: scheduling,
hashing, prefix matching, and physical storage. With a 4096-token block, any
reusable prefix that ends inside a block is missed, so vllm-rbln added an
overlay ([sub-block prefix caching](./sub_block_prefix_caching.md)) -- "the
overlay" below always means that feature, gated by `VLLM_RBLN_SUB_BLOCK_CACHE`.

**Scope.** This design targets the **native vLLM-model path**
(`VLLM_RBLN_USE_VLLM_MODEL=True`), where `--block-size` is still the coarse
unit. The optimum path (the default) already runs a two-level scheme under
different names — `attn_block_size`/`ob_size` is the kernel block,
`cache_config.block_size`/`ib_size` is the page, `get_block_ratio()` is
`pages_per_kernel_block` — implemented in `RBLNPrefixKVCacheManager`. That path is
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
| erase unit (erase block) | **kernel block** (e.g. 4096) |
| LBA space exposed to the host | page-id space exposed to scheduler / connectors |
| physical address `(block, page offset)` | `(kernel_block_id, page_offset)` |
| FTL mapping table | page-hash index (`page_hash → (kernel_block_id, page_offset)`) |
| out-of-place update (no in-place overwrite) | copy-on-write on a shared prefix |
| open block + sequential write pointer | per-request open kernel block, append-only |
| multi-stream (one open block per stream) | one open kernel block per request |
| trim (invalidate) vs erase (reclaim) | request free vs prefix-cache eviction |
| write amplification | copy amplification |
| over-provisioning | reserved free kernel blocks |
| GC victim selection by valid-page count | kernel block compaction (future, not MVP) |

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
| **kernel block** | 4096 | Allocation / reclaim / DMA unit; contiguous token storage |

```text
kernel block (physical, DMA-addressed)
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ page0 │ page1 │ page2 │ page3 │ page4 │ page5 │ page6 │ page7 │
└───────┴───────┴───────┴───────┴───────┴───────┴───────┴───────┘
 ←──── one request's pages, written sequentially from offset 0 ────→
```

### Naming vs today's code

| This design | Current overlay (`RBLNKVCacheManager`) |
|---|---|
| page | sub-block (`sub_block_size`) |
| kernel block | physical / upstream block (`block_size`) |
| page hash chain | `SubBlockHasher` |
| page → kernel block map | `SubBlockIndex` |
| partial merge (CoW) | `KVCacheCopyOp` |
| full dedup | upstream full-block hash hit + collapse of the private copy |

Internal code uses `page` / `kernel_block`. That is deliberately upstream's
word for the same thing, built the opposite way — see
[§ Relation to upstream `kernel_block_size`](#relation-to-upstream-kernel_block_size).

## Spec

### Sizes

```text
chunk_size             = scheduler_config.max_num_batched_tokens
page_size              = cache_config.block_size    # --block-size
kernel_block_size      = backend-derived (not user-visible)
pages_per_kernel_block = kernel_block_size // page_size

REQUIRED: page_size % chunk_size == 0        # a prefill never spans two pages
REQUIRED: kernel_block_size % page_size == 0
REQUIRED: pages_per_kernel_block >= 2        # otherwise use plain upstream
```

`page_size == chunk_size` is the default (one chunk completes one page).
`page_size = k * chunk_size, k > 1` is allowed: the page becomes
hash-complete after `k` chunks. `chunk_size > page_size` is rejected.

### Configuration

User-visible:

| Knob | Role | Note |
|---|---|---|
| `--block-size` | **page** size | must be a multiple of the prefill chunk, must divide `kernel_block_size` |
| `--max-num-batched-tokens` | prefill chunk | `page_size % chunk_size == 0` |
| `--enable-prefix-caching` | on/off | unchanged |
| `VLLM_RBLN_PAGE_LAYOUT` | feature gate | `VLLM_RBLN_PAGE_EXTENT` is accepted as an alias; supersedes `VLLM_RBLN_SUB_BLOCK_CACHE`, which it disables when on |

Published by the model, not tuned per run:

| Knob | Role |
|---|---|
| `additional_config["attn_block_size"]` | `kernel_block_size` — the DMA packing size the attention kernel was compiled for |
| `pages_per_kernel_block` | derived: `kernel_block_size // page_size` |

Do **not** add `VLLM_RBLN_KERNEL_BLOCK_SIZE`. A model that publishes nothing
gets a degenerate geometry (one page per kernel block) and the layer is a no-op, so
enabling the feature is always safe.

Worked example — page 1024, kernel block 8192, eight pages per kernel block:

```bash
VLLM_RBLN_PAGE_LAYOUT=1 vllm serve <model> \
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
| `request → [kernel_block_id]` | **the address** — the block table the worker uses |
| `page_id → {kernel_block_id}` | **content locator** — used only to pick a CoW source |

The locator is *many-valued*: a copy-on-write gives a request its own physical
copy of a shared prefix, so one logical page legitimately has different
physical homes in different requests. Physical address is therefore per-request
and never per-page. Any holder of a page is an equally valid copy source — all
holders have identical bytes.

Within a kernel block, sequential writes (I3) make the slot a pure function of the
logical page index:

```text
slot(page_index) = page_index % pages_per_kernel_block
```

**Page-id recycling.** Upstream reissues a freed page id immediately (unhashed
blocks go to the head of its free queue), so an older kernel block can still claim a
page id whose content has been replaced. Binding therefore distinguishes a
*fresh* page (upstream just handed the id out; every other claim on it is
stale and must be revoked) from a *copy* (content that remains valid
elsewhere). Revoking poisons the stale slot rather than removing it — removing
would shift later slots and break the positional addressing I3 rests on.

### Allocator

The free list holds kernel blocks; reclaim returns whole kernel blocks (I7). A slice of the
pool is withheld as over-provisioning and released only to copy-on-write
destinations, so a pool that is merely full still degrades gracefully.

## Invariants

| # | Invariant | Why |
|---|---|---|
| **I1** | Prefix-cache hits land only on page boundaries (`num_tokens % page_size == 0`). | Hash keys exist only there. |
| **I2** | A request's `num_computed_tokens` is page-aligned throughout prefill. | Keeps pages either complete or untouched. Holds because `page_size % chunk_size == 0`, at most one prefill runs per step, and every per-step token clamp is page-aligned (see [§ Scheduler](#scheduler-integration)). |
| **I3** | Within a kernel block, a request's pages are written sequentially from offset 0. | Sequential-write (ZNS-style) rule; makes `page_offset` derivable and the kernel block DMA-contiguous. |
| **I4** | Never append into a kernel block another request references. A partial match is resolved by copying into a private kernel block. | Out-of-place update: an in-place append would corrupt the sharer's prefix. |
| **I5** | Full kernel blocks are attach targets, shared read-only. A partial kernel block is a **copy source** for anyone, and an **adoption** target for at most one request: an unreferenced one whose pages are exactly the leading pages of the group being bound is extended in place (**I5b**). | A partial kernel block has a live write pointer (I3), so only one writer may hold it; `ref_cnt == 0` means the sharer I4 protects does not exist, and the resumed pages keep their slots. Adoption is what makes the multi-turn append case zero-copy. |
| **I6** | Dedup happens only when a kernel block becomes full, keyed by its kernel-block hash (the chained page hash at its last page boundary). | Partial kernel blocks have no stable identity. |
| **I7** | Reclaim is at kernel block granularity; no mid-kernel-block holes. | Erase-unit asymmetry. Page-only eviction is forbidden. |
| **I8** | Refcounts live on the **kernel block**. The page-hash index is mapping metadata and owns no reference. | By I4/I5, sharing is always whole-kernel-block, so per-page refcounts would be uniform across a kernel block by construction. |
| **I9** | The scheduler always leaves at least one token to recompute (`match ≤ num_tokens - 1`). | Upstream requires the last token's forward pass to produce logits. |
| **I10** | Eligible specs store per-token KV: `FullAttentionSpec` (MVP), later `SlidingWindowSpec` / `ChunkedLocalAttentionSpec`. `MambaSpec` / `CrossAttentionSpec` are ineligible. | Partial copying requires a sliceable token dimension. |

**Two distinct counters — do not conflate.**

- *Kernel block refcount* (I8) governs memory lifetime.
- *Hash multiplicity* — how many kernel blocks currently carry a given page hash —
  governs KV-event emission (Store/Remove fire on 0↔1 transitions). CoW
  duplicates page hashes across two kernel blocks, which is precisely why this
  counter exists.

## Lifecycle

```text
1. schedule / hash   : page unit (--block-size)
2. allocate          : kernel block unit, per-request open kernel block, append-only
3. partial match     : full kernel blocks attached by reference; the straddling
                       kernel block's matched pages copied into a private kernel block
4. kernel block fills      : if its kernel-block hash already exists, collapse to canonical
5. request finishes  : kernel blocks become retained (cached, unreferenced)
6. eviction          : whole kernel block reclaimed, its page hashes unmapped
```

### Allocate

Append into the request's open kernel block; when it fills, allocate a new one.
Capacity is accounted in kernel blocks even though the scheduler counts pages
(see [§ Capacity](#capacity-and-cost-model)).

### Partial merge (CoW)

Split the matched prefix at kernel block boundaries:

```text
matched prefix = [ full kernel block ][ full kernel block ] ... [ k pages of kernel block E ]
                  └── attach by reference ──┘        └── copy into private F,
                                                          slots 0..k-1 ──┘
then continue appending into F at slot k.
```

Consequences:

- Copy cost per match is `< kernel_block_size` tokens regardless of match length —
  hybrid-FTL *partial* merge only; a full merge never occurs.
- Source pages (which may live in a partial kernel block, per I5) are pinned until
  the worker's copy completes — today's `release_copy_ops` contract.
- Lookups may match interior pages of a full kernel block, not just prompt tails.
  This is deliberately denser than upstream and preserves today's hit rate.

### Full dedup

When a private kernel block fills, publish its kernel-block hash. If the hash is already
mapped to a canonical kernel block, repoint the request at the canonical one and
release the private copy when its refcount allows. This bounds how long CoW
duplicates persist.

### Kernel block lifetime

Three states, mirroring trim-vs-erase:

| State | Meaning |
|---|---|
| **live** | refcount > 0; at least one request owns it |
| **retained** | refcount == 0, still reachable through the page-hash index; a prefix-cache hit can revive it |
| **reclaimed** | evicted; page hashes unmapped, kernel block returned to the free list |

The overlay's trick of assigning a synthetic `block_hash` to a finishing
request's partial block exists to make the *retained* state reachable. In this
design it is a first-class transition rather than a workaround.

## Capacity and cost model

Three quantities, all measurable:

**Copy amplification.**

```text
CA = copied tokens / newly computed tokens
per-match bound: < kernel_block_size tokens (one partial merge)
```

Report it; treat a sustained rise as the signal that kernel blocks are too large or
over-provisioning is too low.

**Internal fragmentation.** One open kernel block per concurrent request (the
multi-stream cost):

```text
worst case = concurrent_requests * (kernel_block_size - page_size) tokens
REQUIRED:  max_num_seqs * kernel_block_size  <<  KV pool capacity
```

`RBLN_DEFAULT_MAX_NUM_SEQS` is 1, but serving deployments raise it; at
`max_num_seqs=64, kernel_block_size=4096` this pins ~256K tokens in partial kernel blocks.
Validate this at startup rather than discovering it as preemption thrash.

**Over-provisioning.** The scheduler's page count must not expose the whole
physical pool: CoW needs a destination kernel block, and if free kernel blocks run dry the
system degrades sharply (the SSD write-cliff analogue). Reserve a fraction of
kernel blocks, and round each request's need **up to kernel block granularity** when
reporting free capacity — otherwise upstream's `memory / page_size_bytes`
accounting over-admits by exactly the fragmentation above.

## Scheduler integration

After the CLI flip, `block_size` means *page* everywhere it is read. Every
existing use must be classified; the ones that actually mean *kernel block* have to
be changed. Known sites in `vllm_rbln/v1/core/rbln_scheduler.py`:

| Site | Current meaning | After flip |
|---|---|---|
| `long_prefill_token_threshold` clamp (L221) | arbitrary token cap | must be floored to a page multiple, or it breaks **I2** — the only remaining misalignment path |
| spec-decode "contiguous KV window" clamp (L260-262), `self.block_size` | physical block | must use `kernel_block_size`; using the page is safe but needlessly narrows the decode window |
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
kernel block-major, so gather/scatter still needs `(kernel_block_id, page_offset)`. Since
`pages_per_kernel_block` can no longer be derived from vLLM config (both numbers are
now equal), the backend must expose `kernel_block_size` to the connector through an
explicit channel. This is an open item; see [§ Open questions](#open-questions).

### NIXL / P-D disaggregation (unverified, open)

Not exercised yet, and there is no guard, so this is a code-reading account
rather than a measurement.

`ensure_kv_transfer_initialized` runs *before* the runner's
`initialize_kv_cache` (`rbln_worker.py`), and the runner deepcopies before
restating geometry, so the NIXL worker holds the **pre-restatement, page-sized**
config: `num_blocks` is a page count and `self.block_size` the page. The tensors
it then registers are kernel-block shaped, so
`assert cache.shape[0] == num_blocks` in the NIXL worker looks like it should
fire at startup -- a loud failure rather than silent mis-addressing, but an
unexplained one. `_select_canonical_kv_layers_per_pool` also loses the equality
its docstring rests on: it prefers a Full-attention layer because that view's
`cache.shape[-2]` equals `cache_config.block_size`, which page layout makes
false (kernel block vs page).

**A page-granular descriptor layout should be sound**, so this is a descriptor
question rather than a redesign: pages inside a kernel block are contiguous by
I3 and the pool is one flat token space, so page-space addressing stays correct
as long as registration describes the tensor at page granularity. What needs
settling is whether NIXL keeps the page as its unit (and the `shape[0]` /
stride derivation follows) or is taught the kernel block.

### Cache hierarchy policy

Device kernel blocks plus LMCache CPU/disk form a multi-level cache, and the policy
is currently unspecified. Because the eviction unit is a whole 4096-token
kernel block, these choices matter more here than on GPU vLLM:

- **Write-back vs write-through**: push a page to LMCache when it completes, or
  when its kernel block is evicted? Write-through spends steady bandwidth;
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

CoW makes two kernel blocks carry the same page hashes, so Store/Remove must fire on
0↔1 transitions of *hash multiplicity* (not kernel block refcount, per I8) to keep
llm-d's set-membership index correct. Rationale and the chain-safety argument
are unchanged from
[sub_block_prefix_caching.md § KV cache events](./sub_block_prefix_caching.md#kv-cache-events).

Routers configure their block size to `--block-size`. The "use
`--max-num-batched-tokens` instead" exception goes away.

## Comparison

### vs current overlay

| | Overlay today | Page + kernel block |
|---|---|---|
| CLI `--block-size` | kernel block (large) | **page** |
| Schedule / hash unit | kernel block + sub-block extension | **page** |
| Indexing | `SubBlockIndex` layered on kernel blocks | pages first-class; kernel block is packing |
| Partial hit | memcpy into a new block | partial merge into a private kernel block (same mechanism, named) |
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

### Relation to upstream `kernel_block_size`

Same denotation, opposite construction. Both name *the token extent the
attention kernel addresses*, which is why this design borrows the word (it is
also the vocabulary of the SWA kernel's `sliding_window == kv_cache.size(-2)`
assert). But upstream **splits**: `prepare_kernel_block_sizes` ->
`select_common_block_size` divides a manager block into smaller kernel blocks,
so `kernel_block_size <= cache_config.block_size` is structural
(`num_blocks_per_kv_block = block_size // kernel_block_size`). This design
**groups** pages, so `kernel_block_size >= cache_config.block_size`. **That
inverts upstream's invariant, and it is the thing to watch when reading code
that spans both.**

In practice the two coincide here. The split is inert in vllm-rbln -- no RBLN
backend overrides `get_supported_kernel_block_sizes`, so `select_common_block_size`
returns `block_size` unchanged and the factor is 1 -- and the worker restates
the KV spec to the kernel block before `prepare_kernel_block_sizes` runs, so it
reports exactly the grouped size. For an SWA group it reports `sliding_window`,
which is precisely the per-group physical unit that
[§ Next steps](#next-steps) (2) needs; the two mechanisms are expected to merge
there rather than coexist.

## Non-goals (MVP)

- Sliding-window / Mamba / hybrid fine-grained paths.
- Replacing upstream's hybrid partial-tail machinery for Mamba align mode.
- **Kernel block compaction (GC).** I7 forbids holes, so a kernel block whose early pages
  are hot and whose tail is dead cannot be partially reclaimed. The standard
  answer — victim selection by valid-page ratio (greedy or cost-benefit),
  followed by migrating the valid pages — is deferred. Record the trigger
  metric (per-kernel-block valid-page ratio) now so the hook has a home.
- Shipping code in this document's PR.

## Open questions

1. ~~How is `kernel_block_size` published?~~ **Resolved**: through
   `additional_config["attn_block_size"]` — set by the converter on the
   optimum path, given on the command line on the native path. No new
   environment variable. Follow-up: derive it from the native compile config
   so a mismatch with the compiled kernel is caught at startup.
2. Over-provisioning ratio: fixed fraction, or derived from `max_num_seqs`?
3. Cache hierarchy policy: write-back vs write-through, inclusive vs exclusive,
   admission control.
4. Should the number of concurrently open kernel blocks be capped (bounding
   fragmentation) at the cost of rejecting or downgrading some requests?
5. Measured `kernel_block_size` vs DMA bandwidth curve — the number that justifies
   the whole design and sets the CoW budget.

## Risks

| Risk | Mitigation |
|---|---|
| vLLM assumes `block_size` == KV tensor token dim | Explicit page→kernel block table + kernel block-major tensors; audit every `block_size` read ([§ Scheduler](#scheduler-integration)) |
| Copy amplification | Partial merges only (`< kernel_block_size` per match); dedup promptly on full; track CA as a metric |
| Free-kernel-block exhaustion (write cliff) | Over-provisioning; kernel block-granular capacity accounting |
| Internal fragmentation | Startup check `max_num_seqs * kernel_block_size << pool` |
| Allocator complexity | MVP: append-only open kernel block + CoW on partial match — close to today's always-copy behavior |
| Hybrid / SWA / Mamba | Reject at startup (I10), do not silently degrade |

## Implementation status

Enabled with `VLLM_RBLN_PAGE_LAYOUT=1` (default off). A model that publishes no
kernel block size gets a degenerate geometry and the layer is a no-op, so the flag is
safe to set blindly.

Landed in `afd7142d`: `page_layout.py` (geometry, kernel block pool, page->kernel block
map, binding policy), `rbln_page_layout_kv_cache_manager.py`, scheduler wiring
and kernel block-id block tables, worker geometry restatement and slot-range copies.
409 unit tests.

### Measured 2026-08-13 (MiniMax-M2.5, DP4+EP, 1536-token shared prefix)

| config | match | physical | mean TTFT | mean TPOT | tok/s |
|---|---|---|---|---|---|
| sub-block off | 1024 | 1024 | 877.1 ms | 65.67 ms | 101.8 |
| sub-block on | 512 | 1024 | 493.2 ms | 62.72 ms | 113.9 |
| page layout | 512 | 8192 | 519.2 ms | 51.87 ms | 134.2 |
| sub-block on | 512 | 8192 | 539.0 ms | 51.98 ms | 133.9 |
| sub-block off | 8192 | 8192 | 1330.8 ms | 69.77 ms | 88.8 |

The effect decomposes cleanly and neither half belongs to this design:
**match granularity** (512) buys TTFT (-44%), **physical block size** (8192)
buys TPOT (-17%) and throughput (+18%). Page layout and the overlay are within
noise of each other here, which is why this was first read as "no performance
advantage" -- but the workload is a shared prefix fanned out to four different
questions, and on *that* shape parity is structural (see below). The multi-turn
shape, where the two designs do diverge, is measured further down. Beyond speed
the case is alignment: `--block-size` becomes the unit the scheduler, routers
and connectors share, and upstream's native hashing and events replace the
overlay's reimplementation.

Immediately actionable and independent of this work: the current default
deployment gains ~18% throughput from `--block-size 8192` alone.

### D2D copy volume: parity on fan-out, a win on append

Copy volume depends on the *shape* of the traffic, and the benchmark above only
exercised one of the two shapes.

**Fan-out** — one cached prefix, several different continuations. Here the two
paths move identical bytes. Both share whole physical units by reference and
copy only the sub-unit remainder of the match: the overlay from
`apply_sub_block_match` (`num_matched * sub_block_size` tokens), page layout
from `_place`. Volume is `cached_tokens % physical_block` either way, bounded by
one physical unit. Measured 2026-08-13, Qwen3-0.6B, page/sub-block 256, physical
1024, 1640-token shared prefix (6 pages = 1536 tokens = one full 1024 unit plus
a 512-token remainder), 4 prompts after a warm-up:

| config | copy calls | ops | tokens copied |
|---|---|---|---|
| sub-block on (`--block-size 1024`, sub-block 256) | 4 | 4 | 2048 |
| page layout (page 256, kernel block 1024) | 4 | 4 | 2048 |

The second continuation genuinely must copy in both designs: two requests
writing different next pages cannot share one write pointer.

**Append** — a request ends mid-unit and the *next* one extends that same
prefix, the multi-turn chat and agentic-loop shape. In principle page layout can
keep writing into the kernel block it already holds, where the overlay cannot:
upstream caches only full blocks, so its sub-block match must land in a freshly
allocated block and pay the copy every turn. Adoption (I5b) implements that, and
it eliminates the copy outright:

| `[a,b,c]` finishes, then `[a,b,c,d]` | tokens copied | kernel blocks used |
|---|---|---|
| sub-block on | 768 | 2 blocks |
| page layout, no adoption | 768 | 2 |
| page layout with adoption | **0** | **1** |

Real turns do not end on a page boundary either, and adoption still covers
them, because **upstream hands back the same block ids every turn** -- trailing
block included, since the partial tail block is freed and immediately
reallocated. Observed on hardware: `pages=[1,2,3,4,5,6,7] cached=6` identically
on every turn. So the retained kernel block already holds exactly this group, and only
its tail slot needs rewriting. Adoption therefore rewinds the write pointer to
the cached prefix and re-places the rest; a slot may hold the group's page or
`INVALID_PAGE` (poisoned when that page was last written elsewhere) and either
way the request rewrites it.

Rewinding is safe without consulting global hash state, because it only ever
drops slots upstream handed out for *fresh* writing, and upstream does not hand
out a block that is currently cached -- the same reasoning `_revoke_stale_claims`
already relies on. A slot holding a *different* page is not dropped: those bytes
may still be cached for a longer prefix elsewhere.

Measured 2026-08-13, Qwen3-0.6B, 4 turns each resuming the previous turn's
tokens:

| | copy ops | tokens copied | kernel blocks |
|---|---|---|---|
| sub-block on | 3 | 1536 | one new block per turn |
| page layout, no adoption | 3 | 1536 | one new kernel block per turn |
| page layout with adoption | **0** | **0** | 2, reused every turn |

Greedy output is **bit-identical** with and without adoption across all four
turns, which is the check that matters: the two differ only in whether the KV
bytes are copied elsewhere or left in place, so any mis-aimed kernel block would
diverge immediately.

### Measured 2026-08-13 (Qwen3-0.6B, multi-turn, deployment geometry)

10 turns, each resuming the previous turn's tokens, prompt growing 1654 ->
4048 tokens, 256 output tokens per turn pinned with `ignore_eos` so every config
emits exactly 2560 tokens. Chunk 512 throughout, identical KV pool (262144
tokens), one NPU, runs sequential.

The baseline is **sub-block prefix caching** (`VLLM_RBLN_SUB_BLOCK_CACHE=1`),
which is what "the overlay" means throughout this doc; both baseline runs logged
`Sub-block prefix caching enabled: block_size=<N>, sub_block_size=512`.

| config | tok/s | mean TTFT | copy ops | tokens copied | copy time |
|---|---|---|---|---|---|
| sub-block on, block 1024 (today's default) | 191.9 | 34.8 ms | 5 | 2 560 | 61.2 ms |
| sub-block on, block 8192 | 216.4 | 37.8 ms | 9 | 24 064 | 90.1 ms |
| page layout, page 512 kernel block 8192 | **220.9** | **27.7 ms** | **0** | **0** | **0** |

Against sub-block caching at the same physical block size: **TTFT -26.7%**,
tok/s +2.1%. Against today's default (block 1024): tok/s +15.1%, TTFT -20.4%.
The 10.1 ms TTFT gap is almost exactly the per-turn copy cost sub-block caching
pays (90.1 ms / 10 turns = 9.0 ms), so the attribution is direct.

Two things this changes:

- **The copy is per-op, not per-byte.** 12.2 ms/op for 56 MiB (block 1024) and
  10.0 ms/op for 292 MiB (block 8192) -- 5x the bytes at the same cost. It is
  the 28 per-layer slice assignments that dominate, not bandwidth, so earlier
  GB/s reasoning about this cost was wrong. Expect it to scale with layer count:
  MiniMax-M2.5 has 62.
- **The overlay's cost grows with conversation length.** At block 8192 no full
  block exists until the conversation reaches 8192 tokens, so every turn copies
  the entire matched prefix -- 24 064 tokens over 10 turns here, and rising
  quadratically with turn count. Adoption keeps page layout flat at zero. This
  is the agentic / multi-turn case, and it is where the two designs genuinely
  diverge.

One loose end: `page layout @ 8192` and `sub-block @ 1024` produced
byte-identical text, while `sub-block @ 8192` diverged from both. Cross-geometry text comparison
is not diagnostic (chunked-prefill boundaries move with cache hits, so bitwise
equality is not expected even when both are correct), but the odd one out being
the overlay at the large block size is worth a short greedy probe before
recommending that config.

Beyond copies, page layout's other mechanical edge is match *composability*: the
overlay takes its remainder from a single source block, while `_place` locates
each page independently and can compose one destination from several sources.
That shows up as hit rate on fragmented traffic, not as bytes moved.

### Verification

Wrong KV here is silent *and reads faster* -- fixed output length makes token
counts identical across configs, and every benchmark metric improved while the
model emitted garbage. Only a greedy probe diffed against the overlay catches
it. Do this after any addressing change.

Eviction paths are unit-tested but have never run on hardware: the benchmark
pool (222K tokens) never evicted. Force it with `--num-gpu-blocks-override`.

## Next steps

1. ~~**Pin down what consumes `cache_config.block_size` in the worker.**~~
   **Done 2026-08-13 -- nothing in the worker does.** A read tracer installed
   on `cache_config` at the end of the rescale logged exactly two post-rescale
   readers, both in the *engine core*:
   `resolve_kv_cache_block_sizes` (`vllm/v1/core/kv_cache_utils.py:631`, called
   from `EngineCore.__init__`), where a single group makes it *both* the
   scheduler block size and the hash block size, and the config handshake at
   `core.py:1528`, which reports it to the front end. Zero worker-side readers.

   The overwrite reached them only because at `world_size == 1` vLLM uses
   `UniProcExecutor`, so the worker shares the process -- and the `VllmConfig`
   object -- with the engine core; under a multi-process executor (the DP4+EP
   benchmark) it was inert. It is now also fatal in-process: a kernel block-sized
   value contradicts the page-sized spec the scheduler kept, and
   `UnitaryKVCacheCoordinator` asserts `hash_block_size == block_size` at
   startup. **The overwrite is deleted**; the spec restatement alone is
   correct, and the greedy probe on Qwen3-0.6B is coherent and byte-identical
   with and without the (now removed) write. The earlier corruption that
   motivated it must have had another cause -- most likely the
   `long_prefill_token_threshold` flooring bug fixed in the same commit.
2. **Per-group physical units, for SWA hybrids.** The SWA kernel asserts
   `sliding_window == kv_cache.size(-2)`, so an SWA group's physical unit is
   fixed by its window, not chosen. The page stays global (upstream shares
   `hash_block_size` and `num_computed_tokens` across groups) and the kernel block
   becomes per-group. No longer blocked: with (1) resolved the physical unit
   lives only in the per-group spec, which is already per-group, so nothing has
   to be expressed through the one global `cache_config.block_size`. Then drop
   the single-group restriction in `can_use_page_layout`.
3. ~~**Rename.**~~ **Done 2026-08-13.** `extent` -> `kernel_block`,
   `ExtentGeometry` -> `PageLayout`, module `page_extent.py` -> `page_layout.py`,
   env `VLLM_RBLN_PAGE_EXTENT` -> `VLLM_RBLN_PAGE_LAYOUT` (old name kept as an
   alias). Done ahead of (2) rather than with it, as a pure mechanical commit, so
   (2)'s diff carries only the semantic change. See
   [§ Relation to upstream `kernel_block_size`](#relation-to-upstream-kernel_block_size)
   for the invariant this name inverts.
4. **O(1) `bind()`.** It rewalks every kernel block of a request each step: 0.54 us
   at 4 pages, 20.66 us at 1024. That is 0.08% of TPOT at `max_model_len`, so
   it is scaling, not a present cost. Skip sealed kernel blocks.
5. **Confirm the multi-turn result on MiniMax-M2.5.** The Qwen3-0.6B numbers
   above (TTFT -26.7% against the overlay at the same physical block) should
   grow with layer count, since the copy is per-op dispatch over layers and
   MiniMax has 62 to Qwen's 28. Use the serve path, not the offline API, and a
   conversation long enough to cross an 8192-token kernel block so the overlay's
   full-block sharing also gets exercised.
6. **Short greedy probe on `sub-block caching @ block 8192`.** In the run above it was the
   only config whose text diverged from the other two. Probably benign (moving
   chunked-prefill boundaries), but that is the config recommended for the +18%
   throughput, so it should not go out unchecked.
7. **Decide whether to keep this path**, given ~230 lines in the scheduler and
   worker plus a second KV stack. It is no longer performance-neutral: on
   fan-out it ties, on multi-turn it wins and the gap widens with conversation
   length.

## Migration

1. Agree the invariants and the spec (this doc).
2. Land the addressing layer behind the overlay, without flipping CLI
   semantics. **Done** — `vllm_rbln/v1/core/page_layout/` (geometry, allocator,
   page→kernel block table, binding policy) with unit tests.
3. Flip `--block-size` to the page; derive `kernel_block_size` in the platform;
   update the `block_size` audit sites, tests, llm-d notes, and LMCache paging
   defaults.
4. Enable connector cooperation (drop the mutual-exclusion arbitration).
5. Retire overlay-only APIs (`get_computed_blocks_sub_block`, etc.).
6. Track upstream support for unitary `hash_block_size < block_size`.

## Decision summary

| Decision | Choice |
|---|---|
| Schedule / hash / match unit | page (`--block-size`), `page % chunk == 0` |
| Physical / DMA unit | kernel block (backend-derived, e.g. 4096) |
| Mapping granularity | hybrid: fine map, coarse allocation, merge on partial update |
| New env `VLLM_RBLN_KERNEL_BLOCK_SIZE` | **No** |
| Partial match | partial merge (CoW) into a private kernel block |
| Dedup | full kernel blocks only, by kernel-block hash |
| Refcount granularity | kernel block (I8) |
| Interior page indexing | **Yes** (preserves today's hit rate) |
| Connector interaction | cooperative, not arbitrated |
| GC / compaction | out of MVP; trigger metric recorded |
| MVP specs | full attention only |
