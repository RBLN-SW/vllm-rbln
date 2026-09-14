## Dynamic KV Cache Sizing Overview

By default the KV cache is sized from a pre-compile estimate of free device
memory. That estimate is a whole-card figure and has no notion of chiplets, so on
a quad-chiplet card it can exceed the per-chiplet budget and the engine allocates
a cache that does not fit.

With the flag on and a device present, the pre-compile estimate itself is taken
per chiplet: before the compile, the worker snapshots every chiplet's
`(total, used)` (the same snapshot the final sizing uses, below) and replaces the
whole-card capacity with `chiplets * min_c(total_c * gpu_memory_utilization -
used_c)`; the model-size and buffer terms are subtracted as before, because the
weights reach the device only when the compiled programs load. This estimate is
still only the starting point: the count that serves comes from the placement
after warm-up.

With `VLLM_RBLN_USE_DYNAMIC_KV_CACHE=1` the worker marks the KV cache's
`num_blocks` dimension dynamic at compile time, compiles against a small
compile-time cache, and after warm-up sizes the real cache from two measurements:

- **Growth** -- how many bytes one more block costs on each chiplet. Every
  compiled program reports the device placement of its dynamic-shape inputs
  (`torch.rbln.capture_programs()` -> `CompiledProgram.input_specs[i].physical_placement`):
  the physical shape, dtype and one shard per `(node, chiplet)`, with the dynamic
  dim left symbolic. The KV caches are the only dynamic inputs, so summing the
  shards' extents gives the per-chiplet slope exactly. A layer's dynamic dim is
  its *kernel* block count -- a sliding-window layer splits each manager block
  into `block_size / sliding_window` kernel blocks -- so each input's symbol is
  scaled by its compiled extent over the compile hint. Programs that bind a
  different set of KV tensors (a speculative drafter's) contribute their own
  slope on top of the target's. The fit counts every shard at the size the
  runtime's caching allocator reserves for it (a request up to 1 MiB takes a
  2 MiB block, up to 10 MiB a 20 MiB block, larger ones round up to 2 MiB), so
  a shard size the rounding inflates costs the few blocks that takes below the
  linear answer.
- **Base** -- how many bytes are already spoken for on each chiplet. A per-chiplet
  memory snapshot is taken after the compile-time cache is released, so what
  the runtime does not hand back is measured as base rather than assumed away:
  `torch.rbln.mem_get_info_per_chiplet()` (the driver's view, every process
  included) when the UMD/KMD provide it, otherwise this process's caching
  allocator (`torch.rbln.memory_stats_per_chiplet()`) plus a fixed reserve for the
  runtime's direct allocations and the other tenants' usage sampled at start-up.

The count is the tightest chiplet's
`floor((total * gpu_memory_utilization - base) / per_block)`. When the
scheduler will run sub-block prefix caching, `base` also carries a fixed
per-chiplet reserve (`DYNAMIC_KV_COPY_STREAM_RESERVE_BYTES`) for the command
streams its partial-block copies upload once requests flow; it is independent
of `gpu_memory_utilization`, which is the only other slack. The worker then
reallocates the KV tensors at that size and re-announces the count to the
scheduler. No recompilation happens, because the affected dimension is already
dynamic.

> The dynamic path requires `VLLM_RBLN_USE_VLLM_MODEL=1` and
> `VLLM_RBLN_USE_DEVICE_TENSOR=1`. Only `DynamoRuntime` applies adaptive buffer
> sizes; the other runtimes ignore them silently.

Key components:

- `vllm_rbln.v1.worker.kv_placement` evaluates the placements into per-chiplet
  growth, parses the memory snapshots, and solves for the block count.
- `RBLNWorker` feeds the per-chiplet snapshot into the pre-compile estimate,
  shrinks the cache before the compile, captures the programs warm-up builds,
  takes the snapshot again, and reallocates the KV tensors at the answer.
- `vllm_rbln.patches.dynamic_kv` hands the new block count to the scheduler's
  block pool, which was otherwise sized from the pre-compile estimate.

## Enabling and Configuring

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_RBLN_USE_DYNAMIC_KV_CACHE` | `0` | Size the KV cache from the compiled placement and the device instead of the pre-compile estimate. Off means the estimate, exactly as before. |
| `VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN` | `0` | Compute the count and log how the count vllm sized fits each chiplet, but resize nothing. Implies the flag above, so it is the only variable a trial run needs. |

```bash
export VLLM_RBLN_USE_VLLM_MODEL=1
export VLLM_RBLN_USE_DEVICE_TENSOR=1
export VLLM_RBLN_USE_DYNAMIC_KV_CACHE=1
export VLLM_CACHE_ROOT=<a fresh directory>
```

That is the whole public surface. The number of blocks the cache is shrunk to for
the compile is a module constant
(`vllm_rbln.v1.worker.rbln_worker.COMPILE_KV_CACHE_NUM_BLOCKS`) and cannot be set
from the environment. It is a trace hint for the dynamic dimension, not a
capacity: the count that ends up in service comes from the placement and the
snapshot.

Use a separate `VLLM_CACHE_ROOT` per configuration. The compile cache hash does
not include dynamism, so a static and a dynamic build of the same model share one
signature and can replay each other's codegen.

### Dry run

`VLLM_RBLN_DYNAMIC_KV_CACHE_DRY_RUN=1` (on its own; it implies the flag) keeps every count
as it is today -- the pre-compile estimate or `--num-gpu-blocks-override`, whichever
vllm would have used -- and only reports. The KV dim is still marked dynamic,
since that is what makes the compiled programs carry a placement, but the cache is
not shrunk for the compile and nothing is reallocated after warm-up. Two lines
carry the result:

- at the pre-compile estimate, what the per-chiplet snapshot would have put the
  estimate at, next to the whole-card formula that is kept;
- after warm-up, `dry run: vllm sized N blocks, this feature would set n (+/-d)`
  with, per `(node, chiplet)`, the bytes the current `N` blocks take, the non-KV
  base, the budget, the headroom left in bytes and blocks, and what `n` would
  do: the predicted `used` at `n`, `budget_left` against
  `total * gpu_memory_utilization` and `total_left` against the chiplet's
  physical DRAM. A negative headroom means the current count already exceeds
  the budget on that chiplet. Nothing is allocated at `n`; the real mode's fit
  check line is what validates the prediction.

Unsupported configurations are refused in a dry run too, since the refusals
guard the dynamic compile itself. A sizing failure after warm-up (no placement,
no fit) is logged as a warning instead of raised.

## Unsupported Configurations

The following are rejected at start-up when the flag is on, and are unaffected
when it is off. Run with `VLLM_RBLN_USE_DYNAMIC_KV_CACHE=0` to use them.

| Configuration | Why |
| --- | --- |
| KV transfer connectors | The connector registers the KV cache's physical views during warm-up, and the reallocation invalidates them. |
| DSA indexer cache on a compiler without rebellions-sw/rebel_compiler#13745 | The indexer ops (`sparse_attn_deepseek_indexer*`) are not in the compiler's dynamic-input whitelist there, so the compile refuses with `flows into op 'sparse_attn_deepseek_indexer'`. MLA itself needs nothing extra: the dynamic axis is read off the backend's KV shape, dim 0 for MLA. |
| Cross-layer KV sharing | The compiler admits a dynamic KV input through view ops into several paged naive attention calls (`paged_flash_causal_attention_naive_*`, `paged_sliding_window_attention_naive_*`), which is how a deduped base shared by a full and a sliding-window layer (gpt-oss) compiles; the same view feeding two layers' attention calls is not admitted. |

## When Start-up Refuses

The dynamic path fails loudly rather than falling back, because a silent fallback
would serve from the pre-compile estimate this feature exists to replace.

- **The estimate is already at or below the compile hint.** The estimate is free
  device memory divided by the cost of one block, and that cost scales with
  `--block-size`, so a large block size can put it below the hint. There is
  nothing to shrink, and cancelling the shrink cancels the reallocation too.
  Raise the estimate with a smaller `--block-size`, a higher
  `--gpu-memory-utilization`, or more devices.
- **No compiled program carries a dynamic-shape KV input.** Usually a
  `VLLM_CACHE_ROOT` replaying a static build. Use a fresh directory.
- **Compiled programs disagree on the KV placement.** Two programs bind KV
  inputs of the same shapes and dtypes with different shard layouts, i.e. the
  same tensors placed two ways; the runtime would re-place the cache on every
  switch.
- **The count cannot hold one request or one decode batch.** After the resize
  the pool must hold `1 + max(one request, max_num_seqs decode steps)` blocks,
  the 1 being the null block, summed over the KV cache groups that share the
  pool; a sliding-window group counts vLLM's per-request admission blocks (the
  window plus one unaligned block) rather than the whole context.
- **No KV block fits.** On some chiplet the non-KV base already exceeds
  `total * gpu_memory_utilization`. Raise `--gpu-memory-utilization`, or give the
  model more devices.

Two cases warn and continue on the pre-compile estimate instead, because both are
an explicit request from the caller:

- Compile and warm-up are skipped (`--enforce-eager`, `VLLM_RBLN_COMPILE_MODEL=0`,
  `VLLM_RBLN_ENABLE_WARM_UP=0`). Nothing compiles, so no program carries a placement.
- `--num-gpu-blocks-override` is set. The override pins the count and wins.
- `RBLN_DUMMY_DEVICE` is set (a compile-only run). There is no device to
  measure, so the count stays at the estimate for the scheduler, and the KV
  cache itself stays at the compile hint rather than being restored: the dummy
  UMD still enforces its memory limit, and nothing runs after warm-up.

In both cases `mark_dynamic` is still applied and still logged, so that log line
is not evidence that the block count came from the device.

## Known Limitations

- **Whatever the runtime keeps of the compile-time cache costs blocks.** The
  cache is released before the snapshot, so a runtime that does not hand it
  back (observed on a DP + EP run, where the count dropped by exactly the
  cache's size) is measured, not guessed. A dry run cannot release, so it
  reports the count under both readings.
- **The allocator snapshot is an approximation.** It sees only this process and
  only the caching allocator; the runtime's direct allocations are covered by a
  fixed reserve and other tenants by a start-up sample spread evenly over the
  chiplets. Prefer a stack that answers `mem_get_info_per_chiplet()`.
- **Allocator packing is not modelled.** The growth counts each shard at its
  own reserved size; blocks the allocator keeps cached for later reuse show up
  in the snapshot's `used`, not in the growth, and the fit check line after the
  reallocation is where a mismatch shows.
- **The block count is not perfectly deterministic.** Repeated runs of the same
  configuration occasionally retain an extra arena per chiplet and land above the
  requested budget.
