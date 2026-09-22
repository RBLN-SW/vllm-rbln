## Dynamic KV Cache Sizing Overview

The KV cache is sized from the compiled artifact's placement and a per-chiplet
memory snapshot. The alternative, which `VLLM_RBLN_USE_DYNAMIC_KV_CACHE=0` goes
back to, is a pre-compile estimate of free device memory: a whole-card figure
with no notion of chiplets, so on a quad-chiplet card it can exceed the
per-chiplet budget and the engine allocates a cache that does not fit.

With the feature on and a device present, the pre-compile estimate itself is taken
per chiplet: before the compile, the worker snapshots every chiplet's
`(total, used)` (the same snapshot the final sizing uses, below) and replaces the
whole-card capacity with `chiplets * min_c(total_c * gpu_memory_utilization -
used_c)`; the model-size and buffer terms are subtracted as before, because the
weights reach the device only when the compiled programs load. This estimate is
still only the starting point: the count that serves comes from the placement
after warm-up.

The worker marks the KV cache's
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
  slope on top of the target's. The fit counts the shards at what the runtime's
  caching allocator reserves for them, replayed the way it allocates: requests
  are rounded to 4 KiB and served best-fit from free blocks, a miss maps a
  2 MiB segment (request up to 1 MiB), a 20 MiB segment (up to 10 MiB) or the
  request rounded up to 2 MiB, and a split block's remainder serves later
  requests of the same pool. A model with many mid-size shards therefore packs
  several of them into one segment, and the rounding costs at most a few blocks
  below the linear answer.
- **Base** -- how many bytes are already spoken for on each chiplet. A per-chiplet
  memory snapshot is taken after the compile-time cache is released, so what
  the runtime does not hand back is measured as base rather than assumed away:
  `torch.rbln.mem_get_info_per_chiplet()` (the driver's view, every process
  included) when the UMD/KMD provide it, otherwise this process's caching
  allocator (`torch.rbln.memory_stats_per_chiplet()`) plus a fixed reserve for the
  runtime's direct allocations and the other tenants' usage sampled at start-up.

The count starts at the tightest chiplet's
`floor((total * gpu_memory_utilization - base) / per_block)` and `max_num_blocks`
scans down from there: the allocator rounds each shard up to a segment size, so
the linear figure is an upper bound, not the answer (see Growth below). When the
scheduler will run sub-block prefix caching, `base` also carries a fixed
per-chiplet reserve (`DYNAMIC_KV_COPY_STREAM_RESERVE_BYTES`) for the command
streams its partial-block copies upload once requests flow; it is independent
of `gpu_memory_utilization`, which is the only other slack. The worker then
reallocates the KV tensors at that size and re-announces the count to the
scheduler. No recompilation happens, because the affected dimension is already
dynamic.

### KV transfer connectors

A connector registers the KV cache's physical views, and a resize replaces those
views right after warm-up. So in the one mode that reallocates
(`DynamicKvSizer.defers_kv_registration`, i.e. ACTIVE) the worker skips the
warm-up registration and runs the whole of it from `apply_dynamic_kv_num_blocks`,
after the resize has allocated: `register_kv_caches_with_connector` rebuilds the
mapping from the tensors the runner is holding. The RBLN NIXL connectors take
their block count from `cache_config.num_gpu_blocks` at that point instead of the
pre-compile estimate copied at construction; `RBLNLMCacheConnectorV1` binds the
actual post-resize tensors and does not cache a block count. Nothing is registered
early, so nothing has to be unregistered. Every other mode keeps the start-up
order.

> The dynamic path needs `--model-impl vllm` and `VLLM_RBLN_USE_DEVICE_TENSOR=1`,
> and turns itself off without them. Only `DynamoRuntime` applies adaptive buffer
> sizes; the other runtimes ignore them silently.

Key components:

- `vllm_rbln.v1.worker.kv_placement` evaluates the placements into per-chiplet
  growth, parses the memory snapshots, and solves for the block count.
- `vllm_rbln.v1.worker.dynamic_kv_sizer.DynamicKvSizer` is the state machine
  the worker delegates to: it feeds the per-chiplet snapshot into the pre-compile
  estimate, shrinks the cache before the compile, captures the programs warm-up
  builds, takes the snapshot again, and reallocates the KV tensors at the answer.
  `RBLNWorker` keeps only the hooks that call it (config initialisation,
  warm-up, and the two RPC targets the engine patch invokes).
- `vllm_rbln.patches.dynamic_kv` hands the new block count to the scheduler's
  block pool, which was otherwise sized from the pre-compile estimate.

## Enabling and Configuring

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_RBLN_USE_DYNAMIC_KV_CACHE` | `1` | Size the KV cache from the compiled placement and the device. `0` goes back to the pre-compile estimate. Unset, a configuration the path cannot size turns it off on its own; an explicit `1` refuses such a configuration at start-up. |

```bash
export VLLM_RBLN_USE_DEVICE_TENSOR=1
export VLLM_CACHE_ROOT=<a fresh directory>
```

That is the whole public surface. The number of blocks the cache is shrunk to for
the compile is a module constant
(`vllm_rbln.v1.worker.dynamic_kv_sizer.COMPILE_KV_CACHE_NUM_BLOCKS`) and cannot be set
from the environment. It is a trace hint for the dynamic dimension, not a
capacity: the count that ends up in service comes from the placement and the
snapshot.

The mega-cache bundle key includes the resolved dynamic-KV decision, not only
the environment flag. A configuration that turns the feature off therefore
cannot replay a dynamic artifact, or vice versa.

When the pre-compile estimate falls short of one max-length request, it is
raised to exactly that with a warning instead of letting vllm refuse the compile:
under the shrink the estimate is only the placeholder the model is compiled with,
and the count the device can hold is sized after warm-up, where a pool below one
request is refused with the same message. Only under the shrink -- an override
and a skipped compile both serve this estimate, so raising it there would change
the pool rather than report on it.

## Where It Turns Itself Off

A configuration the mechanism cannot size is not a refusal while the variable is
unset: refusing would stop a run that `VLLM_RBLN_USE_DYNAMIC_KV_CACHE=0` would
have served. The feature logs one warning and the run continues on the
pre-compile estimate, with no KV dimension marked dynamic: `mark_dynamic` follows
this decision, not the flag. An explicit `VLLM_RBLN_USE_DYNAMIC_KV_CACHE=1` is a
request, and start-up refuses it with the same reason.
`dynamic_kv_unsupported_reason` in `v1/worker/utils.py` holds the whole list, and
both the engine patch and the worker read it. The optimum-rbln path is not on it:
it installs neither the engine patch nor a worker that carries a sizer, so the
feature is absent there rather than disabled.

| Configuration | Why |
| --- | --- |
| `VLLM_RBLN_USE_DEVICE_TENSOR=0` | The artifact carries no dynamic KV dimension. |
| `RBLN_USE_CUSTOM_KERNEL=1` | The `rbln_triton_ops` kernels go through the compiler's triton converter, so the KV input never reaches a whitelisted `paged_*` custom op. |
| Flash causal attention disabled | This dispatches to an attention kernel that does not accept a dynamic KV input. |
| A DFlash drafter (`--speculative-config '{"method": "dflash", ...}'`) | The drafter is non-causal on RBLN, and its attention kernel does not accept a dynamic KV input. `use_non_causal` lives on the draft config only, so the method is the signal. |
| `block_size == max_model_len` | This selects the normal-attention kernels, which do not accept a dynamic KV input. |
| A KV transfer connector other than the RBLN NIXL ones (`RblnNixlConnector`, `RblnNixlPullConnector`, `RblnNixlPushConnector`) or `RBLNLMCacheConnectorV1` | The worker registers with the connector only once the resize has allocated (see "KV transfer connectors" above). That ordering is connector-agnostic, so a connector outside `DYNAMIC_KV_SUPPORTED_CONNECTORS` in `v1/worker/utils.py` is untried rather than known broken, and is kept off until it has been. |

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
Three more cases warn and continue on the pre-compile estimate, because each is
an explicit request from the caller:

- Compile and warm-up are skipped (`--enforce-eager`, `--no-rbln-compile-model`,
  `VLLM_RBLN_ENABLE_WARM_UP=0`). Nothing compiles, so no program carries a placement.
- `--num-gpu-blocks-override` is set. The override pins the count and wins.
- `RBLN_DUMMY_DEVICE` is set (a compile-only run). There is no device to
  measure, so the count stays at the estimate for the scheduler, and the KV
  cache itself stays at the compile hint rather than being restored: the dummy
  UMD still enforces its memory limit, and nothing runs after warm-up.

In each case `mark_dynamic` is still applied and still logged, so that log line
is not evidence that the block count came from the device.

## Known Limitations

- **Whatever the runtime keeps of the compile-time cache costs blocks.** The
  cache is released before the snapshot, so a runtime that does not hand it
  back (observed on a DP + EP run, where the count dropped by exactly the
  cache's size) is measured, not guessed.
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
