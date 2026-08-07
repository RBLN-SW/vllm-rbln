## Dynamic KV Cache Sizing Overview

By default the KV cache is sized from a pre-compile estimate of free device
memory. That estimate is a whole-card figure and has no notion of chiplets, so on
a quad-chiplet card it can exceed the per-chiplet budget and the engine allocates
a cache that does not fit.

With `VLLM_RBLN_USE_DYNAMIC_KV_CACHE=1` the worker marks the KV cache's
`num_blocks` dimension dynamic at compile time, asks each compiled artifact how
many blocks actually fit per chiplet, reallocates the KV tensors at that size and
re-announces the count to the scheduler. No recompilation happens, because the
affected dimension is already dynamic.

> The dynamic path requires `VLLM_RBLN_USE_VLLM_MODEL=1` and
> `VLLM_RBLN_USE_DEVICE_TENSOR=1`. Only `DynamoRuntime` applies adaptive buffer
> sizes; the other runtimes ignore them silently.

Key components:

- `vllm_rbln.v1.worker.kv_profile` merges the per-artifact memory profiles that
  `rebel.kv_cache.max_num_blocks` consumes.
- `RBLNWorker` shrinks the cache before the compile, queries the profiles after
  warm-up, and reallocates the KV tensors at the answer.
- `vllm_rbln.patches.dynamic_kv` hands the new block count to the scheduler's
  block pool, which was otherwise sized from the pre-compile estimate.

## Enabling and Configuring

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_RBLN_USE_DYNAMIC_KV_CACHE` | `0` | Size the KV cache from the compiled artifact instead of the pre-compile estimate. Off means the estimate, exactly as before. |

```bash
export VLLM_RBLN_USE_VLLM_MODEL=1
export VLLM_RBLN_USE_DEVICE_TENSOR=1
export VLLM_RBLN_USE_DYNAMIC_KV_CACHE=1
export VLLM_CACHE_ROOT=<a fresh directory>
```

That is the whole public surface. The number of blocks the cache is shrunk to for
the compile is a module constant
(`vllm_rbln.v1.worker.rbln_worker.COMPILE_KV_CACHE_NUM_BLOCKS`) and cannot be set
from the environment. It is a trace hint for the dynamic dimension,
not a capacity: the count that ends up in service comes from the compiled profile.

Use a separate `VLLM_CACHE_ROOT` per configuration. The compile cache hash does
not include dynamism, so a static and a dynamic build of the same model share one
signature and can replay each other's codegen.

## Unsupported Configurations

The following are rejected at start-up when the flag is on, and are unaffected
when it is off. Run with `VLLM_RBLN_USE_DYNAMIC_KV_CACHE=0` to use them.

| Configuration | Why |
| --- | --- |
| MLA models | `num_blocks` is dimension 0 of MLA's KV shape rather than dimension 1, and MLA dispatches `paged_flash_causal_mla_naive_*`, which the compiler's dynamic-input validator does not admit. `VLLM_MLA_DISABLE=1` also works. |
| Speculative decoding | The drafter's profile is merged next to the target's, and the merge cannot tell an artifact re-reporting a shared KV tensor from one owning its own. |
| KV transfer connectors | The connector registers the KV cache's physical views during warm-up, and the reallocation invalidates them. |
| Sliding-window attention, cross-layer KV sharing, KV base deduplication | The compiler requires every dynamic KV input to reach exactly one `paged_flash_causal_attention_naive_{prefill,decode}` call. |

## When Start-up Refuses

The dynamic path fails loudly rather than falling back, because a silent fallback
would serve from the pre-compile estimate this feature exists to replace.

- **The estimate is already at or below the compile hint.** The estimate is free
  device memory divided by the cost of one block, and that cost scales with
  `--block-size`, so a large block size can put it below the hint. There is nothing to
  shrink, and cancelling the shrink cancels the reallocation too. Raise the
  estimate with a smaller `--block-size`, a higher `--gpu-memory-utilization`, or
  more devices.
- **The compiler does not carry the dynamic KV API.** `rebel.kv_cache.max_num_blocks`
  and `DynamoRuntime.reset_adaptive_buffers` are both checked before anything is
  compiled. Upgrade `rebel-compiler`.
- **No artifact reported a memory profile.** Usually a `VLLM_CACHE_ROOT` replaying
  a static build. Use a fresh directory.
- **The per-chiplet budget cannot hold the base usage.** Raise
  `--gpu-memory-utilization`, or lower `--block-size`.

Two cases warn and continue on the pre-compile estimate instead, because both are
an explicit request from the caller:

- Compile and warm-up are skipped (`--enforce-eager`, `VLLM_RBLN_COMPILE_MODEL=0`,
  `VLLM_RBLN_ENABLE_WARM_UP=0`). Nothing compiles, so no artifact reports a profile.
- `--num-gpu-blocks-override` is set. The override pins the count and wins.

In both cases `mark_dynamic` is still applied and still logged, so that log line
is not evidence that the block count came from the device.

## Known Limitations

- **`tensor_parallel_size >= 2` costs blocks.** The compile-time cache is not
  returned to the driver on TP >= 2, and the process cannot observe that it was
  not, so those bytes are charged against the per-chiplet budget. The final count
  loses exactly the compile hint. TP = 1 returns the cache and pays nothing.
- **Data parallel with expert parallel is not charged.** The charge above is
  conditional on `tensor_parallel_size > 1`, but a DP + EP run keeps the outgoing
  cache resident at `tensor_parallel_size = 1`. The budget can be exceeded there.
- **The block count is not perfectly deterministic.** Repeated runs of the same
  configuration occasionally retain an extra arena per chiplet and land above the
  requested budget.
