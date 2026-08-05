## Dynamic KV Cache Sizing

By default the KV cache is sized from a pre-compile *estimate* of free device
memory. That estimate has no notion of chiplets, so on a 4-chiplet card it can
exceed the per-chiplet budget and the engine allocates a cache that does not fit.

With `VLLM_RBLN_USE_DYNAMIC_KV_CACHE=1` the worker instead marks the KV
`num_blocks` dimension dynamic at compile time, asks each compiled artifact's
`kv_cache_memory_profile()` how many blocks actually fit **per chiplet**,
reallocates the KV tensors at that size and tells the scheduler the new number.
No recompilation happens: the affected dimension is `mark_dynamic`'d.

> Requires `VLLM_RBLN_USE_VLLM_MODEL=1` (only `DynamoRuntime` applies adaptive
> buffer sizes) and a `rebel-compiler` carrying `rebel_compiler#10678`
> (`rebel.kv_cache.max_num_blocks`, `DynamoRuntime.reset_adaptive_buffers`).
> Both are checked at start-up, before anything is compiled -- the flag pair in
> `RblnPlatform.check_and_update_config`, the compiler API in
> `_assert_dynamic_kv_compiler_support`.

## Enabling

```bash
export VLLM_RBLN_USE_VLLM_MODEL=1
export VLLM_RBLN_USE_DYNAMIC_KV_CACHE=1
```

That is the whole public surface. The compile-time block count is derived
(`vllm_rbln.envs.DEFAULT_COMPILE_KV_CACHE_NUM_BLOCKS`, currently 8) and does not
have to be chosen.

Use a **separate `VLLM_CACHE_ROOT` per configuration**. The compile cache hash
does not include dynamism, so a static and a dynamic build of the same model
share one signature and can replay each other's codegen.

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_RBLN_USE_DYNAMIC_KV_CACHE` | `0` | The switch. Off means the pre-compile estimate, exactly as before. |
| `VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS` | derived (`8`) | Debug override of the compile-time block count. `0` opts out of the shrink, and therefore out of the resize. Blank (`VAR=`) counts as unset, so exporting it empty is not a way to change the behaviour. |
| `VLLM_RBLN_DYNAMIC_KV_UNPROFILED_RESERVE_BYTES` | `48 MiB` | Held back from every chiplet's budget for device memory the profile does not describe. |

## What the compile-time block count is

It is **not** a capacity. The traced `num_blocks` only supplies the *hint* of the
`mark_dynamic`'d dimension, and the number of blocks that end up in service comes
from the compiled profile and the per-chiplet budget. Building the artifact's
buffers at the full size is therefore wasted work, and warm-up is safe at a small
size because every dummy request points at block id 0.

The cost of the hint is asymmetric:

- **`tensor_parallel_size=1`**: the outgoing cache is returned before the
  replacement is allocated, so the hint costs nothing.
- **`tensor_parallel_size>=2`**: it is *not* returned, and the process cannot
  observe the free from inside, so those bytes are charged against the
  per-chiplet budget. The final block count loses exactly the hint. Measured on
  MiniMax TP4+EP: 62 MiB per block per chiplet, i.e. 8 blocks = 0.484 GiB and a
  final count of **276** where the same configuration shipped 284 before the
  charge existed — and the binding chiplet comes in at 0.9997x of budget instead
  of 1.015x of it, which is the point.

The linearity is exact, and it is what makes a large value survivable. On the same
MiniMax TP4+EP configuration, `VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS=64` serves at
220 blocks where the derived 8 gives 276: `276 - 220 = 56 = 64 - 8`, one block of
service lost per block of hint. Before the retained cache was charged, that same
value was 4 runs out of 4 where the server started and the *first request* died
with `SYS_EBUSY: Lack of device memory` — so the charge turns an over-large hint
from a runtime failure into a predictable price.

A hint large enough to drive a chiplet's budget to zero is refused at start-up
with `the retained compile-time KV cache leaves no budget on ...`. That path has
not been observed on hardware; 64 blocks on the configuration above does not come
close to it.

## When the derived value is wrong

**The estimate is at or below the hint.** vLLM only guarantees
`ceil(max_model_len / block_size)` blocks, so an estimate of 4 blocks
(`--block-size 8192 --max-model-len 32768`) is legal. There is nothing to shrink
there, and cancelling the shrink also cancels the resize — which would leave the
run on the pre-compile estimate this feature exists to replace, without anybody
having asked for that. The worker refuses to start instead, and names the
override. Raise the estimate (`--gpu-memory-utilization`, a smaller
`--block-size`) or set the override below the estimate.

Values **below 8** are unmeasured. `2` is the smallest that is expected to keep
the dimension dynamic; `1` risks dynamo specializing a size-1 dimension, which
produces a static artifact. The profile query then finds no dynamic-shape variable
and the run falls back to the estimate — loudly: a warning per runtime naming the
static artifact, an error that no profile could be collected, and a third warning
when the cache is restored to the estimate.

**Compile and warm-up are skipped** (`--enforce-eager`,
`VLLM_RBLN_COMPILE_MODEL=0`, `VLLM_RBLN_ENABLE_WARM_UP=0`). Nothing compiles, so no
artifact reports a memory profile and there is nothing to resize from. The shrink is
skipped with a warning naming which switch did it, and the run serves the
pre-compile estimate exactly as it does with the feature off. This is the one case
where the derived default cannot work at all rather than working differently.

**You want the pre-compile estimate on purpose** (comparing against the static
path, measuring without the resize): set
`VLLM_RBLN_COMPILE_KV_CACHE_NUM_BLOCKS=0`. The shrink and the resize are both
skipped, with a warning that says so. Note that `mark_dynamic` is still applied
and still logged in this mode, so that log line is never evidence that the block
count came from the device.

## Unsupported combinations

Refused at start-up, with the flag on:

- **MLA models** — `num_blocks` is dimension 0 of MLA's KV shape rather than
  dimension 1, and MLA dispatches `paged_flash_causal_mla_naive_*`, which the
  compiler's dynamic-input validator does not admit. Use `VLLM_MLA_DISABLE=1` or
  run with the flag off.
- **Speculative decoding** — the drafter shares the runner's `runtime_holder`,
  so its profile is merged next to the target's, and the merge key
  `(node, chiplet, base_bytes, bytes_per_block)` cannot tell an artifact
  re-reporting a shared KV tensor from one owning a separate tensor of the same
  size. The joint per-block cost comes out either too low (a cache that does not
  fit) or several times too high.
- **KV transfer connectors** — the connector registers the KV cache's physical
  views during warm-up, and the resize invalidates them.
- **Sliding-window / cross-layer-shared KV / KV base deduplication** — the
  compiler requires every marked input to reach exactly one
  `paged_flash_causal_attention_naive_{prefill,decode}` call.

Known gaps, measured and not yet fixed:

- **DP+EP is not charged.** The retained-cache charge is conditional on
  `tensor_parallel_size > 1`, but MiniMax DP4+EP runs with
  `tensor_parallel_size=1` and sysfs bracketed the outgoing cache as still
  resident on all four cards (1.0138x per chiplet). Which of TP/DP/EP decides
  whether the cache comes back is not in the measurements — both configurations
  that do return it, PP4 and Qwen TP1, are also TP=1 — so the predicate is left
  as measured rather than guessed at.
- **The block count is not perfectly deterministic.** On Qwen at
  `--gpu-memory-utilization 0.9`, 1 run in 3 measured an extra 1490 MiB per chiplet
  and landed at 1.043x of the requested budget. Whether that is a retained arena or
  the measurement artifact below is not settled.

## Measuring per-chiplet residency

The figures above come from summing the runtime's allocation log lines per
`(buffer id, address)`. Frees are not logged, so the sum is an upper bound, and on
a card other tenants are churning it overstates: the outgoing compile-time cache
stops landing at addresses the replacement reuses, so it is counted twice.
Measured, that inflation is exactly the cache's own size — 1152 MiB on Qwen
(8 blocks x 36 growth regions x 4 MiB), which moved one configuration's binding
chiplet from 31.41 GiB to 32.53 GiB with no change in code, block count or
compiler. **Take per-chiplet figures on idle cards**, and treat a residency
argument on a shared card as an upper bound rather than a measurement.
