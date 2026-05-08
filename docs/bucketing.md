## Decode Bucketing Overview

RBLN workers keep their decode graphs compiled by batching requests into a
small, reusable set of bucket sizes. Instead of compiling a new graph whenever
`num_reqs` changes, each incoming decode batch is rounded **up** to the closest
supported bucket and padded before the model forward pass. Prefill always uses
`batch_size = 1` and is unaffected by the settings below.

> Bucketing is automatically enabled when `VLLM_RBLN_USE_VLLM_MODEL=1`.

Key components:

- `vllm_rbln.v1.worker.bucketing` provides the strategy classes
  (`ExponentialBucketingManager`, `LinearBucketingManager`).
- `RBLNModelRunner` builds a manager instance at startup and uses it when
  refreshing metadata, padding decode batches, warming up decode graphs, and
  warming up the RBLN sampler.

## Enabling and Configuring

Bucketing is controlled entirely through environment variables that the runner
reads from `vllm_rbln.envs` at import time. Set them before launching the
worker (e.g. in your launcher script or shell).

| Variable | Default | Description |
| --- | --- | --- |
| `VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY` | `exponential` | Chooses the bucketing implementation. Accepts `exponential` (or `exp`) or `linear`. |
| `VLLM_RBLN_DECODE_BATCH_BUCKET_MIN` | `1` | Smallest allowed decode batch size. Requests smaller than this are still padded to `1`. |
| `VLLM_RBLN_DECODE_BATCH_BUCKET_STEP` | `2` | Controls how aggressively bucket sizes shrink. Meaning depends on the strategy (division factor for exponential, subtraction step for linear). Must be > 0, and > 1 for exponential. |
| `VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT` | `32` | Maximum number of decode buckets to generate. |

## Strategy Details

### Exponential

- Starts at `max_batch_size = max_num_seqs // pipeline_parallel_size`.
- Each additional bucket divides the previous bucket by `step`.
- Stops when `limit` buckets are generated or dropping below `min_batch_size`.
- Use this when you expect wide variance in batch sizes and want denser coverage
  near the top end (e.g. 4096, 2048, 1024, ...).

Example:

```bash
export VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY=exp
export VLLM_RBLN_DECODE_BATCH_BUCKET_MIN=2
export VLLM_RBLN_DECODE_BATCH_BUCKET_STEP=2
export VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT=4
# vllm server launched with max_num_seqs=32, pipeline_parallel_size=1
```

For the configuration above the decode batch buckets are `[32, 16, 8, 4]`.




### Linear

- Starts at `max_batch_size = max_num_seqs // pipeline_parallel_size`.
- Each additional bucket subtracts `step` from the previous bucket.
- Stops when `limit` buckets are generated or the size dips below `min_batch_size`.
- Use this when you want evenly spaced buckets (e.g. 4096, 3840, 3584, ...).

Example:

```bash
export VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY=linear
export VLLM_RBLN_DECODE_BATCH_BUCKET_MIN=2
export VLLM_RBLN_DECODE_BATCH_BUCKET_STEP=4
export VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT=32
# vllm server launched with max_num_seqs=16, pipeline_parallel_size=1
```

For the configuration above the decode batch buckets are `[16, 12, 8, 4]`.

