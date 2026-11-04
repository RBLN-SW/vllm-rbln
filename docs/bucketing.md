## Decode Bucketing Overview

RBLN workers keep their decode graphs compiled by batching requests into a
small, reusable set of bucket sizes. Instead of compiling a new graph whenever
`num_reqs` changes, each incoming decode batch is rounded **up** to the closest
supported bucket and padded before the model forward pass. Prefill always uses
`batch_size = 1` and is unaffected by the settings below.

> Bucketing is automatically enabled on the vllm model path
> (`--model-impl vllm`).

Key components:

- `vllm_rbln.v1.worker.bucketing` provides the strategy classes
  (`ExponentialBucketingManager`, `LinearBucketingManager`,
  `ManualBucketingManager`).
- `RBLNModelRunner` builds a manager instance at startup and uses it when
  refreshing metadata, padding decode batches, warming up decode graphs, and
  warming up the RBLN sampler.

## Enabling and Configuring

Bucketing is controlled entirely through `--rbln-*` flags. `LLM(...)` takes the
same options as `additional_config` keys, spelled without the prefix:
`--rbln-decode-batch-bucket-min 2` is
`additional_config={"decode_batch_bucket_min": 2}`.

| Flag | Default | Description |
| --- | --- | --- |
| `--rbln-decode-batch-bucket-strategy` | `exponential` | Chooses the bucketing implementation. Accepts `exponential`, `linear`, or `manual`. |
| `--rbln-decode-batch-bucket-min` | `1` | Smallest allowed decode batch size. Requests smaller than this are still padded to `1`. |
| `--rbln-decode-batch-bucket-step` | `2` | Controls how aggressively bucket sizes shrink. Meaning depends on the strategy (division factor for exponential, subtraction step for linear). Must be > 0, and > 1 for exponential. |
| `--rbln-decode-batch-bucket-limit` | `1` | Maximum number of decode buckets to generate. The default generates a single bucket, `max_num_seqs // pipeline_parallel_size`. Ignored by `manual`. |
| `--rbln-decode-batch-bucket-manual-buckets` | `[]` | Explicit decode bucket sizes, used only by `manual`. Required when the strategy is `manual`. |

`--rbln-decode-batch-bucket-min`, `--rbln-decode-batch-bucket-step`, and
`--rbln-decode-batch-bucket-limit` apply to `exponential` and `linear` only.

## Strategy Details

### Exponential

- Starts at `max_batch_size = max_num_seqs // pipeline_parallel_size`.
- Each additional bucket divides the previous bucket by `step`.
- Stops when `limit` buckets are generated or dropping below `min_batch_size`.
- Use this when you expect wide variance in batch sizes and want denser coverage
  near the top end (e.g. 4096, 2048, 1024, ...).

Example:

```bash
--rbln-decode-batch-bucket-strategy exp
--rbln-decode-batch-bucket-min 2
--rbln-decode-batch-bucket-step 2
--rbln-decode-batch-bucket-limit 4
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
--rbln-decode-batch-bucket-strategy linear
--rbln-decode-batch-bucket-min 2
--rbln-decode-batch-bucket-step 4
--rbln-decode-batch-bucket-limit 32
# vllm server launched with max_num_seqs=16, pipeline_parallel_size=1
```

For the configuration above the decode batch buckets are `[16, 12, 8, 4]`.

### Manual

- Uses the sizes in `decode_batch_bucket_manual_buckets` as the decode buckets,
  sorted in ascending order.
- Every size must be greater than `0`, and the sizes must be unique.
- The largest size must equal `max_num_seqs // pipeline_parallel_size`;
  otherwise the engine raises an error at startup.
- Use this when you know the batch sizes your workload runs at.

Example:

```bash
--additional-config '{"decode_batch_bucket_strategy": "manual", "decode_batch_bucket_manual_buckets": [4, 12, 32]}'
# vllm server launched with max_num_seqs=32, pipeline_parallel_size=1
```

For the configuration above the decode batch buckets are `[4, 12, 32]`.

