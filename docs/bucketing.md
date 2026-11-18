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
  (`ExponentialBucketingManager`, `LinearBucketingManager`).
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
| `--rbln-decode-batch-bucket-strategy` | `exponential` | Chooses the bucketing implementation. Accepts `exponential` (or `exp`) or `linear`. |
| `--rbln-decode-batch-bucket-min` | `1` | Smallest allowed decode batch size. Requests smaller than this are still padded to `1`. |
| `--rbln-decode-batch-bucket-step` | `2` | Controls how aggressively bucket sizes shrink. Meaning depends on the strategy (division factor for exponential, subtraction step for linear). Must be > 0, and > 1 for exponential. |
| `--rbln-decode-batch-bucket-limit` | `32` | Maximum number of decode buckets to generate. |

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


## Prefill Chunk Bucketing

`--rbln-prefill-chunk-size` sets the lengths of the prefill graphs. Each length
compiles one `[1, length]` graph. A prefill step pads its chunk to the shortest
length that fits, so a short prompt or the last chunk of a long prompt does not
pad to the full length.

Each additional length adds a prefill graph to compile and keep on the device,
leaving less memory for the KV cache.

The longest length is the most tokens the scheduler gives one request per step.

| Flag | Default | Description |
| --- | --- | --- |
| `--rbln-prefill-chunk-size` | `max_num_batched_tokens` | Prefill graph lengths. Every length must be unique, within `(0, max_num_batched_tokens]`, and no longer than `--max-num-scheduled-tokens` when that is set. |

In `additional_config` it takes a list or a single integer, for example
`{"prefill_chunk_size": 512}`.

Example:

```bash
--max-num-batched-tokens 1024
--rbln-prefill-chunk-size 256 512 1024
```

This compiles prefill graphs of 256, 512 and 1024 tokens. A 1200-token prompt
runs a 1024-token chunk on the 1024 graph, then its last 176 tokens on the 256
graph, padded to 256.
