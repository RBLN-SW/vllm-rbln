# PR #1172 Summary

## Overview

This PR brings the Optimum model runner in line with vLLM 0.30.0 and explicitly rejects request features that are not supported on the Optimum path.

The implementation was audited against `GPUModelRunner` in vLLM 0.26.0 and v0.30.0. Two copied methods had diverged from upstream, while three code paths referenced methods that were not implemented by the Optimum runner. This change restores upstream parity where possible and turns the remaining unsupported paths into explicit failures.

## Key Changes

### 1. Align pooling behavior with vLLM 0.30.0

- Updated `_pool()` to use `pooling_metadata.get_pooling_cursor().get_finished_mask()`.
- Removed the previous completion check based on `seq_len == prompt_len`.
- Return copies of `req_ids` and `req_id_to_index` to match the behavior expected by vLLM 0.30.0.

### 2. Fix EC producer handling

- Run the EC producer branch before the idle early-return path.
- Execute the multimodal encoder over the entire prompt for new requests.
- Return `make_empty_encoder_model_runner_output(...)` wrapped with `ModelRunnerOutput.with_ec_conn_output(...)`.
- Forward connector metadata such as `finished_sending` and `ec_connector_worker_meta` to the scheduler.
- Let the scheduler complete encoder-only requests through `is_mm_encoder_only`.
- Remove the custom EOS-synthesizing `_make_producer_output()` implementation.

### 3. Explicitly reject unsupported features

The following features are not supported on the Optimum model path and now raise `NotImplementedError` instead of failing indirectly through missing methods or invalid state:

#### `prompt_logprobs`

The compiled prefill path returns logits only for the final position, so prompt-position log probabilities cannot be generated. The following legacy logic was removed:

- `num_prompt_logprobs`
- `_get_prompt_logprobs_dict()`

#### Streaming input

The streaming-input path previously called `_update_streaming_request()`, which is implemented only by `GPUModelRunner`. It now raises an explicit `NotImplementedError`.

#### `VLLM_COMPUTE_NANS_IN_LOGITS`

The Optimum runner now rejects this configuration during initialization because `_get_nans_in_logits()` is available only on the vLLM path runner.

## Scope

The following areas were reviewed and intentionally left unchanged:

- The new `_update_states` hooks, whose default behavior matches the existing inline implementation
- `_execute_mm_encoder` cache indirection
- `NewRequestData` multimodal-data stripping
- The additional `LogprobsTensors` field, which is already ignored by `postprocess_sampler_output`
- Async scheduling
- Speculative decoding
- Pipeline parallelism
- CUDA graphs
- Mamba
- KV connectors

## Testing

Run the following command:

```bash
uv run --no-sync pytest tests/optimum/v1/worker tests/optimum/v1/distributed
```

Expected results:

- `41 passed` in the worker tests
- `4 passed` in the distributed tests

The new tests cover:

- EC producer output behavior
- Encoder execution across the full prompt
- Rejection of streaming input
- Rejection of `prompt_logprobs`
- Rejection of `VLLM_COMPUTE_NANS_IN_LOGITS`

When combined with `fix/optimum-0.30.0-v2`, the worker tests pass with `42 passed`, exercising the updated `_pool()` path with a tensor-based `seq_lens` value.

## Related Context

This PR is part of the vLLM 0.30.0 update on `dev-0.30.0`.

The `_pool()` change depends on the `seq_lens` tensor fix in `fix/optimum-0.30.0-v2`. The two branches conflict only on one import line in `tests/optimum/v1/worker/test_optimum_model_runner.py`; both changes should be preserved.

## Notes

- The `prompt_logprobs` refusal occurs inside the runner, so it stops EngineCore rather than returning a `400` response to the client.
- Rejecting this feature during request validation would require additional changes to `Platform.validate_request` in `platform/vllm_impl.py`; that work is intentionally left for a separate PR.
- The EC producer path was not verified on a real NPU instance.

## Change Type

- Bug fix
