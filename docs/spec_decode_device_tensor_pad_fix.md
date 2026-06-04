# Spec-decode + device-tensor sampling crash — analysis & fix plan

Tracking: [vllm-rbln-internal#85](https://github.com/rebellions-sw/vllm-rbln-internal/issues/85)
("spec decode + device tensor 호환"), originally surfaced in the
[PR #548 review](https://github.com/RBLN-SW/vllm-rbln/pull/548#discussion_r3179585432).

## Symptom

In production (RBLN NPU, `VLLM_RBLN_USE_DEVICE_TENSOR=1`) with ngram speculative
decoding enabled, the engine crashes with a tensor shape/broadcast mismatch
inside the sampler as soon as **two or more requests reach the sampling step in
the same batch**. A single concurrent request runs fine, which is why the bug
hid until a request burst hit `--max-num-seqs > 1`.

The crash throws at `vllm_rbln/v1/sample/rbln_sampler.py:421`:

```python
return logits.div(temperature.unsqueeze(dim=1))
```

with shapes like `[3, vocab] / [6, 1]` — but that line is only the victim. The
real defect is one layer up, in the model runner.

## Root cause

Two pieces interact badly:

1. **`RBLNModelRunner._sample()` over-pads `SamplingMetadata` for the
   spec-decode path** — `vllm_rbln/v1/worker/rbln_model_runner.py:2084-2091`:

   ```python
   if (envs.VLLM_RBLN_USE_DEVICE_TENSOR
       and logits is not None
       and logits.shape[0] > self.input_batch.num_reqs):
       sampling_metadata = self._pad_sampling_metadata(
           sampling_metadata, logits.shape[0])
   ```

   `_pad_sampling_metadata` was designed for the *non*-spec path, where in
   device-tensor mode `logits` is kept at the padded batch-bucket size and the
   per-request metadata tensors (`temperature`, `top_p`, `top_k`, …) must be
   padded to match.

   But in **speculative decoding** the target logits are shaped
   `[num_tokens + batch_size, vocab]` (per the rejection-sampler contract,
   `vllm_rbln/v1/sample/rbln_rejection_sampler.py:51,65`), where `num_tokens`
   is the total number of draft tokens across the batch. That length is
   *always* `> num_reqs`, so the guard fires and `temperature` (and friends)
   get rebuilt at length `num_tokens + batch_size` instead of staying at
   `num_reqs`.

2. **`RBLNRejectionSampler.forward()` reuses that padded metadata for the
   bonus-token sub-batch** — `vllm_rbln/v1/sample/rbln_rejection_sampler.py:87-100`:

   ```python
   bonus_logits = logits[bonus_logits_indices]   # [batch_size, vocab]
   bonus_sampler_output = self.sampler(
       logits=bonus_logits,
       sampling_metadata=replace(sampling_metadata, max_num_logprobs=-1),
       ...)
   ```

   `bonus_logits` has only `batch_size` rows (one bonus position per request),
   but `sampling_metadata.temperature` is now length `num_tokens + batch_size`.
   Inside `RBLNSampler.apply_temperature`, `bonus_logits.div(temperature.unsqueeze(1))`
   becomes `[batch_size, vocab] / [num_tokens + batch_size, 1]`.

### Why single-request "works"

- `num_reqs == 1`: `bonus_logits = [1, vocab]`, padded `temperature = [N, 1]`.
  `[1, vocab] / [N, 1]` **broadcasts** (the size-1 batch dim expands to `N`),
  so no crash — and row 0 (`bonus_logits[0] / temperature[0]`) is even the
  correct value, so output looks right by coincidence.
- `num_reqs >= 2`: e.g. `[3, vocab] / [6, 1]` — `3 != 6` and neither is 1, so
  broadcasting fails → `RuntimeError`. This matches the reported
  `[3, vocab] / [6, 1]` exactly (3 reqs, `num_tokens + batch_size = 6`).

### The target-logits path is fine (and proves the fix)

`apply_sampling_constraints` (`rbln_rejection_sampler.py:257-312`) expands
per-request params to per-token via `expand_batch_to_tokens(..., cu_num_draft_tokens, ...)`.
That path *requires* unpadded, per-request (`[num_reqs]`) metadata — in fact a
padded `temperature` would trip its `assert cu_num_tokens.shape[0] == x.shape[0]`
(`rbln_rejection_sampler.py:342`). The bonus path just happens to crash first
(line 88 runs before line 115). Both want the original per-request metadata.

### Necessary conditions (all must hold)

1. `--speculative-config` active (ngram, `num_speculative_tokens >= 1`).
2. `--max-num-seqs >= 2`.
3. Two or more requests reach the sampler in the same step.
4. **`VLLM_RBLN_USE_DEVICE_TENSOR=1`** ← the report omitted this; without it no
   padding happens and the bonus path stays `[num_reqs]`-consistent. RBLN NPU
   deployments set it, so it is effectively always on in production.

Confirmed against the reported deployment env (`VLLM_RBLN_USE_DEVICE_TENSOR=1`,
`--speculative-config {"method":"ngram","num_speculative_tokens":3,...}`,
`--max-num-seqs 8`).

## Fix

Primary fix — **do not pad sampling metadata on the spec-decode path** in
`RBLNModelRunner._sample()`:

```python
if (envs.VLLM_RBLN_USE_DEVICE_TENSOR
        and spec_decode_metadata is None          # <-- add
        and logits is not None
        and logits.shape[0] > self.input_batch.num_reqs):
    sampling_metadata = self._pad_sampling_metadata(
        sampling_metadata, logits.shape[0])
```

Rationale: `RBLNRejectionSampler` is a verbatim port of upstream and already
derives both the per-request (bonus) and per-token (target) shapes internally
from `cu_num_draft_tokens`/`bonus_logits_indices`. It needs the *original*
per-request `SamplingMetadata` (length `num_reqs`). Padding only makes sense for
the plain (non-spec) sampler, which consumes `[padded_batch, vocab]` logits
directly.

### Secondary concern (same review thread)

The Copilot review also noted that `_pad_sampling_metadata` rebuilds
`SamplingMetadata` without carrying `spec_token_ids`, so penalties / bad-word
filtering would lose draft-token context even when no crash occurs. Skipping
padding for the spec path (above) sidesteps this too, but we should still add
`spec_token_ids=metadata.spec_token_ids` to the rebuilt object in
`_pad_sampling_metadata` for defense-in-depth.

## Reproduction & test plan

The bug is a pure-Python shape/broadcast contract; it does **not** need NPU
hardware to reproduce. (This dev box has `rebel` 0.10.4 installed but `rbln-smi`
shows no attached device, so full `vllm serve` compile/e2e is not runnable
here.)

### CPU unit/integration test (primary — runnable in CI without NPU)

Add to `tests/torch_compile/unit/v1/sample/test_rejection_sampler.py` (or a new
worker-level test):

1. Build per-request **random**-sampling `SamplingMetadata` with
   `temperature = torch.ones(num_reqs)`, `num_reqs >= 2`.
2. Build a real `SpecDecodeMetadata` via `SpecDecodeMetadata.make_dummy` with
   correct `target_logits_indices` / `bonus_logits_indices` so the bonus path
   slices exactly `batch_size` rows.
3. Drive the **real** `RBLNModelRunner._sample` decision (bound to a minimal
   fake runner exposing `input_batch.{sampling_metadata,num_reqs,vocab_size}`,
   real `_pad_sampling_metadata`, and a real `RBLNRejectionSampler`).
4. Substitute only the device-dependent low-level sampler with a CPU stub that
   reproduces the exact crashing math
   (`logits.div(temperature.unsqueeze(dim=1))` + argmax), mirroring
   `RBLNSampler.apply_temperature` (`rbln_sampler.py:421`).
5. Monkeypatch `envs.VLLM_RBLN_USE_DEVICE_TENSOR = True`.

Assertions:
- **Before fix**: `_sample` pads → bonus path raises `RuntimeError` (batch > 1).
  Guard with `pytest.raises` to lock in the repro.
- **After fix**: no padding for spec path → `temperature` seen by the rejection
  sampler has length `num_reqs`; sampling returns a
  `[num_reqs, max_spec_len + 1]` token tensor with no error.
- Regression: `num_reqs == 1` keeps working both before and after.

### e2e smoke test (when NPU is available)

MiniMax-M2.5 will not load on the bench; use **Qwen3-0.6B** instead. Reuse the
reported launch flags, trimmed to a single DP rank:

```
vllm serve Qwen/Qwen3-0.6B \
  --tensor-parallel-size 1 \
  --max-num-seqs 8 --max-num-batched-tokens 512 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":3,"prompt_lookup_max":5,"prompt_lookup_min":2}'
# env: VLLM_RBLN_USE_DEVICE_TENSOR=1
```

Drive >= 2 concurrent requests with repeated n-grams (to trigger ngram drafts)
and confirm the server no longer crashes and output is coherent.

## Files

- `vllm_rbln/v1/worker/rbln_model_runner.py` — `_sample` (pad guard),
  `_pad_sampling_metadata` (`spec_token_ids` passthrough).
- `vllm_rbln/v1/sample/rbln_rejection_sampler.py` — bonus path
  (reference only; no change required by the primary fix).
- `tests/torch_compile/unit/v1/sample/test_rejection_sampler.py` — new repro
  test.
