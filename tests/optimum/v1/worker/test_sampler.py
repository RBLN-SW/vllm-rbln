# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import torch
from vllm.platforms import current_platform

from .utils import (
    _schedule_cached_reqs,
    _schedule_new_request_from_request,
    create_model_runner,
    fake_load_model,
    forward_steps,
    make_request,
)

DEVICE = current_platform.device_type


@pytest.mark.parametrize(
    "num_seqs, expected_bucket_sizes",
    [
        pytest.param(1, [1], id="1_seq"),
        pytest.param(2, [1, 2], id="2_seq"),
        pytest.param(16, [1, 2, 4, 8, 16], id="16_seq"),
        pytest.param(17, [1, 2, 4, 8, 16, 17], id="17_seq"),
        pytest.param(61, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 61], id="61_seq"),
        pytest.param(
            512,
            [
                1,
                2,
                4,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
                168,
                176,
                184,
                192,
                200,
                208,
                216,
                224,
                232,
                240,
                248,
                256,
                272,
                288,
                304,
                320,
                336,
                352,
                368,
                384,
                400,
                416,
                432,
                448,
                464,
                480,
                496,
                512,
            ],
            id="512_seq",
        ),
        pytest.param(
            515,
            [
                1,
                2,
                4,
                8,
                16,
                24,
                32,
                40,
                48,
                56,
                64,
                72,
                80,
                88,
                96,
                104,
                112,
                120,
                128,
                136,
                144,
                152,
                160,
                168,
                176,
                184,
                192,
                200,
                208,
                216,
                224,
                232,
                240,
                248,
                256,
                272,
                288,
                304,
                320,
                336,
                352,
                368,
                384,
                400,
                416,
                432,
                448,
                464,
                480,
                496,
                512,
                515,
            ],
            id="515_seq",
        ),
    ],
)
def test_get_bucket_sizes(monkeypatch, num_seqs: int, expected_bucket_sizes: list[int]):
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    runner = create_model_runner(max_num_seqs=num_seqs)
    fake_load_model(runner)
    bucket_sizes = runner.get_bucket_sizes(num_seqs)
    assert bucket_sizes == expected_bucket_sizes
    assert len(runner.pooled_tensors) == len(expected_bucket_sizes)


@pytest.mark.parametrize("use_rbln_sampler", [False])
@pytest.mark.parametrize("use_structured_output", [True, False])
def test_forward_sampler_mode_and_structured_output(
    monkeypatch, use_rbln_sampler, use_structured_output
):
    """Test sampler logic for both use_rbln_sampler=True and False."""
    monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1" if use_rbln_sampler else "0")
    reqs = []
    for i in range(3):
        reqs.append(
            make_request(
                request_id=f"req_{i}",
                prompt_token_ids=[1, 2, 3],
                use_structured_output=use_structured_output,
                top_p=0.7,
            )
        )
    forward_steps(reqs)


@pytest.mark.parametrize("top_p", [0.7, 1.0])
@pytest.mark.parametrize("top_k", [0, 3])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize("logprobs", [0, 3])
@pytest.mark.parametrize("presence_penalty", [0.0, 2.0])
@pytest.mark.parametrize("frequency_penalty", [0.0, 2.0])
@pytest.mark.parametrize("repetition_penalty", [1.0, 2.0])
@pytest.mark.parametrize(
    "warm_up", [True, False], ids=["warm_up_true", "warm_up_false"]
)
def test_forward_sampling_parameters(
    monkeypatch,
    top_p,
    top_k,
    temperature,
    logprobs,
    presence_penalty,
    frequency_penalty,
    repetition_penalty,
    warm_up,
):
    monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch.setenv("VLLM_RBLN_ENABLE_WARM_UP", "True" if warm_up else "False")
    reqs = []
    for i in range(3):
        reqs.append(
            make_request(
                request_id=f"req_{i}",
                prompt_token_ids=[1, 2, 3],
                top_p=top_p,
                top_k=top_k,
                temperature=temperature,
                logprobs=logprobs,
                presence_penalty=presence_penalty,
                frequency_penalty=frequency_penalty,
                repetition_penalty=repetition_penalty,
            )
        )
    forward_steps(reqs)


# TODO mix the requests with different sampling parameters


@pytest.mark.parametrize("top_p", [0.7, 1.0])
@pytest.mark.parametrize("top_k", [0, 3])
@pytest.mark.parametrize("temperature", [0.0, 1.0])
@pytest.mark.parametrize(
    "presence_penalty, frequency_penalty, repetition_penalty",
    [(0.0, 0.0, 1.0), (2.0, 2.0, 2.0)],
    ids=["no_penalty", "all_penalty"],
)
def test_no_nan_logits_with_padded_bucket(
    monkeypatch,
    top_p,
    top_k,
    temperature,
    presence_penalty,
    frequency_penalty,
    repetition_penalty,
):
    """When use_rbln_sampler=True and num_reqs < bucket_size, the pooled tensor
    holding padded logits has unused rows that the sampler still processes with
    padded sampling metadata. RBLNInputBatch must explicitly initialize every
    sampling-param tensor's pad rows to safe defaults — otherwise a NaN/garbage
    value from torch.empty() propagates through penalty / top_k / top_p ops
    into NaN logits and out-of-vocab sampled tokens.

    To make this deterministic regardless of allocator state, torch.empty is
    patched during runner construction so every uninitialized float tensor
    starts as NaN. Any missing init guard in RBLNInputBatch will then surface.
    """
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch.setenv("VLLM_RBLN_ENABLE_WARM_UP", "False")

    # max_num_seqs=4 with 3 reqs -> decode uses bucket_size=4, one padded row.
    # Force torch.empty to return NaN-filled float tensors during init so the
    # test does not rely on lucky zero-page allocations.
    real_empty = torch.empty

    def empty_nan(*args, **kwargs):
        t = real_empty(*args, **kwargs)
        if t.is_floating_point():
            t.fill_(float("nan"))
        return t

    torch.empty = empty_nan
    try:
        runner = create_model_runner(max_num_seqs=4)
    finally:
        torch.empty = real_empty

    vocab_size = runner.model_config.get_vocab_size()

    reqs = [
        make_request(
            request_id=f"req_{i}",
            prompt_token_ids=[1, 2, 3],
            top_p=top_p,
            top_k=top_k,
            temperature=temperature,
            presence_penalty=presence_penalty,
            frequency_penalty=frequency_penalty,
            repetition_penalty=repetition_penalty,
        )
        for i in range(3)
    ]

    def assert_no_nan_in_pooled(output):
        # No row of the pooled logits tensor — active or padded — should
        # contain NaN. NaN in pad rows can still propagate into the active
        # sampled token ids through any cross-row sampler op.
        pooled = runner.pooled_tensors[runner.bucket_size]
        assert not torch.isnan(pooled).any(), (
            f"NaN found in pooled logits (bucket_size={runner.bucket_size})"
        )
        for sampled_ids in output.sampled_token_ids:
            for token_id in sampled_ids:
                assert 0 <= token_id < vocab_size, (
                    f"Out-of-vocab sampled token id {token_id} "
                    f"(vocab_size={vocab_size})"
                )

    # Prefill is single-req per step (no padding); just run it.
    for i, req in enumerate(reqs):
        scheduler_output = _schedule_new_request_from_request(
            req, block_ids=([i],), outer_block_ids=[i]
        )
        runner.execute_model(scheduler_output)
        runner.sample_tokens(grammar_output=None)

    for req in reqs:
        req.num_computed_tokens = 3

    # Decode all together: num_reqs=3, bucket_size=4 -> row 3 is padding.
    scheduler_output = _schedule_cached_reqs(reqs, new_block_ids=[None, None, None])
    runner.execute_model(scheduler_output)
    output = runner.sample_tokens(grammar_output=None)
    assert_no_nan_in_pooled(output)


def test_sampler_logits_reshape_keeps_shape_and_stride_stable(monkeypatch):
    """
    Test to ensure that the sampler always receives the same shape and stride
    even when `compute_logits` returns logits with different strides.

    The sampler ops are compiled for the RBLN device, and dynamo guards on
    stride, so a varying stride would recompile them on every other step. This
    test forces `compute_logits` to alternate strides while keeping
    batch_size=1, and asserts the reshape in `sample_tokens` absorbs it.
    """

    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch.setenv("VLLM_RBLN_ENABLE_WARM_UP", "False")

    # Keep max_num_seqs=1 so we always take the non-padding path.
    runner = create_model_runner(max_num_seqs=1)

    # Record what the sampler is actually handed on each step. Patch `forward`
    # rather than the module itself: the runner also reaches the sampler for
    # `compute_logprobs` / `gather_logprobs`.
    seen: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    real_forward = runner.sampler.forward

    def recording_forward(logits, sampling_metadata, *args, **kwargs):
        seen.append((tuple(logits.shape), tuple(logits.stride())))
        return real_forward(logits, sampling_metadata, *args, **kwargs)

    monkeypatch.setattr(runner.sampler, "forward", recording_forward)

    # Alternate logits rank across steps.
    call_count = 0
    real_compute_logits = runner.model.compute_logits

    def compute_logits_flaky(hidden_states, sampling_metadata):
        nonlocal call_count
        call_count += 1
        logits_2d = real_compute_logits(hidden_states, sampling_metadata)
        if call_count % 2 == 1:
            vocab_size = logits_2d.shape[-1]
            # Change stride from (vocab_size, 1) to (vocab_size * 2, 1)
            logits_2d = logits_2d.as_strided(
                size=(1, vocab_size), stride=(2 * vocab_size, 1)
            )
        return logits_2d

    runner.model.compute_logits = compute_logits_flaky

    def run_step(i):
        req = make_request(request_id=f"req_{i}", prompt_token_ids=[1, 2, 3])
        scheduler_output = _schedule_new_request_from_request(
            req, block_ids=([0],), outer_block_ids=[0]
        )
        runner.execute_model(scheduler_output)
        _ = runner.sample_tokens(grammar_output=None)

    # 1st iter: stride-changed logits. 2nd iter: normal-stride logits.
    run_step(0)
    run_step(1)

    assert len(seen) == 2, f"sampler should have run once per step, got {seen}"
    assert seen[0] == seen[1], (
        f"sampler input changed across stride change: {seen[0]} -> {seen[1]}"
    )


def test_min_tokens_masks_stop_tokens_in_half_precision_logits(monkeypatch):
    """
    RBLNSampler hands the logits to the logits processors in the model dtype
    instead of casting them to float32, so a half-precision model produces a
    half-precision logits buffer. MinTokensLogitsProcessor writes its -inf
    constant with `index_put_`, which requires the source dtype to match the
    destination exactly, so a float32 constant raises on that buffer.
    """

    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch.setenv("VLLM_RBLN_ENABLE_WARM_UP", "False")

    stop_token_id = 5
    runner = create_model_runner(max_num_seqs=1, model_dtype=torch.float16)

    # Hold on to the tensor the sampler is handed: the logits processors mutate
    # it in place, so it carries the mask once the step is over.
    seen_logits: list[torch.Tensor] = []
    real_forward = runner.sampler.forward

    def recording_forward(logits, *args, **kwargs):
        seen_logits.append(logits)
        return real_forward(logits, *args, **kwargs)

    monkeypatch.setattr(runner.sampler, "forward", recording_forward)

    req = make_request(
        request_id="req_0",
        prompt_token_ids=[1, 2, 3],
        min_tokens=8,
        stop_token_ids=[stop_token_id],
    )
    scheduler_output = _schedule_new_request_from_request(
        req, block_ids=([0],), outer_block_ids=[0]
    )
    runner.execute_model(scheduler_output)
    assert runner.sample_tokens(grammar_output=None) is not None

    assert len(seen_logits) == 1
    logits = seen_logits[0]
    assert logits.dtype == torch.float16, (
        f"sampler should keep the model dtype, got {logits.dtype}"
    )
    # The request has produced no output token yet, so it is short of its
    # min_tokens and its stop token must be inhibited.
    assert logits[0, stop_token_id] == -float("inf"), (
        "min_tokens did not mask the stop token"
    )
