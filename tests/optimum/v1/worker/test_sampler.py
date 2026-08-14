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
    _schedule_new_request_from_request,
    create_model_runner,
    fake_load_model,
    forward_steps,
    make_request,
    prefill_requests,
    run_decode_steps,
    sampled_token,
)

DEVICE = current_platform.device_type


@pytest.fixture
def rbln_sampler_env(monkeypatch):
    """RBLN sampler on, strict compile, no warm-up — the setup every test
    here shares unless it parametrizes one of these itself."""
    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch.setenv("VLLM_RBLN_ENABLE_WARM_UP", "False")


def set_fixed_logits(runner, favored: dict[int, float]):
    """Make every forward return the same logits: `favored` token values on a
    zero background, for every request row. Greedy sampling then picks the
    highest favored token, letting tests assert exact penalty effects."""
    vocab_size = runner.model_config.get_vocab_size()

    def fixed_forward(model_input, **kwargs):
        logits = torch.zeros(
            (runner.input_batch.num_reqs, 1, vocab_size), dtype=torch.float32
        )
        for token_id, value in favored.items():
            logits[:, :, token_id] = value
        return logits

    runner.model.forward = fixed_forward


@pytest.mark.parametrize(
    "num_seqs, expected_bucket_sizes",
    [
        pytest.param(1, [1], id="1_seq"),
        pytest.param(2, [1, 2], id="2_seq"),
        pytest.param(16, [1, 2, 4, 8, 16], id="16_seq"),
        pytest.param(17, [1, 2, 4, 8, 16, 17], id="17_seq"),
        pytest.param(61, [1, 2, 4, 8, 16, 24, 32, 40, 48, 56, 61], id="61_seq"),
        # Powers of two to 16, then step 8 to 256, then step 16 to 512, and
        # a non-bucket max_num_seqs is appended as its own final bucket.
        pytest.param(
            512,
            [1, 2, 4, 8, *range(16, 257, 8), *range(272, 513, 16)],
            id="512_seq",
        ),
        pytest.param(
            515,
            [1, 2, 4, 8, *range(16, 257, 8), *range(272, 513, 16), 515],
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


@pytest.mark.parametrize("use_rbln_sampler", [True, False])
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
# The three penalties travel one code path (no_penalties on/off), so one
# all-on case suffices here; exact penalty effects have dedicated tests.
@pytest.mark.parametrize(
    "presence_penalty, frequency_penalty, repetition_penalty",
    [(0.0, 0.0, 1.0), (2.0, 2.0, 2.0)],
    ids=["no_penalty", "all_penalty"],
)
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
    rbln_sampler_env,
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
    prefill_requests(runner, reqs)

    # Decode all together: num_reqs=3, bucket_size=4 -> row 3 is padding.
    # Several steps, so the reused pad rows of the pooled buffer and the
    # growing output_token_ids are exercised, not just the first step.
    for output in run_decode_steps(runner, reqs, num_steps=3):
        assert_no_nan_in_pooled(output)


def test_sampler_logits_reshape_keeps_shape_and_stride_stable(
    rbln_sampler_env, monkeypatch
):
    """
    Test to ensure that the sampler always receives the same shape and stride
    even when `compute_logits` returns logits with different strides.

    The sampler ops are compiled for the RBLN device, and dynamo guards on
    stride, so a varying stride would recompile them on every other step. This
    test forces `compute_logits` to alternate strides while keeping
    batch_size=1, and asserts the reshape in `sample_tokens` absorbs it.
    """
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


def test_penalty_decode_steps_do_not_recompile(rbln_sampler_env, monkeypatch):
    """Penalties feed SamplingMetadata.output_token_ids into the sampler — a
    list[list[int]] that grows by one token every decode step. The penalty
    path runs eagerly, fully outside torch.compile, so no dynamo frame may
    specialize on that list: the sampler ops compile once at prefill and
    every decode step after that must hit the cache. A frame that guards on
    the list would recompile per token and eventually kill the engine with
    FailOnRecompileLimitHit.
    """
    # One bucket only, so a changed shape can't excuse a recompile.
    runner = create_model_runner(max_num_seqs=1)
    req = make_request(
        request_id="req_0",
        prompt_token_ids=[1, 2, 3],
        presence_penalty=2.0,
        frequency_penalty=2.0,
        repetition_penalty=2.0,
    )
    # Prefill runs the sampler once: the only legitimate compile.
    prefill_requests(runner, [req])

    # Each decode step grows output_token_ids; any recompile on any step
    # means something specialized on it.
    monkeypatch.setattr(torch._dynamo.config, "error_on_recompile", True)
    run_decode_steps(runner, [req], num_steps=4)


@pytest.mark.parametrize("use_penalty", [True, False], ids=["penalty", "no_penalty"])
def test_presence_frequency_penalty_changes_greedy_pick(rbln_sampler_env, use_penalty):
    """Presence/frequency penalties must actually reach the logits: with
    greedy sampling and fixed logits, the top token wins until it has been
    generated once, after which the penalties push it below the runner-up.
    Without penalties the top token wins every step.
    """
    runner = create_model_runner(max_num_seqs=1)
    # Gaps below 4.0, so presence 2.0 + frequency 2.0 demotes a generated
    # token below the next one.
    set_fixed_logits(runner, {7: 3.0, 11: 1.0, 23: 0.5})
    req = make_request(
        request_id="req_0",
        prompt_token_ids=[1, 2, 3],
        temperature=0.0,
        presence_penalty=2.0 if use_penalty else 0.0,
        frequency_penalty=2.0 if use_penalty else 0.0,
    )

    # Prefill has no output tokens yet, so both cases pick the top token.
    (prefill_output,) = prefill_requests(runner, [req])
    assert sampled_token(prefill_output, "req_0") == 7

    step1, step2 = run_decode_steps(runner, [req], num_steps=2)
    if use_penalty:
        # 7 was generated at prefill: 3.0 - 4.0 < 1.0, so 11 wins, then 23.
        assert sampled_token(step1, "req_0") == 11
        assert sampled_token(step2, "req_0") == 23
    else:
        assert sampled_token(step1, "req_0") == 7
        assert sampled_token(step2, "req_0") == 7


@pytest.mark.parametrize(
    "repetition_penalty", [2.0, 1.0], ids=["penalty", "no_penalty"]
)
def test_repetition_penalty_applies_to_prompt_tokens(
    rbln_sampler_env, repetition_penalty
):
    """Repetition penalty covers prompt tokens, not just generated ones: a
    prompt token holding the top logit must lose to the runner-up already at
    the prefill sample when the penalty halves its positive logit.
    """
    runner = create_model_runner(max_num_seqs=1)
    # Token 3 is in the prompt: 2.0 / 2.0 = 1.0 < 1.5, so 11 wins.
    set_fixed_logits(runner, {3: 2.0, 11: 1.5})
    req = make_request(
        request_id="req_0",
        prompt_token_ids=[1, 2, 3],
        temperature=0.0,
        repetition_penalty=repetition_penalty,
    )

    (prefill_output,) = prefill_requests(runner, [req])
    expected = 11 if repetition_penalty == 2.0 else 3
    assert sampled_token(prefill_output, "req_0") == expected


def test_mixed_penalty_batch_isolates_requests(rbln_sampler_env):
    """One penalized request in a batch must not disturb the others: the
    batch-level no_penalties flag turns the penalty path on for every row,
    and the unpenalized rows rely on their 0.0/1.0 defaults being no-ops.
    """
    # 3 reqs pad to bucket_size=4, covering the padded-metadata path too.
    runner = create_model_runner(max_num_seqs=4)
    set_fixed_logits(runner, {7: 3.0, 11: 1.0, 23: 0.5})

    penalized = make_request(
        request_id="req_0",
        prompt_token_ids=[1, 2, 3],
        temperature=0.0,
        presence_penalty=2.0,
        frequency_penalty=2.0,
    )
    plain = [
        make_request(request_id=f"req_{i}", prompt_token_ids=[1, 2, 3], temperature=0.0)
        for i in (1, 2)
    ]
    reqs = [penalized, *plain]

    prefill_requests(runner, reqs)

    step1, step2 = run_decode_steps(runner, reqs, num_steps=2)
    for output in (step1, step2):
        for req in plain:
            assert sampled_token(output, req.request_id) == 7
    assert sampled_token(step1, "req_0") == 11
    assert sampled_token(step2, "req_0") == 23
