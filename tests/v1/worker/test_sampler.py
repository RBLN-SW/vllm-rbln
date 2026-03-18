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
from torch._dynamo.testing import CompileCounter
from vllm.platforms import current_platform

from .utils import (
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
@pytest.mark.parametrize("warm_up", [True, False], ids=["warm_upTrue", "warm_upFalse"])
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


def test_sampler_logits_reshape_prevents_torch_compile_recompile(monkeypatch):
    """
    Test to ensure that the sampler does not recompile
    when `compute_logits` returns logits with different strides.

    This test forces `compute_logits` to return logits with different strides
    while keeping batch_size=1, and asserts the sampler compiles only once.
    """

    monkeypatch.setenv("VLLM_RBLN_SAMPLER", "1")
    monkeypatch.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    monkeypatch.setenv("VLLM_RBLN_ENABLE_WARM_UP", "False")
    monkeypatch.setenv("TORCH_LOGS", "recompiles")

    compile_counter = CompileCounter()
    real_torch_compile = torch.compile

    def torch_compile_with_counter(fn, *args, **kwargs):
        kwargs.setdefault("backend", compile_counter)
        return real_torch_compile(fn, *args, **kwargs)

    monkeypatch.setattr(torch, "compile", torch_compile_with_counter)

    # Keep max_num_seqs=1 so we always take the non-padding path.
    runner = create_model_runner(max_num_seqs=1)

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

    for i in range(2):
        req = make_request(request_id=f"req_{i}", prompt_token_ids=[1, 2, 3])
        scheduler_output = _schedule_new_request_from_request(
            req, block_ids=([0],), outer_block_ids=[0]
        )
        runner.execute_model(scheduler_output)
        _ = runner.sample_tokens(grammar_output=None)

    # Should compile only one graph for the sampler.
    # If recompilation happens, the counter will be more than 3 and fail the assertion.
    assert compile_counter.frame_count == 3
