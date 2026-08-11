# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from unittest.mock import Mock

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_rbln.v1.sample import rbln_sampler as module
from vllm_rbln.v1.sample.ops.top_k_top_p import GREEDY_TOP_K, GREEDY_TOP_P
from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler

VOCAB_SIZE = 8


def make_sampling_metadata(
    temperature: torch.Tensor | None,
    all_greedy: bool,
    all_random: bool,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
) -> SamplingMetadata:
    return SamplingMetadata(
        temperature=temperature,
        all_greedy=all_greedy,
        all_random=all_random,
        top_p=top_p,
        top_k=top_k,
        generators={},
        max_num_logprobs=None,
        no_penalties=True,
        prompt_token_ids=None,
        frequency_penalties=torch.tensor([]),
        presence_penalties=torch.tensor([]),
        repetition_penalties=torch.tensor([]),
        output_token_ids=[],
        allowed_token_ids_mask=None,
        bad_words_token_ids={},
        logitsprocs=LogitsProcessors(),
    )


@pytest.fixture
def sampler(monkeypatch) -> RBLNSampler:
    monkeypatch.setattr(module, "compile", lambda target, **kwargs: target)
    return RBLNSampler(compile_context=Mock())


@pytest.fixture
def op_args(sampler) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Record the (top_k, top_p) tensors the sampling op receives."""
    recorded: list[tuple[torch.Tensor, torch.Tensor]] = []
    original = sampler.topk_topp_sampler.top_k_top_p_sample

    def spy(logits, k, p):
        recorded.append((k, p))
        return original(logits, k, p)

    sampler.topk_topp_sampler.top_k_top_p_sample = spy
    return recorded


def near_tied_logits() -> torch.Tensor:
    """Two rows whose top two logits are almost equal.

    Under the previous encoding -- dividing a greedy row by 1e-3 to collapse its
    softmax toward a one-hot -- a gap this small survives the scaling (0.0001 *
    1000 = 0.1), leaving the argmax only ~52% likely. Narrowing the candidate
    set instead makes the outcome exact.
    """
    logits = torch.full((2, VOCAB_SIZE), -100.0)
    logits[0, 4] = 10.0
    logits[0, 6] = 10.0001  # row 0 argmax
    logits[1, 1] = 10.0001  # row 1 argmax
    logits[1, 5] = 10.0
    return logits


def test_greedy_row_of_a_mixed_batch_samples_its_argmax(sampler):
    metadata = make_sampling_metadata(
        temperature=torch.tensor([0.0, 1.0]),
        all_greedy=False,
        all_random=False,
    )

    for _ in range(50):
        sampled, _ = sampler.sample(near_tied_logits(), metadata)
        assert sampled[0].item() == 6


def test_mixed_batch_narrows_only_the_greedy_row(sampler, op_args):
    metadata = make_sampling_metadata(
        temperature=torch.tensor([0.0, 1.0]),
        all_greedy=False,
        all_random=False,
    )

    sampler.sample(near_tied_logits(), metadata)

    top_k, top_p = op_args[0]
    # The random row keeps the values that disable both filters.
    assert torch.equal(
        top_k, torch.tensor([GREEDY_TOP_K, VOCAB_SIZE], dtype=torch.int32)
    )
    assert torch.equal(top_p, torch.tensor([GREEDY_TOP_P, 1.0], dtype=torch.float32))


def test_random_batch_keeps_request_top_k_top_p(sampler, op_args):
    metadata = make_sampling_metadata(
        temperature=torch.tensor([1.0, 2.0]),
        all_greedy=False,
        all_random=True,
        top_k=torch.tensor([2, VOCAB_SIZE], dtype=torch.int32),
        top_p=torch.tensor([1.0, 0.9], dtype=torch.float32),
    )

    sampler.sample(near_tied_logits(), metadata)

    top_k, top_p = op_args[0]
    assert torch.equal(top_k, torch.tensor([2, VOCAB_SIZE], dtype=torch.int32))
    assert torch.equal(top_p, torch.tensor([1.0, 0.9], dtype=torch.float32))


def test_all_greedy_batch_takes_the_argmax_op(sampler, op_args):
    metadata = make_sampling_metadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
    )

    sampled, _ = sampler.sample(near_tied_logits(), metadata)

    # An all-greedy batch skips the top-k/top-p op entirely.
    assert op_args == []
    assert sampled.tolist() == [6, 1]


def test_greedy_row_temperature_is_replaced_by_one(sampler):
    """The temperature substitution only has to avoid dividing by zero, so the
    greedy row's logits reach the op unscaled."""
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    scaled = sampler.apply_temperature(
        logits.clone(), torch.tensor([0.0, 2.0]), all_random=False
    )

    expected = torch.stack([logits[0], logits[1] / 2.0])
    assert torch.equal(scaled, expected)
