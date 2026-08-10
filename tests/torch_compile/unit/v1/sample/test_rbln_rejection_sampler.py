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

"""Tests for how RBLNRejectionSamplerImpl hands greedy requests to the NPU
`rbln::rejection_sample` op.

The op has no greedy kernel: it draws one token per row from `target_probs`
under that row's top-k/top-p and accepts the draft only if the two match. Greedy
rows are therefore narrowed to their top-1 candidate, which is what these tests
pin down. `compile` is replaced by the identity so the impl calls the op eagerly
and runs against its CPU reference implementation.
"""

from unittest.mock import Mock

import pytest
import torch
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata

from vllm_rbln.v1.sample import rbln_rejection_sampler as module
from vllm_rbln.v1.sample.ops.top_k_top_p import (
    GREEDY_TOP_K,
    GREEDY_TOP_P,
    build_op_top_k_top_p,
)
from vllm_rbln.v1.sample.rbln_rejection_sampler import (
    PLACEHOLDER_TOKEN_ID,
    RBLNRejectionSamplerImpl,
)

DEVICE = torch.device("cpu")
VOCAB_SIZE = 8
NUM_SPEC_TOKENS = 2


def make_sampling_metadata(
    temperature: torch.Tensor | None,
    all_greedy: bool,
    all_random: bool,
    top_k: torch.Tensor | None = None,
    top_p: torch.Tensor | None = None,
) -> SamplingMetadata:
    """Build a SamplingMetadata carrying only the fields this impl reads.

    Unlike the helper in test_torch_rejection_sampler, this one allows a mixed
    batch (`all_greedy` and `all_random` both False).
    """
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
def impl(monkeypatch) -> RBLNRejectionSamplerImpl:
    monkeypatch.setattr(module, "compile", lambda target, **kwargs: target)
    return RBLNRejectionSamplerImpl(
        compile_context=Mock(),
        num_spec_tokens=NUM_SPEC_TOKENS,
    )


########################### build_op_top_k_top_p ###########################
def test_all_greedy_batch_is_encoded_as_argmax():
    metadata = make_sampling_metadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 3, VOCAB_SIZE, DEVICE)

    assert torch.equal(top_k, torch.tensor([GREEDY_TOP_K] * 3, dtype=torch.int32))
    assert torch.equal(top_p, torch.tensor([GREEDY_TOP_P] * 3, dtype=torch.float32))


def test_all_random_batch_without_top_k_top_p_disables_both():
    metadata = make_sampling_metadata(
        temperature=torch.tensor([1.0, 2.0]),
        all_greedy=False,
        all_random=True,
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 2, VOCAB_SIZE, DEVICE)

    # `top_k == vocab_size` and `top_p == 1.0` are the values that leave the
    # candidate set untouched.
    assert torch.equal(top_k, torch.tensor([VOCAB_SIZE] * 2, dtype=torch.int32))
    assert torch.equal(top_p, torch.tensor([1.0, 1.0], dtype=torch.float32))


def test_all_random_batch_keeps_request_top_k_top_p():
    metadata = make_sampling_metadata(
        temperature=torch.tensor([1.0, 2.0]),
        all_greedy=False,
        all_random=True,
        top_k=torch.tensor([4, VOCAB_SIZE], dtype=torch.int32),
        top_p=torch.tensor([0.9, 1.0], dtype=torch.float32),
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 2, VOCAB_SIZE, DEVICE)

    assert torch.equal(top_k, torch.tensor([4, VOCAB_SIZE], dtype=torch.int32))
    assert torch.equal(top_p, torch.tensor([0.9, 1.0], dtype=torch.float32))


def test_mixed_batch_overrides_only_greedy_rows():
    # Row 0 is greedy, and vLLM rewrites its params to the same values a random
    # request without top-k/top-p carries -- which is why the op cannot tell the
    # two apart from `sampling_metadata` alone.
    metadata = make_sampling_metadata(
        temperature=torch.tensor([0.0, 1.0, 2.0]),
        all_greedy=False,
        all_random=False,
        top_k=torch.tensor([VOCAB_SIZE, 4, VOCAB_SIZE], dtype=torch.int32),
        top_p=torch.tensor([1.0, 1.0, 0.9], dtype=torch.float32),
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 3, VOCAB_SIZE, DEVICE)

    assert torch.equal(
        top_k, torch.tensor([GREEDY_TOP_K, 4, VOCAB_SIZE], dtype=torch.int32)
    )
    assert torch.equal(
        top_p, torch.tensor([GREEDY_TOP_P, 1.0, 0.9], dtype=torch.float32)
    )


def test_mixed_batch_without_request_top_k_top_p():
    metadata = make_sampling_metadata(
        temperature=torch.tensor([2.0, 0.0]),
        all_greedy=False,
        all_random=False,
    )

    top_k, top_p = build_op_top_k_top_p(metadata, 2, VOCAB_SIZE, DEVICE)

    assert torch.equal(
        top_k, torch.tensor([VOCAB_SIZE, GREEDY_TOP_K], dtype=torch.int32)
    )
    assert torch.equal(top_p, torch.tensor([1.0, GREEDY_TOP_P], dtype=torch.float32))


########################### apply_sampling_constraints ###########################
def test_apply_sampling_constraints_leaves_all_greedy_logits_untouched(impl):
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    metadata = make_sampling_metadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
    )

    out = impl.apply_sampling_constraints(
        logits.clone(),
        cu_num_draft_tokens=torch.tensor([1, 2]),
        sampling_metadata=metadata,
    )

    # The op resolves these rows to their argmax, so no host-side rewrite is
    # needed -- not even temperature scaling.
    assert torch.equal(out, logits)


def test_apply_sampling_constraints_scales_only_random_rows(impl):
    logits = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    metadata = make_sampling_metadata(
        temperature=torch.tensor([0.0, 2.0]),
        all_greedy=False,
        all_random=False,
    )

    out = impl.apply_sampling_constraints(
        logits.clone(),
        cu_num_draft_tokens=torch.tensor([1, 2]),
        sampling_metadata=metadata,
    )

    # The greedy row is divided by 1 (scaling cannot move an argmax); the random
    # row by its own temperature.
    expected = torch.stack([logits[0], logits[1] / 2.0])
    assert torch.equal(out, expected)


########################### rejection_sample ###########################
def make_target_probs(argmax_token_ids: list[int]) -> torch.Tensor:
    """One row per draft position, peaked at the given token but far from
    one-hot: the argmax holds 0.4 of the mass and the rest is spread evenly."""
    num_tokens = len(argmax_token_ids)
    probs = torch.full(
        (num_tokens, VOCAB_SIZE),
        0.6 / (VOCAB_SIZE - 1),
        dtype=torch.float32,
    )
    for row, token_id in enumerate(argmax_token_ids):
        probs[row, token_id] = 0.4
    return probs


def run_rejection_sample(
    impl: RBLNRejectionSamplerImpl,
    draft_token_ids: list[int],
    target_argmax_token_ids: list[int],
    bonus_token_ids: list[int],
    metadata: SamplingMetadata,
    num_draft_tokens: list[int] | None = None,
) -> torch.Tensor:
    """Run the impl on a batch of drafts packed in `draft_token_ids` order.

    `num_draft_tokens` defaults to every request holding NUM_SPEC_TOKENS drafts.
    """
    if num_draft_tokens is None:
        num_draft_tokens = [NUM_SPEC_TOKENS] * (len(draft_token_ids) // NUM_SPEC_TOKENS)
    return impl.rejection_sample(
        draft_token_ids=torch.tensor(draft_token_ids, dtype=torch.int32),
        num_draft_tokens=num_draft_tokens,
        max_spec_len=NUM_SPEC_TOKENS,
        cu_num_draft_tokens=torch.cumsum(
            torch.tensor(num_draft_tokens, dtype=torch.int32), dim=0
        ),
        draft_probs=None,
        target_probs=make_target_probs(target_argmax_token_ids),
        bonus_token_ids=torch.tensor(bonus_token_ids, dtype=torch.int64).unsqueeze(-1),
        sampling_metadata=metadata,
    )


@pytest.mark.parametrize("trial", range(20))
def test_greedy_rows_accept_exactly_the_target_argmax(impl, trial):
    """A greedy row accepts iff its draft is the target argmax, and recovers the
    argmax at the first mismatch.

    `target_probs` puts only 0.4 on the argmax, so a row treated as random would
    diverge from this expectation within a few trials.
    """
    metadata = make_sampling_metadata(
        # Row 0 greedy, row 1 greedy, row 2 random.
        temperature=torch.tensor([0.0, 0.0, 1.0]),
        all_greedy=False,
        all_random=False,
        # The greedy rows carry the params vLLM assigns them, so only the
        # temperature marks them as greedy.
        top_k=torch.tensor([VOCAB_SIZE, VOCAB_SIZE, 1], dtype=torch.int32),
        top_p=torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32),
    )

    output = run_rejection_sample(
        impl,
        # Row 0 matches both argmaxes; row 1 mismatches at position 1; row 2
        # (random, top_k=1) matches both argmaxes.
        draft_token_ids=[3, 5, 2, 4, 6, 1],
        target_argmax_token_ids=[3, 5, 2, 7, 6, 1],
        bonus_token_ids=[10, 11, 12],
        metadata=metadata,
    )

    expected = torch.tensor(
        [
            [3, 5, 10],  # all accepted -> bonus token
            [2, 7, PLACEHOLDER_TOKEN_ID],  # accept one, then recover the argmax
            [6, 1, 12],  # top_k=1 row behaves like the greedy rows
        ],
        dtype=torch.int32,
    )
    assert torch.equal(output, expected)


def test_all_greedy_rows_accept_exactly_the_target_argmax(impl):
    metadata = make_sampling_metadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
    )

    output = run_rejection_sample(
        impl,
        draft_token_ids=[3, 5, 2, 4],
        target_argmax_token_ids=[3, 5, 2, 7],
        bonus_token_ids=[10, 11],
        metadata=metadata,
    )

    expected = torch.tensor(
        [
            [3, 5, 10],
            [2, 7, PLACEHOLDER_TOKEN_ID],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(output, expected)


def test_requests_with_fewer_drafts_than_the_padded_length(impl):
    """A request may bring fewer drafts than `num_spec_tokens`, which the impl
    pads to. Its bonus token then lands right after its own last draft."""
    metadata = make_sampling_metadata(
        temperature=None,
        all_greedy=True,
        all_random=False,
    )

    output = run_rejection_sample(
        impl,
        draft_token_ids=[3, 2, 4],
        target_argmax_token_ids=[3, 2, 7],
        bonus_token_ids=[10, 11],
        metadata=metadata,
        num_draft_tokens=[1, 2],
    )

    expected = torch.tensor(
        [
            # One draft, accepted -> bonus token at column 1, not at column 2.
            [3, 10, PLACEHOLDER_TOKEN_ID],
            [2, 7, PLACEHOLDER_TOKEN_ID],
        ],
        dtype=torch.int32,
    )
    assert torch.equal(output, expected)
