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

from types import SimpleNamespace

import pytest
import torch

from vllm_rbln.v1.sample.cpu_rejection_sampler import (
    rejection_sample as cpu_rejection_sample,
)
from vllm_rbln.v1.sample.rbln_rejection_sampler import RBLNRejectionSampler
from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler


@pytest.fixture(scope="module")
def rejection_sampler(monkeypatch_module):
    monkeypatch_module.setenv("VLLM_RBLN_COMPILE_STRICT_MODE", "1")
    from rebel.compile_context import CompileContext

    seed = 42
    compile_context = CompileContext(use_global_ctx=True, use_weight_sharing=True)
    sampler = RBLNSampler(
        logprobs_mode="raw_logprobs",
        seed=seed,
        compile_context=compile_context,
    )

    return RBLNRejectionSampler(sampler, seed=seed, compile_context=compile_context)


def _one_hot(target_tokens: list[int], vocab_size: int) -> torch.Tensor:
    """(num_tokens, vocab) one-hot probs -> deterministic argmax sampling."""
    probs = torch.zeros((len(target_tokens), vocab_size), dtype=torch.float32)
    for row, tok in enumerate(target_tokens):
        probs[row, tok] = 1.0
    return probs


def _build_inputs(
    *,
    num_draft_tokens,
    draft_token_ids,
    target_tokens,
    max_spec_len,
    bonus_token_ids,
    vocab_size=64,
    top_k=None,
    top_p=None,
):
    # NOTE: vocab_size must be a multiple of 64 because of primitive constraint
    """Build the shared argument tuple for both rejection_sample entry points.

    `draft_token_ids` / `target_tokens` are in packed layout (concatenation
    over requests, length N = sum(num_draft_tokens)).

    When `top_k` / `top_p` (per-batch lists) are given, the metadata switches
    to all_random sampling. The probs stay one-hot, so both paths remain
    deterministic and comparable:
      - RBLN primitive: any top_k >= 1 / top_p filter keeps the hot token
        (it is the row max) and every other surviving prob is 0, so the
        probs/q argmax always picks the hot token.
      - CPU reference: prob[draft] is exactly 1 (accept for any u) or 0
        (reject for any u), and the recovered-token argmax picks the hot
        token, independent of the uniform/exponential draws.
    """
    draft = torch.tensor(draft_token_ids, dtype=torch.int32)
    num_tokens = draft.shape[0]
    batch_size = len(num_draft_tokens)
    assert num_tokens == sum(num_draft_tokens) == len(target_tokens)

    cu_num_draft_tokens = torch.tensor(num_draft_tokens, dtype=torch.int64).cumsum(0)
    target_probs = _one_hot(target_tokens, vocab_size)
    bonus = torch.tensor(bonus_token_ids, dtype=torch.int64).reshape(-1, 1)

    if top_k is None and top_p is None:
        # all_greedy -> CPU takes the argmax branch; top_k/top_p None -> RBLN NPU
        # primitive samples the one-hot row verbatim (== argmax).
        sampling_metadata = SimpleNamespace(
            all_greedy=True,
            all_random=False,
            top_k=None,
            top_p=None,
            generators={},
        )
    else:
        # all_random -> CPU takes the accept-threshold branch
        # (target_probs[draft] >= u); RBLN NPU primitive applies the
        # top_k/top_p masks before sampling. See the docstring above for why
        # one-hot probs keep both paths deterministic.
        sampling_metadata = SimpleNamespace(
            all_greedy=False,
            all_random=True,
            # temperature != 0 -> no request is treated as greedy.
            temperature=torch.ones(batch_size, dtype=torch.float32),
            top_k=(
                torch.tensor(top_k, dtype=torch.int32) if top_k is not None else None
            ),
            top_p=(
                torch.tensor(top_p, dtype=torch.float32) if top_p is not None else None
            ),
            generators={},
        )
    return (
        draft,
        list(num_draft_tokens),
        max_spec_len,
        cu_num_draft_tokens,
        None,  # draft_probs (unused for ngram)
        target_probs,
        bonus,
        sampling_metadata,
    )


def _assert_rbln_matches_cpu(sampler, inputs):
    out_rbln = sampler.rejection_sample(*inputs).cpu().to(torch.int64)
    out_cpu = cpu_rejection_sample(*inputs).cpu().to(torch.int64)
    assert torch.equal(out_rbln, out_cpu), (
        f"\nRBLN:\n{out_rbln}\nCPU (reference):\n{out_cpu}"
    )


def test_uniform_num_draft_tokens(rejection_sampler):
    """Same num_draft_tokens (== max_spec_len) for every request.

    b0: all targets match -> all accepted. b1: mismatch at pos 1 -> recover.
    RBLN output must equal the CPU reference output.
    """
    inputs = _build_inputs(
        num_draft_tokens=[3, 3],  # identical across requests
        draft_token_ids=[1, 2, 3, 4, 5, 6],  # packed: b0=[1,2,3], b1=[4,5,6]
        target_tokens=[1, 2, 3, 4, 7, 0],  # b1 pos1 mismatches (7 != 5)
        max_spec_len=3,
        bonus_token_ids=[9, 8],
    )
    _assert_rbln_matches_cpu(rejection_sampler, inputs)


def test_varying_num_draft_tokens(rejection_sampler):
    """Different num_draft_tokens across requests, including a zero-draft one.

    b0: 3 drafts all accepted. b1: 0 drafts (bonus only). b2: 2 drafts, reject
    at pos 0. RBLN output must equal the CPU reference output.
    """
    inputs = _build_inputs(
        num_draft_tokens=[3, 0, 2],  # differ across requests
        draft_token_ids=[1, 2, 3, 4, 5],  # packed: b0=[1,2,3], b1=[], b2=[4,5]
        target_tokens=[1, 2, 3, 6, 0],  # b2 pos0 mismatches (6 != 4)
        max_spec_len=3,
        bonus_token_ids=[9, 8, 7],
    )
    _assert_rbln_matches_cpu(rejection_sampler, inputs)
