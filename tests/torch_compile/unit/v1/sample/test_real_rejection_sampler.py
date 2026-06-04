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

"""Parity tests for RBLNRejectionSampler against the CPU reference sampler.

Instead of hardcoding expected outputs, we treat the CPU rejection sampler
(`cpu_rejection_sampler.rejection_sample`, upstream-faithful) as the oracle and
assert the RBLN sampler (real compiled NPU primitive — NOT mocked) produces the
SAME `output_token_ids` for identical inputs.

Both `rejection_sample` entry points share the same signature, so we feed them
the same tensors. Determinism: with `all_greedy=True` the CPU path takes the
`target_probs.argmax` branch, and feeding one-hot `target_probs` makes the RBLN
NPU primitive sample that same argmax token — so a correct RBLN implementation
must match the CPU output bit-for-bit.

Two cases:
  - test_uniform_*  : every request has the SAME num_draft_tokens (== max_spec_len)
  - test_varying_*  : requests have DIFFERENT num_draft_tokens, including a
                      zero-draft request that only emits its bonus token

NOTE: requires the `rbln` runtime (NPU), like the other tests under
tests/torch_compile (they import vllm_rbln modules that pull in `rebel`).
"""

from types import SimpleNamespace
import pytest
import torch

from vllm_rbln.v1.sample.cpu_rejection_sampler import (
    rejection_sample as cpu_rejection_sample,
)
from vllm_rbln.v1.sample.rbln_rejection_sampler import RBLNRejectionSampler
from vllm_rbln.v1.sample.rbln_sampler import RBLNSampler


# Module-scoped: CompileContext(use_global_ctx=True) is one-per-process, so
# building a fresh context (and device runtime) per test makes the second
# test's device jobs abort (SYS_TASK_ABORTED / SYS_ENODEV). Share a single
# sampler across the module instead.
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


def _build_inputs(*, num_draft_tokens, draft_token_ids, target_tokens,
                  max_spec_len, bonus_token_ids, vocab_size=64):
    # NOTE: vocab_size must be a multiple of 64 — the rejection_sample NPU
    # primitive (contrib_top_k_top_p_sample) is built on 64-lane splats and
    # the rebel compiler aborts (core dump) on non-multiple-of-64 vocab.
    """Build the shared argument tuple for both rejection_sample entry points.

    `draft_token_ids` / `target_tokens` are in packed layout (concatenation
    over requests, length N = sum(num_draft_tokens)).
    """
    draft = torch.tensor(draft_token_ids, dtype=torch.int32)
    num_tokens = draft.shape[0]
    assert num_tokens == sum(num_draft_tokens) == len(target_tokens)

    cu_num_draft_tokens = torch.tensor(num_draft_tokens, dtype=torch.int64).cumsum(0)
    target_probs = _one_hot(target_tokens, vocab_size)
    bonus = torch.tensor(bonus_token_ids, dtype=torch.int64).reshape(-1, 1)

    # all_greedy -> CPU takes the argmax branch; top_k/top_p None -> RBLN NPU
    # primitive samples the one-hot row verbatim (== argmax).
    sampling_metadata = SimpleNamespace(
        all_greedy=True,
        all_random=False,
        top_k=None,
        top_p=None,
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
        num_draft_tokens=[3, 3],            # identical across requests
        draft_token_ids=[1, 2, 3, 4, 5, 6],  # packed: b0=[1,2,3], b1=[4,5,6]
        target_tokens=[1, 2, 3, 4, 7, 0],   # b1 pos1 mismatches (7 != 5)
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
        num_draft_tokens=[3, 0, 2],      # differ across requests
        draft_token_ids=[1, 2, 3, 4, 5],  # packed: b0=[1,2,3], b1=[], b2=[4,5]
        target_tokens=[1, 2, 3, 6, 0],   # b2 pos0 mismatches (6 != 4)
        max_spec_len=3,
        bonus_token_ids=[9, 8, 7],
    )
    _assert_rbln_matches_cpu(rejection_sampler, inputs)
