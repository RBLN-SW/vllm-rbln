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

"""Compile-path and eager coverage tests for sampler operations.

- TestSamplerCompile: strict mode, compile must succeed
- TestSamplerEager: ops not compilable by RBLN (e.g. exponential_),
  run eagerly for code coverage"""

import rebel  # noqa: F401 -- registers "rbln" backend
import torch

COMPILE_ATOL = 5e-3
COMPILE_RTOL = 5e-3


def _compile(fn):
    """Compile with RBLN backend in strict mode.
    Equivalent to VLLM_RBLN_COMPILE_STRICT_MODE=1."""
    return torch.compile(fn, backend="rbln", dynamic=False,
                         options={"mode": "strict"})


class TestSamplerCompile:
    """Tests that go through torch.compile(backend='rbln') in strict mode."""

    def test_apply_top_k_top_p_greedy(self):
        from vllm_rbln.v1.sample.rbln_sampler import apply_top_k_top_p

        logits = torch.randn(4, 32)

        def fn(l):
            return apply_top_k_top_p(l, k=None, p=None)

        ref = fn(logits.clone())
        compiled = _compile(fn)(logits.clone())
        torch.testing.assert_close(compiled, ref)


class TestSamplerEager:
    """Tests for ops not compilable by RBLN (exponential_ etc.).
    Run eagerly for code coverage."""

    def test_apply_top_k_top_p_with_k(self):
        from vllm_rbln.v1.sample.rbln_sampler import apply_top_k_top_p

        torch.manual_seed(42)
        logits = torch.randn(4, 32)
        k = torch.tensor([3, 3, 3, 3])

        result = apply_top_k_top_p(logits, k=k, p=None)
        assert result.shape == (4,)

    def test_apply_top_k_top_p_with_p(self):
        from vllm_rbln.v1.sample.rbln_sampler import apply_top_k_top_p

        logits = torch.tensor([[10.0, 1.0, 0.5, 0.1, 0.01]])
        p = torch.tensor([0.95])

        result = apply_top_k_top_p(logits, k=None, p=p)
        assert result.shape == (1,)

    def test_apply_top_k_top_p_combined(self):
        from vllm_rbln.v1.sample.rbln_sampler import apply_top_k_top_p

        torch.manual_seed(0)
        logits = torch.randn(4, 20)
        k = torch.tensor([5, 5, 5, 5])
        p = torch.tensor([0.9, 0.9, 0.9, 0.9])

        result = apply_top_k_top_p(logits, k=k, p=p)
        assert result.shape == (4,)

    def test_random_sample(self):
        from vllm_rbln.v1.sample.rbln_sampler import random_sample

        probs = torch.softmax(torch.randn(4, 32), dim=-1)
        result = random_sample(probs, generators={})
        assert result.shape == (4,)
        assert (result >= 0).all() and (result < 32).all()

    def test_random_sample_with_generator(self):
        from vllm_rbln.v1.sample.rbln_sampler import random_sample

        probs = torch.softmax(torch.randn(4, 32), dim=-1)
        gen = torch.Generator().manual_seed(42)
        result = random_sample(probs, generators={1: gen})
        assert result.shape == (4,)
