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

"""Compile-path tests for sampler operations: verify eager vs
torch.compile(backend='rbln') produce identical results."""

import rebel  # noqa: F401 -- registers "rbln" backend
import torch


def _compile(fn):
    return torch.compile(fn, backend="rbln", dynamic=False)


COMPILE_ATOL = 5e-3
COMPILE_RTOL = 5e-3


class TestSamplerCompile:
    def test_apply_top_k_top_p_greedy(self):
        from vllm_rbln.v1.sample.rbln_sampler import apply_top_k_top_p

        logits = torch.randn(4, 32)

        def fn(l):
            return apply_top_k_top_p(l, k=None, p=None)

        ref = fn(logits.clone())
        compiled = _compile(fn)(logits.clone())
        torch.testing.assert_close(compiled, ref)

    def test_apply_top_k_top_p_with_k(self):
        from vllm_rbln.v1.sample.rbln_sampler import apply_top_k_top_p

        torch.manual_seed(42)
        logits = torch.randn(4, 32)
        k = torch.tensor([3, 3, 3, 3])

        def fn(l, k_val):
            return apply_top_k_top_p(l, k=k_val, p=None)

        compiled = _compile(fn)(logits.clone(), k)
        assert compiled.shape == (4,)
