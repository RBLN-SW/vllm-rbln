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

"""Compile-path tests for LoRA operations: verify eager vs
torch.compile(backend='rbln') produce identical results."""

import rebel  # noqa: F401 -- registers "rbln" backend
import pytest
import torch

from vllm_rbln.lora.inputs import LoRAInputs
from vllm_rbln.lora.mask import LoRAMask

COMPILE_ATOL = 5e-3
COMPILE_RTOL = 5e-3


def _compile(fn):
    return torch.compile(fn, backend="rbln", dynamic=False)


class TestLoRACompile:
    @pytest.fixture
    def wrapper(self):
        from vllm_rbln.lora.punica_wrapper.punica_rbln import PunicaWrapperRBLN

        return PunicaWrapperRBLN(
            max_num_batched_tokens=256, max_batches=8, device="cpu"
        )

    def test_add_lora_embedding(self, wrapper):
        max_loras, rank, hidden, num_tokens = 2, 8, 16, 6
        LoRAMask.set_lora_mask(torch.ones(num_tokens, max_loras * rank))

        lora_b = torch.randn(max_loras, 1, hidden, rank)
        x = torch.randn(num_tokens, rank)

        def fn(y, x_in):
            return wrapper.add_lora_embedding(y, x_in, lora_b)

        ref = fn(torch.zeros(num_tokens, hidden), x.clone())
        compiled = _compile(fn)(torch.zeros(num_tokens, hidden), x.clone())
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_add_lora_logits(self, wrapper):
        max_loras, rank, h_in, vocab, num_tokens = 2, 4, 8, 16, 3
        LoRAMask.set_lora_mask(torch.ones(num_tokens, max_loras * rank))

        lora_a = torch.randn(max_loras, 1, rank, h_in)
        lora_b = torch.randn(max_loras, 1, vocab, rank)
        x = torch.randn(num_tokens, h_in)

        def fn(y, x_in):
            return wrapper.add_lora_logits(y, x_in, lora_a, lora_b, scale=1.0)

        ref = fn(torch.zeros(num_tokens, vocab), x.clone())
        compiled = _compile(fn)(torch.zeros(num_tokens, vocab), x.clone())
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_add_lora_linear_single_slice(self, wrapper):
        max_loras, rank, h_in, h_out, num_tokens = 2, 4, 8, 16, 3
        LoRAMask.set_lora_mask(torch.ones(num_tokens, max_loras * rank))

        lora_a = (torch.randn(max_loras, 1, rank, h_in),)
        lora_b = (torch.randn(max_loras, 1, h_out, rank),)
        x = torch.randn(num_tokens, h_in)

        def fn(y, x_in):
            return wrapper.add_lora_linear(
                y, x_in, lora_a, lora_b, scale=1.0, output_slices=(h_out,)
            )

        ref = fn(torch.zeros(num_tokens, h_out), x.clone())
        compiled = _compile(fn)(torch.zeros(num_tokens, h_out), x.clone())
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_add_lora_linear_multi_slice(self, wrapper):
        max_loras, rank, h_in, h_out = 2, 4, 8, 8
        num_tokens, num_slices = 3, 3
        LoRAMask.set_lora_mask(torch.ones(num_tokens, max_loras * rank))

        lora_a = tuple(torch.randn(max_loras, 1, rank, h_in) for _ in range(num_slices))
        lora_b = tuple(torch.randn(max_loras, 1, h_out, rank) for _ in range(num_slices))
        x = torch.randn(num_tokens, h_in)

        def fn(y, x_in):
            return wrapper.add_lora_linear(
                y, x_in, lora_a, lora_b, scale=0.5,
                output_slices=(h_out,) * num_slices
            )

        ref = fn(torch.zeros(num_tokens, h_out * num_slices), x.clone())
        compiled = _compile(fn)(torch.zeros(num_tokens, h_out * num_slices), x.clone())
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)
