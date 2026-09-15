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

# forward_oot is RBLN's compile-friendly rewrite of the DeepSeek scaling RoPE and
# must stay numerically equal to upstream's forward_native. Both read the same
# cos/sin cache, so a bare instance with just those attributes suffices.

from typing import Any

import pytest
import torch
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekScalingRotaryEmbedding,
)

from vllm_rbln.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    RBLNDeepseekScalingRotaryEmbedding,
)

_MAX_POS = 32


def _rope(*, head_size=64, rotary_dim=64, is_neox_style=True) -> Any:
    rope: Any = object.__new__(RBLNDeepseekScalingRotaryEmbedding)
    rope.head_size = head_size
    rope.rotary_dim = rotary_dim
    rope.is_neox_style = is_neox_style
    rope.use_aiter = False  # RBLN default; the native path reads it on half dtypes
    # Deterministic fp32 stand-in for the precomputed [max_pos, rotary_dim] cos|sin
    # cache; forward_oot coerces it to the activation dtype at call time.
    rope.cos_sin_cache = torch.linspace(-1.0, 1.0, _MAX_POS * rotary_dim).reshape(
        _MAX_POS, rotary_dim
    )
    return rope


def _qk(num_tokens=4, num_heads=2, head_size=64):
    n = num_tokens * num_heads * head_size
    q = torch.linspace(-2.0, 2.0, n).reshape(num_tokens, num_heads, head_size)
    k = torch.linspace(2.0, -2.0, n).reshape(num_tokens, num_heads, head_size)
    return q, k


class TestForwardOotMatchesNative:
    @pytest.mark.parametrize("is_neox_style", [True, False])
    def test_full_rotary(self, is_neox_style):
        rope = _rope(head_size=64, rotary_dim=64, is_neox_style=is_neox_style)
        q, k = _qk(head_size=64)
        pos = torch.arange(4)
        qo, ko = rope.forward_oot(pos, q.clone(), k.clone())
        qn, kn = rope.forward_native(pos, q.clone(), k.clone())
        assert torch.allclose(qo, qn, atol=1e-5)
        assert torch.allclose(ko, kn, atol=1e-5)

    @pytest.mark.parametrize("is_neox_style", [True, False])
    def test_partial_rotary(self, is_neox_style):
        # rotary_dim < head_size: only the first rotary_dim channels are rotated.
        rope = _rope(head_size=64, rotary_dim=32, is_neox_style=is_neox_style)
        q, k = _qk(head_size=64)
        pos = torch.arange(4)
        qo, ko = rope.forward_oot(pos, q.clone(), k.clone())
        qn, kn = rope.forward_native(pos, q.clone(), k.clone())
        assert torch.allclose(qo, qn, atol=1e-5)
        assert torch.allclose(ko, kn, atol=1e-5)

    def test_with_offsets(self):
        rope = _rope()
        q, k = _qk()
        pos = torch.arange(4)
        offsets = torch.tensor([1, 1, 1, 1])
        qo, ko = rope.forward_oot(pos, q.clone(), k.clone(), offsets)
        qn, kn = rope.forward_native(pos, q.clone(), k.clone(), offsets)
        assert torch.allclose(qo, qn, atol=1e-5)
        assert torch.allclose(ko, kn, atol=1e-5)


class TestForwardOotBehavior:
    def test_non_rotary_tail_passes_through_unchanged(self):
        # For partial rotary the [rotary_dim:] channels are copied through as-is.
        rope = _rope(head_size=64, rotary_dim=32)
        q, k = _qk(head_size=64)
        qo, ko = rope.forward_oot(torch.arange(4), q.clone(), k.clone())
        assert torch.equal(qo[..., 32:], q[..., 32:])
        assert torch.equal(ko[..., 32:], k[..., 32:])

    def test_offsets_are_added_to_positions(self):
        # forward_oot(pos, offsets=d) indexes the cache at pos+d, so it equals
        # forward_oot(pos+d) with no offset.
        rope = _rope()
        q, k = _qk()
        with_offset, _ = rope.forward_oot(
            torch.arange(4), q.clone(), k.clone(), torch.tensor([2, 2, 2, 2])
        )
        shifted, _ = rope.forward_oot(torch.arange(4) + 2, q.clone(), k.clone())
        assert torch.allclose(with_offset, shifted, atol=1e-5)

    def test_neox_and_gptj_styles_differ(self):
        # The two rotation conventions are genuinely different transforms.
        q, k = _qk()
        pos = torch.arange(4)
        neox, _ = _rope(is_neox_style=True).forward_oot(pos, q.clone(), k.clone())
        gptj, _ = _rope(is_neox_style=False).forward_oot(pos, q.clone(), k.clone())
        assert not torch.allclose(neox, gptj)


class TestDtypeHandling:
    @pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_output_dtype_follows_activation(self, dtype):
        # The fp32 cache is coerced to the activation dtype, so a fp16/bf16 query
        # yields that dtype and still matches upstream in it.
        rope = _rope()
        q, k = _qk()
        q, k = q.to(dtype), k.to(dtype)
        pos = torch.arange(4)
        qo, ko = rope.forward_oot(pos, q.clone(), k.clone())
        qn, kn = rope.forward_native(pos, q.clone(), k.clone())
        assert qo.dtype == dtype and ko.dtype == dtype
        assert torch.allclose(qo.float(), qn.float(), atol=5e-3)
        assert torch.allclose(ko.float(), kn.float(), atol=5e-3)


class TestRegistration:
    def test_registered_as_oot_rotary_embedding(self):
        from vllm.model_executor.custom_op import maybe_get_oot_by_class

        assert (
            maybe_get_oot_by_class(DeepseekScalingRotaryEmbedding)
            is RBLNDeepseekScalingRotaryEmbedding
        )
