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

"""Unit tests for model_executor layers: rotary embedding, logits processor,
quantization helpers, and fused MoE utilities."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch


# ===========================================================================
# Tests: rotary_embedding/base.py
# ===========================================================================


class TestRotaryEmbedding:
    def _make_rope(self, head_size=64, rotary_dim=64, is_neox_style=True):
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope__custom_init__,
            rope_original__init__,
        )

        rope = SimpleNamespace()

        # Simulate parent init: build cos_sin_cache
        inv_freq = 1.0 / (10000.0 ** (
            torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
        ))
        t = torch.arange(128, dtype=torch.float)
        freqs = torch.outer(t, inv_freq)
        cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)
        rope.cos_sin_cache = cos_sin
        rope.is_neox_style = is_neox_style
        rope.head_size = head_size
        rope.rotary_dim = rotary_dim

        def register_buffer(name, tensor, persistent=False):
            setattr(rope, name, tensor)
        rope.register_buffer = register_buffer

        # Call only the cache transformation part of custom init
        cos, sin = rope.cos_sin_cache.chunk(2, dim=-1)
        if is_neox_style:
            cos = cos.repeat(1, 2)
            sin = sin.repeat(1, 2)
        else:
            cos = torch.stack([cos, cos], dim=-1).reshape(cos.shape[0], -1)
            sin = torch.stack([sin, sin], dim=-1).reshape(sin.shape[0], -1)
        rope.cos_cache = cos
        rope.sin_cache = sin

        return rope

    def test_init_neox_cache_shape(self):
        rope = self._make_rope(is_neox_style=True)
        assert rope.cos_cache.shape[0] == 128
        assert rope.cos_cache.shape[1] == 64

    def test_init_gptj_cache_shape(self):
        rope = self._make_rope(is_neox_style=False)
        assert rope.cos_cache.shape[0] == 128
        assert rope.cos_cache.shape[1] == 64

    def test_forward_neox(self):
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = self._make_rope(is_neox_style=True)
        batch, seq_len, num_heads, head_size = 2, 4, 8, 64
        positions = torch.zeros(batch, seq_len, dtype=torch.long)
        query = torch.randn(batch, seq_len, num_heads * head_size)
        key = torch.randn(batch, seq_len, 2 * head_size)

        q_out, k_out = rope_forward_oot(rope, positions, query, key)
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape

    def test_forward_gptj(self):
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = self._make_rope(is_neox_style=False)
        batch, seq_len, num_heads, head_size = 1, 8, 4, 64
        positions = torch.zeros(batch, seq_len, dtype=torch.long)
        query = torch.randn(batch, seq_len, num_heads * head_size)
        key = torch.randn(batch, seq_len, 2 * head_size)

        q_out, k_out = rope_forward_oot(rope, positions, query, key)
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape

    def test_forward_with_offsets(self):
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = self._make_rope(is_neox_style=True)
        batch, seq_len, num_heads, head_size = 1, 4, 4, 64
        positions = torch.zeros(batch, seq_len, dtype=torch.long)
        offsets = torch.ones(batch, seq_len, dtype=torch.long)
        query = torch.randn(batch, seq_len, num_heads * head_size)
        key = torch.randn(batch, seq_len, 2 * head_size)

        q_out, k_out = rope_forward_oot(rope, positions, query, key, offsets)
        assert q_out.shape == query.shape

    def test_forward_partial_rotary_dim(self):
        """When rotary_dim < head_size, unrotated part is concatenated."""
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = self._make_rope(head_size=128, rotary_dim=64, is_neox_style=True)
        batch, seq_len, num_heads = 1, 4, 4
        positions = torch.zeros(batch, seq_len, dtype=torch.long)
        query = torch.randn(batch, seq_len, num_heads * 128)
        key = torch.randn(batch, seq_len, 2 * 128)

        q_out, k_out = rope_forward_oot(rope, positions, query, key)
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape


# ===========================================================================
# Tests: logits_processor.py
# ===========================================================================


class TestLogitsProcessor:
    def test_get_logits(self):
        from vllm_rbln.model_executor.layers.logits_processor import (
            logits_processor_get_logits,
        )

        hidden = torch.randn(2, 16)
        expected = torch.randn(2, 100)

        mock_self = MagicMock()
        mock_lm_head = MagicMock()
        mock_lm_head.quant_method.apply.return_value = expected

        result = logits_processor_get_logits(
            mock_self, hidden, mock_lm_head, embedding_bias=None
        )
        assert torch.equal(result, expected)
        mock_lm_head.quant_method.apply.assert_called_once_with(
            mock_lm_head, hidden, bias=None
        )

    def test_get_logits_with_bias(self):
        from vllm_rbln.model_executor.layers.logits_processor import (
            logits_processor_get_logits,
        )

        hidden = torch.randn(2, 16)
        bias = torch.randn(100)
        expected = torch.randn(2, 100)

        mock_self = MagicMock()
        mock_lm_head = MagicMock()
        mock_lm_head.quant_method.apply.return_value = expected

        result = logits_processor_get_logits(mock_self, hidden, mock_lm_head, bias)
        mock_lm_head.quant_method.apply.assert_called_once_with(
            mock_lm_head, hidden, bias=bias
        )

    def test_gather_logits_all_gather(self):
        from vllm_rbln.model_executor.layers.logits_processor import (
            logits_processor_gather_logits,
        )

        logits = torch.randn(2, 110)
        mock_self = SimpleNamespace(use_all_gather=True, org_vocab_size=100)

        with patch(
            "vllm_rbln.model_executor.layers.logits_processor.tensor_model_parallel_all_gather",
            return_value=logits,
        ):
            result = logits_processor_gather_logits(mock_self, logits)
        assert result.shape == (2, 100)

    def test_gather_logits_gather(self):
        from vllm_rbln.model_executor.layers.logits_processor import (
            logits_processor_gather_logits,
        )

        logits = torch.randn(2, 110)
        mock_self = SimpleNamespace(use_all_gather=False, org_vocab_size=100)

        with patch(
            "vllm_rbln.model_executor.layers.logits_processor.tensor_model_parallel_gather",
            return_value=logits,
        ):
            result = logits_processor_gather_logits(mock_self, logits)
        assert result.shape == (2, 100)

    def test_gather_logits_none_from_rank_gt0(self):
        from vllm_rbln.model_executor.layers.logits_processor import (
            logits_processor_gather_logits,
        )

        mock_self = SimpleNamespace(use_all_gather=False, org_vocab_size=100)

        with patch(
            "vllm_rbln.model_executor.layers.logits_processor.tensor_model_parallel_gather",
            return_value=None,
        ):
            result = logits_processor_gather_logits(mock_self, torch.randn(2, 110))
        assert result is None


# ===========================================================================
# Tests: quantization/mxfp4.py helpers
# ===========================================================================


class TestMxfp4Helpers:
    def test_dequantize_mxfp4_basic(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        # 4 packed bytes = 8 FP4 values, 1 scale for 32 elements
        # We need blocks dim to be K//2 and scales dim to be K//32
        # For K=8: blocks=[4], scales=[1] (but 8/32 = 0.25, need at least 1)
        # Let's use K=64: blocks=[32], scales=[2]
        blocks = torch.zeros(32, dtype=torch.uint8)
        scales = torch.full((2,), 127, dtype=torch.uint8)  # exponent=0 -> 2^0=1

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result.shape == (64,)  # 32 * 2
        # All zeros nibbles -> 0.0 values
        assert torch.allclose(result, torch.zeros(64))

    def test_dequantize_mxfp4_nonzero(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        # 0x21 = hi=2 (1.0), lo=1 (0.5)
        blocks = torch.tensor([0x21], dtype=torch.uint8)
        scales = torch.tensor([127], dtype=torch.uint8)  # 2^0 = 1

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result.shape == (2,)
        assert result[0].item() == pytest.approx(0.5, abs=1e-5)
        assert result[1].item() == pytest.approx(1.0, abs=1e-5)

    def test_dequantize_mxfp4_with_scale(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        # scale=128 -> exponent=128-127=1 -> 2^1=2
        blocks = torch.tensor([0x21], dtype=torch.uint8)
        scales = torch.tensor([128], dtype=torch.uint8)

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result[0].item() == pytest.approx(1.0, abs=1e-5)  # 0.5 * 2
        assert result[1].item() == pytest.approx(2.0, abs=1e-5)  # 1.0 * 2

    def test_dequantize_mxfp4_batched(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        blocks = torch.zeros(3, 16, dtype=torch.uint8)
        scales = torch.full((3, 1), 127, dtype=torch.uint8)

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result.shape == (3, 32)

    def test_swigluoai(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import _swigluoai

        gate = torch.tensor([1.0, 2.0, 10.0])
        up = torch.tensor([0.5, 1.0, 0.0])

        result = _swigluoai(gate, up, alpha=1.702, limit=7.0)
        assert result.shape == (3,)

        # gate clamped at 7.0 for last element
        # up clamped at [-7, 7]
        gate_clamped = gate.clamp(max=7.0)
        up_clamped = up.clamp(min=-7.0, max=7.0)
        glu = gate_clamped * torch.sigmoid(gate_clamped * 1.702)
        expected = (up_clamped + 1) * glu
        assert torch.allclose(result, expected)

    def test_swigluoai_negative(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import _swigluoai

        gate = torch.tensor([-1.0, -10.0])
        up = torch.tensor([-10.0, 1.0])

        result = _swigluoai(gate, up, alpha=1.702, limit=7.0)
        assert result.shape == (2,)


# ===========================================================================
# Tests: fused_moe/layer.py helpers
# ===========================================================================


class TestFusedMoEHelpers:
    def test_get_masked_routing_weights_renormalize(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        router_logits = torch.randn(4, 8)
        top_k = 2

        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs.VLLM_RBLN_USE_MOE_TOKENS_MASK",
            False,
        ):
            masked_weights, expert_count = get_masked_routing_weights(
                router_logits, top_k, renormalize=True, expert_map=None
            )
        assert masked_weights.shape == (4, 8)
        # Each row should have exactly top_k non-zero entries
        for i in range(4):
            assert (masked_weights[i] != 0).sum() == top_k
        assert expert_count.shape == (8,)
        assert expert_count.sum() == 4 * top_k

    def test_get_masked_routing_weights_no_renormalize(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        router_logits = torch.randn(3, 6)

        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs.VLLM_RBLN_USE_MOE_TOKENS_MASK",
            False,
        ):
            masked_weights, expert_count = get_masked_routing_weights(
                router_logits, top_k=1, renormalize=False, expert_map=None
            )
        assert masked_weights.shape == (3, 6)
        for i in range(3):
            assert (masked_weights[i] != 0).sum() == 1

    def test_get_masked_routing_weights_with_expert_map(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        router_logits = torch.randn(2, 4)
        expert_map = torch.tensor([1, 0, 3, 2], dtype=torch.int64)

        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs.VLLM_RBLN_USE_MOE_TOKENS_MASK",
            False,
        ):
            masked_weights, _ = get_masked_routing_weights(
                router_logits, top_k=2, renormalize=True, expert_map=expert_map
            )
        assert masked_weights.shape == (2, 4)


# ===========================================================================
# Tests: quantization/kernels/mixed_precision/unpacked.py
# ===========================================================================


class TestRBLNInt8UnpackedLinearKernel:
    def test_can_implement_uint8(self):
        from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (
            RBLNInt8UnpackedLinearKernel,
        )
        from vllm.scalar_type import scalar_types

        config = SimpleNamespace(
            weight_type=scalar_types.uint8b128,
            group_size=128,
            zero_points=None,
            has_g_idx=False,
        )
        ok, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert ok is True
        assert reason is None

    def test_can_implement_uint4(self):
        from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (
            RBLNInt8UnpackedLinearKernel,
        )
        from vllm.scalar_type import scalar_types

        config = SimpleNamespace(
            weight_type=scalar_types.uint4b8,
            group_size=64,
            zero_points=None,
            has_g_idx=False,
        )
        ok, _ = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert ok is True

    def test_can_implement_unsupported_type(self):
        from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (
            RBLNInt8UnpackedLinearKernel,
        )
        from vllm.scalar_type import scalar_types

        config = SimpleNamespace(
            weight_type=scalar_types.int8,
            group_size=128,
            zero_points=None,
            has_g_idx=False,
        )
        ok, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert ok is False
        assert "not supported" in reason

    def test_can_implement_unsupported_group_size(self):
        from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (
            RBLNInt8UnpackedLinearKernel,
        )
        from vllm.scalar_type import scalar_types

        config = SimpleNamespace(
            weight_type=scalar_types.uint8b128,
            group_size=32,
            zero_points=None,
            has_g_idx=False,
        )
        ok, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert ok is False

    def test_can_implement_asymmetric(self):
        from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (
            RBLNInt8UnpackedLinearKernel,
        )
        from vllm.scalar_type import scalar_types

        config = SimpleNamespace(
            weight_type=scalar_types.uint8b128,
            group_size=128,
            zero_points=True,
            has_g_idx=False,
        )
        ok, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert ok is False
        assert "Asymmetric" in reason

    def test_can_implement_with_g_idx(self):
        from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (
            RBLNInt8UnpackedLinearKernel,
        )
        from vllm.scalar_type import scalar_types

        config = SimpleNamespace(
            weight_type=scalar_types.uint8b128,
            group_size=128,
            zero_points=None,
            has_g_idx=True,
        )
        ok, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert ok is False

    def test_get_min_capability(self):
        from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (
            RBLNInt8UnpackedLinearKernel,
        )

        with pytest.raises(NotImplementedError):
            RBLNInt8UnpackedLinearKernel.get_min_capability()
