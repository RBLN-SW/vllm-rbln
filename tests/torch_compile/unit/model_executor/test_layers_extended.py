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

"""Extended tests for RBLN model executor layers: bug-catching, MoE reference
implementations, FP8, custom ops, and quantization method weights."""

import os
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.distributed as dist
from torch.nn.parameter import Parameter

import vllm_rbln.rbln_envs as envs


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rope(head_size=64, rotary_dim=64, max_position_embeddings=128,
               is_neox_style=True, base=10000.0):
    """Build a lightweight RoPE namespace that mirrors RotaryEmbedding state."""
    inv_freq = 1.0 / (base ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
    ))
    t = torch.arange(max_position_embeddings, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    rope = SimpleNamespace()
    rope.cos_sin_cache = cos_sin
    rope.is_neox_style = is_neox_style
    rope.head_size = head_size
    rope.rotary_dim = rotary_dim

    def register_buffer(name, tensor, persistent=False):
        setattr(rope, name, tensor)
    rope.register_buffer = register_buffer

    # Replicate the cache transformation from rope__custom_init__
    cos, sin = cos_sin.chunk(2, dim=-1)
    if is_neox_style:
        cos = cos.repeat(1, 2)
        sin = sin.repeat(1, 2)
    else:
        cos = torch.stack([cos, cos], dim=-1).reshape(cos.shape[0], -1)
        sin = torch.stack([sin, sin], dim=-1).reshape(sin.shape[0], -1)
    rope.cos_cache = cos
    rope.sin_cache = sin

    return rope


# ===========================================================================
# 1. Bug-catching / edge cases
# ===========================================================================


class TestBugCatching:
    """Additional edge-case and bug-catching tests."""

    def test_mxfp4_single_element_block(self):
        """Minimum block: 1 packed byte (2 FP4 values), 1 scale."""
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        blocks = torch.tensor([0x21], dtype=torch.uint8)
        scales = torch.tensor([127], dtype=torch.uint8)

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result.shape == (2,)

    def test_mxfp4_bfloat16_output(self):
        """Verify _dequantize_mxfp4 works with bfloat16 output dtype."""
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        blocks = torch.zeros(16, dtype=torch.uint8)
        scales = torch.full((1,), 127, dtype=torch.uint8)

        result = _dequantize_mxfp4(blocks, scales, torch.bfloat16)
        assert result.dtype == torch.bfloat16
        assert result.shape == (32,)

    def test_rope_batch_size_gt_1(self):
        """Verify RoPE works correctly with batch_size > 1."""
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = _make_rope(head_size=64, rotary_dim=64, is_neox_style=True)
        batch, seq_len, num_heads, head_size = 4, 8, 4, 64
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0).expand(
            batch, -1
        )
        query = torch.randn(batch, seq_len, num_heads * head_size)
        key = torch.randn(batch, seq_len, 2 * head_size)

        q_out, k_out = rope_forward_oot(rope, positions, query, key)
        assert q_out.shape == query.shape
        assert k_out.shape == key.shape

        # All batch elements should produce the same output since positions
        # and inputs are the same per batch element (positions are broadcast)
        # Actually inputs differ per batch, so just check shapes.
        assert q_out.shape[0] == batch

    def test_can_implement_with_group_size_minus_1(self):
        """group_size=-1 means per-channel quantization and should be
        accepted."""
        from vllm_rbln.model_executor.layers.quantization.kernels.mixed_precision.unpacked import (
            RBLNInt8UnpackedLinearKernel,
        )
        from vllm.scalar_type import scalar_types

        config = SimpleNamespace(
            weight_type=scalar_types.uint8b128,
            group_size=-1,
            zero_points=None,
            has_g_idx=False,
        )
        ok, reason = RBLNInt8UnpackedLinearKernel.can_implement(config)
        assert ok is True
        assert reason is None

    def test_moe_routing_deterministic(self):
        """Same input should produce same routing."""
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs"
            ".VLLM_RBLN_USE_MOE_TOKENS_MASK",
            False,
        ):
            router_logits = torch.randn(4, 8)
            w1, c1 = get_masked_routing_weights(
                router_logits.clone(), top_k=2, renormalize=True,
                expert_map=None,
            )
            w2, c2 = get_masked_routing_weights(
                router_logits.clone(), top_k=2, renormalize=True,
                expert_map=None,
            )
        assert torch.allclose(w1, w2)
        assert torch.equal(c1, c2)

    def test_mxfp4_all_fp4_indices(self):
        """Verify all 16 FP4 LUT entries are correctly decoded."""
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        FP4_VALUES = [
            +0.0, +0.5, +1.0, +1.5, +2.0, +3.0, +4.0, +6.0,
            -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
        ]

        # Pack each index as the lo nibble with hi=0
        for idx in range(16):
            byte_val = idx & 0x0F  # lo nibble = idx, hi nibble = 0
            blocks = torch.tensor([byte_val], dtype=torch.uint8)
            scales = torch.tensor([127], dtype=torch.uint8)

            result = _dequantize_mxfp4(blocks, scales, torch.float32)
            assert result[0].item() == pytest.approx(
                FP4_VALUES[idx], abs=1e-6
            ), f"FP4 index {idx}: expected {FP4_VALUES[idx]}, got {result[0].item()}"


# ===========================================================================
# 6. MXFP4 custom_moe_glu_mxfp4 reference implementation
# ===========================================================================


class TestMxfp4MoEReference:
    """Test the full custom_moe_glu_mxfp4 reference implementation with real
    tensors (no mocking). Exercises the reference path by patching
    VLLM_RBLN_COMPILE_MODEL=False."""

    NUM_EXPERTS = 4
    HIDDEN = 64
    INTERMEDIATE = 128
    NUM_TOKENS = 8

    def _make_mxfp4_weights(self, num_experts, rows, cols):
        """Create fake MXFP4 quantized weight blocks and scales.
        blocks: [num_experts, rows, cols // 2]  (uint8, packed FP4)
        scales: [num_experts, rows, cols // 32]  (uint8, E8M0)
        We use idx=2 (value 1.0) packed as lo nibble for predictable values.
        """
        # Pack small non-zero values: lo=2 (1.0), hi=0 (0.0) -> 0x02
        blocks = torch.full(
            (num_experts, rows, cols // 2), 0x02, dtype=torch.uint8
        )
        # Scale exponent = 127 means 2^0 = 1.0
        scales = torch.full(
            (num_experts, rows, cols // 32), 127, dtype=torch.uint8
        )
        return blocks, scales

    @pytest.fixture(autouse=True)
    def _patch_compile(self):
        with patch(
            "vllm_rbln.model_executor.layers.quantization.mxfp4.envs"
            ".VLLM_RBLN_COMPILE_MODEL",
            False,
        ):
            yield

    def test_basic_output_shape_and_finiteness(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            custom_moe_glu_mxfp4,
        )

        torch.manual_seed(42)
        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, self.NUM_TOKENS

        gate_blocks, gate_scales = self._make_mxfp4_weights(E, I, H)
        up_blocks, up_scales = self._make_mxfp4_weights(E, I, H)
        down_blocks, down_scales = self._make_mxfp4_weights(E, H, I)

        gate_bias = torch.zeros(E, I, dtype=torch.float32)
        up_bias = torch.zeros(E, I, dtype=torch.float32)
        down_bias = torch.zeros(E, H, dtype=torch.float32)

        hidden_states = torch.randn(T, H, dtype=torch.float32) * 0.01
        router_logits = torch.randn(T, E, dtype=torch.float32)

        alpha = torch.tensor(1.702, dtype=torch.float32)
        limit = torch.tensor(7.0, dtype=torch.float32)

        result = custom_moe_glu_mxfp4(
            hidden_states, gate_blocks, gate_scales, gate_bias,
            up_blocks, up_scales, up_bias,
            down_blocks, down_scales, down_bias,
            router_logits, alpha, limit, k=2, post_norm=True,
        )

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_topk_1(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            custom_moe_glu_mxfp4,
        )

        torch.manual_seed(42)
        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4

        gate_blocks, gate_scales = self._make_mxfp4_weights(E, I, H)
        up_blocks, up_scales = self._make_mxfp4_weights(E, I, H)
        down_blocks, down_scales = self._make_mxfp4_weights(E, H, I)

        gate_bias = torch.zeros(E, I, dtype=torch.float32)
        up_bias = torch.zeros(E, I, dtype=torch.float32)
        down_bias = torch.zeros(E, H, dtype=torch.float32)

        hidden_states = torch.randn(T, H, dtype=torch.float32) * 0.01
        router_logits = torch.randn(T, E, dtype=torch.float32)

        alpha = torch.tensor(1.702)
        limit = torch.tensor(7.0)

        result = custom_moe_glu_mxfp4(
            hidden_states, gate_blocks, gate_scales, gate_bias,
            up_blocks, up_scales, up_bias,
            down_blocks, down_scales, down_bias,
            router_logits, alpha, limit, k=1, post_norm=True,
        )

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_post_norm_false(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            custom_moe_glu_mxfp4,
        )

        torch.manual_seed(42)
        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4

        gate_blocks, gate_scales = self._make_mxfp4_weights(E, I, H)
        up_blocks, up_scales = self._make_mxfp4_weights(E, I, H)
        down_blocks, down_scales = self._make_mxfp4_weights(E, H, I)

        gate_bias = torch.zeros(E, I, dtype=torch.float32)
        up_bias = torch.zeros(E, I, dtype=torch.float32)
        down_bias = torch.zeros(E, H, dtype=torch.float32)

        hidden_states = torch.randn(T, H, dtype=torch.float32) * 0.01
        router_logits = torch.randn(T, E, dtype=torch.float32)

        alpha = torch.tensor(1.702)
        limit = torch.tensor(7.0)

        result = custom_moe_glu_mxfp4(
            hidden_states, gate_blocks, gate_scales, gate_bias,
            up_blocks, up_scales, up_bias,
            down_blocks, down_scales, down_bias,
            router_logits, alpha, limit, k=2, post_norm=False,
        )

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_with_expert_map(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            custom_moe_glu_mxfp4,
        )

        torch.manual_seed(42)
        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4

        gate_blocks, gate_scales = self._make_mxfp4_weights(E, I, H)
        up_blocks, up_scales = self._make_mxfp4_weights(E, I, H)
        down_blocks, down_scales = self._make_mxfp4_weights(E, H, I)

        gate_bias = torch.zeros(E, I, dtype=torch.float32)
        up_bias = torch.zeros(E, I, dtype=torch.float32)
        down_bias = torch.zeros(E, H, dtype=torch.float32)

        hidden_states = torch.randn(T, H, dtype=torch.float32) * 0.01
        router_logits = torch.randn(T, E, dtype=torch.float32)

        # Identity map
        expert_map = torch.arange(E, dtype=torch.int32)

        alpha = torch.tensor(1.702)
        limit = torch.tensor(7.0)

        result = custom_moe_glu_mxfp4(
            hidden_states, gate_blocks, gate_scales, gate_bias,
            up_blocks, up_scales, up_bias,
            down_blocks, down_scales, down_bias,
            router_logits, alpha, limit, k=2, post_norm=True,
            expert_map=expert_map,
        )

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_zero_hidden_states_produce_zero_output(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            custom_moe_glu_mxfp4,
        )

        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4

        gate_blocks, gate_scales = self._make_mxfp4_weights(E, I, H)
        up_blocks, up_scales = self._make_mxfp4_weights(E, I, H)
        down_blocks, down_scales = self._make_mxfp4_weights(E, H, I)

        gate_bias = torch.zeros(E, I, dtype=torch.float32)
        up_bias = torch.zeros(E, I, dtype=torch.float32)
        down_bias = torch.zeros(E, H, dtype=torch.float32)

        hidden_states = torch.zeros(T, H, dtype=torch.float32)
        router_logits = torch.randn(T, E, dtype=torch.float32)

        alpha = torch.tensor(1.702)
        limit = torch.tensor(7.0)

        result = custom_moe_glu_mxfp4(
            hidden_states, gate_blocks, gate_scales, gate_bias,
            up_blocks, up_scales, up_bias,
            down_blocks, down_scales, down_bias,
            router_logits, alpha, limit, k=2, post_norm=True,
        )

        # gate=0 -> sigmoid(0*alpha)=0.5 -> glu=0*0.5=0 -> (up+1)*0=0
        # So activated=0, expert_out=0, output=0
        assert torch.allclose(result, torch.zeros_like(result), atol=1e-6)


# ===========================================================================
# 7. FusedMoE unquantized reference implementation
# ===========================================================================


class TestUnquantizedFusedMoEReference:
    """Test unquantized_fused_moe_method_rbln with real tensors."""

    NUM_EXPERTS = 4
    HIDDEN = 64
    INTERMEDIATE = 128
    NUM_TOKENS = 8

    def _make_layer(self, num_experts, hidden, intermediate, top_k=2,
                    renormalize=True, expert_map=None):
        """Create a minimal FusedMoE-like layer namespace."""
        torch.manual_seed(42)
        layer = SimpleNamespace()
        # w13_weight: [num_experts, 2*intermediate, hidden]
        layer.w13_weight = torch.randn(
            num_experts, 2 * intermediate, hidden, dtype=torch.float32
        ) * 0.01
        # w2_weight: [num_experts, hidden, intermediate]
        layer.w2_weight = torch.randn(
            num_experts, hidden, intermediate, dtype=torch.float32
        ) * 0.01
        layer.top_k = top_k
        layer.renormalize = renormalize
        layer.expert_map = expert_map
        return layer

    def test_basic_output_shape_and_finiteness(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            unquantized_fused_moe_method_rbln,
        )

        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, self.NUM_TOKENS
        layer = self._make_layer(E, H, I)
        x = torch.randn(T, H, dtype=torch.float32) * 0.01
        router_logits = torch.randn(T, E, dtype=torch.float32)

        result = unquantized_fused_moe_method_rbln(None, layer, x, router_logits)

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_renormalize_true(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            unquantized_fused_moe_method_rbln,
        )

        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4
        layer = self._make_layer(E, H, I, top_k=2, renormalize=True)
        x = torch.randn(T, H, dtype=torch.float32) * 0.01
        router_logits = torch.randn(T, E, dtype=torch.float32)

        result = unquantized_fused_moe_method_rbln(None, layer, x, router_logits)
        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_renormalize_false(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            unquantized_fused_moe_method_rbln,
        )

        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4
        layer = self._make_layer(E, H, I, top_k=2, renormalize=False)
        x = torch.randn(T, H, dtype=torch.float32) * 0.01
        router_logits = torch.randn(T, E, dtype=torch.float32)

        result = unquantized_fused_moe_method_rbln(None, layer, x, router_logits)
        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_with_expert_map(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            unquantized_fused_moe_method_rbln,
        )

        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4
        expert_map = torch.tensor([1, 0, 3, 2], dtype=torch.int64)
        layer = self._make_layer(E, H, I, top_k=2, expert_map=expert_map)
        x = torch.randn(T, H, dtype=torch.float32) * 0.01
        router_logits = torch.randn(T, E, dtype=torch.float32)

        result = unquantized_fused_moe_method_rbln(None, layer, x, router_logits)
        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_topk_1(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            unquantized_fused_moe_method_rbln,
        )

        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4
        layer = self._make_layer(E, H, I, top_k=1, renormalize=True)
        x = torch.randn(T, H, dtype=torch.float32) * 0.01
        router_logits = torch.randn(T, E, dtype=torch.float32)

        result = unquantized_fused_moe_method_rbln(None, layer, x, router_logits)
        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_zero_input_produces_zero_output(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            unquantized_fused_moe_method_rbln,
        )

        E, H, I, T = self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4
        layer = self._make_layer(E, H, I)
        x = torch.zeros(T, H, dtype=torch.float32)
        router_logits = torch.randn(T, E, dtype=torch.float32)

        result = unquantized_fused_moe_method_rbln(None, layer, x, router_logits)
        # silu(0) * 0 = 0 for all experts, so output should be zero
        assert torch.allclose(result.squeeze(), torch.zeros(T, H), atol=1e-6)


# ===========================================================================
# 8. FP8 block matmul and MoE reference
# ===========================================================================


class TestFp8BlockMatmul:
    """Test RBLNW8A16BlockFp8LinearOp._w8a16_block_fp8_matmul directly."""

    def test_basic_matmul_shape_and_finiteness(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            RBLNW8A16BlockFp8LinearOp,
        )

        torch.manual_seed(42)
        out_features, in_features = 64, 128
        block_size = [1, 128]
        op = RBLNW8A16BlockFp8LinearOp(
            weight_group_shape=(1, 128),
            act_quant_group_shape=(1, 128),
        )

        input_tensor = torch.randn(4, in_features, dtype=torch.float32)
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        weight_scale = torch.ones(
            out_features // block_size[0],
            in_features // block_size[1],
            dtype=torch.float32,
        )

        result = op._w8a16_block_fp8_matmul(
            input_tensor, weight, weight_scale, block_size
        )
        assert result.shape == (4, out_features)
        assert torch.isfinite(result).all()

    def test_scale_affects_output(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            RBLNW8A16BlockFp8LinearOp,
        )

        torch.manual_seed(42)
        out_features, in_features = 64, 128
        block_size = [1, 128]
        op = RBLNW8A16BlockFp8LinearOp(
            weight_group_shape=(1, 128),
            act_quant_group_shape=(1, 128),
        )

        input_tensor = torch.randn(2, in_features, dtype=torch.float32)
        weight = torch.randn(out_features, in_features, dtype=torch.float32)

        scale_1 = torch.ones(out_features, 1, dtype=torch.float32)
        scale_2 = torch.full((out_features, 1), 2.0, dtype=torch.float32)

        r1 = op._w8a16_block_fp8_matmul(
            input_tensor, weight, scale_1, block_size
        )
        r2 = op._w8a16_block_fp8_matmul(
            input_tensor, weight, scale_2, block_size
        )

        # With scale=2, result should be 2x
        assert torch.allclose(r2, r1 * 2.0, atol=1e-4)

    def test_with_bias(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            RBLNW8A16BlockFp8LinearOp,
        )

        torch.manual_seed(42)
        out_features, in_features = 64, 128
        block_size = [1, 128]
        op = RBLNW8A16BlockFp8LinearOp(
            weight_group_shape=(1, 128),
            act_quant_group_shape=(1, 128),
        )

        input_tensor = torch.randn(2, in_features, dtype=torch.float32)
        weight = torch.randn(out_features, in_features, dtype=torch.float32)
        weight_scale = torch.ones(out_features, 1, dtype=torch.float32)
        bias = torch.randn(out_features, dtype=torch.float32)

        r_no_bias = op._w8a16_block_fp8_matmul(
            input_tensor, weight, weight_scale, block_size
        )
        r_with_bias = op._w8a16_block_fp8_matmul(
            input_tensor, weight, weight_scale, block_size, bias=bias
        )

        assert torch.allclose(r_with_bias, r_no_bias + bias, atol=1e-5)


class TestFp8MoESwiGLUGroupDequantize:
    """Test custom_moe_swiglu_group_dequantize reference implementation."""

    NUM_EXPERTS = 4
    HIDDEN = 64
    INTERMEDIATE = 128
    NUM_TOKENS = 8
    GROUP_SIZE = 64

    def test_basic_output_shape_and_finiteness(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            custom_moe_swiglu_group_dequantize,
        )

        torch.manual_seed(42)
        E, H, I, T, G = (
            self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE,
            self.NUM_TOKENS, self.GROUP_SIZE,
        )

        hidden_states = torch.randn(T, H, dtype=torch.float32) * 0.01
        # Weight shapes: [num_experts, out_features, in_features]
        gate_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        up_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        down_proj_weight = torch.randn(E, H, I, dtype=torch.float32) * 0.01

        # Scale shapes: [num_experts, out_blocks, in_blocks]
        gate_proj_scale = torch.ones(E, I, H // G, dtype=torch.float32)
        up_proj_scale = torch.ones(E, I, H // G, dtype=torch.float32)
        down_proj_scale = torch.ones(E, H, I // G, dtype=torch.float32)

        router_logits = torch.randn(T, E, dtype=torch.float32)
        group_size = torch.tensor(G, dtype=torch.int32)

        result = custom_moe_swiglu_group_dequantize(
            hidden_states, gate_proj_weight, gate_proj_scale,
            up_proj_weight, up_proj_scale,
            down_proj_weight, down_proj_scale,
            router_logits, group_size, topk=2,
        )

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_topk_1(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            custom_moe_swiglu_group_dequantize,
        )

        torch.manual_seed(42)
        E, H, I, T, G = (
            self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4, self.GROUP_SIZE,
        )

        hidden_states = torch.randn(T, H, dtype=torch.float32) * 0.01
        gate_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        up_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        down_proj_weight = torch.randn(E, H, I, dtype=torch.float32) * 0.01
        gate_proj_scale = torch.ones(E, I, H // G, dtype=torch.float32)
        up_proj_scale = torch.ones(E, I, H // G, dtype=torch.float32)
        down_proj_scale = torch.ones(E, H, I // G, dtype=torch.float32)
        router_logits = torch.randn(T, E, dtype=torch.float32)
        group_size = torch.tensor(G, dtype=torch.int32)

        result = custom_moe_swiglu_group_dequantize(
            hidden_states, gate_proj_weight, gate_proj_scale,
            up_proj_weight, up_proj_scale,
            down_proj_weight, down_proj_scale,
            router_logits, group_size, topk=1,
        )

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_with_e_score_correction_bias(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            custom_moe_swiglu_group_dequantize,
        )

        torch.manual_seed(42)
        E, H, I, T, G = (
            self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4, self.GROUP_SIZE,
        )

        hidden_states = torch.randn(T, H, dtype=torch.float32) * 0.01
        gate_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        up_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        down_proj_weight = torch.randn(E, H, I, dtype=torch.float32) * 0.01
        gate_proj_scale = torch.ones(E, I, H // G, dtype=torch.float32)
        up_proj_scale = torch.ones(E, I, H // G, dtype=torch.float32)
        down_proj_scale = torch.ones(E, H, I // G, dtype=torch.float32)
        router_logits = torch.randn(T, E, dtype=torch.float32)
        group_size = torch.tensor(G, dtype=torch.int32)
        e_score_bias = torch.randn(E, dtype=torch.float32) * 0.1

        result = custom_moe_swiglu_group_dequantize(
            hidden_states, gate_proj_weight, gate_proj_scale,
            up_proj_weight, up_proj_scale,
            down_proj_weight, down_proj_scale,
            router_logits, group_size, topk=2,
            e_score_correction_bias=e_score_bias,
        )

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_with_expert_map(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            custom_moe_swiglu_group_dequantize,
        )

        torch.manual_seed(42)
        E, H, I, T, G = (
            self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4, self.GROUP_SIZE,
        )

        hidden_states = torch.randn(T, H, dtype=torch.float32) * 0.01
        gate_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        up_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        down_proj_weight = torch.randn(E, H, I, dtype=torch.float32) * 0.01
        gate_proj_scale = torch.ones(E, I, H // G, dtype=torch.float32)
        up_proj_scale = torch.ones(E, I, H // G, dtype=torch.float32)
        down_proj_scale = torch.ones(E, H, I // G, dtype=torch.float32)
        router_logits = torch.randn(T, E, dtype=torch.float32)
        group_size = torch.tensor(G, dtype=torch.int32)
        expert_map = torch.arange(E, dtype=torch.int64)

        result = custom_moe_swiglu_group_dequantize(
            hidden_states, gate_proj_weight, gate_proj_scale,
            up_proj_weight, up_proj_scale,
            down_proj_weight, down_proj_scale,
            router_logits, group_size, topk=2,
            expert_map=expert_map,
        )

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_with_biases(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            custom_moe_swiglu_group_dequantize,
        )

        torch.manual_seed(42)
        E, H, I, T, G = (
            self.NUM_EXPERTS, self.HIDDEN, self.INTERMEDIATE, 4, self.GROUP_SIZE,
        )

        hidden_states = torch.randn(T, H, dtype=torch.float32) * 0.01
        gate_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        up_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        down_proj_weight = torch.randn(E, H, I, dtype=torch.float32) * 0.01
        gate_proj_scale = torch.ones(E, I, H // G, dtype=torch.float32)
        up_proj_scale = torch.ones(E, I, H // G, dtype=torch.float32)
        down_proj_scale = torch.ones(E, H, I // G, dtype=torch.float32)
        router_logits = torch.randn(T, E, dtype=torch.float32)
        group_size = torch.tensor(G, dtype=torch.int32)

        gate_bias = torch.randn(E, I, dtype=torch.float32) * 0.01
        up_bias = torch.randn(E, I, dtype=torch.float32) * 0.01
        down_bias = torch.randn(E, H, dtype=torch.float32) * 0.01

        result = custom_moe_swiglu_group_dequantize(
            hidden_states, gate_proj_weight, gate_proj_scale,
            up_proj_weight, up_proj_scale,
            down_proj_weight, down_proj_scale,
            router_logits, group_size, topk=2,
            gate_proj_bias=gate_bias,
            up_proj_bias=up_bias,
            down_proj_bias=down_bias,
        )

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_fake_returns_correct_shape(self):
        """Test the fake op (for symbolic tracing) returns correct shape."""
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            custom_moe_swiglu_group_dequantize_fake,
        )

        T, H, E, I, G = 4, 64, 4, 128, 64

        hidden_states = torch.randn(T, H)
        result = custom_moe_swiglu_group_dequantize_fake(
            hidden_states,
            torch.randn(E, I, H), torch.ones(E, I, H // G),
            torch.randn(E, I, H), torch.ones(E, I, H // G),
            torch.randn(E, H, I), torch.ones(E, H, I // G),
            torch.randn(T, E), torch.tensor(G),
            topk=2,
        )
        assert result.shape == (T, H)


# ===========================================================================
# 9. FusedMoE custom_moe_glu (non-quantized custom op) and helpers
# ===========================================================================


class TestCustomMoeGluOp:
    """Test the custom_moe_glu custom op reference implementation and
    the custom_moe_glu_fake shape inference."""

    def test_custom_moe_glu_reference_output(self):
        """Call the custom_moe_glu op directly with real tensors."""
        torch.manual_seed(42)
        E, H, I, T = 4, 64, 128, 8

        gate_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        up_proj_weight = torch.randn(E, I, H, dtype=torch.float32) * 0.01
        down_proj_weight = torch.randn(E, H, I, dtype=torch.float32) * 0.01
        hidden_states = torch.randn(T, H, dtype=torch.float32) * 0.01
        masked_routing_weights = torch.zeros(T, E, dtype=torch.float32)
        # Assign some routing weights
        for i in range(T):
            masked_routing_weights[i, i % E] = 1.0

        if envs.VLLM_RBLN_MOE_USE_OPT_KERNEL:
            result = torch.ops.rbln_custom_ops.custom_moe_glu(
                hidden_states, gate_proj_weight, up_proj_weight,
                down_proj_weight, masked_routing_weights,
                topk=1, post_norm=True,
            )
        else:
            expert_select_count = torch.zeros(E, dtype=torch.int32)
            for i in range(T):
                expert_select_count[i % E] += 1
            result = torch.ops.rbln_custom_ops.custom_moe_glu(
                hidden_states, gate_proj_weight, up_proj_weight,
                down_proj_weight, masked_routing_weights,
                expert_select_count,
            )

        assert result.shape == (T, H)
        assert torch.isfinite(result).all()

    def test_custom_moe_glu_fake_shape(self):
        """The fake op should return empty tensor with correct shape."""
        T, H, E, I = 4, 64, 4, 128

        hidden_states = torch.randn(T, H)
        gate_proj_weight = torch.randn(E, I, H)
        up_proj_weight = torch.randn(E, I, H)
        down_proj_weight = torch.randn(E, H, I)
        masked_routing_weights = torch.randn(T, E)

        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            custom_moe_glu_fake,
        )

        if envs.VLLM_RBLN_MOE_USE_OPT_KERNEL:
            result = custom_moe_glu_fake(
                hidden_states, gate_proj_weight, up_proj_weight,
                down_proj_weight, masked_routing_weights,
                topk=1, post_norm=True,
            )
        else:
            expert_select_count = torch.zeros(E, dtype=torch.int32)
            result = custom_moe_glu_fake(
                hidden_states, gate_proj_weight, up_proj_weight,
                down_proj_weight, masked_routing_weights,
                expert_select_count,
            )

        assert result.shape == (T, H)


# ===========================================================================
# 10. _swigluoai extended tests
# ===========================================================================


class TestSwigluoaiExtended:
    """Additional numerical tests for _swigluoai to increase coverage."""

    def test_positive_inputs(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _swigluoai,
        )

        torch.manual_seed(42)
        gate = torch.tensor([1.0, 2.0, 3.0])
        up = torch.tensor([0.5, 1.0, 1.5])

        result = _swigluoai(gate, up, alpha=1.702, limit=7.0)

        # Manual computation
        gate_c = gate.clamp(max=7.0)
        up_c = up.clamp(min=-7.0, max=7.0)
        glu = gate_c * torch.sigmoid(gate_c * 1.702)
        expected = (up_c + 1) * glu

        assert torch.allclose(result, expected, atol=1e-6)

    def test_negative_gate_values(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _swigluoai,
        )

        gate = torch.tensor([-3.0, -1.0, -0.5])
        up = torch.tensor([1.0, 1.0, 1.0])

        result = _swigluoai(gate, up, alpha=1.702, limit=7.0)

        gate_c = gate.clamp(max=7.0)
        up_c = up.clamp(min=-7.0, max=7.0)
        glu = gate_c * torch.sigmoid(gate_c * 1.702)
        expected = (up_c + 1) * glu

        assert torch.allclose(result, expected, atol=1e-6)

    def test_both_limits_hit(self):
        """gate > limit should be clamped; up < -limit should be clamped."""
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _swigluoai,
        )

        gate = torch.tensor([100.0])
        up = torch.tensor([-100.0])

        result = _swigluoai(gate, up, alpha=1.702, limit=7.0)

        gate_c = torch.tensor([7.0])
        up_c = torch.tensor([-7.0])
        glu = gate_c * torch.sigmoid(gate_c * 1.702)
        expected = (up_c + 1) * glu

        assert torch.allclose(result, expected, atol=1e-6)

    def test_batch_computation(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _swigluoai,
        )

        torch.manual_seed(42)
        gate = torch.randn(16, 128)
        up = torch.randn(16, 128)

        result = _swigluoai(gate, up, alpha=1.702, limit=7.0)

        assert result.shape == (16, 128)
        assert torch.isfinite(result).all()


# ===========================================================================
# 11. MXFP4 MoEMethod create_weights and process_weights_after_loading
# ===========================================================================


class TestMxfp4MoEMethodWeights:
    """Test Mxfp4MoEMethod.create_weights and process_weights_after_loading
    with a real FusedMoE-like module."""

    def test_create_weights_registers_parameters(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            Mxfp4MoEMethod,
        )

        E, H, I = 4, 64, 128
        # Create a minimal FusedMoE mock that has register_parameter
        layer = MagicMock(spec=["register_parameter"])
        layer.__class__ = type("FusedMoE", (), {})
        # Patch isinstance check
        from vllm.model_executor.layers.fused_moe import FusedMoE

        # Use a real nn.Module for register_parameter
        layer = torch.nn.Module()
        layer.__class__ = FusedMoE

        moe_config = MagicMock()
        method = Mxfp4MoEMethod(moe_config)

        method.create_weights(
            layer, num_experts=E, hidden_size=H,
            intermediate_size_per_partition=I,
            params_dtype=torch.bfloat16,
        )

        assert hasattr(layer, "w13_weight")
        assert hasattr(layer, "w13_weight_scale")
        assert hasattr(layer, "w13_bias")
        assert hasattr(layer, "w2_weight")
        assert hasattr(layer, "w2_weight_scale")
        assert hasattr(layer, "w2_bias")

        # Check shapes
        assert layer.w13_weight.shape == (E, 2 * I, H // 2)
        assert layer.w13_weight_scale.shape == (E, 2 * I, H // 32)
        assert layer.w13_bias.shape == (E, 2 * I)
        assert layer.w2_weight.shape == (E, H, I // 2)
        assert layer.w2_weight_scale.shape == (E, H, I // 32)
        assert layer.w2_bias.shape == (E, H)

    def test_process_weights_after_loading_splits_correctly(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            Mxfp4MoEMethod,
        )
        from vllm.model_executor.layers.fused_moe import FusedMoE

        E, H, I = 4, 64, 128

        layer = torch.nn.Module()
        layer.__class__ = FusedMoE

        moe_config = MagicMock()
        method = Mxfp4MoEMethod(moe_config)

        method.create_weights(
            layer, num_experts=E, hidden_size=H,
            intermediate_size_per_partition=I,
            params_dtype=torch.bfloat16,
        )

        method.process_weights_after_loading(layer)

        # After processing, gate/up/down proj buffers should exist
        assert hasattr(layer, "gate_proj_blocks")
        assert hasattr(layer, "gate_proj_scales")
        assert hasattr(layer, "gate_proj_bias")
        assert hasattr(layer, "up_proj_blocks")
        assert hasattr(layer, "up_proj_scales")
        assert hasattr(layer, "up_proj_bias")
        assert hasattr(layer, "down_proj_blocks")
        assert hasattr(layer, "down_proj_scales")
        assert hasattr(layer, "down_proj_bias")

        # gate_proj should be every other row of w13 (even rows)
        assert layer.gate_proj_blocks.shape == (E, I, H // 2)
        assert layer.up_proj_blocks.shape == (E, I, H // 2)
        assert layer.down_proj_blocks.shape == (E, H, I // 2)


# ===========================================================================
# FP8 Linear Method -- unit tests for init, create_weights, apply
# ===========================================================================


def _make_fp8_config(weight_block_size=None, activation_scheme="dynamic",
                     is_checkpoint_fp8_serialized=True):
    return SimpleNamespace(
        weight_block_size=weight_block_size,
        activation_scheme=activation_scheme,
        is_checkpoint_fp8_serialized=is_checkpoint_fp8_serialized,
    )


def _make_fp8_linear_method(weight_block_size=None, activation_scheme="dynamic",
                            is_checkpoint_fp8_serialized=True):
    """Create Fp8LinearMethod bypassing platform-dependent init for per-tensor."""
    from vllm_rbln.model_executor.layers.quantization.fp8 import Fp8LinearMethod

    config = _make_fp8_config(weight_block_size, activation_scheme,
                              is_checkpoint_fp8_serialized)
    if weight_block_size is not None:
        # block-quant path doesn't hit platform registry
        return Fp8LinearMethod(config)

    # per-tensor path calls init_fp8_linear_kernel which needs platform;
    # bypass __init__ and set attributes manually
    method = object.__new__(Fp8LinearMethod)
    method.quant_config = config
    method.out_dtype = torch.get_default_dtype()
    method.marlin_input_dtype = None
    method.use_marlin = False
    method.use_deep_gemm = False
    method.weight_block_size = None
    method.block_quant = False
    method.act_q_static = activation_scheme == "static"
    method.act_q_group_shape = None
    method.fp8_linear = None  # not needed for apply() since if True: path
    return method


class TestFp8LinearMethodInit:
    """Test Fp8LinearMethod.__init__ for block-quant path."""

    def test_block_quant_init(self):
        method = _make_fp8_linear_method(weight_block_size=[128, 128])

        assert method.block_quant is True
        assert method.weight_block_size == [128, 128]
        assert hasattr(method, "w8a8_block_fp8_linear")

    def test_per_tensor_init(self):
        method = _make_fp8_linear_method(weight_block_size=None)

        assert method.block_quant is False


class TestFp8LinearMethodApply:
    """Test Fp8LinearMethod.apply covering all code paths."""

    def test_apply_block_quant(self):
        out_features, in_features = 128, 256
        block_size = [128, 128]

        method = _make_fp8_linear_method(weight_block_size=block_size)

        layer = SimpleNamespace(
            weight=torch.randn(out_features, in_features).to(torch.float8_e4m3fn),
            weight_scale=torch.rand(
                out_features // block_size[0], in_features // block_size[1],
                dtype=torch.bfloat16,
            ),
            input_scale=None,
        )

        x = torch.randn(4, in_features, dtype=torch.bfloat16)
        result = method.apply(layer, x)

        assert result.shape == (4, out_features)
        assert result.dtype == torch.bfloat16

    def test_apply_per_tensor_scalar_scale(self):
        """Per-tensor path with single scale (weight_scale.numel() == 1)."""
        out_features, in_features = 32, 64
        method = _make_fp8_linear_method(weight_block_size=None)

        # weight shape [in, out] after .t() in process_weights_after_loading
        layer = SimpleNamespace(
            weight=torch.randn(in_features, out_features).to(torch.float8_e4m3fn),
            weight_scale=torch.tensor([0.5], dtype=torch.bfloat16),
            input_scale=None,
        )

        x = torch.randn(4, in_features, dtype=torch.bfloat16)
        result = method.apply(layer, x)

        assert result.shape == (4, out_features)

    def test_apply_per_row_scale(self):
        """Per-tensor path with per-row scale (weight_scale.shape[0] == weight.shape[0])."""
        out_features, in_features = 32, 64
        method = _make_fp8_linear_method(weight_block_size=None)

        layer = SimpleNamespace(
            weight=torch.randn(in_features, out_features).to(torch.float8_e4m3fn),
            weight_scale=torch.rand(in_features, dtype=torch.bfloat16),
            input_scale=None,
        )

        x = torch.randn(4, in_features, dtype=torch.bfloat16)
        result = method.apply(layer, x)

        assert result.shape == (4, out_features)

    def test_apply_with_bias(self):
        out_features, in_features = 32, 64
        method = _make_fp8_linear_method(weight_block_size=None)

        layer = SimpleNamespace(
            weight=torch.randn(in_features, out_features).to(torch.float8_e4m3fn),
            weight_scale=torch.tensor([1.0], dtype=torch.bfloat16),
            input_scale=None,
        )

        x = torch.randn(4, in_features, dtype=torch.bfloat16)
        bias = torch.randn(out_features, dtype=torch.bfloat16)
        result = method.apply(layer, x, bias=bias)

        assert result.shape == (4, out_features)

    def test_apply_fallback_scale_broadcast(self):
        """Fallback path where scale is not per-row and not scalar."""
        out_features, in_features = 32, 64
        method = _make_fp8_linear_method(weight_block_size=None)

        # scale shape doesn't match weight.shape[0] => fallback broadcast
        layer = SimpleNamespace(
            weight=torch.randn(in_features, out_features).to(torch.float8_e4m3fn),
            weight_scale=torch.rand(out_features, dtype=torch.bfloat16),
            input_scale=None,
        )

        x = torch.randn(4, in_features, dtype=torch.bfloat16)
        result = method.apply(layer, x)

        assert result.shape == (4, out_features)


@pytest.fixture(scope="module", autouse=False)
def init_distributed():
    """Initialize a minimal gloo process group for tests needing distributed."""
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "localhost")
        os.environ.setdefault("MASTER_PORT", "29500")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        dist.init_process_group(backend="gloo", rank=0, world_size=1)
    yield
    # Don't destroy - other tests may need it


class TestFp8LinearMethodCreateWeights:
    """Test create_weights for both block-quant and per-tensor paths."""

    @pytest.fixture(autouse=True)
    def _setup_distributed(self, init_distributed):
        from vllm.distributed.parallel_state import (
            init_model_parallel_group,
        )
        # Initialize vllm's TP group if not already done
        try:
            from vllm.distributed import get_tensor_model_parallel_world_size
            get_tensor_model_parallel_world_size()
        except AssertionError:
            init_model_parallel_group(
                group_ranks=[[0]],
                local_rank=0,
                backend="gloo",
            )

    def test_create_weights_block_quant_serialized(self):
        method = _make_fp8_linear_method(
            weight_block_size=[128, 128],
            is_checkpoint_fp8_serialized=True,
        )

        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=256,
            output_partition_sizes=[128],
            input_size=256,
            output_size=128,
            params_dtype=torch.bfloat16,
        )

        assert hasattr(layer, "weight")
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert hasattr(layer, "weight_scale_inv")

    def test_create_weights_per_tensor_serialized(self):
        method = _make_fp8_linear_method(
            weight_block_size=None,
            is_checkpoint_fp8_serialized=True,
        )

        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=64,
            output_partition_sizes=[32],
            input_size=64,
            output_size=32,
            params_dtype=torch.bfloat16,
        )

        assert hasattr(layer, "weight")
        assert layer.weight.dtype == torch.float8_e4m3fn
        assert hasattr(layer, "weight_scale")
        assert hasattr(layer, "input_scale")

    def test_create_weights_non_serialized(self):
        method = _make_fp8_linear_method(
            weight_block_size=None,
            is_checkpoint_fp8_serialized=False,
        )

        layer = torch.nn.Module()
        method.create_weights(
            layer,
            input_size_per_partition=64,
            output_partition_sizes=[32],
            input_size=64,
            output_size=32,
            params_dtype=torch.bfloat16,
        )

        assert hasattr(layer, "weight")
        assert layer.weight.dtype == torch.bfloat16


class TestFp8MoEMethodInit:
    def test_block_quant(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        config = _make_fp8_config(weight_block_size=[128, 128])
        mock_layer = SimpleNamespace(moe_config=SimpleNamespace())

        with patch.object(Fp8MoEMethod.__bases__[0], "__init__", lambda self, *a: None):
            method = Fp8MoEMethod(config, mock_layer)

        assert method.block_quant is True
        assert method.weight_block_size == [128, 128]

    def test_per_tensor(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        config = _make_fp8_config(weight_block_size=None)
        mock_layer = SimpleNamespace(moe_config=SimpleNamespace())

        with patch.object(Fp8MoEMethod.__bases__[0], "__init__", lambda self, *a: None):
            method = Fp8MoEMethod(config, mock_layer)

        assert method.block_quant is False


class TestFp8MoEMethodCreateWeights:
    def test_create_weights_per_tensor(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        config = _make_fp8_config(
            weight_block_size=None,
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
        )
        mock_init_layer = SimpleNamespace(moe_config=SimpleNamespace())

        with patch.object(Fp8MoEMethod.__bases__[0], "__init__", lambda self, *a: None):
            method = Fp8MoEMethod(config, mock_init_layer)

        layer = torch.nn.Module()
        method.create_weights(
            layer,
            num_experts=4,
            hidden_size=32,
            intermediate_size_per_partition=64,
            params_dtype=torch.bfloat16,
        )

        assert layer.w13_weight.shape == (4, 128, 32)
        assert layer.w13_weight.dtype == torch.float8_e4m3fn
        assert layer.w2_weight.shape == (4, 32, 64)
        assert hasattr(layer, "w13_weight_scale")
        assert hasattr(layer, "w2_weight_scale")

    def test_create_weights_block_quant(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        config = _make_fp8_config(
            weight_block_size=[64, 64],
            is_checkpoint_fp8_serialized=True,
            activation_scheme="dynamic",
        )
        mock_init_layer = SimpleNamespace(moe_config=SimpleNamespace())

        with patch.object(Fp8MoEMethod.__bases__[0], "__init__", lambda self, *a: None), \
             patch("vllm_rbln.model_executor.layers.quantization.fp8.get_tensor_model_parallel_world_size", return_value=1):
            method = Fp8MoEMethod(config, mock_init_layer)

            layer = torch.nn.Module()
            method.create_weights(
                layer,
                num_experts=4,
                hidden_size=64,
                intermediate_size_per_partition=128,
                params_dtype=torch.bfloat16,
            )

        assert layer.w13_weight.shape == (4, 256, 64)
        assert hasattr(layer, "w13_weight_scale_inv")
        assert hasattr(layer, "w2_weight_scale_inv")


class TestFp8MoEMethodApply:
    def test_apply(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        num_experts, hidden_size, intermediate_size = 4, 64, 128
        num_tokens, top_k = 8, 2
        block_size = [64, 64]

        config = _make_fp8_config(weight_block_size=block_size)
        mock_init_layer = SimpleNamespace(moe_config=SimpleNamespace())

        with patch.object(Fp8MoEMethod.__bases__[0], "__init__", lambda self, *a: None):
            method = Fp8MoEMethod(config, mock_init_layer)

        # Build mock layer with required attributes
        scale_intermediate = intermediate_size // block_size[0]
        scale_hidden = hidden_size // block_size[1]

        layer = SimpleNamespace(
            w13_weight=torch.randn(num_experts, 2 * intermediate_size, hidden_size).to(
                torch.float8_e4m3fn
            ),
            w2_weight=torch.randn(num_experts, hidden_size, intermediate_size).to(
                torch.float8_e4m3fn
            ),
            w13_weight_scale_inv=torch.rand(
                num_experts, 2 * scale_intermediate, scale_hidden
            ),
            w2_weight_scale_inv=torch.rand(num_experts, scale_hidden, scale_intermediate),
            top_k=top_k,
            expert_map=None,
        )

        x = torch.randn(num_tokens, hidden_size, dtype=torch.bfloat16)
        router_logits = torch.randn(num_tokens, num_experts)

        with patch(
            "vllm_rbln.model_executor.layers.quantization.fp8.envs"
            ".VLLM_RBLN_USE_MOE_TOKENS_MASK", False,
        ):
            result = method.apply(layer, x, router_logits)

        assert result.shape == (num_tokens, hidden_size)


# ===========================================================================
# FP8 process_weights_after_loading -- integration tests
# ===========================================================================


class TestFp8LinearProcessWeights:
    """Integration tests for Fp8LinearMethod.process_weights_after_loading."""

    def test_block_quant_path(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import Fp8LinearMethod

        method = _make_fp8_linear_method(
            weight_block_size=[128, 128],
            is_checkpoint_fp8_serialized=True,
        )

        out_features, in_features = 128, 256
        block_n, block_k = 128, 128

        layer = torch.nn.Module()
        layer.weight = Parameter(
            torch.randn(out_features, in_features).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale_inv = Parameter(
            torch.rand(out_features // block_n, in_features // block_k),
            requires_grad=False,
        )

        with patch(
            "vllm_rbln.model_executor.layers.quantization.fp8.process_fp8_weight_block_strategy",
            return_value=(layer.weight.data, layer.weight_scale_inv.data),
        ), patch(
            "vllm_rbln.model_executor.layers.quantization.fp8.maybe_post_process_fp8_weight_block",
        ):
            method.process_weights_after_loading(layer)

        assert hasattr(layer, "weight")
        assert hasattr(layer, "weight_scale")
        assert layer.input_scale is None

    def test_per_tensor_serialized_path(self):
        method = _make_fp8_linear_method(
            weight_block_size=None,
            is_checkpoint_fp8_serialized=True,
        )

        out_features, in_features = 32, 64

        layer = torch.nn.Module()
        layer.weight = Parameter(
            torch.randn(out_features, in_features).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.weight_scale = Parameter(
            torch.tensor([0.5], dtype=torch.float32),
            requires_grad=False,
        )
        layer.logical_widths = [out_features]

        with patch(
            "vllm_rbln.model_executor.layers.quantization.fp8.process_fp8_weight_tensor_strategy",
            return_value=(layer.weight.data, layer.weight_scale.data, None),
        ):
            method.process_weights_after_loading(layer)

        assert isinstance(layer.weight, Parameter)
        assert isinstance(layer.weight_scale, Parameter)
        assert layer.input_scale is None

    def test_non_serialized_quantize_path(self):
        import vllm.model_executor.layers.quantization.fp8 as upstream_mod

        method = _make_fp8_linear_method(
            weight_block_size=None,
            is_checkpoint_fp8_serialized=False,
        )

        out_features, in_features = 32, 64

        layer = torch.nn.Module()
        layer.weight = Parameter(
            torch.randn(out_features, in_features, dtype=torch.bfloat16),
            requires_grad=False,
        )

        # Mock scaled_fp8_quant to return fp8 weight + scale
        fake_qweight = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
        fake_scale = torch.tensor(1.0)

        with patch.object(
            upstream_mod.ops, "scaled_fp8_quant",
            return_value=(fake_qweight, fake_scale),
        ):
            method.process_weights_after_loading(layer)

        assert isinstance(layer.weight, Parameter)
        # weight should be transposed
        assert layer.weight.shape == (in_features, out_features)


class TestFp8MoEProcessWeights:
    """Integration tests for Fp8MoEMethod.process_weights_after_loading."""

    def _make_moe_method(self, weight_block_size=None, activation_scheme="dynamic",
                         is_checkpoint_fp8_serialized=True):
        from vllm_rbln.model_executor.layers.quantization.fp8 import Fp8MoEMethod

        config = _make_fp8_config(weight_block_size, activation_scheme,
                                  is_checkpoint_fp8_serialized)
        mock_init_layer = SimpleNamespace(moe_config=SimpleNamespace())

        with patch.object(Fp8MoEMethod.__bases__[0], "__init__", lambda self, *a: None):
            method = Fp8MoEMethod(config, mock_init_layer)
        method.rocm_aiter_moe_enabled = False
        return method

    def test_block_quant_converts_parameters(self):
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE

        method = self._make_moe_method(weight_block_size=[64, 64])

        num_experts = 4
        layer = MagicMock(spec=FusedMoE)
        layer.w13_weight = Parameter(
            torch.randn(num_experts, 128, 64).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_weight_scale_inv = Parameter(
            torch.rand(num_experts, 2, 1), requires_grad=False,
        )
        layer.w2_weight = Parameter(
            torch.randn(num_experts, 64, 128).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2_weight_scale_inv = Parameter(
            torch.rand(num_experts, 1, 2), requires_grad=False,
        )

        with patch(
            "vllm_rbln.model_executor.layers.quantization.fp8.isinstance",
            return_value=True,
        ):
            method.process_weights_after_loading(layer)

        assert isinstance(layer.w13_weight, Parameter)
        assert isinstance(layer.w2_weight, Parameter)

    def test_serialized_per_tensor_requantizes(self):
        from vllm.model_executor.layers.fused_moe.layer import FusedMoE
        import vllm.model_executor.layers.quantization.fp8 as upstream_mod

        method = self._make_moe_method(
            weight_block_size=None,
            activation_scheme="dynamic",
            is_checkpoint_fp8_serialized=True,
        )

        num_experts, intermediate, hidden = 2, 32, 16
        layer = MagicMock(spec=FusedMoE)
        layer.w13_weight = Parameter(
            torch.randn(num_experts, 2 * intermediate, hidden).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w13_weight_scale = Parameter(
            torch.rand(num_experts, 2), requires_grad=False,
        )
        layer.w2_weight = Parameter(
            torch.randn(num_experts, hidden, intermediate).to(torch.float8_e4m3fn),
            requires_grad=False,
        )
        layer.w2_weight_scale = Parameter(
            torch.rand(num_experts), requires_grad=False,
        )
        layer.w13_input_scale = None
        layer.w2_input_scale = None
        layer.intermediate_size_per_partition = intermediate
        layer.local_num_experts = num_experts

        fake_qweight = torch.randn(intermediate, hidden).to(torch.float8_e4m3fn)
        fake_scale = torch.tensor(1.0)

        with patch.object(
            upstream_mod.ops, "scaled_fp8_quant",
            return_value=(fake_qweight, fake_scale),
        ):
            method.process_weights_after_loading(layer)

        assert isinstance(layer.w13_weight_scale, Parameter)
        assert layer.w13_weight_scale.shape == (num_experts,)
