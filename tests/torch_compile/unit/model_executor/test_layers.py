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

"""Unit tests for model_executor layers.

Tests that can go through torch.compile(backend='rbln') use the compile path
to verify both correctness and compile compatibility. Tests that require mocks
or only validate configs/shapes remain as eager-only tests."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import rebel  # noqa: F401 -- registers "rbln" backend
import pytest
import torch

import vllm_rbln.rbln_envs as envs

COMPILE_ATOL = 5e-3
COMPILE_RTOL = 5e-3


def _compile(fn):
    return torch.compile(fn, backend="rbln", dynamic=False)


# ===========================================================================
# Helpers
# ===========================================================================


def _make_rope(head_size=64, rotary_dim=64, max_position_embeddings=128,
               is_neox_style=True):
    rope = SimpleNamespace()
    inv_freq = 1.0 / (10000.0 ** (
        torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim
    ))
    t = torch.arange(max_position_embeddings, dtype=torch.float)
    freqs = torch.outer(t, inv_freq)
    cos_sin = torch.cat([freqs.cos(), freqs.sin()], dim=-1)

    rope.cos_sin_cache = cos_sin
    rope.is_neox_style = is_neox_style
    rope.head_size = head_size
    rope.rotary_dim = rotary_dim

    def register_buffer(name, tensor, persistent=False):
        setattr(rope, name, tensor)
    rope.register_buffer = register_buffer

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
# Tests: rotary_embedding -- compile path
# ===========================================================================


class TestRotaryEmbedding:
    def test_init_neox_cache_shape(self):
        rope = _make_rope(is_neox_style=True)
        assert rope.cos_cache.shape == (128, 64)

    def test_init_gptj_cache_shape(self):
        rope = _make_rope(is_neox_style=False)
        assert rope.cos_cache.shape == (128, 64)

    def test_forward_neox_compile(self):
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = _make_rope(is_neox_style=True)
        positions = torch.arange(4, dtype=torch.long).unsqueeze(0)
        query = torch.randn(1, 4, 8 * 64)
        key = torch.randn(1, 4, 2 * 64)

        def fn(pos, q, k):
            return rope_forward_oot(rope, pos, q, k)

        q_ref, k_ref = fn(positions, query.clone(), key.clone())
        q_compiled, k_compiled = _compile(fn)(positions, query.clone(), key.clone())

        torch.testing.assert_close(q_compiled, q_ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)
        torch.testing.assert_close(k_compiled, k_ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_forward_gptj_compile(self):
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = _make_rope(is_neox_style=False)
        positions = torch.arange(4, dtype=torch.long).unsqueeze(0)
        query = torch.randn(1, 4, 4 * 64)
        key = torch.randn(1, 4, 2 * 64)

        def fn(pos, q, k):
            return rope_forward_oot(rope, pos, q, k)

        q_ref, k_ref = fn(positions, query.clone(), key.clone())
        q_compiled, k_compiled = _compile(fn)(positions, query.clone(), key.clone())

        torch.testing.assert_close(q_compiled, q_ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)
        torch.testing.assert_close(k_compiled, k_ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_forward_with_offsets_compile(self):
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = _make_rope(max_position_embeddings=256)
        positions = torch.arange(4, dtype=torch.long).unsqueeze(0)
        offsets = torch.full((1, 4), 10, dtype=torch.long)
        query = torch.randn(1, 4, 4 * 64)
        key = torch.randn(1, 4, 2 * 64)

        def fn(pos, q, k, off):
            return rope_forward_oot(rope, pos, q, k, off)

        q_ref, k_ref = fn(positions, query.clone(), key.clone(), offsets)
        q_compiled, k_compiled = _compile(fn)(
            positions, query.clone(), key.clone(), offsets
        )

        torch.testing.assert_close(q_compiled, q_ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)
        torch.testing.assert_close(k_compiled, k_ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_forward_partial_rotary_dim_compile(self):
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = _make_rope(head_size=128, rotary_dim=64)
        positions = torch.arange(4, dtype=torch.long).unsqueeze(0)
        query = torch.randn(1, 4, 2 * 128)
        key = torch.randn(1, 4, 128)

        def fn(pos, q, k):
            return rope_forward_oot(rope, pos, q, k)

        q_ref, k_ref = fn(positions, query.clone(), key.clone())
        q_compiled, k_compiled = _compile(fn)(positions, query.clone(), key.clone())

        torch.testing.assert_close(q_compiled, q_ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)
        torch.testing.assert_close(k_compiled, k_ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)


# ===========================================================================
# Tests: logits_processor -- mock-based (not compilable)
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
# Tests: quantization/mxfp4 helpers -- compile path
# ===========================================================================


class TestMxfp4Helpers:
    def test_dequantize_zeros_compile(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        blocks = torch.zeros(16, dtype=torch.uint8)
        scales = torch.full((1,), 127, dtype=torch.uint8)

        def fn(b, s):
            return _dequantize_mxfp4(b, s, torch.float32)

        ref = fn(blocks, scales)
        compiled = _compile(fn)(blocks, scales)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_dequantize_known_values_compile(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        blocks = torch.tensor([0x21], dtype=torch.uint8)
        scales = torch.tensor([127], dtype=torch.uint8)

        def fn(b, s):
            return _dequantize_mxfp4(b, s, torch.float32)

        ref = fn(blocks, scales)
        compiled = _compile(fn)(blocks, scales)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_dequantize_with_scale_compile(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        blocks = torch.tensor([0x21], dtype=torch.uint8)
        scales = torch.tensor([128], dtype=torch.uint8)

        def fn(b, s):
            return _dequantize_mxfp4(b, s, torch.float32)

        ref = fn(blocks, scales)
        compiled = _compile(fn)(blocks, scales)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_dequantize_batched_compile(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        blocks = torch.randint(0, 256, (4, 8, 32), dtype=torch.uint8)
        scales = torch.randint(100, 140, (4, 8, 2), dtype=torch.uint8)

        def fn(b, s):
            return _dequantize_mxfp4(b, s, torch.float32)

        ref = fn(blocks, scales)
        compiled = _compile(fn)(blocks, scales)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_swigluoai_compile(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import _swigluoai

        gate = torch.randn(8)
        up = torch.randn(8)

        def fn(g, u):
            return _swigluoai(g, u, alpha=1.702, limit=7.0)

        ref = fn(gate, up)
        compiled = _compile(fn)(gate, up)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_swigluoai_negative_compile(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import _swigluoai

        gate = torch.tensor([-1.0, -10.0])
        up = torch.tensor([-10.0, 1.0])

        def fn(g, u):
            return _swigluoai(g, u, alpha=1.702, limit=7.0)

        ref = fn(gate, up)
        compiled = _compile(fn)(gate, up)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)


# ===========================================================================
# Tests: fused_moe routing -- compile path
# ===========================================================================


class TestFusedMoEHelpers:
    def test_routing_renormalize_compile(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        router_logits = torch.randn(4, 8)

        def fn(logits):
            return get_masked_routing_weights(logits, top_k=2, renormalize=True,
                                              expert_map=None)

        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs"
            ".VLLM_RBLN_USE_MOE_TOKENS_MASK", False,
        ):
            ref_w, ref_c = fn(router_logits.clone())
            compiled_w, compiled_c = _compile(fn)(router_logits.clone())

        torch.testing.assert_close(compiled_w, ref_w, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)
        torch.testing.assert_close(compiled_c, ref_c)

    def test_routing_no_renormalize_compile(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        router_logits = torch.randn(4, 8)

        def fn(logits):
            return get_masked_routing_weights(logits, top_k=1, renormalize=False,
                                              expert_map=None)

        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs"
            ".VLLM_RBLN_USE_MOE_TOKENS_MASK", False,
        ):
            ref_w, ref_c = fn(router_logits.clone())
            compiled_w, compiled_c = _compile(fn)(router_logits.clone())

        torch.testing.assert_close(compiled_w, ref_w, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)
        torch.testing.assert_close(compiled_c, ref_c)

    def test_routing_with_expert_map_compile(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        router_logits = torch.randn(2, 4)
        expert_map = torch.tensor([1, 0, 3, 2], dtype=torch.int64)

        def fn(logits):
            return get_masked_routing_weights(logits, top_k=2, renormalize=True,
                                              expert_map=expert_map)

        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs"
            ".VLLM_RBLN_USE_MOE_TOKENS_MASK", False,
        ):
            ref_w, ref_c = fn(router_logits.clone())
            compiled_w, compiled_c = _compile(fn)(router_logits.clone())

        torch.testing.assert_close(compiled_w, ref_w, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)
        torch.testing.assert_close(compiled_c, ref_c)


# ===========================================================================
# Tests: fused MoE forward -- compile path
# ===========================================================================


class TestFusedMoEForwardCompile:
    """Test full MoE forward through compile path."""

    def _make_moe_weights(self, num_experts=4, hidden_size=32,
                          intermediate_size=64):
        """Create fake MoE layer weights."""
        # w13 = [gate_proj; up_proj] fused: [num_experts, 2*intermediate, hidden]
        w13 = torch.randn(num_experts, 2 * intermediate_size, hidden_size)
        # w2 = down_proj: [num_experts, hidden, intermediate]
        w2 = torch.randn(num_experts, hidden_size, intermediate_size)
        return w13, w2

    def test_unquantized_moe_rbln(self):
        """Full unquantized MoE forward (rbln path) through compile."""
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            unquantized_fused_moe_method_rbln,
        )

        num_experts, hidden_size, intermediate_size = 4, 32, 64
        num_tokens, top_k = 8, 2
        w13, w2 = self._make_moe_weights(num_experts, hidden_size, intermediate_size)

        layer = SimpleNamespace(
            w13_weight=w13,
            w2_weight=w2,
            top_k=top_k,
            renormalize=True,
            expert_map=None,
        )
        self_stub = SimpleNamespace()

        x = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def fn(inp, logits):
            return unquantized_fused_moe_method_rbln(self_stub, layer, inp, logits)

        ref = fn(x.clone(), router_logits.clone())
        compiled = _compile(fn)(x.clone(), router_logits.clone())

        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_unquantized_moe_rbln_no_renormalize(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            unquantized_fused_moe_method_rbln,
        )

        num_experts, hidden_size, intermediate_size = 4, 32, 64
        num_tokens, top_k = 4, 1
        w13, w2 = self._make_moe_weights(num_experts, hidden_size, intermediate_size)

        layer = SimpleNamespace(
            w13_weight=w13,
            w2_weight=w2,
            top_k=top_k,
            renormalize=False,
            expert_map=None,
        )
        self_stub = SimpleNamespace()

        x = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def fn(inp, logits):
            return unquantized_fused_moe_method_rbln(self_stub, layer, inp, logits)

        ref = fn(x.clone(), router_logits.clone())
        compiled = _compile(fn)(x.clone(), router_logits.clone())

        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_unquantized_moe_rbln_with_expert_map(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            unquantized_fused_moe_method_rbln,
        )

        num_experts, hidden_size, intermediate_size = 4, 32, 64
        num_tokens, top_k = 6, 2
        w13, w2 = self._make_moe_weights(num_experts, hidden_size, intermediate_size)

        layer = SimpleNamespace(
            w13_weight=w13,
            w2_weight=w2,
            top_k=top_k,
            renormalize=True,
            expert_map=torch.tensor([1, 0, 3, 2], dtype=torch.int64),
        )
        self_stub = SimpleNamespace()

        x = torch.randn(num_tokens, hidden_size)
        router_logits = torch.randn(num_tokens, num_experts)

        def fn(inp, logits):
            return unquantized_fused_moe_method_rbln(self_stub, layer, inp, logits)

        ref = fn(x.clone(), router_logits.clone())
        compiled = _compile(fn)(x.clone(), router_logits.clone())

        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_custom_moe_glu_reference(self):
        """Test custom_moe_glu reference impl through compile path."""
        from vllm_rbln.model_executor.layers.fused_moe.layer import custom_moe_glu

        num_experts, hidden_size, intermediate_size = 4, 32, 64
        num_tokens = 8

        hidden_states = torch.randn(num_tokens, hidden_size)
        gate_proj = torch.randn(num_experts, intermediate_size, hidden_size)
        up_proj = torch.randn(num_experts, intermediate_size, hidden_size)
        down_proj = torch.randn(num_experts, hidden_size, intermediate_size)
        masked_routing = torch.randn(num_tokens, num_experts).softmax(dim=-1)

        if envs.VLLM_RBLN_MOE_USE_OPT_KERNEL:
            def fn(h, g, u, d, r):
                return custom_moe_glu(h, g, u, d, r, topk=2, post_norm=True)
        else:
            expert_count = torch.ones(num_experts, dtype=torch.int32)

            def fn(h, g, u, d, r):
                return custom_moe_glu(h, g, u, d, r, expert_count)

        ref = fn(
            hidden_states.clone(), gate_proj, up_proj, down_proj,
            masked_routing.clone(),
        )
        compiled = _compile(fn)(
            hidden_states.clone(), gate_proj, up_proj, down_proj,
            masked_routing.clone(),
        )

        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)


# ===========================================================================
# Tests: FP8 quantization -- compile path
# ===========================================================================


class TestFp8Compile:
    def test_w8a16_block_fp8_matmul(self):
        """Block FP8 dequant + matmul through compile path."""
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            RBLNW8A16BlockFp8LinearOp,
        )

        out_features, in_features = 64, 128
        block_size = [32, 64]
        batch = 4

        op = RBLNW8A16BlockFp8LinearOp(
            weight_group_shape=(block_size[0], block_size[1]),
            act_quant_group_shape=(1, 128),
        )

        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        weight = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
        weight_scale = torch.rand(
            out_features // block_size[0], in_features // block_size[1],
            dtype=torch.bfloat16,
        )

        def fn(inp, w, ws):
            return op.apply(inp, w, ws)

        ref = fn(x.clone(), weight, weight_scale)
        compiled = _compile(fn)(x.clone(), weight, weight_scale)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_w8a16_block_fp8_matmul_with_bias(self):
        from vllm_rbln.model_executor.layers.quantization.fp8 import (
            RBLNW8A16BlockFp8LinearOp,
        )

        out_features, in_features = 64, 128
        block_size = [32, 64]

        op = RBLNW8A16BlockFp8LinearOp(
            weight_group_shape=(block_size[0], block_size[1]),
            act_quant_group_shape=(1, 128),
        )

        x = torch.randn(2, in_features, dtype=torch.bfloat16)
        weight = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
        weight_scale = torch.rand(
            out_features // block_size[0], in_features // block_size[1],
            dtype=torch.bfloat16,
        )
        bias = torch.randn(out_features, dtype=torch.bfloat16)

        def fn(inp, w, ws, b):
            return op.apply(inp, w, ws, bias=b)

        ref = fn(x.clone(), weight, weight_scale, bias)
        compiled = _compile(fn)(x.clone(), weight, weight_scale, bias)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_fp8_per_tensor_dequant_linear(self):
        """Per-tensor FP8 dequant path: weight * scale → bf16 linear."""
        out_features, in_features = 32, 64
        batch = 4

        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        weight_fp8 = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
        weight_scale = torch.tensor([0.5], dtype=torch.bfloat16)

        def fn(inp, w, ws):
            w_bf16 = w.to(torch.bfloat16) * ws
            return torch.nn.functional.linear(inp, w_bf16)

        ref = fn(x.clone(), weight_fp8, weight_scale)
        compiled = _compile(fn)(x.clone(), weight_fp8, weight_scale)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_fp8_per_row_dequant_linear(self):
        """Per-row FP8 dequant path: weight * scale.unsqueeze(1) → bf16 linear."""
        out_features, in_features = 32, 64
        batch = 4

        x = torch.randn(batch, in_features, dtype=torch.bfloat16)
        weight_fp8 = torch.randn(out_features, in_features).to(torch.float8_e4m3fn)
        weight_scale = torch.rand(out_features, dtype=torch.bfloat16)

        def fn(inp, w, ws):
            w_bf16 = w.to(torch.bfloat16) * ws.unsqueeze(1)
            return torch.nn.functional.linear(inp, w_bf16)

        ref = fn(x.clone(), weight_fp8, weight_scale)
        compiled = _compile(fn)(x.clone(), weight_fp8, weight_scale)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)

    def test_fp8_blockwise_dequantize(self):
        """Blockwise weight dequantization used inside FP8 MoE."""
        num_experts, out_features, in_features = 4, 64, 128
        in_block_size = 64
        out_block_size = 32
        out_blocks = out_features // out_block_size
        in_blocks = in_features // in_block_size

        weight = torch.randn(num_experts, out_features, in_features).to(
            torch.float8_e4m3fn
        )
        scale = torch.rand(num_experts, out_blocks, in_blocks, dtype=torch.bfloat16)

        def fn(w, s):
            # Mirrors _dequantize_blockwise_weight from fp8.py
            expanded = s.repeat_interleave(out_block_size, dim=1).repeat_interleave(
                in_block_size, dim=2
            )
            expanded = expanded[:, :out_features, :in_features]
            return w.to(torch.bfloat16) * expanded
        ref = fn(weight, scale)
        compiled = _compile(fn)(weight, scale)
        torch.testing.assert_close(compiled, ref, atol=COMPILE_ATOL, rtol=COMPILE_RTOL)


# ===========================================================================
# Tests: quantization kernels -- config validation (not compilable)
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
