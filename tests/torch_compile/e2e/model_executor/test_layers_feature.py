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

"""Feature tests for RBLN model executor layers: rotary embedding correctness,
logits processing with scaling, MXFP4 quantization helpers, MoE routing, and
edge-case / bug-catching scenarios."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch

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
# 1. RoPE correctness
# ===========================================================================


class TestRoPECorrectness:
    """Verify rotary embedding numerical correctness and edge cases."""

    def test_neox_cos_sin_cache_shape_matches_rotary_dim(self):
        """cos/sin cache second dim should equal rotary_dim (doubled then
        collapsed back)."""
        for rotary_dim in [32, 64, 128]:
            rope = _make_rope(head_size=rotary_dim, rotary_dim=rotary_dim,
                              is_neox_style=True)
            assert rope.cos_cache.shape[1] == rotary_dim
            assert rope.sin_cache.shape[1] == rotary_dim

    def test_gptj_cos_sin_cache_shape_matches_rotary_dim(self):
        for rotary_dim in [32, 64, 128]:
            rope = _make_rope(head_size=rotary_dim, rotary_dim=rotary_dim,
                              is_neox_style=False)
            assert rope.cos_cache.shape[1] == rotary_dim
            assert rope.sin_cache.shape[1] == rotary_dim

    def test_neox_rotation_identity_at_position_zero(self):
        """At position 0 the frequencies are all cos=1, sin=0 for the base
        component; the doubled cache should still leave query/key unchanged
        when the underlying freqs evaluate to cos=1, sin=0."""
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = _make_rope(head_size=64, rotary_dim=64, is_neox_style=True)
        batch, seq_len, num_heads, head_size = 1, 1, 1, 64
        positions = torch.zeros(batch, seq_len, dtype=torch.long)
        query = torch.randn(batch, seq_len, num_heads * head_size)
        key = torch.randn(batch, seq_len, head_size)

        q_out, k_out = rope_forward_oot(rope, positions, query, key)

        # At position 0, freqs = outer(0, inv_freq) = 0, so cos=1, sin=0
        # rotate_neox(x) flips halves and negates first half, but sin=0
        # so result = query * 1 + rotate(query) * 0 = query
        assert torch.allclose(q_out, query, atol=1e-5)
        assert torch.allclose(k_out, key, atol=1e-5)

    def test_gptj_rotation_identity_at_position_zero(self):
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = _make_rope(head_size=64, rotary_dim=64, is_neox_style=False)
        batch, seq_len, num_heads, head_size = 1, 1, 1, 64
        positions = torch.zeros(batch, seq_len, dtype=torch.long)
        query = torch.randn(batch, seq_len, num_heads * head_size)
        key = torch.randn(batch, seq_len, head_size)

        q_out, k_out = rope_forward_oot(rope, positions, query, key)
        assert torch.allclose(q_out, query, atol=1e-5)
        assert torch.allclose(k_out, key, atol=1e-5)

    def test_neox_vs_gptj_produce_different_embeddings(self):
        """neox and gptj styles should produce different outputs for the same
        input at non-zero positions."""
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        torch.manual_seed(42)
        batch, seq_len, num_heads, head_size = 1, 4, 2, 64
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        query = torch.randn(batch, seq_len, num_heads * head_size)
        key = torch.randn(batch, seq_len, head_size)

        rope_neox = _make_rope(head_size=head_size, rotary_dim=head_size,
                               is_neox_style=True)
        q_neox, k_neox = rope_forward_oot(rope_neox, positions, query.clone(),
                                          key.clone())

        rope_gptj = _make_rope(head_size=head_size, rotary_dim=head_size,
                               is_neox_style=False)
        q_gptj, k_gptj = rope_forward_oot(rope_gptj, positions, query.clone(),
                                           key.clone())

        # They should differ at non-zero positions
        assert not torch.allclose(q_neox[:, 1:], q_gptj[:, 1:], atol=1e-5)

    def test_different_positions_produce_different_embeddings(self):
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = _make_rope(head_size=64, rotary_dim=64, is_neox_style=True)
        batch, num_heads, head_size = 1, 2, 64
        query = torch.randn(batch, 1, num_heads * head_size)
        key = torch.randn(batch, 1, head_size)

        pos0 = torch.zeros(batch, 1, dtype=torch.long)
        pos5 = torch.full((batch, 1), 5, dtype=torch.long)

        q0, _ = rope_forward_oot(rope, pos0, query.clone(), key.clone())
        q5, _ = rope_forward_oot(rope, pos5, query.clone(), key.clone())

        assert not torch.allclose(q0, q5, atol=1e-5)

    def test_offsets_shift_positions(self):
        """positions + offsets should equal using shifted positions directly."""
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        rope = _make_rope(head_size=64, rotary_dim=64, is_neox_style=True,
                          max_position_embeddings=256)
        batch, seq_len, num_heads, head_size = 1, 4, 2, 64
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        offsets = torch.full((batch, seq_len), 10, dtype=torch.long)
        shifted_positions = positions + offsets

        query = torch.randn(batch, seq_len, num_heads * head_size)
        key = torch.randn(batch, seq_len, head_size)

        q_off, k_off = rope_forward_oot(rope, positions, query.clone(),
                                        key.clone(), offsets)
        q_shift, k_shift = rope_forward_oot(rope, shifted_positions,
                                            query.clone(), key.clone())

        assert torch.allclose(q_off, q_shift, atol=1e-5)
        assert torch.allclose(k_off, k_shift, atol=1e-5)

    def test_partial_rotary_dim_preserves_unrotated_part(self):
        """When rotary_dim < head_size, the tail portion of each head should
        remain unchanged."""
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        head_size, rotary_dim = 128, 64
        rope = _make_rope(head_size=head_size, rotary_dim=rotary_dim,
                          is_neox_style=True)
        batch, seq_len, num_heads = 1, 4, 2
        positions = torch.arange(seq_len, dtype=torch.long).unsqueeze(0)
        query = torch.randn(batch, seq_len, num_heads * head_size)
        key = torch.randn(batch, seq_len, head_size)

        q_out, k_out = rope_forward_oot(rope, positions, query.clone(),
                                        key.clone())

        # Unrotated tail of each head should be identical
        q_orig = query.view(batch, seq_len, num_heads, head_size)
        q_result = q_out.view(batch, seq_len, num_heads, head_size)
        assert torch.allclose(q_orig[..., rotary_dim:],
                              q_result[..., rotary_dim:], atol=1e-6)

        k_orig = key.view(batch, seq_len, 1, head_size)
        k_result = k_out.view(batch, seq_len, 1, head_size)
        assert torch.allclose(k_orig[..., rotary_dim:],
                              k_result[..., rotary_dim:], atol=1e-6)

    def test_positions_exceeding_max_position_embeddings_raises(self):
        """Bug scenario: if positions exceed max_position_embeddings, the
        index_select into cos/sin cache should raise an IndexError."""
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        max_pos = 64
        rope = _make_rope(head_size=64, rotary_dim=64,
                          max_position_embeddings=max_pos, is_neox_style=True)
        batch, seq_len, num_heads, head_size = 1, 1, 1, 64
        positions = torch.tensor([[max_pos]], dtype=torch.long)  # out of range
        query = torch.randn(batch, seq_len, num_heads * head_size)
        key = torch.randn(batch, seq_len, head_size)

        with pytest.raises(IndexError):
            rope_forward_oot(rope, positions, query, key)

    def test_head_dim_not_divisible_by_2_raises(self):
        """Bug scenario: rotary_dim must be even for the inv_freq arange.
        With an odd rotary_dim the helper still builds, but forward should
        produce mismatched shapes if rotary_dim is odd."""
        # An odd rotary_dim causes arange(0, rotary_dim, 2) to produce
        # floor(rotary_dim/2) entries, so cos_cache has that width.
        # The forward expects cos_cache width == rotary_dim, leading to a
        # shape mismatch in the multiply.
        from vllm_rbln.model_executor.layers.rotary_embedding.base import (
            rope_forward_oot,
        )

        odd_dim = 63
        rope = _make_rope(head_size=odd_dim, rotary_dim=odd_dim,
                          is_neox_style=True)
        batch, seq_len, num_heads = 1, 1, 1
        positions = torch.zeros(batch, seq_len, dtype=torch.long)
        query = torch.randn(batch, seq_len, num_heads * odd_dim)
        key = torch.randn(batch, seq_len, odd_dim)

        # cos_cache width is 2 * floor(63/2) = 62, but query_rot has dim 63
        # This should fail due to shape mismatch
        with pytest.raises(RuntimeError):
            rope_forward_oot(rope, positions, query, key)


# ===========================================================================
# 2. Logits processor
# ===========================================================================


class TestLogitsProcessorScaling:
    """Test LogitsProcessor scaling behaviour via the upstream forward path."""

    def _build_processor(self, scale=1.0, vocab_size=100, soft_cap=None):
        """Build a real LogitsProcessor with a mock _get_logits."""
        from vllm.model_executor.layers.logits_processor import LogitsProcessor

        proc = LogitsProcessor.__new__(LogitsProcessor)
        # Minimal init without calling super().__init__ which needs platform
        proc.scale = scale
        proc.vocab_size = vocab_size
        proc.org_vocab_size = vocab_size
        proc.logits_as_input = True  # skip _get_logits entirely
        proc.soft_cap = soft_cap
        proc.use_all_gather = False
        return proc

    def test_scale_factor_1_no_change(self):
        proc = self._build_processor(scale=1.0)
        logits = torch.randn(4, 100)
        original = logits.clone()
        result = proc.forward(None, logits)
        assert torch.allclose(result, original)

    def test_scale_factor_2(self):
        proc = self._build_processor(scale=2.0)
        logits = torch.randn(4, 100)
        original = logits.clone()
        result = proc.forward(None, logits)
        assert torch.allclose(result, original * 2.0)

    def test_scale_factor_fractional(self):
        proc = self._build_processor(scale=0.5)
        logits = torch.randn(4, 100)
        original = logits.clone()
        result = proc.forward(None, logits)
        assert torch.allclose(result, original * 0.5)

    def test_soft_cap_applied_before_scale(self):
        """soft_cap: logits = tanh(logits / cap) * cap, then scale."""
        cap = 30.0
        scale = 2.0
        proc = self._build_processor(scale=scale, soft_cap=cap)
        logits = torch.tensor([[50.0, -50.0, 0.0, 10.0]])
        result = proc.forward(None, logits.clone())

        expected = logits / cap
        expected = torch.tanh(expected) * cap
        expected = expected * scale
        assert torch.allclose(result, expected, atol=1e-5)

    def test_zero_scale_zeros_logits(self):
        proc = self._build_processor(scale=0.0)
        logits = torch.randn(4, 100)
        result = proc.forward(None, logits)
        assert torch.allclose(result, torch.zeros_like(logits))


# ===========================================================================
# 3. MXFP4 quantization helpers
# ===========================================================================


class TestMxfp4QuantizationHelpers:
    """Test _dequantize_mxfp4 and _swigluoai for correctness and edge cases."""

    def test_zero_blocks_produce_zero_output(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        blocks = torch.zeros(16, dtype=torch.uint8)
        scales = torch.full((1,), 127, dtype=torch.uint8)  # 2^0
        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result.shape == (32,)
        assert torch.allclose(result, torch.zeros(32))

    def test_roundtrip_known_values(self):
        """Pack known FP4 values and verify dequantize recovers them."""
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        # FP4 LUT index 2 = 1.0, index 1 = 0.5
        # Pack: lo=2 (1.0), hi=1 (0.5) -> byte = (1 << 4) | 2 = 0x12
        blocks = torch.tensor([0x12], dtype=torch.uint8)
        scales = torch.tensor([127], dtype=torch.uint8)  # 2^0 = 1

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result[0].item() == pytest.approx(1.0, abs=1e-6)  # lo nibble idx 2
        assert result[1].item() == pytest.approx(0.5, abs=1e-6)  # hi nibble idx 1

    def test_scale_exponent_effect(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        # idx 2 = 1.0 with scale exponent 3 -> 1.0 * 2^3 = 8.0
        blocks = torch.tensor([0x02], dtype=torch.uint8)
        scales = torch.tensor([130], dtype=torch.uint8)  # 130 - 127 = 3

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result[0].item() == pytest.approx(8.0, abs=1e-5)

    def test_negative_fp4_values(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        # FP4 LUT index 10 = -1.0, index 8 = -0.0
        # Pack: lo=10 (-1.0), hi=8 (-0.0) -> byte = (8 << 4) | 10 = 0x8A
        blocks = torch.tensor([0x8A], dtype=torch.uint8)
        scales = torch.tensor([127], dtype=torch.uint8)

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result[0].item() == pytest.approx(-1.0, abs=1e-6)
        # -0.0 should compare equal to 0.0
        assert result[1].item() == pytest.approx(0.0, abs=1e-6)

    def test_multidimensional_batch(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        # Shape: [num_experts, rows, K//2] with K=64
        blocks = torch.zeros(4, 8, 32, dtype=torch.uint8)
        scales = torch.full((4, 8, 2), 127, dtype=torch.uint8)

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result.shape == (4, 8, 64)

    def test_non_power_of_2_row_dimension(self):
        """Edge case: non-power-of-2 number of rows should still work since
        the quant operates on the last dimension."""
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        # 3 rows, K=32 => blocks=[3, 16], scales=[3, 1]
        blocks = torch.zeros(3, 16, dtype=torch.uint8)
        scales = torch.full((3, 1), 127, dtype=torch.uint8)

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        assert result.shape == (3, 32)

    def test_large_scale_value(self):
        """Large exponent (127 + 10 = 137) -> 2^10 = 1024.  The FP4 value
        0.5 should become 0.5 * 1024 = 512."""
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _dequantize_mxfp4,
        )

        # idx 1 = 0.5
        blocks = torch.tensor([0x01], dtype=torch.uint8)
        scales = torch.tensor([137], dtype=torch.uint8)  # 137 - 127 = 10

        result = _dequantize_mxfp4(blocks, scales, torch.float32)
        expected = 0.5 * (2.0 ** 10)  # 512.0
        assert result[0].item() == pytest.approx(expected, rel=1e-5)

    def test_swigluoai_basic_computation(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _swigluoai,
        )

        gate = torch.tensor([0.0])
        up = torch.tensor([0.0])
        # gate=0 -> sigmoid(0)=0.5 -> glu=0*0.5=0 -> (0+1)*0=0
        result = _swigluoai(gate, up, alpha=1.702, limit=7.0)
        assert result.item() == pytest.approx(0.0, abs=1e-6)

    def test_swigluoai_clamping(self):
        from vllm_rbln.model_executor.layers.quantization.mxfp4 import (
            _swigluoai,
        )

        gate = torch.tensor([100.0])
        up = torch.tensor([100.0])
        result = _swigluoai(gate, up, alpha=1.702, limit=7.0)

        # gate clamped to 7, up clamped to 7
        gate_c = torch.tensor([7.0])
        up_c = torch.tensor([7.0])
        glu = gate_c * torch.sigmoid(gate_c * 1.702)
        expected = (up_c + 1) * glu
        assert torch.allclose(result, expected)


# ===========================================================================
# 4. MoE routing
# ===========================================================================


class TestMoERouting:
    """Test get_masked_routing_weights for correctness."""

    @pytest.fixture(autouse=True)
    def _patch_tokens_mask(self):
        with patch(
            "vllm_rbln.model_executor.layers.fused_moe.layer.envs"
            ".VLLM_RBLN_USE_MOE_TOKENS_MASK",
            False,
        ):
            yield

    def test_returns_correct_tuple(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        router_logits = torch.randn(4, 8)
        weights, counts = get_masked_routing_weights(
            router_logits, top_k=2, renormalize=True, expert_map=None
        )
        assert isinstance(weights, torch.Tensor)
        assert isinstance(counts, torch.Tensor)
        assert weights.shape == (4, 8)
        assert counts.shape == (8,)

    def test_only_top_k_experts_nonzero(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        num_tokens, num_experts, top_k = 8, 16, 3
        router_logits = torch.randn(num_tokens, num_experts)

        weights, counts = get_masked_routing_weights(
            router_logits, top_k=top_k, renormalize=True, expert_map=None
        )

        for i in range(num_tokens):
            assert (weights[i] != 0).sum().item() == top_k

    def test_expert_counts_sum_to_tokens_times_topk(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        num_tokens, num_experts, top_k = 6, 4, 2
        router_logits = torch.randn(num_tokens, num_experts)

        _, counts = get_masked_routing_weights(
            router_logits, top_k=top_k, renormalize=True, expert_map=None
        )
        assert counts.sum().item() == num_tokens * top_k

    def test_renormalize_weights_sum_to_one(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        router_logits = torch.randn(4, 8)
        weights, _ = get_masked_routing_weights(
            router_logits, top_k=2, renormalize=True, expert_map=None
        )

        # Each row's non-zero weights should sum to ~1
        for i in range(4):
            row_sum = weights[i].sum().item()
            assert row_sum == pytest.approx(1.0, abs=1e-5)

    def test_no_renormalize_weights_do_not_sum_to_one(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        # With renormalize=False, softmax is applied over all experts first,
        # then top-k is selected. The selected weights generally don't sum to 1.
        torch.manual_seed(0)
        router_logits = torch.randn(4, 8)
        weights, _ = get_masked_routing_weights(
            router_logits, top_k=2, renormalize=False, expert_map=None
        )
        row_sums = weights.sum(dim=1)
        # At least one row should not sum to 1
        assert not all(
            abs(s.item() - 1.0) < 1e-4 for s in row_sums
        )

    def test_expert_map_remaps_indices(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        # 4 global experts, map: global 0->1, 1->0, 2->3, 3->2
        expert_map = torch.tensor([1, 0, 3, 2], dtype=torch.int64)
        router_logits = torch.randn(2, 4)

        weights, _ = get_masked_routing_weights(
            router_logits, top_k=1, renormalize=True, expert_map=expert_map
        )
        assert weights.shape == (2, 4)
        for i in range(2):
            assert (weights[i] != 0).sum().item() == 1

    def test_varying_num_experts_and_topk(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        for num_experts in [2, 5, 16, 64]:
            for top_k in [1, 2, min(4, num_experts)]:
                router_logits = torch.randn(3, num_experts)
                weights, counts = get_masked_routing_weights(
                    router_logits, top_k=top_k, renormalize=True,
                    expert_map=None,
                )
                assert weights.shape == (3, num_experts)
                assert counts.shape == (num_experts,)
                assert counts.sum().item() == 3 * top_k

    def test_single_token(self):
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        router_logits = torch.randn(1, 8)
        weights, counts = get_masked_routing_weights(
            router_logits, top_k=3, renormalize=True, expert_map=None
        )
        assert (weights[0] != 0).sum().item() == 3
        assert counts.sum().item() == 3

    def test_topk_equals_num_experts(self):
        """When top_k == num_experts, all experts should be selected."""
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        num_experts = 4
        router_logits = torch.randn(2, num_experts)
        weights, counts = get_masked_routing_weights(
            router_logits, top_k=num_experts, renormalize=True,
            expert_map=None,
        )
        for i in range(2):
            assert (weights[i] != 0).sum().item() == num_experts

    def test_expert_map_with_negative_entries(self):
        """Negative expert_map entries (unassigned experts) should be handled
        via the where-clause remapping to n_expert-1."""
        from vllm_rbln.model_executor.layers.fused_moe.layer import (
            get_masked_routing_weights,
        )

        # 4 global experts, only 0 and 2 assigned locally
        expert_map = torch.tensor([0, -1, 1, -1], dtype=torch.int64)
        router_logits = torch.randn(3, 4)

        weights, counts = get_masked_routing_weights(
            router_logits, top_k=2, renormalize=True, expert_map=expert_map
        )
        assert weights.shape == (3, 4)
        assert counts.shape == (4,)


# ===========================================================================
# 5. Bug-catching / edge cases
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
