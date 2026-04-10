"""E2E compile tests for attention operations on RBLN NPU.

Compiles pure-PyTorch attention models via torch.compile(backend="rbln"),
runs on NPU hardware, and compares output with host-computed reference.

This verifies that the RBLN compiler correctly handles attention-pattern
computations (matmul-scale-mask-softmax-matmul) across different head
configurations that correspond to various TP sizes.
"""

import pytest
import torch

# RBLN NPU accumulates in FP16; expect ~5% relative error vs FP32 host reference
NPU_ATOL = 5e-2
NPU_RTOL = 5e-2

# TP head configs: (n_kv_heads, n_groups, head_dim) simulating TP=1,2,4
TP_CONFIGS = [
    pytest.param(1, 4, 64, id="tp1-kv1-g4-d64"),
    pytest.param(2, 2, 64, id="tp2-kv2-g2-d64"),
    pytest.param(4, 1, 64, id="tp4-kv4-g1-d64"),
]


@pytest.fixture(autouse=True)
def reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ---------------------------------------------------------------------------
# Attention models using pure PyTorch ops (compilable to NPU)
# ---------------------------------------------------------------------------

class ScaledDotProductAttention(torch.nn.Module):
    """Standard attention: Q @ K^T -> softmax -> @ V."""

    def forward(self, q, k, v):
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


class MaskedAttention(torch.nn.Module):
    """Attention with explicit mask: Q @ K^T + mask -> softmax -> @ V."""

    def forward(self, q, k, v, mask):
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = attn_weights + mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


class GroupedQueryAttention(torch.nn.Module):
    """GQA: q=[B,H,G,L,D], k/v=[B,H,1,S,D] -> broadcast k/v to G groups."""

    def forward(self, q, k, v):
        n_groups = q.shape[2]
        k = k.expand(-1, -1, n_groups, -1, -1)
        v = v.expand(-1, -1, n_groups, -1, -1)
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


class CausalAttention(torch.nn.Module):
    """Causal attention with triangular mask."""

    def forward(self, q, k, v):
        seq_len = q.shape[-2]
        attn_weights = torch.matmul(q, k.transpose(-2, -1))
        causal_mask = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=q.device), diagonal=1
        )
        attn_weights = attn_weights + causal_mask
        attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_weights, v)


# ---------------------------------------------------------------------------
# E2E compile tests
# ---------------------------------------------------------------------------

class TestScaledDotProductAttentionCompile:

    @pytest.mark.parametrize("n_kv_heads,n_groups,head_dim", TP_CONFIGS)
    def test_sdpa_matches_host(self, n_kv_heads, n_groups, head_dim):
        torch.manual_seed(42)
        n_heads = n_kv_heads * n_groups
        seq_len = 8

        q = torch.randn(1, n_heads, seq_len, head_dim)
        k = torch.randn(1, n_heads, seq_len, head_dim)
        v = torch.randn(1, n_heads, seq_len, head_dim)

        model = ScaledDotProductAttention()
        expected = model(q, k, v)

        compiled = torch.compile(model, backend="rbln", dynamic=False)
        npu_output = compiled(q, k, v)

        assert torch.allclose(npu_output, expected, atol=NPU_ATOL, rtol=NPU_RTOL), (
            f"Max diff: {(npu_output - expected).abs().max().item()}"
        )


class TestMaskedAttentionCompile:

    @pytest.mark.parametrize("n_kv_heads,n_groups,head_dim", TP_CONFIGS)
    def test_masked_attention_matches_host(self, n_kv_heads, n_groups, head_dim):
        torch.manual_seed(42)
        n_heads = n_kv_heads * n_groups
        seq_len = 8

        q = torch.randn(1, n_heads, seq_len, head_dim)
        k = torch.randn(1, n_heads, seq_len, head_dim)
        v = torch.randn(1, n_heads, seq_len, head_dim)
        mask = torch.triu(torch.full((seq_len, seq_len), float("-inf")), diagonal=1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        model = MaskedAttention()
        expected = model(q, k, v, mask)

        compiled = torch.compile(model, backend="rbln", dynamic=False)
        npu_output = compiled(q, k, v, mask)

        assert torch.allclose(npu_output, expected, atol=NPU_ATOL, rtol=NPU_RTOL), (
            f"Max diff: {(npu_output - expected).abs().max().item()}"
        )


class TestGroupedQueryAttentionCompile:

    @pytest.mark.parametrize("n_kv_heads,n_groups,head_dim", TP_CONFIGS)
    def test_gqa_matches_host(self, n_kv_heads, n_groups, head_dim):
        torch.manual_seed(42)
        seq_len = 8

        q = torch.randn(1, n_kv_heads, n_groups, seq_len, head_dim)
        k = torch.randn(1, n_kv_heads, 1, seq_len, head_dim)
        v = torch.randn(1, n_kv_heads, 1, seq_len, head_dim)

        model = GroupedQueryAttention()
        expected = model(q, k, v)

        compiled = torch.compile(model, backend="rbln", dynamic=False)
        npu_output = compiled(q, k, v)

        assert torch.allclose(npu_output, expected, atol=NPU_ATOL, rtol=NPU_RTOL), (
            f"Max diff: {(npu_output - expected).abs().max().item()}"
        )


class TestCausalAttentionCompile:

    @pytest.mark.parametrize("n_kv_heads,n_groups,head_dim", TP_CONFIGS)
    def test_causal_attention_matches_host(self, n_kv_heads, n_groups, head_dim):
        torch.manual_seed(42)
        n_heads = n_kv_heads * n_groups
        seq_len = 8

        q = torch.randn(1, n_heads, seq_len, head_dim)
        k = torch.randn(1, n_heads, seq_len, head_dim)
        v = torch.randn(1, n_heads, seq_len, head_dim)

        model = CausalAttention()
        expected = model(q, k, v)

        compiled = torch.compile(model, backend="rbln", dynamic=False)
        npu_output = compiled(q, k, v)

        assert torch.allclose(npu_output, expected, atol=NPU_ATOL, rtol=NPU_RTOL), (
            f"Max diff: {(npu_output - expected).abs().max().item()}"
        )
