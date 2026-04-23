# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# RBLN MLA backend: minimal port of vLLM FlashAttn MLA using custom ops for
# compilation.  Reuses vLLM MLA common types and metadata builder.

from typing import ClassVar

import torch
from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.attention.mla_attention import (
    MLACommonBackend,
)
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.v1.attention.backend import (
    AttentionType,
    MLAAttentionImpl,
    is_quantized_kv_cache,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum, register_backend

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

from ..flash_attention import (
    RBLNFlashAttentionMetadata,
    RBLNFlashAttentionMetadataBuilder,
)

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Custom ops (stubs for torch.compile — actual kernel provided by RBLN runtime)
# ---------------------------------------------------------------------------


def _fake_mla_output(q: torch.Tensor, kv_c_normed: torch.Tensor) -> torch.Tensor:
    """Return shape: [batch, num_heads, seq_len, kv_lora_rank]."""
    b, num_heads, seq_len, _ = q.shape
    kv_lora_rank = kv_c_normed.shape[-1]
    return torch.empty(
        (b, num_heads, seq_len, kv_lora_rank), device=q.device, dtype=q.dtype
    )


@torch.library.custom_op(
    "rbln_custom_ops::paged_flash_causal_mla_naive_prefill",
    mutates_args=["kv_cache"],
)
def paged_flash_causal_mla_naive_prefill_impl(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _fake_mla_output(q, kv_c_normed)


@torch.library.register_fake("rbln_custom_ops::paged_flash_causal_mla_naive_prefill")
def _(q, kv_c_normed, k_pe, kv_cache, seq_idx, block_tables, scale):
    return _fake_mla_output(q, kv_c_normed)


@torch.library.custom_op(
    "rbln_custom_ops::paged_flash_causal_mla_naive_decode",
    mutates_args=["kv_cache"],
)
def paged_flash_causal_mla_naive_decode_impl(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _fake_mla_output(q, kv_c_normed)


@torch.library.register_fake("rbln_custom_ops::paged_flash_causal_mla_naive_decode")
def _(q, kv_c_normed, k_pe, kv_cache, seq_idx, block_tables, scale):
    return _fake_mla_output(q, kv_c_normed)


# ---------------------------------------------------------------------------
# Backend / Impl
# ---------------------------------------------------------------------------


@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA)
class RBLNFlashAttnMLABackend(MLACommonBackend):
    """MLA backend for RBLN."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]
    accept_output_buffer: bool = False

    @staticmethod
    def get_name() -> str:
        return "RBLN_FLASH_ATTN_MLA"

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNFlashAttentionMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["RBLNFlashAttnMLAImpl"]:
        return RBLNFlashAttnMLAImpl

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [576]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, block_size, head_size)


class RBLNFlashAttnMLAImpl(MLAAttentionImpl[RBLNFlashAttentionMetadata]):
    """RBLN MLA implementation.

    Inherits from MLAAttentionImpl directly because MLACommonImpl.__init__
    requires FlashAttention/FlashInfer which are unavailable on RBLN.
    """

    can_return_lse_for_decode: bool = True

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None,
        attn_type: str,
        kv_sharing_target_layer_name: str | None,
        # MLA specific
        q_lora_rank: int | None,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        qk_head_dim: int,
        v_head_dim: int,
        kv_b_proj: ColumnParallelLinear,
        indexer=None,
        q_pad_num_heads: int | None = None,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        self.kv_cache_dtype = kv_cache_dtype
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.qk_head_dim = qk_head_dim
        self.v_head_dim = v_head_dim
        self.kv_b_proj = kv_b_proj
        self.indexer = indexer
        self.q_pad_num_heads = q_pad_num_heads

        unsupported = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported):
            raise NotImplementedError(
                "FlashAttnMLAImpl does not support alibi_slopes, "
                "sliding_window, or logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Only decoder self-attention is implemented for FlashAttnMLAImpl"
            )
        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashAttnMLA with FP8 KV cache not yet supported"
            )
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in RBLN.")

        vllm_config = get_current_vllm_config()
        self.device = vllm_config.device_config.device
        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len
        self.attn_type = attn_type

        supported_head_sizes = RBLNFlashAttnMLABackend.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by MLA backend. "
                f"Supported: {supported_head_sizes}."
            )

        self.sliding_window = sliding_window
        self.is_causal = envs.VLLM_RBLN_FLASH_CAUSAL_ATTN
        self.scale_tensor = torch.tensor(scale, device=self.device)

    # -- stubs required by MLAAttentionImpl interface -----------------------
    def forward_mha(
        self, q, kv_c_normed, k_pe, kv_c_and_k_pe_cache, attn_metadata, k_scale, output
    ):
        raise NotImplementedError("RBLN MLA backend uses forward() directly")

    def forward_mqa(self, q, kv_c_and_k_pe_cache, attn_metadata, layer):
        raise NotImplementedError("RBLN MLA backend uses forward() directly")

    def process_weights_after_loading(self, act_dtype: torch.dtype):
        pass

    # -- helpers ------------------------------------------------------------
    def _v_up_proj(self, x: torch.Tensor, W_UV: torch.Tensor) -> torch.Tensor:
        """V-up projection.

        Args:
            x: [batch, num_heads, seq_len, kv_lora_rank]
            W_UV: [1, num_heads, kv_lora_rank, v_head_dim]

        Returns:
            [batch, seq_len, num_heads * v_head_dim]
        """
        b_size, num_heads, seq_len, _ = x.size()
        x = torch.matmul(x, W_UV)
        x = x.transpose(1, 2).reshape(b_size, seq_len, num_heads * self.v_head_dim)
        return x

    # -- main forward -------------------------------------------------------
    def forward(
        self,
        layer: torch.nn.Module,
        q: torch.Tensor,
        kv_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RBLNFlashAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward pass.

        Tensor shapes (RBLN convention — batch dim preserved):
            q:           [batch, seq_len, num_heads, qk_head_dim]
            kv_c_normed: [batch, seq_len, kv_lora_rank]
            k_pe:        [batch, seq_len, qk_rope_head_dim]
            kv_cache:    [num_blocks, block_size, head_size]
        """
        b_size, q_len, _, _ = q.size()

        # Q → latent space via W_UK_T: project q_nope down to kv_lora_rank
        # q: [B, S, H, D] → transpose to [B, H, S, D] for matmul
        decode_q_nope, decode_q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        decode_q_nope = decode_q_nope.transpose(1, 2)  # [B, H, S, nope]
        decode_ql_nope = torch.matmul(
            decode_q_nope, layer.W_UK_T
        )  # [B, H, S, lora_rank]
        decode_q_pe = decode_q_pe.transpose(1, 2)  # [B, H, S, rope]
        q = torch.cat(
            [decode_ql_nope, decode_q_pe], dim=-1
        )  # [B, H, S, lora_rank+rope]

        # Dispatch to custom kernel
        if attn_metadata.is_prefill:
            kernel = torch.ops.rbln_custom_ops.paged_flash_causal_mla_naive_prefill
        else:
            kernel = torch.ops.rbln_custom_ops.paged_flash_causal_mla_naive_decode

        attn_output = kernel(
            q,
            kv_c_normed,
            k_pe,
            kv_cache,
            attn_metadata.seq_lens.to(torch.int16),
            attn_metadata.block_tables.to(torch.int16),
            self.scale_tensor,
        )

        # attn_output: [B, H, S, kv_lora_rank] → V-up projection → [B, S, H*v_head_dim]
        return self._v_up_proj(attn_output, layer.W_UV)
