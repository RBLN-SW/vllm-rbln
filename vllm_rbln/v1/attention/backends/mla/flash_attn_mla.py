# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# RBLN MLA backend: minimal port of vLLM FlashAttn MLA using torch SDPA and
# Python cache write. Reuses vLLM MLA common types and metadata builder.

from typing import ClassVar

import torch
from vllm.attention.backends.abstract import AttentionType, is_quantized_kv_cache
from vllm.attention.backends.registry import AttentionBackendEnum, register_backend
from vllm.config import get_current_vllm_config
from vllm.config.cache import CacheDType
from vllm.model_executor.layers.linear import ColumnParallelLinear
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonBaseImpl,
)

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger

from ..flash_attention import (
    RBLNFlashAttentionMetadata,
    RBLNFlashAttentionMetadataBuilder,
)

logger = init_logger(__name__)

import logging

log = logging.getLogger("torch._dynamo")


def _empty_mla_attention_output(
    q: torch.Tensor, kv_c_normed: torch.Tensor | None
) -> torch.Tensor:
    """Shape for MLA kernel output consumed by o_proj: [B, seq, H * v_head_dim]."""
    b, seq_len, num_heads, _ = q.shape
    kv_lora_rank = kv_c_normed.shape[-1]

    device = q.device
    dtype = q.dtype
    return torch.empty(
        (b, seq_len, num_heads, kv_lora_rank), device=device, dtype=dtype
    )


# RBLN custom op (flash causal attention naive prefill/decode w/o attn mask)
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
    """
    Expected tensor shapes:
    - q: [batch, num_tokens, num_heads, _ ]
    - kv_c_normed: [batch, num_tokens, kv_lora_rank]
    - k_pe: [batch, num_tokens, qk_rope_head_dim]
    - kv_cache: [num_blocks, block_size, num_kv_heads(=kv_lora_rank+qk_rope_head_dim)]
      Key and value cache
    - seq_idx: [batch, num_partitions]
      number of already cached tokens in each partition
    - block_tables: [num_partitions,] for prefill,
                    [batch, num_partitions] for decode
    - scale: []
    Returns:
        Tensor: attn_output [batch, seq_len, num_heads, kv_lora_rank]

    batch size is assumed to be 1 for prefill.
    """
    return _empty_mla_attention_output(q, kv_c_normed)


@torch.library.register_fake("rbln_custom_ops::paged_flash_causal_mla_naive_prefill")
def _(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _empty_mla_attention_output(q, kv_c_normed)


@torch.library.custom_op(
    "rbln_custom_ops::flash_causal_mla_naive_decode",
    mutates_args=["kv_cache"],
)
def flash_causal_mla_naive_decode_impl(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _empty_mla_attention_output(q, kv_c_normed)


@torch.library.register_fake("rbln_custom_ops::flash_causal_mla_naive_decode")
def _(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    return _empty_mla_attention_output(q, kv_c_normed)


@register_backend(AttentionBackendEnum.FLASH_ATTN_MLA)
class RBLNFlashAttnMLABackend(MLACommonBackend):
    """MLA backend for RBLN: uses torch SDPA and Python cache write."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

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
        num_kv_heads: int,  # assumed to be 1 for MLA
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        """kv cache shape
        # B - num_blocks
        # S - block_size
        # H - head_size
        """
        return (num_blocks, block_size, head_size)


class RBLNFlashAttnMLAImpl(MLACommonBaseImpl[RBLNFlashAttentionMetadata]):
    """RBLN MLA impl: Python cache write + torch SDPA for decode (prefill TODO).

    Subclasses MLACommonBaseImpl only: MLACommonImpl.__init__ always enters a
    FlashAttention/FlashInfer prefill branch that requires flash_attn_varlen_func,
    which is absent on RBLN. Prefill is not supported here anyway; decode uses
    _forward_decode and forward below.
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
        # MLA Specific Arguments
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
        # Copy from FlashAttnMLAImpl.__init__
        super().__init__(
            num_heads,
            head_size,
            scale,
            num_kv_heads,
            alibi_slopes,
            sliding_window,
            kv_cache_dtype,
            logits_soft_cap,
            attn_type,
            kv_sharing_target_layer_name,
            q_lora_rank,
            kv_lora_rank,
            qk_nope_head_dim,
            qk_rope_head_dim,
            qk_head_dim,
            v_head_dim,
            kv_b_proj,
            indexer,
            q_pad_num_heads,
        )

        unsupported_features = [alibi_slopes, sliding_window, logits_soft_cap]
        if any(unsupported_features):
            raise NotImplementedError(
                "FlashAttnMLAImpl does not support one of the following: "
                "alibi_slopes, sliding_window, logits_soft_cap"
            )
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError(
                "Encoder self-attention and "
                "encoder/decoder cross-attention "
                "are not implemented for "
                "FlashAttnMLAImpl"
            )

        if is_quantized_kv_cache(self.kv_cache_dtype):
            raise NotImplementedError(
                "FlashAttnMLA V1 with FP8 KV cache not yet supported"
            )

        # Copy from RBLNFlashAttentionImpl.__init__
        vllm_config = get_current_vllm_config()
        self.enforce_eager = vllm_config.model_config.enforce_eager
        self.device = vllm_config.device_config.device
        self.block_size = vllm_config.cache_config.block_size
        self.max_model_len = vllm_config.model_config.max_model_len

        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError("KV sharing is not supported in RBLN.")
        if logits_soft_cap is not None:
            logger.warning_once(
                "RBLN Attention Backend does not support logits soft cap. "
                "Outputs may be slightly off."
            )
            logits_soft_cap = None

        supported_head_sizes = RBLNFlashAttnMLABackend.get_supported_head_sizes()
        if head_size not in supported_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by PagedAttention. "
                f"Supported head sizes are: {supported_head_sizes}."
            )
        self.attn_type = attn_type

        # TODO(RBLN): We need to apply sinks attn kernel.
        self.sinks = None
        self.sliding_window = sliding_window
        self.is_causal = envs.VLLM_RBLN_FLASH_CAUSAL_ATTN
        self.is_batch_attention_opt = envs.VLLM_RBLN_BATCH_ATTN_OPT
        self.is_normal = False
        self.scale = torch.tensor([scale], device=self.device)

        if not self.is_batch_attention_opt:
            raise NotImplementedError(
                "Batch attention non-optimization is not supported for MLA"
            )

    def _v_up_proj(self, x: torch.Tensor):
        """V-up projection.

        Args:
            x: torch.size([batch, num_tokens, num_heads, kv_lora_rank])

        Returns:
            torch.size([batch, num_tokens, num_heads, v_head_dim])
        """
        b_size, q_len, num_heads, _ = x.size()
        x = x.view(b_size * q_len, num_heads, -1).transpose(0, 1)
        x = torch.bmm(x, self.W_UV)
        x = x.transpose(0, 1).view(b_size, q_len, num_heads, self.v_head_dim)

        return x

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
        """Forward pass with xFormers and PagedAttention.

        Args:
            q : torch.size([batch, num_tokens, num_heads, qk_head_dim])
            kv_c_normed: torch.size([batch, num_tokens, kv_lora_rank])
            k_pe: torch.size([batch, num_tokens, 1, qk_rope_head_dim])
            kv_cache: torch.size([num_blocks, block_size, num_kv_heads_dim])
            attn_metadata: RBLNFlashAttentionMetadata
            output: torch.size([batch, num_tokens, num_heads * v_head_dim])
            output_scale: torch.size([batch, num_tokens, num_heads * v_head_dim])
            output_block_scale: torch.size([batch, num_tokens, num_heads * v_head_dim])

        Returns:
            attn_out  = (batch_size, seq_len, num_heads, v_head_dim)
        """
        b_size, q_len, num_heads, _ = q.size()

        decode_q_nope, decode_q_pe = q.split(
            [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
        )
        decode_q_nope = decode_q_nope.view(b_size * q_len, num_heads, -1).transpose(
            0, 1
        )

        # Multiply (N, B, P) x (N, P, L) -> (N, B, L)
        decode_ql_nope = torch.bmm(decode_q_nope, self.W_UK_T).view(
            b_size, q_len, num_heads, -1
        )

        q = torch.cat([decode_ql_nope, decode_q_pe], dim=-1)
        k_pe = k_pe.squeeze(2)

        if self.sliding_window is not None:
            raise NotImplementedError(
                "Sliding window attention is not supported for MLA"
            )
        elif not self.is_causal:
            raise NotImplementedError("Non-causal attention is not supported for MLA")
        # actually non-flash paged attention DOES NOT use slot_mapping
        else:
            if self.is_normal:
                raise NotImplementedError("Normal attention is not supported for MLA")
            else:
                if envs.VLLM_RBLN_COMPILE_MODEL:
                    if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
                        raise NotImplementedError(
                            "Triton Custom kernel is not supported for MLA"
                        )
                    else:
                        paged_flash_causal_mla_naive_prefill = (  # noqa: E501
                            torch.ops.rbln_custom_ops.paged_flash_causal_mla_naive_prefill
                        )
                        flash_causal_mla_naive_decode = (  # noqa: E501
                            torch.ops.rbln_custom_ops.flash_causal_mla_naive_decode
                        )
                else:
                    raise NotImplementedError(
                        "Eager execution is not supported for MLA custom kernel"
                    )

                # * batched attention - seq_lens[B, 1] == seq_idx,
                #   original sequence index
                # * otherwise         - seq_lens[B, P] == dyn_size_for_partitions,
                #   dynamic size for each partition
                if not attn_metadata.is_prefill:
                    decode_args = [
                        q,
                        kv_c_normed,
                        k_pe,
                        kv_cache,
                        attn_metadata.seq_lens.to(torch.int16),
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,
                    ]
                    attn_output = flash_causal_mla_naive_decode(  # noqa: E501
                        *decode_args,
                    )
                else:
                    prefill_args = [
                        q,
                        kv_c_normed,
                        k_pe,
                        kv_cache,
                        attn_metadata.seq_lens.to(torch.int16),
                        attn_metadata.block_tables.to(torch.int16),
                        self.scale,
                    ]
                    attn_output = paged_flash_causal_mla_naive_prefill(  # noqa: E501
                        *prefill_args,
                    )
                    # log.info(f"[thkim_debug] q : {q.shape}, kv_c_normed : {kv_c_normed.shape}, k_pe : {k_pe.shape}, kv_cache : {kv_cache.shape}, seq_idx : {attn_metadata.seq_lens.shape}, block_tables : {attn_metadata.block_tables.shape}, scale : {self.scale.shape}")
                    # q : torch.Size([1, 128, 16, 576]),
                    # kv_c_normed : torch.Size([1, 128, 512]),
                    # k_pe : torch.Size([1, 128, 64]),
                    # kv_cache : torch.Size([1461, 8192, 576]),
                    # seq_idx : torch.Size([1, 1]),
                    # block_tables : torch.Size([1]),
                    # scale : torch.Size([])

        # Custom ops return [batch, seq, num_heads, kv_lora_rank] (MLA / o_proj layout).
        expected = (b_size, q_len, self.num_heads, self.kv_lora_rank)
        if attn_output.shape != expected:
            raise ValueError(
                f"MLA attention output shape {tuple(attn_output.shape)} != expected "
                f"{expected}; kernel must return V-space layout for o_proj."
            )

        return self._v_up_proj(attn_output)
