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

from dataclasses import dataclass
from typing import ClassVar

import torch

from vllm.attention.backends.abstract import AttentionLayer, AttentionType
from vllm.config import VllmConfig
from vllm.config.cache import CacheDType
from vllm.v1.attention.backends.mla.common import (
    MLACommonBackend,
    MLACommonDecodeMetadata,
    MLACommonImpl,
    MLACommonMetadata,
    MLACommonMetadataBuilder,
    MLACommonPrefillMetadata,
    QueryLenSupport,
)
from vllm.v1.kv_cache_interface import AttentionSpec

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


def _concat_and_cache_mla_python(
    kv_c: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Python fallback for writing MLA latent + rope into paged cache."""
    # kv_c: (num_tokens, kv_lora_rank), k_pe: (num_tokens, qk_rope_head_dim)
    kv_merged = torch.cat([kv_c, k_pe], dim=-1)
    flat = kv_cache.reshape(-1, kv_cache.size(-1))
    flat[slot_mapping] = kv_merged.to(flat.dtype)


@dataclass
class RBLNFlashAttnMLADecodeMetadata(MLACommonDecodeMetadata):
    """Decode metadata for RBLN MLA (no CUDA graph / FA AOT)."""

    query_start_loc: torch.Tensor
    max_query_len: int
    max_seq_len: int
    scheduler_metadata: torch.Tensor | None = None
    max_num_splits: int = 0


@dataclass
class RBLNFlashAttnMLAMetadata(MLACommonMetadata[RBLNFlashAttnMLADecodeMetadata]):
    """Metadata for RBLN MLA attention."""

    pass


class RBLNFlashAttnMLABackend(MLACommonBackend):
    """MLA backend for RBLN: uses torch SDPA and Python cache write."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.float16, torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["auto"]

    @staticmethod
    def get_name() -> str:
        return "RBLN_FLASH_ATTN_MLA"

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttnMLAMetadataBuilder"]:
        return RBLNFlashAttnMLAMetadataBuilder

    @staticmethod
    def get_impl_cls() -> type["RBLNFlashAttnMLAImpl"]:
        return RBLNFlashAttnMLAImpl


class RBLNFlashAttnMLAMetadataBuilder(MLACommonMetadataBuilder[RBLNFlashAttnMLAMetadata]):
    """Builds RBLN MLA metadata (no FA AOT / cuda graph)."""

    query_len_support: ClassVar[QueryLenSupport] = QueryLenSupport.VARLEN
    reorder_batch_threshold: int = 512

    def __init__(
        self,
        kv_cache_spec: AttentionSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        interleave_size = vllm_config.parallel_config.cp_kv_cache_interleave_size
        super().__init__(
            kv_cache_spec,
            layer_names,
            vllm_config,
            device,
            metadata_cls=RBLNFlashAttnMLAMetadata,
            supports_dcp_with_varlen=(interleave_size == 1),
        )

    def _build_decode(
        self,
        block_table_tensor: torch.Tensor,
        seq_lens_cpu: torch.Tensor,
        seq_lens_device: torch.Tensor,
        query_start_loc_cpu: torch.Tensor,
        query_start_loc_device: torch.Tensor,
        num_decode_tokens: int,
        dcp_tot_seq_lens_device: torch.Tensor | None,
    ) -> RBLNFlashAttnMLADecodeMetadata:
        base = super()._build_decode(
            block_table_tensor=block_table_tensor,
            seq_lens_cpu=seq_lens_cpu,
            seq_lens_device=seq_lens_device,
            query_start_loc_cpu=query_start_loc_cpu,
            query_start_loc_device=query_start_loc_device,
            num_decode_tokens=num_decode_tokens,
            dcp_tot_seq_lens_device=dcp_tot_seq_lens_device,
        )
        query_lens_cpu = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]
        max_query_len = int(query_lens_cpu.max().item())
        max_seq_len = int(seq_lens_cpu.max().item())
        return RBLNFlashAttnMLADecodeMetadata(
            block_table=base.block_table,
            seq_lens=base.seq_lens,
            dcp_tot_seq_lens=base.dcp_tot_seq_lens,
            query_start_loc=query_start_loc_device,
            max_query_len=max_query_len,
            max_seq_len=max_seq_len,
            scheduler_metadata=None,
            max_num_splits=0,
        )


class RBLNFlashAttnMLAImpl(MLACommonImpl[RBLNFlashAttnMLAMetadata]):
    """RBLN MLA impl: Python cache write + torch SDPA for decode (prefill TODO)."""

    can_return_lse_for_decode: bool = True

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.kv_cache_dtype.startswith("fp8"):
            raise NotImplementedError(
                "RBLN MLA does not support FP8 KV cache yet."
            )

    def _forward_decode(
        self,
        q: torch.Tensor | tuple[torch.Tensor, torch.Tensor],
        kv_c_and_k_pe_cache: torch.Tensor,
        attn_metadata: RBLNFlashAttnMLAMetadata,
        layer: AttentionLayer,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Decode path using torch SDPA (no flash kernel)."""
        assert attn_metadata.decode is not None
        dec = attn_metadata.decode

        if isinstance(q, tuple):
            q_nope, q_pe = q
        else:
            q_nope, q_pe = torch.split(
                q, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1
            )

        kv_c_cache = kv_c_and_k_pe_cache[..., : self.kv_lora_rank]
        k_pe_cache = kv_c_and_k_pe_cache[..., self.kv_lora_rank :]

        # Gather from paged cache for each request and run SDPA
        batch = q_pe.size(0)
        max_seqlen_k = dec.max_seq_len
        head_dim_qk = self.qk_rope_head_dim
        head_dim_v = self.kv_lora_rank

        # Simplified: assume contiguous cache layout per request for now
        # (num_blocks, block_size, head_size) -> gather by block_table + seq_lens
        num_blocks, block_size, head_size = kv_c_and_k_pe_cache.shape
        scale = self.scale

        outputs = []
        lse_list = []

        for b in range(batch):
            seq_len_k = int(dec.seq_lens[b].item())
            if seq_len_k == 0:
                out_b = torch.zeros(
                    1, self.num_heads, head_dim_v,
                    device=q_pe.device, dtype=q_pe.dtype
                )
                outputs.append(out_b)
                if self.need_to_return_lse_for_decode:
                    lse_list.append(torch.zeros(1, self.num_heads, device=q_pe.device, dtype=torch.float32))
                continue

            # Gather K,V for this request from paged blocks
            block_table_b = dec.block_table[b]
            n_blocks_b = block_table_b.numel()
            k_pe_b = torch.zeros(
                1, seq_len_k, head_dim_qk,
                device=kv_c_and_k_pe_cache.device, dtype=kv_c_and_k_pe_cache.dtype
            )
            kv_c_b = torch.zeros(
                1, seq_len_k, head_dim_v,
                device=kv_c_and_k_pe_cache.device, dtype=kv_c_and_k_pe_cache.dtype
            )
            written = 0
            for blk_idx in range(n_blocks_b):
                if written >= seq_len_k:
                    break
                slot = int(block_table_b[blk_idx].item())
                take = min(block_size, seq_len_k - written)
                k_pe_b[0, written : written + take] = k_pe_cache[slot, :take, :]
                kv_c_b[0, written : written + take] = kv_c_cache[slot, :take, :]
                written += take

            q_pe_b = q_pe[b : b + 1]
            q_nope_b = q_nope[b : b + 1]

            # MQA-style: q (1, N, L+R) @ k (1, seq_len_k, L+R)^T; then softmax @ v
            q_b = torch.cat([q_nope_b, q_pe_b], dim=-1)
            k_b = torch.cat([kv_c_b, k_pe_b], dim=-1)
            v_b = kv_c_b

            attn_weights = torch.matmul(q_b, k_b.transpose(-2, -1)) * scale
            # Decode: single query attends to all keys (no causal mask)
            attn_weights = torch.nn.functional.softmax(attn_weights, dim=-1)
            out_b = torch.matmul(attn_weights, v_b)
            outputs.append(out_b)

            if self.need_to_return_lse_for_decode:
                lse_b = torch.log(attn_weights.sum(dim=-1).clamp(min=1e-10))
                lse_list.append(lse_b)

        attn_out = torch.cat(outputs, dim=0)
        lse_out = torch.cat(lse_list, dim=0) if lse_list else None
        return attn_out, lse_out

    def forward(
        self,
        layer: AttentionLayer,
        q: torch.Tensor,
        k_c_normed: torch.Tensor,
        k_pe: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: RBLNFlashAttnMLAMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Forward with Python cache write instead of ops.concat_and_cache_mla."""
        assert output is not None
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError(
                "fused output quantization not supported for RBLN MLA"
            )

        if attn_metadata is None:
            return output.fill_(0)

        num_actual_toks = attn_metadata.num_actual_tokens
        output_padded = output
        output = output[:num_actual_toks, ...]
        q = q[:num_actual_toks, ...]
        k_c_normed = k_c_normed[:num_actual_toks, ...]
        k_pe = k_pe[:num_actual_toks, ...]

        num_decodes = attn_metadata.num_decodes
        num_prefills = attn_metadata.num_prefills
        num_decode_tokens = attn_metadata.num_decode_tokens

        decode_q = q[:num_decode_tokens]
        prefill_q = q[num_decode_tokens:]
        prefill_k_pe = k_pe[num_decode_tokens:]
        prefill_k_c_normed = k_c_normed[num_decode_tokens:]

        # Python cache write (RBLN has no concat_and_cache_mla op)
        if kv_cache.numel() > 0:
            _concat_and_cache_mla_python(
                k_c_normed,
                k_pe.squeeze(1) if k_pe.dim() == 3 else k_pe,
                kv_cache,
                attn_metadata.slot_mapping.flatten(),
            )

        if num_prefills > 0:
            raise NotImplementedError(
                "RBLN MLA prefill is not implemented yet; use decode-only."
            )

        if num_decodes > 0:
            assert attn_metadata.decode is not None
            decode_q_nope, decode_q_pe = decode_q.split(
                [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1
            )
            decode_q_nope = decode_q_nope.transpose(0, 1)
            N, B, P = decode_q_nope.shape
            _, _, L = self.W_UK_T.shape
            decode_ql_nope = decode_q_nope.new_empty((N, B, L))
            torch.bmm(decode_q_nope, self.W_UK_T, out=decode_ql_nope)
            decode_ql_nope = decode_ql_nope.transpose(0, 1)
            decode_q_tuple = (decode_ql_nope, decode_q_pe)

            attn_out, lse = self._forward_decode(
                decode_q_tuple, kv_cache, attn_metadata, layer
            )
            self._v_up_proj(attn_out, out=output[:num_decode_tokens])

        return output_padded
