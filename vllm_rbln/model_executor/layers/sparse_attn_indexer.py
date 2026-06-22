# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# RBLN backend for the sparse-attention lightning indexer key cache.

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV32IndexerCache,
    Indexer,
)

from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.layers.attention.attention import _resolve_kv_cache
from vllm_rbln.v1.attention.backends.mla.indexer import (
    RBLNDeepseekV32IndexerBackend,
)

logger = init_logger(__name__)


@torch.library.custom_op(
    "rbln_custom_ops::sparse_attn_deepseek_indexer",
    mutates_args=["k_indexer_cache"],
)
def sparse_attn_deepseek_indexer_impl(
    q_indexer: torch.Tensor,  # [B, n_head, T, head_dim] (device layout)
    k_indexer_cur: torch.Tensor,  # [B, T, head_dim]
    k_indexer_cache: torch.Tensor,  # [num_block, partition_size, head_dim]
    seq_idx: torch.Tensor,
    # TODO(kblee): sync with args
    block_idx: torch.Tensor,
    block_offset: torch.Tensor,
    block_table: torch.Tensor,
    topk: int,
) -> torch.Tensor:
    b, _, t, _ = q_indexer.shape
    return torch.empty((b, t, topk), device=q_indexer.device, dtype=torch.int32)


@torch.library.register_fake("rbln_custom_ops::sparse_attn_deepseek_indexer")
def _(
    q_indexer,
    k_indexer_cur,
    k_indexer_cache,
    seq_idx,
    block_idx,
    block_offset,
    block_table,
    topk,
):
    b, _, t, _ = q_indexer.shape
    return torch.empty((b, t, topk), device=q_indexer.device, dtype=torch.int32)


_original_indexer_cache_init = DeepseekV32IndexerCache.__init__


def _rbln_indexer_cache_init(self, *args, **kwargs) -> None:
    _original_indexer_cache_init(self, *args, **kwargs)
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    self.head_dim = vllm_config.model_config.hf_text_config.index_head_dim
    self.dtype = torch.bfloat16

    from vllm_rbln.models.utils import (
        rbln_extract_layer_index,
        rbln_num_attn_module,
    )

    model_config = vllm_config.model_config
    num_attn_module = rbln_num_attn_module(model_config)
    self.layer_index = rbln_extract_layer_index(self.prefix, num_attn_module)
    if model_config is not None:
        start, _end = model_config.get_layers_start_end_indices(
            vllm_config.parallel_config
        )
        self.layer_index -= start * num_attn_module


def _rbln_indexer_cache_get_attn_backend(self):
    return RBLNDeepseekV32IndexerBackend


def _resolve_indexer_index_tensors(
    self, attn_metadata
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    block_size = self.k_cache.cache_config.block_size
    seq_lens = attn_metadata.seq_lens
    block_table = attn_metadata.block_tables
    slot_mapping = attn_metadata.slot_mapping

    seq_idx = seq_lens.view(-1, 1).to(torch.int16)
    block_idx = (slot_mapping // block_size).view(-1, 1).to(torch.int16)
    block_offset = (slot_mapping % block_size).view(-1, 1).to(torch.int16)
    block_table = block_table.to(torch.int16)
    return seq_idx, block_idx, block_offset, block_table


def _rbln_indexer_forward(
    self,
    hidden_states: torch.Tensor,
    qr: torch.Tensor,
    positions: torch.Tensor,
    rotary_emb,
) -> torch.Tensor:
    batch_size, seq_len, _ = hidden_states.shape

    # q [B, S, n_head, head_dim]
    q, _ = self.wq_b(qr)
    q = q.view(batch_size, seq_len, self.n_head, self.head_dim)
    q_pe, q_nope = torch.split(
        q, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
    )

    # k [B, S, head_dim]
    k, _ = self.wk(hidden_states)
    k = self.k_norm(k)
    k_pe, k_nope = torch.split(
        k, [self.rope_dim, self.head_dim - self.rope_dim], dim=-1
    )

    # q,k rope
    k_pe = k_pe.unsqueeze(2)
    q_pe, k_pe = rotary_emb(positions, q_pe, k_pe)
    k_pe = k_pe.squeeze(2)

    q = torch.cat([q_pe, q_nope], dim=-1)
    k = torch.cat([k_pe, k_nope], dim=-1)  # [B, S, head_dim]

    # TODO(kblee): need transpose?
    q_indexer = q.transpose(1, 2).contiguous()
    k_indexer_cur = k.contiguous()

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[self.k_cache.prefix]
    k_cache = _resolve_kv_cache(attn_metadata, self.k_cache.layer_index)

    seq_idx, block_idx, block_offset, block_table = _resolve_indexer_index_tensors(
        self, attn_metadata
    )

    topk_index = torch.ops.rbln_custom_ops.sparse_attn_deepseek_indexer(
        q_indexer,
        k_indexer_cur,
        k_cache,
        seq_idx,
        block_idx,
        block_offset,
        block_table,
        self.topk_tokens,
    )
    return topk_index


DeepseekV32IndexerCache.__init__ = _rbln_indexer_cache_init
DeepseekV32IndexerCache.get_attn_backend = _rbln_indexer_cache_get_attn_backend
Indexer.forward = _rbln_indexer_forward
