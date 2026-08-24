# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import torch
from rebel.ops.torch_custom_ops import attn as _rbln_attn_ops  # noqa: F401
from vllm.forward_context import get_forward_context
from vllm.model_executor.models.deepseek_v2 import (
    DeepseekV32IndexerCache,
)

from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch
from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.patches.models_utils import (
    rbln_extract_layer_index,
    rbln_num_attn_module,
)
from vllm_rbln.v1.attention.backends.mla.indexer import (
    RBLNDeepseekV32IndexerBackend,
    RBLNDeepseekV32IndexerScaleBackend,
)

logger = init_logger(__name__)
_original_indexer_cache_init = DeepseekV32IndexerCache.__init__


@register_patch(
    target="vllm.model_executor.models.deepseek_v2.DeepseekV32IndexerCache.__init__",
    reason=(
        "Give the indexer KV cache the RBLN head_dim/dtype "
        "and a layer_index that accounts for the extra indexer module per decoder "
        "layer (num_attn_module=2), so the runner binds a distinct cache for it."
    ),
)
def rbln_indexer_cache_init(self, *args, **kwargs) -> None:
    _original_indexer_cache_init(self, *args, **kwargs)
    from vllm.config import get_current_vllm_config

    vllm_config = get_current_vllm_config()
    self.head_dim = vllm_config.model_config.hf_text_config.index_head_dim
    cache_dtype = vllm_config.cache_config.cache_dtype
    self.is_fp8_cache = bool(cache_dtype) and cache_dtype.startswith("fp8")
    self.dtype = torch.float8_e4m3fn if self.is_fp8_cache else torch.bfloat16

    model_config = vllm_config.model_config
    num_attn_module = rbln_num_attn_module(model_config)
    start = 0
    if model_config is not None:
        start, _end = model_config.get_layers_start_end_indices(
            vllm_config.parallel_config
        )
    self.layer_index = (
        rbln_extract_layer_index(self.prefix, num_attn_module) - start * num_attn_module
    )

    # register fp16 scale cache (2-cache spec).
    self.scale_cache = None
    if self.is_fp8_cache:
        scale = DeepseekV32IndexerCache.__new__(DeepseekV32IndexerCache)
        _original_indexer_cache_init(
            scale,
            head_dim=1,
            dtype=torch.float16,
            prefix=f"{self.prefix}.scale_cache",
            cache_config=self.cache_config,
        )
        scale.get_attn_backend = lambda: RBLNDeepseekV32IndexerScaleBackend
        scale.layer_index = (
            rbln_extract_layer_index(scale.prefix, num_attn_module)
            - start * num_attn_module
        )
        self.scale_cache = scale


@register_patch(
    target="vllm.model_executor.models.deepseek_v2.DeepseekV32IndexerCache.get_attn_backend",
    reason="Use in the RBLN indexer backend.",
)
def rbln_indexer_cache_get_attn_backend(self):
    return RBLNDeepseekV32IndexerBackend


@register_patch(
    target="vllm.model_executor.models.deepseek_v2.Indexer.forward",
    reason=("Run the lightning indexer through the RBLN. "),
)
def rbln_indexer_forward(
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
    kw, _ = self.wk_weights_proj(hidden_states)
    k = kw[..., : self.head_dim]
    weights = kw[..., self.head_dim :] * (self.n_head**-0.5)
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

    q_indexer = q.transpose(1, 2).contiguous()
    k_indexer_cur = k.contiguous()

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[self.k_cache.prefix]
    k_cache = _resolve_kv_cache(attn_metadata, self.k_cache.layer_index)

    scale_cache = None
    if getattr(self.k_cache, "scale_cache", None) is not None:
        scale_cache = _resolve_kv_cache(
            attn_metadata, self.k_cache.scale_cache.layer_index
        ).squeeze(-1)  # [num_block, ps, 1] -> [num_block, ps]

    weights = weights.contiguous()  # [B, T, n_head]
    softmax_scale = torch.tensor(
        self.softmax_scale, dtype=torch.float32, device=q_indexer.device
    )
    topk_index = torch.ops.rbln_custom_ops.sparse_attn_deepseek_indexer(
        q_indexer,
        k_indexer_cur,
        k_cache,
        softmax_scale,
        weights,
        attn_metadata.seq_lens.to(torch.int32),
        attn_metadata.block_tables,
        self.topk_tokens,
        scale_cache,
    )
    return topk_index
