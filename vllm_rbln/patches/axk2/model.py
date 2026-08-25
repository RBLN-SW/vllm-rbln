# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.forward_context import get_forward_context

from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.patches.axk2.loader import load_frozen_module

CANONICAL_NAME = "vllm.model_executor.models.axk2"


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

    q_indexer = q.transpose(1, 2).contiguous()
    k_indexer_cur = k.contiguous()

    weights, _ = self.weights_proj(hidden_states)
    weights = (weights * (self.n_head**-0.5)).contiguous()  # [B, S, n_head]

    forward_context = get_forward_context()
    attn_metadata = forward_context.attn_metadata
    if isinstance(attn_metadata, dict):
        attn_metadata = attn_metadata[self.k_cache.prefix]
    k_cache = _resolve_kv_cache(attn_metadata, self.k_cache.layer_index)

    # fp8 indexer cache is a 2-cache spec: fp8_e4m3 values here plus a companion
    # fp16 per-position scale cache. Registered by the DeepseekV32IndexerCache
    # patch only when --kv-cache-dtype is fp8*, so this stays None on the bf16 path.
    scale_cache = None
    if getattr(self.k_cache, "scale_cache", None) is not None:
        scale_cache = _resolve_kv_cache(
            attn_metadata, self.k_cache.scale_cache.layer_index
        ).squeeze(-1)  # [num_block, ps, 1] -> [num_block, ps]

    softmax_scale = torch.tensor(
        self.softmax_scale, dtype=torch.float32, device=q_indexer.device
    )
    return torch.ops.rbln_custom_ops.sparse_attn_deepseek_indexer(
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


def _rbln_gated_mla_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    assert not self.dsv3_pad, (
        "axk2 DSv3 dim padding is not supported on RBLN yet "
        f"(native kv_lora={self.native_kv_lora_rank} vs {self.kv_lora_rank}, "
        f"native v_head={self.native_v_head_dim} vs {self.v_head_dim})"
    )

    batch_size, seq_len, _ = hidden_states.shape

    assert self.q_lora_rank is not None, "axk2 always uses a q_lora bottleneck"
    assert self.fused_qkv_a_proj is not None
    assert self.q_a_layernorm is not None
    assert self.q_b_proj is not None

    qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
    q_c, kv_lora = qkv_lora.split(
        [
            self.q_lora_rank,
            self.native_kv_lora_rank + self.native_qk_rope_head_dim,
        ],
        dim=-1,
    )

    q_c_prenorm = q_c
    q_c = self.q_a_layernorm(q_c)

    if self.attn_gate_fused:
        q_b_out = self.q_b_proj(torch.cat([q_c, q_c_prenorm], dim=-1))[0]
        per_head = self.qk_head_dim + self.native_v_head_dim
        q_gate = q_b_out.view(batch_size, seq_len, self.num_heads, per_head)
        q, gate = q_gate.split([self.qk_head_dim, self.native_v_head_dim], dim=-1)
        q = q.reshape(batch_size, seq_len, self.num_heads, self.qk_head_dim)
        attn_gate = gate.reshape(
            batch_size, seq_len, self.num_heads * self.native_v_head_dim
        )
    else:
        q = self.q_b_proj(q_c)[0].reshape(
            batch_size, seq_len, self.num_heads, self.qk_head_dim
        )
        attn_gate = None

    kv_c, k_pe = kv_lora.split(
        [self.native_kv_lora_rank, self.native_qk_rope_head_dim], dim=-1
    )
    kv_c_normed = self.kv_a_layernorm(kv_c)
    k_pe = k_pe.unsqueeze(2)

    if self.rotary_emb is not None:
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q = torch.cat([q_nope, q_pe], dim=-1)
    k_pe = k_pe.squeeze(2)

    topk_indices = None
    if getattr(self, "indexer", None) is not None and getattr(self, "is_sparse", False):
        topk_indices = self.indexer(
            hidden_states, q_c, positions, self.indexer_rope_emb
        )

    if llama_4_scaling is not None:
        q *= llama_4_scaling

    attn_out = self.mla_attn(
        q,
        kv_c_normed,
        k_pe,
        output_shape=(batch_size, seq_len, self.num_heads * self.v_head_dim),
        topk_indices=topk_indices,
    )

    if attn_gate is not None:
        attn_out = attn_out * torch.sigmoid(attn_gate)

    return self.o_proj(attn_out)[0]


def _rbln_axk2_moe_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    final_hidden_states = self.experts(
        hidden_states=hidden_states, router=lambda x: self.gate(x)[0]
    )
    if hidden_states.dtype != torch.float16:
        final_hidden_states *= self.routed_scaling_factor

    shared_output = None
    if self.shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)
        if hidden_states.dtype == torch.float16:
            shared_output *= 1.0 / self.routed_scaling_factor
        if not self.is_sequence_parallel:
            final_hidden_states = final_hidden_states + shared_output
            shared_output = None

    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)

    if shared_output is not None:
        final_hidden_states = final_hidden_states + shared_output

    return final_hidden_states


from vllm_rbln.patches.axk2 import config as _config  # noqa: E402, F401

_module = load_frozen_module(CANONICAL_NAME, "_skt_model.py")

_module.Indexer.forward = _rbln_indexer_forward
_module.AXK2GatedMultiHeadLatentAttentionWrapper.forward = _rbln_gated_mla_forward
_module.AXK2MoE.forward = _rbln_axk2_moe_forward

AXK2ForCausalLM = _module.AXK2ForCausalLM

__all__ = ["AXK2ForCausalLM", "CANONICAL_NAME"]
