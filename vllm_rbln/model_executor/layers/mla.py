# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper


def __MLAAttentionWrapper_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    q_c = None
    kv_lora = None

    # RBLN fix
    batch_size, seq_len, _ = hidden_states.shape

    if self.q_lora_rank is not None:
        assert self.fused_qkv_a_proj is not None, (
            "fused_qkv_a_proj is required when q_lora_rank is not None"
        )
        assert self.q_a_layernorm is not None, (
            "q_a_layernorm is required when q_lora_rank is not None"
        )
        assert self.q_b_proj is not None, (
            "q_b_proj is required when q_lora_rank is not None"
        )
        qkv_lora = self.fused_qkv_a_proj(hidden_states)[
            0
        ]  # (batch_size, seq_len, q_lora_rank + kv_lora_rank + qk_rope_head_dim)
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]  # (batch_size, seq_len, num_heads * qk_head_dim)
    else:
        assert self.kv_a_proj_with_mqa is not None, (
            "kv_a_proj_with_mqa is required when q_lora_rank is None"
        )
        assert self.q_proj is not None, "q_proj is required when q_lora_rank is None"
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        q = self.q_proj(hidden_states)[0]

    kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_c_normed = self.kv_a_layernorm(kv_c)  # (batch_size, seq_len, kv_lora_rank)

    # RBLN fix: view as batch dim & add head dim of 1 to k_pe
    q = q.view(batch_size, seq_len, self.num_heads, self.qk_head_dim)
    k_pe = k_pe.unsqueeze(2)  # (batch_size, seq_len, 1, qk_rope_head_dim)

    if self.rotary_emb is not None:
        q[..., self.qk_nope_head_dim :], k_pe = self.rotary_emb(
            positions, q[..., self.qk_nope_head_dim :], k_pe
        )
    k_pe.squeeze_(2)

    if self.indexer and self.is_sparse:
        _topk_indices = self.indexer(
            hidden_states, q_c, positions, self.indexer_rope_emb
        )

    if llama_4_scaling is not None:
        q *= llama_4_scaling

    attn_out = self.mla_attn(
        q,
        kv_c_normed,
        k_pe,
        output_shape=(hidden_states.shape[0], self.num_heads * self.v_head_dim),
    )

    return self.o_proj(attn_out)[0]


MultiHeadLatentAttentionWrapper.forward_oot = __MLAAttentionWrapper_forward
