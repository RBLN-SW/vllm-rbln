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

import torch
from vllm.model_executor.layers.mla import MultiHeadLatentAttentionWrapper


def __MLAAttentionWrapper_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    """RBLN override of MultiHeadLatentAttentionWrapper.forward.

    Follows the same logic as upstream but preserves explicit batch and
    seq_len dimensions (4-D tensors) which the RBLN compiler requires.
    Upstream flattens to ``[num_tokens, ...]`` throughout.
    """
    q_c = None
    kv_lora = None

    batch_size, seq_len, _ = hidden_states.shape

    if self.q_lora_rank is not None:
        assert self.fused_qkv_a_proj is not None
        assert self.q_a_layernorm is not None
        assert self.q_b_proj is not None

        qkv_lora = self.fused_qkv_a_proj(hidden_states)[0]
        q_c, kv_lora = qkv_lora.split(
            [self.q_lora_rank, self.kv_lora_rank + self.qk_rope_head_dim],
            dim=-1,
        )
        q_c = self.q_a_layernorm(q_c)
        q = self.q_b_proj(q_c)[0]
    else:
        assert self.kv_a_proj_with_mqa is not None
        assert self.q_proj is not None
        kv_lora = self.kv_a_proj_with_mqa(hidden_states)[0]
        q = self.q_proj(hidden_states)[0]

    kv_c, k_pe = kv_lora.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_c_normed = self.kv_a_layernorm(kv_c)

    # NOTE(RBLN): reshape to 4-D [batch, seq, heads, head_dim]
    # upstream uses 3-D [num_tokens, heads, head_dim]
    q = q.reshape(batch_size, seq_len, self.num_heads, self.qk_head_dim)
    # Add head dim — upstream unsqueeze(1), RBLN unsqueeze(2) due to batch dim
    k_pe = k_pe.unsqueeze(2)

    if self.rotary_emb is not None:
        # Avoid in-place slice assignment (q[..., N:] = ...) which creates a
        # view->slice->copy_ chain that process_inplace_copy_ops cannot handle
        # when the view changes the number of dimensions.
        q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
        q = torch.cat([q_nope, q_pe], dim=-1)
    # Remove the head dim added above so downstream gets [batch, seq, rope_dim]
    k_pe = k_pe.squeeze(2)

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
        output_shape=(batch_size, seq_len, self.num_heads * self.v_head_dim),
    )

    return self.o_proj(attn_out)[0]


MultiHeadLatentAttentionWrapper.forward = __MLAAttentionWrapper_forward
