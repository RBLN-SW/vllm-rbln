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
from vllm.distributed import tensor_model_parallel_all_gather
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Attention, DeepseekV2MoE


def __deepseek_v2_moe_forward_rsd(self, hidden_states: torch.Tensor) -> torch.Tensor:
    shared_output, final_hidden_states = self.experts(
        hidden_states=hidden_states, router=lambda x: self.gate(x)[0]
    )
    # Fix FP16 overflow
    # See DeepseekV2DecoderLayer for more details.
    if hidden_states.dtype != torch.float16:
        final_hidden_states *= self.routed_scaling_factor
    elif self.shared_experts is not None:
        shared_output *= 1.0 / self.routed_scaling_factor

    if self.shared_experts is not None:
        final_hidden_states += shared_output

    if self.tp_size > 1:
        final_hidden_states = self.experts.maybe_all_reduce_tensor_model_parallel(
            final_hidden_states
        )
    # FIXME(RBLN) - DO NOT reshape
    # return final_hidden_states.view(orig_shape)
    return final_hidden_states


def __deepseek_v2_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, _, _ = hidden_states.shape
    if self.q_lora_rank is not None:
        q = self.q_a_proj(hidden_states)[0]
        q = self.q_a_layernorm(q)
        q = self.q_b_proj(q)[0].view(-1, self.num_local_heads, self.qk_head_dim)
    else:
        q = self.q_proj(hidden_states)[0].view(
            -1, self.num_local_heads, self.qk_head_dim
        )
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]
    kv_a, k_pe = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    kv_a = self.kv_a_layernorm(kv_a.contiguous())
    kv = self.kv_b_proj(kv_a)[0]
    kv = kv.view(-1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_pe = k_pe.view(-1, 1, self.qk_rope_head_dim)

    q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)
    if q_nope.dim() != q_pe.dim():
        q_pe = q_pe.squeeze(0)
    if k_nope.dim() != k_pe.dim():
        k_pe = k_pe.squeeze(0)

    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.repeat(1, self.num_local_heads, 1)], dim=-1)
    # padding value to qk_head_dim for alignment
    if self.qk_head_dim != self.v_head_dim:
        v = torch.nn.functional.pad(
            v, [0, self.qk_head_dim - self.v_head_dim], value=0
        ).view(-1, self.num_local_heads * self.qk_head_dim)
    q = q.reshape(batch, -1, self.num_local_heads * self.qk_head_dim)
    k = k.reshape(batch, -1, self.num_local_heads * self.qk_head_dim)
    v = v.reshape(batch, -1, self.num_local_heads * self.qk_head_dim)
    attn_output = self.attn(q, k, v)
    if self.qk_head_dim != self.v_head_dim:
        attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
            ..., : self.v_head_dim
        ].reshape(batch, -1, self.num_local_heads * self.v_head_dim)

    output, _ = self.o_proj(attn_output)
    return output


# reference is from DeepseekV2MoE.forward
DeepseekV2MoE.forward = __deepseek_v2_moe_forward_rsd
DeepseekV2Attention.forward = __deepseek_v2_attention_forward
