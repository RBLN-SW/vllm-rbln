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
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.AXK1 import AXK1Attention, AXK1MoE
import logging
log = logging.getLogger("torch._dynamo")

def __AXK1_moe_forward_rsd(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
        final_hidden_states = tensor_model_parallel_all_reduce(
            final_hidden_states
        )
    # FIXME(RBLN) - DO NOT reshape
    # return final_hidden_states.view(orig_shape)
    return final_hidden_states


def __AXK1_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, num_tokens, _ = hidden_states.shape # 1, 128, 7168
    if self.q_lora_rank is not None:
        q = self.q_a_proj(hidden_states)[0]
        q = self.q_a_layernorm(q) # torch.Size([1, 128, 1536])
        q = self.q_b_proj(q)[0].view(batch, num_tokens, self.num_local_heads, self.qk_head_dim) # torch.size([b, 128, 16, 192])
    else:
        q = self.q_proj(hidden_states)[0].view(
            batch, num_tokens, self.num_local_heads, self.qk_head_dim
        )
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1) # torch.size([128, 16, 128]), torch.size([128, 16, 64])
    latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0] # torch.size([1, 128, 576])
    
    # #rbln_fix: remove the first dimension of latent_cache
    # latent_cache = latent_cache.squeeze(0)
    
    kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1) # torch.Size([1, 128, 512])
    latent_cache = latent_cache.unsqueeze(2) # torch.size([1, 128, 1, 576])
    kv_a = self.kv_a_layernorm(kv_a)
    kv = self.kv_b_proj(kv_a)[0] # torch.size([1,128, 4096])
    kv = kv.view(batch, -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim)
    k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_pe = latent_cache[..., self.kv_lora_rank :] # torch.size([128, 1, 64])
    q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)


# (*([FakeTensor(..., size=(2, 1, 16, 128), dtype=torch.bfloat16), FakeTensor(..., size=(2, 2, 16, 64), dtype=torch.bfloat16)],), **{'dim': -1}): got RuntimeError('Sizes of tensors must match except in dimension 3. Expected 1 in dimension 1 but got 2 for tensor number 1 in the list') 

    #repeat(*(FakeTensor(..., size=(2, 2, 0, 64), dtype=torch.bfloat16), 1, 16, 1)
    k = torch.cat([k_nope, k_pe.repeat(1, 1, self.num_local_heads, 1)], dim=-1) # torch.size([b, 128, 16, 192])
    q = torch.cat([q_nope, q_pe], dim=-1) # torch.size([b, 128, 16, 192])
    
    # Apply llama 4 scaling if provided
    if llama_4_scaling is not None:
        q *= llama_4_scaling
        
    q = q.view(batch, -1, self.num_local_heads * self.qk_head_dim)
    k = k.view(batch, -1, self.num_local_heads * self.qk_head_dim)
    # padding value to qk_head_dim for alignment
    v = torch.nn.functional.pad(
        v, [0, self.qk_head_dim - self.v_head_dim], value=0
    ).view(-1, self.num_local_heads * self.qk_head_dim)
    attn_output = self.attn(q, k, v)
    attn_output = attn_output.view(-1, self.num_local_heads, self.qk_head_dim)[
        ..., : self.v_head_dim
    ].reshape(-1, self.num_local_heads * self.v_head_dim)
    output, _ = self.o_proj(attn_output)
    return output


# reference is from DeepseekV2MoE.forward
AXK1MoE.forward = __AXK1_moe_forward_rsd
AXK1Attention.forward = __AXK1_attention_forward