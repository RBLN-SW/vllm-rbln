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

import logging

import torch
import torch.nn.functional as F
from torch import nn
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.models import AXK1 as _axk1_mod
from vllm.model_executor.models.AXK1 import AXK1Attention, AXK1MoE

log = logging.getLogger("torch._dynamo")


def __AXK1_moe_forward_rsd(self, hidden_states: torch.Tensor) -> torch.Tensor:
    shared_output, final_hidden_states = self.experts(
        hidden_states=hidden_states, router=lambda x: self.gate(x)[0].sigmoid()
    )
    if hidden_states.dtype != torch.float16:
        final_hidden_states *= self.routed_scaling_factor
    elif self.shared_experts is not None:
        shared_output *= 1.0 / self.routed_scaling_factor

    if self.shared_experts is not None:
        final_hidden_states += shared_output

    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states


def __AXK1_attention_forward(
    self,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    llama_4_scaling: torch.Tensor | None = None,
) -> torch.Tensor:
    batch, num_tokens, _ = hidden_states.shape
    if self.q_lora_rank is not None:
        q = self.q_a_proj(hidden_states)[0]
        q = self.q_a_layernorm(q)
        q = self.q_b_proj(q)[0].reshape(
            batch, num_tokens, self.num_local_heads, self.qk_head_dim
        )
    else:
        q = self.q_proj(hidden_states)[0].reshape(
            batch, num_tokens, self.num_local_heads, self.qk_head_dim
        )
    q_nope, q_pe = q.split([self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
    latent_cache = self.kv_a_proj_with_mqa(hidden_states)[0]

    kv_a, _ = latent_cache.split([self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
    latent_cache = latent_cache.unsqueeze(2)
    kv_a = self.kv_a_layernorm(kv_a)
    kv = self.kv_b_proj(kv_a)[0]
    kv = kv.reshape(
        batch, -1, self.num_local_heads, self.qk_nope_head_dim + self.v_head_dim
    )
    k_nope, v = kv.split([self.qk_nope_head_dim, self.v_head_dim], dim=-1)
    k_pe = latent_cache[..., self.kv_lora_rank :]
    q_pe, k_pe = self.rotary_emb(positions, q_pe, k_pe)

    q = torch.cat([q_nope, q_pe], dim=-1)
    k = torch.cat([k_nope, k_pe.repeat(1, 1, self.num_local_heads, 1)], dim=-1)

    if llama_4_scaling is not None:
        q *= llama_4_scaling

    q = q.reshape(batch, -1, self.num_local_heads * self.qk_head_dim)
    k = k.reshape(-1, self.num_local_heads * self.qk_head_dim)
    v = torch.nn.functional.pad(
        v, [0, self.qk_head_dim - self.v_head_dim], value=0
    ).reshape(-1, self.num_local_heads * self.qk_head_dim)
    attn_output = self.attn(q, k, v)
    attn_output = attn_output.reshape(
        batch, -1, self.num_local_heads, self.qk_head_dim
    )[..., : self.v_head_dim].reshape(batch, -1, self.num_local_heads * self.v_head_dim)
    output, _ = self.o_proj(attn_output)
    return output


class AXK1MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        reduce_results: bool = True,
        is_sequence_parallel: bool = False,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.gate_proj",
        )
        self.up_proj = ColumnParallelLinear(
            hidden_size,
            intermediate_size,
            bias=False,
            quant_config=quant_config,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            reduce_results=reduce_results,
            disable_tp=is_sequence_parallel,
            prefix=f"{prefix}.down_proj",
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate, _ = self.gate_proj(x)
        up, _ = self.up_proj(x)
        x = F.silu(gate) * up
        x, _ = self.down_proj(x)
        return x


AXK1MoE.forward = __AXK1_moe_forward_rsd
AXK1Attention.forward = __AXK1_attention_forward
_axk1_mod.AXK1MLP = AXK1MLP
