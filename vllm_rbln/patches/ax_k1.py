# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import typing
from collections.abc import Callable, Iterable

import torch
import torch.nn.functional as F
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.layers.fused_moe import fused_moe_make_expert_params_mapping
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.model_executor.models.AXK1 import (
    AXK1Attention,
    AXK1ForCausalLM,
    AXK1MoE,
    get_spec_layer_idx_from_weight_name,
)
from vllm.model_executor.models.utils import is_pp_missing_parameter

from vllm_rbln.patches import register_patch


class AXK1MLP(torch.nn.Module):
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


register_patch(
    target="vllm.model_executor.models.AXK1.AXK1MLP",
    reason="",
)(AXK1MLP)


@register_patch(
    target="vllm.model_executor.models.AXK1.AXK1Attention.forward",
    reason="",
)
def patched_ax_k1_attention_forward(
    self: AXK1Attention,
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


@register_patch(
    target="vllm.model_executor.models.AXK1.AXK1MoE.forward",
    reason="",
)
def patched_ax_k1_moe_foward_rsd(
    self: AXK1MoE,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    # RBLN's fused_moe_forward_rbln returns only the routed (fused) output, so
    # compute the shared experts separately here (matches dev-0.22.0's
    # DeepseekV2 RBLN MoE forward). self.experts was constructed with
    # shared_experts= for the GPU path, but the RBLN path does not fuse them.
    shared_output = None
    if self.shared_experts is not None:
        shared_output = self.shared_experts(hidden_states)

    final_hidden_states = self.experts(
        hidden_states=hidden_states, router=lambda x: self.gate(x)[0]
    )
    if hidden_states.dtype != torch.float16:
        final_hidden_states = final_hidden_states * self.routed_scaling_factor
    elif self.shared_experts is not None:
        shared_output = shared_output * (1.0 / self.routed_scaling_factor)

    if self.shared_experts is not None:
        final_hidden_states = final_hidden_states + shared_output

    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states


@register_patch(
    target="vllm.model_executor.models.AXK1.AXK1ForCausalLM.load_weights",
    reason=(
        "The AXK1MLP replacement above keeps gate_proj and up_proj as separate "
        "layers, but the upstream loader stacks checkpoint gate_proj/up_proj "
        "into gate_up_proj and its `name not in params_dict` guard then drops "
        "every dense and shared-expert MLP weight without an error. Load them "
        "under their own names and split a fused shared_experts.gate_up_proj "
        "checkpoint tensor back into its two halves."
    ),
)
def patched_ax_k1_load_weights(
    self: AXK1ForCausalLM, weights: Iterable[tuple[str, torch.Tensor]]
) -> set[str]:
    stacked_params_mapping: list[tuple[str, str, str | int]] = []
    mla_params_mapping: list[tuple[str, str, str | int]] = [
        ("fused_qkv_a_proj", "q_a_proj", 0),
        ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
    ]
    mha_params_mapping: list[tuple[str, str, str | int]] = [
        ("qkv_proj", "q_proj", "q"),
        ("qkv_proj", "k_proj", "k"),
        ("qkv_proj", "v_proj", "v"),
    ]

    if self.use_mha:
        stacked_params_mapping.extend(mha_params_mapping)
    else:
        stacked_params_mapping.extend(mla_params_mapping)

    expert_params_mapping = fused_moe_make_expert_params_mapping(
        self,
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts,
        num_redundant_experts=self.num_redundant_experts,
    )

    params_dict = dict(self.named_parameters())
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue

        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        if spec_layer is not None:
            continue

        if "shared_experts.gate_up_proj" in name:
            gate_name = name.replace(
                "shared_experts.gate_up_proj", "shared_experts.gate_proj"
            )
            up_name = name.replace(
                "shared_experts.gate_up_proj", "shared_experts.up_proj"
            )
            if loaded_weight.ndim > 0 and loaded_weight.shape[0] > 1:
                half = loaded_weight.shape[0] // 2
                gate_weight, up_weight = loaded_weight[:half], loaded_weight[half:]
            else:
                gate_weight = up_weight = loaded_weight
            for split_name, split_weight in (
                (gate_name, gate_weight),
                (up_name, up_weight),
            ):
                if is_pp_missing_parameter(split_name, self):
                    continue
                if split_name not in params_dict:
                    continue
                param = params_dict[split_name]
                weight_loader_fn = getattr(
                    param, "weight_loader", default_weight_loader
                )
                weight_loader_fn(param, split_weight)
                loaded_params.add(split_name)
            continue

        for param_name, weight_name, shard_id in stacked_params_mapping:
            if weight_name not in name:
                continue
            if ("mlp.experts." in name) and name not in params_dict:
                continue
            name_mapped = name.replace(weight_name, param_name)

            if (param_name == "fused_qkv_a_proj") and name_mapped not in params_dict:
                continue
            else:
                name = name_mapped
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            is_expert_weight = False

            num_chunks = 1
            for j in range(num_chunks):
                chunk_name = name
                weight_to_load = loaded_weight

                for mapping in expert_params_mapping:
                    param_name, weight_name, expert_id, shard_id = mapping
                    if weight_name not in chunk_name:
                        continue

                    is_expert_weight = True

                    name_mapped = chunk_name.replace(weight_name, param_name)

                    if is_pp_missing_parameter(name_mapped, self):
                        continue

                    param = params_dict[name_mapped]
                    weight_loader = typing.cast(
                        Callable[..., bool], param.weight_loader
                    )
                    success = weight_loader(
                        param,
                        weight_to_load,
                        name_mapped,
                        shard_id=shard_id,
                        expert_id=expert_id,
                        return_success=True,
                    )
                    if success:
                        name = name_mapped
                        break
                else:
                    if is_expert_weight:
                        continue

                    if name.endswith(".bias") and name not in params_dict:
                        continue

                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue

                    if is_pp_missing_parameter(name, self):
                        continue

                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
        loaded_params.add(name)

    return loaded_params
