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

from itertools import islice

import torch
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.distributed.parallel_state import get_pp_group
from vllm.logger import init_logger
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.models.nemotron_h import NemotronHModel, NemotronHMoE
from vllm.sequence import IntermediateTensors

from vllm_rbln.patches import register_patch
from vllm_rbln.patches.registry import add_registration

logger = init_logger(__name__)


@register_patch(
    target="vllm.model_executor.layers.fused_moe.config.FusedMoEConfig.__post_init__",
    reason=(
        "Upstream refuses non-gated experts (is_act_and_mul=False, Nemotron-H's "
        "relu2) on any platform but CUDA, XPU and ROCm, while RBLN's custom_moe "
        "runs them with no gate weight. The body is upstream's without that "
        "platform check."
    ),
)
def patched_fused_moe_config_post_init(self: FusedMoEConfig) -> None:
    from vllm._aiter_ops import rocm_aiter_ops

    tp_size = self.moe_parallel_config.tp_size
    assert self.intermediate_size % tp_size == 0
    self.intermediate_size_per_partition = self.intermediate_size // tp_size

    if self.dp_size > 1:
        logger.debug_once(
            "Using FusedMoEConfig::max_num_tokens=%d", self.max_num_tokens
        )

    assert self.max_num_tokens > 0

    if self.router_logits_dtype is None:
        self.router_logits_dtype = self.in_dtype

    if self.hidden_dim_unpadded is None:
        self.hidden_dim_unpadded = self.hidden_dim
    if self.intermediate_size_per_partition_unpadded is None:
        self.intermediate_size_per_partition_unpadded = (
            self.intermediate_size_per_partition
        )

    if self.is_act_and_mul:
        self.rocm_aiter_fmoe_enabled = rocm_aiter_ops.is_fused_moe_enabled()
        self.aiter_fmoe_shared_expert_enabled = (
            rocm_aiter_ops.is_fusion_moe_shared_experts_enabled()
        )

    if self.use_mori_kernels:
        assert self.rocm_aiter_fmoe_enabled, (
            "Mori needs to be used with aiter fused_moe for now."
        )
        assert not self.aiter_fmoe_shared_expert_enabled, (
            "Mori does not support fusion shared expert now. "
            "Turn it off by setting VLLM_ROCM_USE_AITER_FUSION_SHARED_EXPERTS=0"
        )


@add_registration(
    reason=(
        "Upstream sizes the hybrid pool page from the Mamba-2 conv and SSM state "
        "shapes; on RBLN the conv ring is a block-aux cache and the SSM state a "
        "full-attention KV block in the activation dtype, which the RBLN class "
        "declares instead."
    )
)
def register_nemotron_h() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "NemotronHForCausalLM",
        "vllm_rbln.model_executor.models.nemotron_h:RBLNNemotronHForCausalLM",
    )


@register_patch(
    target="vllm.model_executor.models.nemotron_h.NemotronHMoE.forward",
    reason=(
        "RBLNMoERunner takes a router callable and runs neither the routed "
        "input/output transforms, the routed scaling nor the shared experts that "
        "upstream's MoERunner applies; the latent MoE does all four around it."
    ),
)
def patched_nemotron_h_moe_forward(
    self: NemotronHMoE, hidden_states: torch.Tensor
) -> torch.Tensor:
    assert not self.is_sequence_parallel, "sequence-parallel MoE is not used on RBLN"
    routed = hidden_states
    if self.fc1_latent_proj is not None:
        routed = self.fc1_latent_proj(hidden_states)[0]

    output = self.experts(
        hidden_states=routed, router=lambda _: self.gate(hidden_states)[0]
    )
    output = output * self.routed_scaling_factor
    if self.fc2_latent_proj is not None:
        output = self.fc2_latent_proj(output)[0]
    if self.shared_experts is not None:
        output = output + self.shared_experts(hidden_states)
    if self.tp_size > 1:
        output = tensor_model_parallel_all_reduce(output)
    return output


@register_patch(
    target="vllm.model_executor.models.nemotron_h.NemotronHModel.forward",
    reason=(
        "A Nemotron-H layer holds one mixer and one norm, so the first returns "
        "its input as the residual: an input-output alias a nested compile "
        "region refuses. A zero residual makes the first norm's fused add build "
        "it instead (h + 0 == h). The body is upstream's otherwise."
    ),
)
def patched_nemotron_h_model_forward(
    self: NemotronHModel,
    input_ids: torch.Tensor | None,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
) -> torch.Tensor | IntermediateTensors | tuple[torch.Tensor, list[torch.Tensor]]:
    if get_pp_group().is_first_rank:
        if inputs_embeds is not None:
            hidden_states = inputs_embeds
        else:
            hidden_states = self.embed_input_ids(input_ids)
        residual = torch.zeros_like(hidden_states)
    else:
        assert intermediate_tensors is not None
        hidden_states = intermediate_tensors["hidden_states"]
        residual = intermediate_tensors["residual"]

    aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
    for idx, layer in enumerate(islice(self.layers, self.start_layer, self.end_layer)):
        hidden_states, residual = layer(
            positions=positions,
            hidden_states=hidden_states,
            residual=residual,
        )
        self._maybe_add_hidden_state(
            aux_hidden_states, idx + 1, hidden_states, residual
        )

    if not get_pp_group().is_last_rank:
        return IntermediateTensors(
            {"hidden_states": hidden_states, "residual": residual}
        )
    hidden_states, _ = self.norm_f(hidden_states, residual)

    if len(aux_hidden_states) > 0:
        return hidden_states, aux_hidden_states
    return hidden_states
