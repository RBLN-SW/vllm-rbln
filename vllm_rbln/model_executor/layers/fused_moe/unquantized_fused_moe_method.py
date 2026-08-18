# Copyright 2025 Rebellions Inc. All rights reserved.
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

import torch
from vllm.model_executor.layers.fused_moe import UnquantizedFusedMoEMethod
from vllm.model_executor.layers.fused_moe.routed_experts import RoutedExperts


class RBLNUnquantizedFusedMoEMethod(UnquantizedFusedMoEMethod):
    """Unquantized MoE method for the RBLNMoERunner forward path.

    vLLM creates UnquantizedFusedMoEMethod directly when no quantization config
    is provided. Registering this OOT implementation preserves that selection
    path while routing execution through the RBLN MoE custom op.
    """

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        router_logits: torch.Tensor,
    ) -> torch.Tensor:
        # RBLNMoERunner computes router logits after RBLN DP multicast and then calls
        # quant_method.apply(layer, x, router_logits). Keep this router-logits interface
        # instead of upstream MoERunner's topk_weights/topk_ids interface.
        assert isinstance(w13 := layer.w13_weight, torch.Tensor)
        assert isinstance(w2 := layer.w2_weight, torch.Tensor)
        intermediate_size = w2.shape[-1]

        gate_proj_weight = w13[:, :intermediate_size, :]
        up_proj_weight = w13[:, intermediate_size:, :]
        down_proj_weight = w2

        orig_shape = x.shape
        num_tokens = orig_shape[:-1].numel()
        hidden_states = x.reshape(num_tokens, -1)
        masked_routing_weights = router_logits

        final_hidden_states = torch.ops.rbln_custom_ops.custom_moe_glu(
            hidden_states,
            gate_proj_weight,
            up_proj_weight,
            down_proj_weight,
            masked_routing_weights,
            layer.activation.value,
            layer.expert_map,
            None,
            None,
            None,
        )
        return final_hidden_states.reshape(orig_shape)
