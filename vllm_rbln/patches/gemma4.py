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

import torch
import torch.nn.functional as F
from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4 import Gemma4DecoderLayer, Gemma4MLP

from vllm_rbln.patches import register_patch
from vllm_rbln.patches.registry import add_registration


@register_patch(
    target="vllm.model_executor.models.config.Gemma4Config.verify_and_update_config",
    reason=(
        "Upstream picks a CUDA attention backend for Gemma4's mixed head sizes "
        "(256 sliding, 512 full) and falls back to TRITON_ATTN when FA4 is "
        "missing, which RBLN refuses. RBLN's one backend serves both sizes, and "
        "the hook runs inside VllmConfig.__post_init__, before any platform hook "
        "could undo the choice."
    ),
)
def keep_the_rbln_attention_backend(vllm_config: VllmConfig) -> None:
    """Leave the attention backend to the RBLN platform."""


@register_patch(
    target="vllm.model_executor.models.gemma4.Gemma4DecoderLayer.forward",
    reason=(
        "Gemma4 routes on the residual, not on the experts' input, and hands "
        "upstream's MoE runner precomputed logits. RBLNMoERunner takes a router "
        "callable instead, as the patched qwen/deepseek/minimax/gpt-oss MoE "
        "blocks pass it; this forward is upstream's with that one call changed."
    ),
)
def patched_gemma4_decoder_layer_forward(
    self: Gemma4DecoderLayer,
    positions: torch.Tensor,
    hidden_states: torch.Tensor,
    residual: torch.Tensor | None,
    per_layer_input: torch.Tensor | None = None,
    **kwargs,
) -> tuple[torch.Tensor, None]:
    residual = hidden_states
    hidden_states = self.input_layernorm(residual)
    hidden_states = self.self_attn(
        positions=positions, hidden_states=hidden_states, **kwargs
    )
    hidden_states = self.post_attention_layernorm(hidden_states)
    hidden_states = hidden_states + residual
    residual = hidden_states

    hidden_states = self.pre_feedforward_layernorm(hidden_states)
    hidden_states = self.mlp(hidden_states)

    if self.enable_moe_block:
        hidden_states_1 = self.post_feedforward_layernorm_1(hidden_states)
        hidden_states_2 = self.pre_feedforward_layernorm_2(residual)
        router_logits = self.router(residual)
        # The logits ignore the runner's input, so they would miss its DP
        # multicast; RBLNGemma4ForConditionalGeneration refuses DP.
        hidden_states_2 = self.experts(
            hidden_states=hidden_states_2, router=lambda _: router_logits
        )
        hidden_states_2 = self.post_feedforward_layernorm_2(hidden_states_2)
        hidden_states = hidden_states_1 + hidden_states_2

    hidden_states = self.post_feedforward_layernorm(hidden_states)
    hidden_states = hidden_states + residual

    if per_layer_input is not None and self.per_layer_input_gate is not None:
        gate = self.per_layer_input_gate(hidden_states)
        gate = torch.nn.functional.gelu(gate, approximate="tanh")
        per_layer_contribution = self.per_layer_projection(gate * per_layer_input)
        hidden_states = hidden_states + self.post_per_layer_input_norm(
            per_layer_contribution
        )

    return hidden_states * self.layer_scalar, None


@add_registration(
    reason=(
        "Upstream's Gemma4ForConditionalGeneration merges image embeddings "
        "outside forward() and runs its vision tower through HF's dynamic "
        "attention dispatch. The RBLN subclass merges them in the compiled "
        "graph and compiles the tower at one patch count."
    )
)
def register_gemma4() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "Gemma4ForConditionalGeneration",
        "vllm_rbln.model_executor.models.gemma4:RBLNGemma4ForConditionalGeneration",
    )


@register_patch(
    target="vllm.model_executor.models.gemma4.Gemma4MLP.forward",
    reason=(
        "Upstream halves one merged gate_up matmul; separate gate and up "
        "matmuls compile better on RBLN. RBLNGemma4 splits the weight at load "
        "(split_gate_up_weights) and this forward runs the two halves."
    ),
)
def split_gate_up_mlp_forward(self: Gemma4MLP, x: torch.Tensor) -> torch.Tensor:
    gate = F.linear(x, self.gate_weight)
    up = F.linear(x, self.up_weight)
    x, _ = self.down_proj(F.gelu(gate, approximate=self.act_fn.approximate) * up)
    return x
