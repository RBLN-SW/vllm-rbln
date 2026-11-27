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
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextAttention,
    Qwen3NextSparseMoeBlock,
)

from vllm_rbln.patches import register_patch
from vllm_rbln.patches.registry import add_registration


@add_registration(
    reason=(
        "Upstream's Qwen3.5 merges image embeddings outside forward() and runs its "
        "vision tower with cu_seqlens attention; the RBLN class merges them inside "
        "the compiled graph and compiles the tower at one fixed patch bucket. "
        "Upstream's MTP drafter numbers its layer 0, as the target's first: the "
        "RBLN class numbers it past the target's, as the runner's KV cache order "
        "needs one number a layer."
    )
)
def register_qwen3_5() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "vllm_rbln.model_executor.models.qwen3_5:RBLNQwen3_5ForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Qwen3_5MTP", "vllm_rbln.model_executor.models.qwen3_5:RBLNQwen3_5MTP"
    )
    ModelRegistry.register_model(
        "Qwen3_5MoeForConditionalGeneration",
        "vllm_rbln.model_executor.models.qwen3_5:RBLNQwen3_5MoeForConditionalGeneration",
    )
    ModelRegistry.register_model(
        "Qwen3_5MoeMTP", "vllm_rbln.model_executor.models.qwen3_5:RBLNQwen3_5MoeMTP"
    )


@register_patch(
    target="vllm.model_executor.models.qwen3_next.Qwen3NextAttention._project_qkv_gate",
    reason=(
        "Upstream flattens q and k to [tokens, ...] around the QK norms while the "
        "gate and v keep the RBLN [batch, seq, ...] layout, so the gated product "
        "broadcasts [batch, hidden] against [batch, 1, hidden] into a wrong shape."
    ),
)
def patched_qwen3_next_project_qkv_gate(
    self: Qwen3NextAttention, qkv: torch.Tensor, positions: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
    """The eager split + QK-RMSNorm + RoPE path, with every tensor kept in the
    leading dims of `qkv` (the fused Triton path is CUDA/XPU only)."""
    lead = qkv.shape[:-1]
    if self.attn_output_gate:
        q_gate, k, v = qkv.split([self.q_size * 2, self.kv_size, self.kv_size], dim=-1)
        q, gate = torch.chunk(q_gate.view(*lead, self.num_heads, -1), 2, dim=-1)
        q = q.reshape(*lead, -1)
        gate = gate.reshape(*lead, -1)
    else:
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        gate = None
    q = self.q_norm(q.view(-1, self.num_heads, self.head_dim)).view(*lead, -1)
    k = self.k_norm(k.view(-1, self.num_kv_heads, self.head_dim)).view(*lead, -1)
    q, k = self.rotary_emb(positions, q, k)
    return q, k, v, gate


@register_patch(
    target="vllm.model_executor.models.qwen3_next.Qwen3NextSparseMoeBlock.forward",
    reason=(
        "Upstream unpacks the hidden states as [tokens, hidden] and hands its "
        "FusedMoE the hidden states as the router input; the RBLN layout is "
        "[batch, seq, hidden], RBLNMoERunner takes the router as a callable and "
        "computes the routed experts only, so the shared expert is added here."
    ),
)
def patched_qwen3_next_moe_forward(
    self: Qwen3NextSparseMoeBlock,
    hidden_states: torch.Tensor,
    already_sequence_parallel: bool = False,
) -> torch.Tensor:
    assert hidden_states.dim() == 3  # [B, L, H]

    def router(h: torch.Tensor) -> torch.Tensor:
        return self.gate(h)[0]

    final_hidden_states = self.experts(hidden_states=hidden_states, router=router)
    if self.shared_expert is not None:
        # Qwen2MoeMLP applies the shared expert's sigmoid gate itself.
        final_hidden_states = final_hidden_states + self.shared_expert(hidden_states)
    if self.tp_size > 1:
        final_hidden_states = tensor_model_parallel_all_reduce(final_hidden_states)
    return final_hidden_states
