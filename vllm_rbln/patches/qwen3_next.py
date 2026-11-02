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
from rebel.ops.torch_custom_ops.causal_conv1d_ring import pack_weight
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from vllm_rbln.patches import register_patch
from vllm_rbln.patches.registry import add_registration
from vllm_rbln.v1.attention.backends.gdn import (
    GDN_HEAD_DIM,
    GDN_PREFILL_CHUNK,
    RBLNGDNAttentionMetadata,
)

_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_TARGET = f"{_MODULE}.QwenGatedDeltaNetAttention"


@add_registration(
    reason=(
        "Upstream's Qwen3.5 merges image embeddings outside forward() and runs its "
        "vision tower with cu_seqlens attention; the RBLN class merges them inside "
        "the compiled graph and compiles the tower at one fixed patch bucket."
    )
)
def register_qwen3_5() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "Qwen3_5ForConditionalGeneration",
        "vllm_rbln.model_executor.models.qwen3_5:RBLNQwen3_5ForConditionalGeneration",
    )


@register_patch(
    target=f"{_TARGET}.forward",
    reason=(
        "Upstream picks a Triton (CUDA), AITER or CPU-op forward by platform and "
        "has no out-of-tree branch; this one runs the short convolution and the "
        "delta rule as RBLN custom kernels on the backend's state rows."
    ),
)
def patched_gdn_forward(
    self: QwenGatedDeltaNetAttention, hidden_states: torch.Tensor
) -> torch.Tensor:
    """
    Args:
        hidden_states: [batch, seq, hidden] or [batch * seq, hidden]; the batch
            and seq come from the step's GDN metadata
    """
    attn_metadata = get_forward_context().attn_metadata
    metadata = (
        attn_metadata.get(self.prefix) if isinstance(attn_metadata, dict) else None
    )
    if not isinstance(metadata, RBLNGDNAttentionMetadata):
        raise RuntimeError(f"{self.prefix}: this step carries no RBLN GDN metadata")
    batch, seq_len = metadata.batch_size, metadata.seq_len
    if hidden_states.numel() != batch * seq_len * hidden_states.shape[-1]:
        raise RuntimeError(
            f"{self.prefix}: hidden_states {tuple(hidden_states.shape)} do not hold "
            f"the step's batch {batch} x seq {seq_len} the GDN metadata describes"
        )
    state = metadata.states[self.prefix]
    tp_size = self.tp_size
    heads_k = self.num_k_heads // tp_size
    heads_v = self.num_v_heads // tp_size
    ratio = heads_v // heads_k
    key_dim = self.key_dim // tp_size
    value_dim = self.value_dim // tp_size
    channels = key_dim * 2 + value_dim
    head_dim = GDN_HEAD_DIM
    kernel = self.conv_kernel_size

    # Keep the leading dims the model uses ([batch, seq] on RBLN); only the
    # kernels need the explicit [batch, seq, ...] views.
    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    gates, _ = self.in_proj_ba(hidden_states)
    q, k, v, z = mixed_qkvz.split([key_dim, key_dim, value_dim, value_dim], dim=-1)
    b, a = gates.chunk(2, dim=-1)
    conv_weight = self.conv1d.weight.view(channels, kernel).to(mixed_qkvz.dtype)

    # One conv per q / k / v: RSD then cuts each one's channels on its heads,
    # where a cut of the q|k|v channels would not fall on head boundaries. The
    # weights are sliced, not split: the compiler folds a slice into a constant
    # RSD can cut, but leaves the last part of a split as an op it cannot.
    def conv(x, weight, ring, heads):
        return torch.ops.rbln_custom_ops.causal_conv1d_ring_tabled(
            x.reshape(batch, seq_len, heads, head_dim),
            pack_weight(weight.reshape(heads, head_dim, kernel)),
            ring,
            *metadata.conv_tables,
            state.conv_width,
        )

    # The cores take q / k per key head (they L2-normalize them) and v, g and
    # beta per value head grouped under its key head: value head h is key head
    # h // ratio's (h % ratio)-th, as HF's repeat_interleave pairs them.
    query, key = (
        conv(x, w, ring, heads_k)
        .view(batch, seq_len, heads_k, head_dim)
        .transpose(1, 2)
        .contiguous()
        for x, w, ring in (
            (q, conv_weight[:key_dim], state.conv_q),
            (k, conv_weight[key_dim : 2 * key_dim], state.conv_k),
        )
    )
    value = conv(v, conv_weight[2 * key_dim :], state.conv_v, heads_v)
    value = value.view(batch, seq_len, heads_k, ratio, head_dim)
    value = value.permute(0, 2, 3, 1, 4).contiguous()

    beta = torch.sigmoid(b).reshape(batch, seq_len, heads_k, ratio)
    g = -torch.exp(self.A_log.float()) * F.softplus(a.float() + self.dt_bias.float())
    g = g.reshape(batch, seq_len, heads_k, ratio)
    beta, g = (x.permute(0, 2, 3, 1) for x in (beta, g))
    if metadata.is_prefill:
        # Padded tokens must neither decay the state nor write to it.
        assert metadata.valid_mask is not None
        g = g * metadata.valid_mask
        beta = beta * metadata.valid_mask.to(beta.dtype)
    beta, g = beta.contiguous(), g.contiguous()
    if metadata.is_prefill:
        core = torch.ops.rbln_custom_ops.chunk_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            state.ssm_state,
            metadata.core_state_index,
            metadata.core_has_initial_state,
            GDN_PREFILL_CHUNK,
        )
    else:
        core = torch.ops.rbln_custom_ops.recurrent_gated_delta_rule(
            query,
            key,
            value,
            g,
            beta,
            state.ssm_state,
            metadata.core_state_index,
        )
    # [batch, heads_k, ratio, seq, 128] -> [batch, seq, heads_v, 128]
    core = core.permute(0, 3, 1, 2, 4).reshape(batch, seq_len, heads_v, head_dim)
    z = z.reshape(batch, seq_len, heads_v, head_dim)
    gated = self.norm(core, z)
    output, _ = self.out_proj(gated.reshape(batch, seq_len, heads_v * head_dim))
    return output.reshape(*hidden_states.shape[:-1], output.shape[-1])


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
