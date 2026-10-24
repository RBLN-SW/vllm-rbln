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
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from vllm_rbln.patches import register_patch
from vllm_rbln.v1.attention.backends.gdn import RBLNGDNAttentionMetadata

_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_TARGET = f"{_MODULE}.QwenGatedDeltaNetAttention"


@register_patch(
    target=f"{_TARGET}.get_kv_cache_spec",
    reason=(
        "The RBLN core kernels update contiguous state rows, so a request's state "
        "has to follow its batch row rather than a KV cache block; the RBLN GDN "
        "backend owns the state outside the KV cache groups."
    ),
)
def patched_gdn_get_kv_cache_spec(self: QwenGatedDeltaNetAttention, vllm_config):
    return None


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
    state = metadata.states[self.prefix]
    batch, seq_len = metadata.batch_size, metadata.seq_len
    if hidden_states.numel() != batch * seq_len * hidden_states.shape[-1]:
        raise RuntimeError(
            f"{self.prefix}: hidden_states {tuple(hidden_states.shape)} do not hold "
            f"the step's batch {batch} x seq {seq_len} the GDN metadata describes"
        )
    tp_size = self.tp_size
    heads_k = self.num_k_heads // tp_size
    heads_v = self.num_v_heads // tp_size
    ratio = heads_v // heads_k
    key_dim = self.key_dim // tp_size
    value_dim = self.value_dim // tp_size
    channels = key_dim * 2 + value_dim

    # Keep the leading dims the model uses ([batch, seq] on RBLN); only the
    # kernels need the explicit [batch, seq, ...] views.
    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    gates, _ = self.in_proj_ba(hidden_states)
    mixed_qkv, z = mixed_qkvz.split([channels, value_dim], dim=-1)
    b, a = gates.chunk(2, dim=-1)

    conv_weight = self.conv1d.weight.view(channels, self.conv_kernel_size)
    conv_out = torch.ops.rbln_custom_ops.kda_conv(
        mixed_qkv.reshape(batch, seq_len, channels),
        conv_weight.to(mixed_qkv.dtype),
        state.conv_state,
        metadata.state_index,
        metadata.seq_idx,
        metadata.query_len,
        metadata.has_initial_state_int,
        state.conv_width,
    )
    query, key, value = conv_out.split([key_dim, key_dim, value_dim], dim=-1)
    query = query.reshape(batch, seq_len, heads_k, self.head_k_dim).transpose(1, 2)
    key = key.reshape(batch, seq_len, heads_k, self.head_k_dim).transpose(1, 2)
    value = value.reshape(batch, seq_len, heads_k, ratio, self.head_v_dim)
    value = value.permute(0, 2, 3, 1, 4)

    beta = torch.sigmoid(b).reshape(batch, seq_len, heads_k, ratio).permute(0, 2, 3, 1)
    g = -torch.exp(self.A_log.float()) * F.softplus(a.float() + self.dt_bias.float())
    g = g.reshape(batch, seq_len, heads_k, ratio).permute(0, 2, 3, 1)
    if metadata.is_prefill:
        # Padded tokens must neither decay the state nor write to it.
        assert metadata.valid_mask is not None
        g = g * metadata.valid_mask
        beta = beta * metadata.valid_mask.to(beta.dtype)
        core = torch.ops.rbln_custom_ops.gdn_prefill(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            state.ssm_state,
            metadata.slot,
            metadata.has_initial_state,
        )
    else:
        core = torch.ops.rbln_custom_ops.gdn_decode(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g.contiguous(),
            beta.contiguous(),
            state.ssm_state,
            metadata.slot,
        )
    # [batch, heads_k, ratio, seq, dv] -> [batch, seq, heads_v, dv]
    core = core.permute(0, 3, 1, 2, 4).reshape(batch, seq_len, heads_v, self.head_v_dim)
    z = z.reshape(batch, seq_len, heads_v, self.head_v_dim)
    gated = self.norm(core, z)
    output, _ = self.out_proj(gated.reshape(batch, seq_len, heads_v * self.head_v_dim))
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
