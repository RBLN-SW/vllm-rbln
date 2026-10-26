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
from vllm import ir
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from vllm_rbln.patches import register_patch
from vllm_rbln.v1.attention.backends.gdn import (
    GDN_HEAD_DIM,
    RBLNGDNAttentionMetadata,
)

_MODULE = "vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn"
_TARGET = f"{_MODULE}.QwenGatedDeltaNetAttention"


def _l2norm(x: torch.Tensor) -> torch.Tensor:
    """x * rsqrt(sum(x^2) + 1e-6) as vLLM's rms_norm op (eps / D) scaled by
    D ** -0.5: the compiler lowers that op as one norm that stays cut on the
    heads, where a [..., 1] sum is packed across tokens and heads."""
    dim = x.shape[-1]
    return ir.ops.rms_norm(x, None, 1e-6 / dim) * dim**-0.5


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

    # Keep the leading dims the model uses ([batch, seq] on RBLN); only the
    # kernels need the explicit [batch, seq, ...] views.
    mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
    gates, _ = self.in_proj_ba(hidden_states)
    q, k, v, z = mixed_qkvz.split([key_dim, key_dim, value_dim, value_dim], dim=-1)
    b, a = gates.chunk(2, dim=-1)
    conv_weight = self.conv1d.weight.view(channels, self.conv_kernel_size)
    conv_weight = conv_weight.to(mixed_qkvz.dtype)
    # Sliced, not split: the compiler folds a slice into a constant RSD can
    # cut, but leaves the last part of a split as an op it cannot.
    w_q = conv_weight[:key_dim]
    w_k = conv_weight[key_dim : 2 * key_dim]
    w_v = conv_weight[2 * key_dim :]

    # One conv per q / k / v: RSD then cuts each one's channels on its heads,
    # where a cut of the q|k|v channels would not fall on head boundaries.
    def conv(x, weight, ring):
        return torch.ops.rbln_custom_ops.kda_conv(
            x.reshape(batch, seq_len, -1),
            weight,
            ring,
            metadata.state_index,
            metadata.seq_idx,
            metadata.query_len,
            metadata.has_initial_state,
            state.conv_width,
        )

    # The KDA core takes one q / k per value head, L2-normalized by the caller.
    # Not the kernel's norm_inputs: its zero-row constant shares a content key
    # with decode-only constants, which then stay on the device and fail the
    # weight reuse check.
    query, key = (
        _l2norm(conv(x, w, ring).view(batch * seq_len, heads_k, head_dim))
        .view(batch, seq_len, heads_k, 1, head_dim)
        .expand(batch, seq_len, heads_k, ratio, head_dim)
        .reshape(batch, seq_len, heads_v, head_dim)
        .transpose(1, 2)
        for x, w, ring in ((q, w_q, state.conv_q), (k, w_k, state.conv_k))
    )
    value = conv(v, w_v, state.conv_v).view(batch, seq_len, heads_v, head_dim)
    value = value.transpose(1, 2)

    beta = torch.sigmoid(b).reshape(batch, seq_len, heads_v).transpose(1, 2)
    g = -torch.exp(self.A_log.float()) * F.softplus(a.float() + self.dt_bias.float())
    g = g.reshape(batch, seq_len, heads_v).transpose(1, 2)
    if metadata.is_prefill:
        # Padded tokens must neither decay the state nor write to it.
        assert metadata.valid_mask is not None
        g = g * metadata.valid_mask
        beta = beta * metadata.valid_mask.to(beta.dtype)
    # A per-channel decay, as the KDA kernel reads it: the same over the head.
    g = g.unsqueeze(-1).expand(batch, heads_v, seq_len, head_dim).contiguous()
    if metadata.is_prefill:
        core = torch.ops.rbln_custom_ops.kda_prefill(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g,
            beta.contiguous(),
            state.ssm_state,
            metadata.core_state_index,
            metadata.core_has_initial_state,
        )
    else:
        core = torch.ops.rbln_custom_ops.kda_decode(
            query.contiguous(),
            key.contiguous(),
            value.contiguous(),
            g,
            beta.contiguous(),
            state.ssm_state,
            metadata.core_state_index,
        )
    core = core.transpose(1, 2).reshape(batch, seq_len, heads_v, head_dim)
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
