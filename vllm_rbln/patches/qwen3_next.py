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
from rebel.ops.torch_custom_ops.gdn import CHUNK as GDN_PREFILL_CHUNK
from rebel.ops.torch_custom_ops.gdn import STATE_LANES
from rebel.ops.torch_custom_ops.kda import CONV_LANES
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.model_executor.models.qwen3_next import Qwen3NextAttention

from vllm_rbln.patches import register_patch
from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.patches.registry import add_registration
from vllm_rbln.v1.attention.backends.gdn_attn import (
    RBLNGDNAttentionBackend,
    RBLNGDNAttentionMetadata,
    gdn_state_shape,
)
from vllm_rbln.v1.kv_cache import RBLNBlockAuxSpec


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


class RBLNGDNConvRingCache(nn.Module, AttentionLayerBase):
    """A GatedDeltaNet layer's short-conv ring: a block-aux cache beside the
    layer's state, one ring a state block, addressed by the layer's block ids."""

    def __init__(
        self, prefix: str, owner: str, block_shape: tuple[int, ...], dtype: torch.dtype
    ) -> None:
        super().__init__()
        self.prefix = prefix
        self.owner = owner
        self.block_shape = block_shape
        self.dtype = dtype
        self.kv_cache = torch.tensor([])
        # This cache's position among the graph's KV cache inputs: the runner
        # assigns it once the caches exist.
        self.layer_index: int | None = None
        context = get_current_vllm_config().compilation_config.static_forward_context
        if prefix in context:
            raise ValueError(f"Duplicate layer name: {prefix}")
        context[prefix] = self

    def forward(self) -> None: ...

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> None:
        return None

    def get_aux_cache_spec(self, vllm_config: VllmConfig) -> RBLNBlockAuxSpec:
        return RBLNBlockAuxSpec(self.owner, self.block_shape, self.dtype)

    def get_attn_backend(self) -> type[RBLNGDNAttentionBackend]:
        return RBLNGDNAttentionBackend

    def inputs(self) -> torch.Tensor:
        metadata = get_forward_context().attn_metadata[self.owner]
        assert self.layer_index is not None, self.prefix
        return _resolve_kv_cache(metadata, self.layer_index)


@register_patch(
    target="vllm.model_executor.models.qwen3_next.QwenGatedDeltaNetAttention",
    key=f"{__name__}.RBLNQwenGatedDeltaNetAttention.qwen3_next",
    reason=(
        "Qwen3-Next builds its GatedDeltaNet layer from this module's name; the "
        "RBLN layer declares the delta-rule state as a full-attention KV block and "
        "runs the conv and the delta rule as RBLN custom kernels."
    ),
)
@register_patch(
    target="vllm.model_executor.models.qwen3_5.QwenGatedDeltaNetAttention",
    reason=(
        "Upstream's GatedDeltaNet layer declares a conv + ssm state page its "
        "Triton/CUDA kernels address and picks a forward by platform, with no "
        "out-of-tree branch; the RBLN layer declares the delta-rule state as a "
        "full-attention KV block, keeps the conv ring in a block-aux cache, and runs "
        "both as RBLN custom kernels on the group's block ids."
    ),
)
class RBLNQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    def __init__(
        self, config, vllm_config: VllmConfig, prefix: str = "", **kwargs
    ) -> None:
        super().__init__(config, vllm_config, prefix, **kwargs)
        if self.head_k_dim != self.head_v_dim or self.head_k_dim % STATE_LANES:
            raise NotImplementedError(
                "The RBLN GatedDeltaNet kernels take one head dim for keys and "
                f"values, a multiple of {STATE_LANES}, got {self.head_k_dim} / "
                f"{self.head_v_dim}"
            )
        # The state cache's position among the graph's KV cache inputs: the
        # runner assigns it once the caches exist.
        self.layer_index: int | None = None
        model_config = vllm_config.model_config
        self.num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.kv_head_size = model_config.get_head_size()
        # One ring a block, in kda_conv's stacked form ([W, P=1, C / 64, 64]): the
        # kernel takes it as allocated, and the converter's reshape of the
        # unstacked form cannot carry the cache's dynamic block count.
        channels = (2 * self.key_dim + self.value_dim) // self.tp_size
        self.conv_ring = RBLNGDNConvRingCache(
            f"{prefix}.conv_ring",
            owner=prefix,
            block_shape=(
                self.conv_kernel_size - 1,
                1,
                channels // CONV_LANES,
                CONV_LANES,
            ),
            dtype=vllm_config.model_config.dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        return (
            gdn_state_shape(
                self.num_kv_heads, self.cache_config.block_size, self.kv_head_size
            ),
        )

    def get_state_dtype(self) -> tuple[torch.dtype]:
        # The delta-rule kernels keep the state in the activation dtype (dlfloat16
        # inside the kernel), so the cache dtype options cannot move it.
        _, state_dtype = super().get_state_dtype()
        if state_dtype != self.model_config.dtype:
            raise NotImplementedError(
                "The RBLN GatedDeltaNet kernels keep the state in the activation "
                f"dtype {self.model_config.dtype}, got {state_dtype}"
            )
        return (state_dtype,)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
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
                f"{self.prefix}: hidden_states {tuple(hidden_states.shape)} do not "
                f"hold the step's batch {batch} x seq {seq_len} the GDN metadata "
                "describes"
            )
        assert self.layer_index is not None, self.prefix
        state = _resolve_kv_cache(metadata, self.layer_index)
        ring = self.conv_ring.inputs()
        tp_size = self.tp_size
        heads_k = self.num_k_heads // tp_size
        heads_v = self.num_v_heads // tp_size
        ratio = heads_v // heads_k
        key_dim = self.key_dim // tp_size
        value_dim = self.value_dim // tp_size
        channels = key_dim * 2 + value_dim
        head_dim = self.head_k_dim
        kernel = self.conv_kernel_size

        # Keep the leading dims the model uses ([batch, seq] on RBLN); only the
        # kernels need the explicit [batch, seq, ...] views.
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        gates, _ = self.in_proj_ba(hidden_states)
        qkv, z = mixed_qkvz.split([channels, value_dim], dim=-1)
        b, a = gates.chunk(2, dim=-1)

        # One conv over q | k | v, whose rows the model regrouped by key head at
        # load time (grouped_qkv_rows): RSD cuts the conv's channels where it cuts
        # the delta-rule kernels' heads, so a chiplet's conv output is what they read.
        convolved = torch.ops.rbln_custom_ops.kda_conv(
            qkv.reshape(batch, seq_len, 1, channels),
            self.conv1d.weight.view(1, channels, kernel).to(mixed_qkvz.dtype),
            ring,
            metadata.conv_state_index,
            metadata.conv_seq_idx,
            metadata.conv_query_len,
            metadata.conv_has_initial_state,
            kernel - 1,
            store_index=metadata.conv_store_index,
        )
        # The delta-rule kernels take q / k a key head (L2-normalized inside), v, g and
        # beta per value head grouped under its key head: value head h is key head
        # h // ratio's (h % ratio)-th, as HF's repeat_interleave pairs them.
        grouped = convolved.view(batch, seq_len, heads_k, 2 + ratio, head_dim)
        grouped = grouped.permute(0, 2, 3, 1, 4)
        query, key = (grouped[:, :, i].contiguous() for i in (0, 1))
        value = grouped[:, :, 2:].contiguous()

        beta = torch.sigmoid(b).reshape(batch, seq_len, heads_k, ratio)
        g = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias.float()
        )
        g = g.reshape(batch, seq_len, heads_k, ratio)
        beta, g = (x.permute(0, 2, 3, 1) for x in (beta, g))
        if metadata.num_prefills > 0:
            # Padded tokens must neither decay the state nor write to it.
            assert metadata.valid_mask is not None
            g = g * metadata.valid_mask
            beta = beta * metadata.valid_mask.to(beta.dtype)
        beta, g = beta.contiguous(), g.contiguous()
        if metadata.num_prefills > 0:
            core = torch.ops.rbln_custom_ops.chunk_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                state,
                metadata.prefill_state_indices,
                metadata.prefill_has_initial_state,
                GDN_PREFILL_CHUNK,
                store_index=metadata.store_state_indices,
            )
        else:
            core = torch.ops.rbln_custom_ops.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g,
                beta,
                state,
                metadata.non_spec_state_indices_tensor,
                store_index=metadata.store_state_indices,
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
