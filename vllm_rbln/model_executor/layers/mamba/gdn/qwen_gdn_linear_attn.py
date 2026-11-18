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
from rebel.ops.torch_custom_ops.gdn import STATE_LANES
from rebel.ops.torch_custom_ops.kda import CONV_LANES
from torch import nn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)

from vllm_rbln.patches.attention import _resolve_kv_cache
from vllm_rbln.v1.attention.backends.gdn_attn import (
    RBLNGDNAttentionBackend,
    RBLNGDNAttentionMetadata,
)
from vllm_rbln.v1.kv_cache import RBLNBlockAuxSpec


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
        # Index among the graph's KV cache inputs, assigned by the runner.
        self.layer_index: int | None = None
        self.kv_cache_subindex = 1
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


class RBLNQwenGatedDeltaNetAttention(QwenGatedDeltaNetAttention):
    """GatedDeltaNet for RBLN: the delta-rule state is declared as a
    full-attention KV block, the conv ring sits in a block-aux cache, and both
    run as RBLN custom kernels on the group's block ids."""

    def __init__(
        self, config, vllm_config: VllmConfig, prefix: str = "", **kwargs
    ) -> None:
        # Registered out of tree, so other models that build upstream's layer land
        # here too; only the RBLN Qwen3.5 class regroups the weights for it.
        architecture = vllm_config.model_config.architecture
        if architecture != "Qwen3_5ForConditionalGeneration":
            raise NotImplementedError(
                "The RBLN GatedDeltaNet layer covers the dense Qwen3.5 models "
                f"only; got {architecture}."
            )
        super().__init__(config, vllm_config, prefix, **kwargs)
        if self.head_k_dim != self.head_v_dim or self.head_k_dim % STATE_LANES:
            raise NotImplementedError(
                "The RBLN GatedDeltaNet kernels take one head dim for keys and "
                f"values, a multiple of {STATE_LANES}, got {self.head_k_dim} / "
                f"{self.head_v_dim}"
            )
        # Index among the graph's KV cache inputs, assigned by the runner.
        self.layer_index: int | None = None
        model_config = vllm_config.model_config
        self.num_kv_heads = model_config.get_num_kv_heads(vllm_config.parallel_config)
        self.kv_head_size = model_config.get_head_size()
        # One ring a block, in kda_conv's stacked form [W, 1, C / 64, 64]: a
        # reshape of the unstacked form cannot carry the dynamic block count.
        channels = self.conv_dim // self.tp_size
        self.conv_ring = RBLNGDNConvRingCache(
            f"{prefix}.conv_ring",
            owner=prefix,
            block_shape=(
                self.conv_kernel_size - 1,
                1,
                channels // CONV_LANES,
                CONV_LANES,
            ),
            dtype=model_config.dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        # A full-attention KV block, the flash backend's without the custom
        # kernel (which the platform refuses for a GDN model).
        # TODO(RBLN): use the linear value heads instead of self.num_kv_heads.
        return (
            (2, self.num_kv_heads, 1, self.cache_config.block_size, self.kv_head_size),
        )

    def get_state_dtype(self) -> tuple[torch.dtype]:
        # The delta-rule kernels take the state in the activation dtype.
        # TODO(RBLN): support cache_config.mamba_ssm_cache_dtype (float32 in the
        # Qwen3.5 HF config).
        return (self.model_config.dtype,)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch, seq, hidden] or [batch * seq, hidden]; the batch
                comes from the step's GDN metadata
        """
        metadata = get_forward_context().attn_metadata[self.prefix]
        assert isinstance(metadata, RBLNGDNAttentionMetadata), type(metadata)
        batch = metadata.batch_size
        num_tokens = hidden_states.numel() // hidden_states.shape[-1]
        assert num_tokens % batch == 0, (
            f"{self.prefix}: hidden_states {tuple(hidden_states.shape)} do not hold "
            f"the step's batch {batch}"
        )
        seq_len = num_tokens // batch
        assert self.layer_index is not None, self.prefix
        state = _resolve_kv_cache(metadata, self.layer_index)
        ring = self.conv_ring.inputs()
        tp_size = self.tp_size
        num_k_heads = self.num_k_heads // tp_size
        num_v_heads = self.num_v_heads // tp_size
        ratio = num_v_heads // num_k_heads
        value_dim = self.value_dim // tp_size
        channels = self.conv_dim // tp_size
        head_dim = self.head_k_dim
        kernel = self.conv_kernel_size

        # Keep the leading dims the model uses ([batch, seq] on RBLN); only the
        # kernels need the explicit [batch, seq, ...] views.
        mixed_qkvz, _ = self.in_proj_qkvz(hidden_states)
        gates, _ = self.in_proj_ba(hidden_states)
        qkv, z = mixed_qkvz.split([channels, value_dim], dim=-1)
        b, a = gates.chunk(2, dim=-1)

        # One conv over q | k | v in key-head order (grouped_qkv_rows).
        convolved = torch.ops.rbln_custom_ops.kda_conv(
            qkv.reshape(batch, seq_len, 1, channels),
            self.conv1d.weight.view(1, channels, kernel),
            ring,
            metadata.conv_load_blocks,
            metadata.conv_start_positions,
            metadata.conv_query_lens,
            metadata.has_initial_state,
            kernel - 1,
            store_index=metadata.conv_store_blocks,
        )
        # The delta-rule kernels take q / k per key head (L2-normalized inside) and
        # v, g, beta per value head, grouped under its key head as HF pairs them.
        grouped = convolved.view(batch, seq_len, num_k_heads, 2 + ratio, head_dim)
        grouped = grouped.permute(0, 2, 3, 1, 4)
        query, key = (grouped[:, :, i].contiguous() for i in (0, 1))
        value = grouped[:, :, 2:].contiguous()

        beta = torch.sigmoid(b).reshape(batch, seq_len, num_k_heads, ratio)
        g = -torch.exp(self.A_log.float()) * F.softplus(
            a.float() + self.dt_bias.float()
        )
        g = g.reshape(batch, seq_len, num_k_heads, ratio)
        beta, g = (x.permute(0, 2, 3, 1) for x in (beta, g))
        if metadata.is_prefill:
            # Padded tokens must neither decay the state nor write to it.
            assert metadata.valid_mask is not None
            valid_mask = metadata.valid_mask[..., :seq_len]
            g = g * valid_mask
            beta = beta * valid_mask.to(beta.dtype)
            core = torch.ops.rbln_custom_ops.chunk_gated_delta_rule(
                query,
                key,
                value,
                g.contiguous(),
                beta.contiguous(),
                state,
                metadata.load_blocks,
                metadata.has_initial_state,
                store_index=metadata.store_blocks,
            )
        else:
            core = torch.ops.rbln_custom_ops.recurrent_gated_delta_rule(
                query,
                key,
                value,
                g.contiguous(),
                beta.contiguous(),
                state,
                metadata.load_blocks,
                store_index=metadata.store_blocks,
            )
        # [batch, num_k_heads, ratio, seq, 128] -> [batch, seq, num_v_heads, 128]
        core = core.permute(0, 3, 1, 2, 4).reshape(
            batch, seq_len, num_v_heads, head_dim
        )
        z = z.reshape(batch, seq_len, num_v_heads, head_dim)
        gated = self.norm(core, z)
        output, _ = self.out_proj(gated.reshape(batch, seq_len, num_v_heads * head_dim))
        return output.reshape(*hidden_states.shape[:-1], output.shape[-1])
