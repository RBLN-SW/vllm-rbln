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

from dataclasses import replace

import torch
import torch.nn.functional as F
from rebel.ops.torch_custom_ops.gdn import STATE_LANES
from rebel.ops.torch_custom_ops.kda import CONV_LANES
from vllm.config import get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_mixer2 import MambaMixer2
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_rbln.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    RBLNGDNBlockAuxCache,
)
from vllm_rbln.v1.attention.backends.mamba2_attn import (
    RBLNMamba2AttentionBackend,
    RBLNMamba2AttentionMetadata,
)
from vllm_rbln.v1.attention.kv_cache_bindings import resolve_kv_cache


class RBLNMamba2BlockAuxCache(RBLNGDNBlockAuxCache):
    """A short-conv ring of a Mamba-2 layer, under its owner's backend."""

    def get_attn_backend(self) -> type[RBLNMamba2AttentionBackend]:
        return RBLNMamba2AttentionBackend


class RBLNMambaMixer2(MambaMixer2):
    def __init__(self, *args, **kwargs) -> None:
        # Registered out of tree, so every model that builds upstream's layer
        # lands here; only Nemotron-H's forward and platform rules are covered.
        architecture = get_current_vllm_config().model_config.architecture
        if architecture != "NemotronHForCausalLM":
            raise NotImplementedError(
                f"The RBLN Mamba-2 layer covers Nemotron-H only; got {architecture}."
            )
        super().__init__(*args, **kwargs)
        if self.n_groups % self.tp_size:
            raise NotImplementedError(
                "The RBLN Mamba-2 layer shards whole groups over TP ranks; got "
                f"{self.n_groups} groups over tensor_parallel_size={self.tp_size}"
            )
        for label, dim in (("state", self.ssm_state_size), ("head", self.head_dim)):
            if dim % STATE_LANES:
                raise NotImplementedError(
                    f"The RBLN Mamba-2 kernels take a {label} size multiple of "
                    f"{STATE_LANES}, got {dim}"
                )
        if self.activation != "silu":
            raise NotImplementedError(
                f"The RBLN short conv applies SiLU, got activation {self.activation}"
            )
        model_config = self.model_config
        assert model_config is not None
        parallel_config = get_current_vllm_config().parallel_config
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.kv_head_size = model_config.get_head_size()
        self.local_groups = self.n_groups // self.tp_size
        self.local_heads = self.num_heads // self.tp_size
        # W holds the taps before a step's first token and its MTP drafts.
        width = self.conv_kernel_size - 1 + self.num_spec
        group_channels = self.groups_ssm_state_size // self.tp_size
        self.conv_ring_x = RBLNMamba2BlockAuxCache(
            f"{self.prefix}.conv_ring_x",
            owner=self.prefix,
            block_shape=(
                width,
                1,
                self.tped_intermediate_size // CONV_LANES,
                CONV_LANES,
            ),
            dtype=model_config.dtype,
        )
        self.conv_ring_bc = RBLNMamba2BlockAuxCache(
            f"{self.prefix}.conv_ring_bc",
            owner=self.prefix,
            block_shape=(width, 2, group_channels // CONV_LANES, CONV_LANES),
            dtype=model_config.dtype,
        )

    def get_state_shape(self) -> tuple[tuple[int, ...]]:
        # A full-attention KV block, as RBLNQwenGatedDeltaNetAttention declares
        # its state: the kernels keep a rank's states at the start of the page.
        assert self.cache_config is not None
        return (
            (2, self.num_kv_heads, 1, self.cache_config.block_size, self.kv_head_size),
        )

    def get_state_dtype(self) -> tuple[torch.dtype]:
        # The Mamba-2 kernels take the state in the activation dtype.
        # TODO(RBLN): support cache_config.mamba_ssm_cache_dtype (float32 in the
        # Nemotron-H HF configs).
        assert self.model_config is not None
        return (self.model_config.dtype,)

    def get_kv_cache_spec(self, vllm_config) -> KVCacheSpec | None:
        spec = super().get_kv_cache_spec(vllm_config)
        return replace(spec, num_speculative_blocks=0)

    def forward(
        self,
        hidden_states: torch.Tensor,
        mup_vector: torch.Tensor | None = None,
    ) -> torch.Tensor:
        projected_states, _ = self.in_proj(hidden_states)
        if mup_vector is not None:
            projected_states = projected_states * mup_vector
        ssm_output = torch.empty(
            [*hidden_states.shape[:-1], self.tped_intermediate_size],
            dtype=hidden_states.dtype,
            device=hidden_states.device,
        )
        self.conv_ssm_forward(projected_states, ssm_output)
        gate = projected_states[..., : self.tped_intermediate_size]
        hidden_states = self.norm(ssm_output, gate)
        output, _ = self.out_proj(hidden_states)
        return output

    def conv_ssm_forward(
        self,
        projected_states: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        metadata = get_forward_context().attn_metadata[self.prefix]
        assert isinstance(metadata, RBLNMamba2AttentionMetadata), type(metadata)
        batch = metadata.batch_size
        num_tokens = projected_states.numel() // projected_states.shape[-1]
        assert num_tokens % batch == 0, (
            f"{self.prefix}: {tuple(projected_states.shape)} do not hold the "
            f"step's batch {batch}"
        )
        seq_len = num_tokens // batch
        groups, heads = self.local_groups, self.local_heads
        ratio = heads // groups
        inner = self.tped_intermediate_size
        group_channels = self.groups_ssm_state_size // self.tp_size
        state_dim, head_dim = self.ssm_state_size, self.head_dim
        kernel = self.conv_kernel_size

        hidden_states_B_C, dt = torch.split(
            projected_states[..., inner:],
            [self.tped_conv_size, self.tped_dt_size],
            dim=-1,
        )
        hidden_states, B, C = self.split_hidden_states_B_C_fn(hidden_states_B_C)

        # 1. Convolution sequence transformation, x and B | C on rings of
        # their own, each conv1d row's taps and bias with its channel.
        conv_weights = self.conv1d.weight.view(-1, kernel)
        conv_bias = self.conv1d.bias
        conv_operands = (
            metadata.conv_load_blocks,
            metadata.conv_start_positions,
            metadata.conv_query_lens,
            metadata.has_initial_state,
            kernel - 1 + self.num_spec,
        )
        hidden_states = torch.ops.rbln_custom_ops.kda_conv(
            hidden_states.reshape(batch, seq_len, 1, inner),
            conv_weights[:inner].view(1, inner, kernel),
            self.conv_ring_x.inputs(),
            *conv_operands,
            store_index=metadata.conv_store_blocks,
            bias=None if conv_bias is None else conv_bias[:inner].view(1, inner),
        )
        B_C = torch.ops.rbln_custom_ops.kda_conv(
            torch.stack(
                [
                    B.reshape(batch, seq_len, group_channels),
                    C.reshape(batch, seq_len, group_channels),
                ],
                dim=2,
            ),
            conv_weights[inner:].view(2, group_channels, kernel),
            self.conv_ring_bc.inputs(),
            *conv_operands,
            store_index=metadata.conv_store_blocks,
            bias=None if conv_bias is None else conv_bias[inner:].view(2, -1),
        )

        # 2. State space model sequence transformation: q = C, k = B and
        # v = dt * x a group, g = dt * A a head; the D skip outside the kernel.
        B_C = B_C.view(batch, seq_len, 2, groups, state_dim).permute(2, 0, 3, 1, 4)
        key, query = B_C[0].contiguous(), B_C[1].contiguous()
        hidden_states = hidden_states.view(batch, seq_len, groups, ratio, head_dim)
        dt = F.softplus(dt.float() + self.dt_bias.float())
        dt = dt.reshape(batch, seq_len, groups, ratio)
        g = (dt * self.A.float().view(groups, ratio)).permute(0, 2, 3, 1)
        value = (hidden_states.float() * dt.unsqueeze(-1)).to(hidden_states.dtype)
        value = value.permute(0, 2, 3, 1, 4)
        state = resolve_kv_cache(metadata, self.prefix)
        if metadata.is_prefill:
            assert metadata.valid_mask is not None
            valid_mask = metadata.valid_mask[..., :seq_len]
            g = g * valid_mask
            value = value * valid_mask.unsqueeze(-1).to(value.dtype)
            core = torch.ops.rbln_custom_ops.mamba2_prefill(
                query,
                key,
                value.contiguous(),
                g.contiguous(),
                state,
                metadata.load_blocks,
                metadata.has_initial_state,
                store_index=metadata.store_blocks,
            )
        else:
            core = torch.ops.rbln_custom_ops.mamba2_decode(
                query,
                key,
                value.contiguous(),
                g.contiguous(),
                state,
                metadata.load_blocks,
                store_index=metadata.store_blocks,
            )
        # [batch, groups, ratio, seq, P] -> [batch, seq, heads, P]
        core = core.permute(0, 3, 1, 2, 4).reshape(batch, seq_len, heads, head_dim)
        skip = hidden_states.reshape(batch, seq_len, heads, head_dim) * self.D.view(
            heads, 1
        ).to(hidden_states.dtype)
        output.copy_((core + skip).reshape(output.shape))
