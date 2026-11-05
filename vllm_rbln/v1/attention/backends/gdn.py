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

"""GatedDeltaNet (Qwen3-Next, Qwen3.5 `linear_attention`) state on RBLN.

A layer's recurrent state is a `MambaSpec` block in a pool of its own
(`RBLNGDNStateSpec`, v1/kv_cache.py): one [Hk, R, 2, 128, 64] state a block
in the activation dtype (dlfloat16 inside the kernels), addressed by the
group's block table. Its short-conv ring is a block-aux cache on the same
block ids. The builder turns the block table into the rows the kernels read
and write: a request's one block (`mamba_block_size` is the model length),
read and written in place; a padding row reads and writes block 0, vLLM's
null block.

The delta-rule kernels take q / k per key head, L2-normalized inside, and v, g and
beta per value head grouped under its key head. The short conv
(`rbln_custom_ops::kda_conv`, fed as GLM-5.3 feeds its own) runs once over
q | k | v in that same key-head order (`grouped_qkv_rows`); the converter builds
its ring tables in the graph from the step's blocks and positions, and the ring
is stored to the block the state is.
"""

from dataclasses import dataclass

import torch
from rebel.ops.torch_custom_ops.gdn import STATE_BLOCKS
from vllm.config import VllmConfig
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.registry import (
    MambaAttentionBackendEnum,
    register_backend,
)
from vllm.v1.kv_cache_interface import KVCacheSpec


def gdn_state_shape(heads_k: int, heads_v: int) -> tuple[int, ...]:
    """One block's state as the delta-rule kernels keep it: each key head's
    `heads_v / heads_k` [HEAD_DIM, HEAD_DIM] states in the kernels' lane blocks."""
    # TODO(rebel_compiler#15378): the kernels then read the head dim off the
    # query (any multiple of STATE_LANES, Dk == Dv) and STATE_BLOCKS goes away:
    # take the head dim from the config and build [D / STATE_LANES, D, STATE_LANES].
    return (heads_k, heads_v // heads_k, *STATE_BLOCKS)


def grouped_qkv_rows(heads_k: int, ratio: int, head_dim: int) -> torch.Tensor:
    """The rows of [q heads | k heads | v heads] regrouped by key head: q_h, k_h,
    then key head h's `ratio` value heads, for each h. The short conv runs one
    call over this order, so RSD's cut of its channels is the delta-rule
    kernels' cut of their heads and no chiplet reads another's output."""
    q = torch.arange(heads_k).view(heads_k, 1)
    k = q + heads_k
    v = (torch.arange(heads_k * ratio) + 2 * heads_k).view(heads_k, ratio)
    heads = torch.cat([q, k, v], dim=1).reshape(-1, 1)
    return (heads * head_dim + torch.arange(head_dim)).reshape(-1)


@dataclass
class RBLNGDNAttentionMetadata:
    is_prefill: bool
    batch_size: int
    seq_len: int
    # [batch] int16 on the host: the state block each row loads, the one it
    # stores, and 1 where the loaded block carries a state. The delta-rule
    # kernels read them through the const buffer (a device copy crashes the
    # runtime).
    ssm_state_index: torch.Tensor
    ssm_store_index: torch.Tensor
    ssm_has_initial_state: torch.Tensor
    # [batch] int32 on the device: what kda_conv reads a row's ring by (its
    # block, the step's first position, its valid tokens, 1 where the ring
    # carries a state) and the block it stores the ring to.
    conv_state_index: torch.Tensor
    conv_seq_idx: torch.Tensor
    conv_query_len: torch.Tensor
    conv_has_initial_state: torch.Tensor
    conv_store_index: torch.Tensor
    # [1, 1, seq_len] float32 on prefill: 1 for valid tokens, 0 for padding
    valid_mask: torch.Tensor | None


@register_backend(MambaAttentionBackendEnum.GDN_ATTN, is_mamba=True)
class RBLNGDNAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name() -> str:
        return "RBLN_GDN_ATTN"

    @staticmethod
    def get_builder_cls() -> type["RBLNGDNAttentionMetadataBuilder"]:
        return RBLNGDNAttentionMetadataBuilder

    @classmethod
    def is_ssm(cls) -> bool:
        return True


class RBLNGDNAttentionMetadataBuilder(
    AttentionMetadataBuilder[RBLNGDNAttentionMetadata]
):
    def __init__(
        self,
        kv_cache_spec: KVCacheSpec,
        layer_names: list[str],
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)
        self.prefill_chunk_size = vllm_config.scheduler_config.max_num_batched_tokens
        self._staged: dict[tuple, torch.Tensor] = {}

    def _stage(self, t: torch.Tensor, slot: str) -> torch.Tensor:
        """Copy a host tensor into this builder's persistent device buffer, keyed
        by name so same-shaped tensors never share one (see the flash builder)."""
        key = (slot, t.shape, t.dtype)
        if (buf := self._staged.get(key)) is None:
            buf = torch.empty(t.shape, dtype=t.dtype, device=self.device)
            self._staged[key] = buf
        buf.copy_(t)
        return buf

    def build(  # type: ignore[override]
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
    ) -> RBLNGDNAttentionMetadata:
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc[: num_reqs + 1]
        query_lens = query_start_loc[1:] - query_start_loc[:-1]
        seq_lens = common_attn_metadata.seq_lens[:num_reqs]
        num_computed = seq_lens - query_lens
        first_positions = positions[query_start_loc[:num_reqs]]

        if is_prefill:
            assert num_reqs == 1, "RBLN prefills one request per step"
            batch, seq_len = 1, self.prefill_chunk_size
        else:
            batch, seq_len = batch_pad, 1
            assert int(query_lens.max()) == 1, (
                "a decode step takes one token per request"
            )

        def padded(values: torch.Tensor) -> torch.Tensor:
            out = torch.zeros(batch, dtype=torch.int32)
            out[:num_reqs] = values.to(torch.int32)
            return out

        # TODO(prefix caching): one column while mamba_block_size is the model
        # length, so a row stores to the block it loads; a step crossing into a
        # new block stores to the block of its last scheduled token instead.
        blocks = padded(common_attn_metadata.block_table_tensor[:num_reqs, 0])

        valid_mask = None
        if is_prefill:
            valid_mask = (torch.arange(seq_len) < query_lens[0]).to(torch.float32)
            valid_mask = valid_mask.view(1, 1, seq_len)
        has_initial_state = padded(num_computed > 0)
        return RBLNGDNAttentionMetadata(
            is_prefill=is_prefill,
            batch_size=batch,
            seq_len=seq_len,
            ssm_state_index=blocks.to(torch.int16),
            ssm_store_index=blocks.to(torch.int16),
            ssm_has_initial_state=has_initial_state.to(torch.int16),
            conv_state_index=self._stage(blocks, "conv_state_index"),
            conv_seq_idx=self._stage(padded(first_positions), "conv_seq_idx"),
            conv_query_len=self._stage(padded(query_lens), "conv_query_len"),
            conv_has_initial_state=self._stage(
                has_initial_state, "conv_has_initial_state"
            ),
            conv_store_index=self._stage(blocks, "conv_store_index"),
            valid_mask=None
            if valid_mask is None
            else self._stage(valid_mask, "valid_mask"),
        )
