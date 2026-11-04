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
(`RBLNGDNStateSpec`, v1/kv_cache.py): one [Hk, R, 2, 128, 64] state of
dlfloat16 bits (int16) a block, addressed by the group's block table. Its
short-conv ring is a block-aux cache on the same block ids. The builder turns
the block table into the rows the kernels read and write: a request loads the
block of its last computed token and stores the block of its last scheduled
one; a padding row does both on block 0, vLLM's null block.

The cores take q / k per key head, L2-normalized in the kernel, and v, g and
beta per value head grouped under its key head. The short conv
(`rbln_custom_ops::causal_conv1d_ring_tabled`) runs once over q | k | v in that
same key-head order (`grouped_qkv_rows`); its index tables are built here, once
a step for every layer of the group.
"""

from dataclasses import dataclass

import torch
from rebel.ops.torch_custom_ops.causal_conv1d_ring import ring_tables
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

# The core kernels take 128-wide heads; the CR13 prefill kernel runs 128-token
# chunks only.
GDN_HEAD_DIM = 128
GDN_PREFILL_CHUNK = 128


def gdn_state_shape(heads_k: int, heads_v: int) -> tuple[int, ...]:
    """One block's state as the CR13 cores keep it: each key head's
    `heads_v / heads_k` [128, 128] states in [2, 128, 64] dlfloat16 lanes."""
    return (heads_k, heads_v // heads_k, GDN_HEAD_DIM // 64, GDN_HEAD_DIM, 64)


def grouped_qkv_rows(heads_k: int, ratio: int, head_dim: int) -> torch.Tensor:
    """The rows of [q heads | k heads | v heads] regrouped by key head: q_h, k_h,
    then key head h's `ratio` value heads, for each h. The short conv runs one
    call over this order, so RSD's cut of its channels is the cores' cut of
    their heads and no chiplet reads another's output."""
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
    # stores, and 1 where the loaded block carries a state. The core kernels
    # read them through the const buffer (a device copy crashes the runtime).
    core_state_index: torch.Tensor
    core_store_index: torch.Tensor
    core_has_initial_state: torch.Tensor
    # The conv ring's tables (ring_tables): hist_idx / ring_idx staged on the
    # device, block / count on the host (the kernel's const buffers)
    conv_tables: tuple[torch.Tensor, ...]
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
        layers = vllm_config.compilation_config.static_forward_context
        self.conv_kernel_size = layers[layer_names[0]].conv_kernel_size
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

        # A fresh request (no computed token) takes column 0 for its load,
        # which has_initial_state tells the kernel not to read.
        cols = torch.stack([num_computed - 1, seq_lens - 1], dim=1).clamp(min=0)
        cols = cols // self.kv_cache_spec.block_size
        blocks = common_attn_metadata.block_table_tensor[:num_reqs].gather(
            1, cols.long()
        )
        load_blocks = padded(blocks[:, 0])
        store_blocks = padded(blocks[:, 1])
        # The conv ring's tables name one block, read and written in place; a
        # store block of its own waits on the ring kernel's store operand.
        assert torch.equal(load_blocks, store_blocks), (
            "a GatedDeltaNet step crossing into a new state block needs the conv "
            "ring kernel's store operand; keep mamba_block_size at max_model_len"
        )

        valid_mask = None
        if is_prefill:
            valid_mask = (torch.arange(seq_len) < query_lens[0]).to(torch.float32)
            valid_mask = valid_mask.view(1, 1, seq_len)
        has_initial_state = padded(num_computed > 0)
        # A ring block holds one ring of width K - 1.
        width = self.conv_kernel_size - 1
        hist_idx, ring_idx, block, count = ring_tables(
            load_blocks,
            padded(first_positions),
            padded(query_lens),
            has_initial_state,
            seq_len,
            width,
            self.conv_kernel_size,
            width,
        )
        return RBLNGDNAttentionMetadata(
            is_prefill=is_prefill,
            batch_size=batch,
            seq_len=seq_len,
            core_state_index=load_blocks.to(torch.int16),
            core_store_index=store_blocks.to(torch.int16),
            core_has_initial_state=has_initial_state.to(torch.int16),
            conv_tables=(
                self._stage(hist_idx, "hist_idx"),
                self._stage(ring_idx, "ring_idx"),
                block,
                count,
            ),
            valid_mask=None
            if valid_mask is None
            else self._stage(valid_mask, "valid_mask"),
        )
