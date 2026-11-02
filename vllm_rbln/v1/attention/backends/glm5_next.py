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

from dataclasses import dataclass, fields
from typing import ClassVar

import torch
from vllm.config import VllmConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.v1.attention.backend import AttentionBackend
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.kv_cache_layout import KVCacheLayout

import vllm_rbln.utils as rbln_utils

from .flash_attention import (
    RBLNFlashAttentionBackend,
    RBLNFlashAttentionMetadata,
    RBLNFlashAttentionMetadataBuilder,
)


def kda_state_layout(
    block_size: int, kv_lora_rank: int, num_heads: int, state_bytes: int
) -> tuple[int, int] | None:
    """(heads a state group holds, groups a block holds) for a rank's `num_heads`
    KDA states in an fp8 MLA latent block, or None when whole states do not tile it.

    A block too big for the rank's heads holds several groups' worth and the
    rank uses one of them (`kda_state_slot`), so the block size can stay the one
    a peer of another TP degree runs.
    """
    group_heads, rem = divmod(block_size * kv_lora_rank, state_bytes)
    if rem or not group_heads:
        return None
    if num_heads % group_heads == 0:
        return group_heads, 1
    if group_heads % num_heads == 0:
        return num_heads, group_heads // num_heads
    return None


def kda_state_slot(vllm_config: VllmConfig) -> tuple[int, int]:
    """(state slots a block holds, the one this rank uses)."""
    text_config = vllm_config.model_config.hf_text_config
    if text_config.model_type != "glm5_next_text":
        return 1, 0
    layout = kda_state_layout(
        vllm_config.cache_config.block_size,
        text_config.kv_lora_rank,
        text_config.linear_num_heads
        // vllm_config.parallel_config.tensor_parallel_size,
        text_config.linear_head_dim**2 * vllm_config.model_config.dtype.itemsize,
    )
    assert layout is not None
    slots = layout[1]
    return slots, get_tensor_model_parallel_rank() % slots


def kda_state_band(vllm_config: VllmConfig) -> tuple[int, int] | None:
    """(model-wide KDA heads, bytes one head's state takes), or None off GLM-5.3."""
    text_config = vllm_config.model_config.hf_text_config
    if text_config.model_type != "glm5_next_text":
        return None
    return (
        text_config.linear_num_heads,
        text_config.linear_head_dim**2 * vllm_config.model_config.dtype.itemsize,
    )


@dataclass
class RBLNGlm5NextStateMetadata(RBLNFlashAttentionMetadata):
    load_blocks: torch.Tensor | None = None
    store_blocks: torch.Tensor | None = None
    # load / store_blocks in units of a block's state slots, for the KDA kernels.
    load_slots: torch.Tensor | None = None
    store_slots: torch.Tensor | None = None


class RBLNGlm5NextMetadataBuilder(RBLNFlashAttentionMetadataBuilder):
    def build(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
        skip_attn_masks: bool = False,
        back_pad: torch.Tensor | None = None,
        is_prompt_sorted: bool = False,
        num_accepted: torch.Tensor | None = None,
        state_load_pos: torch.Tensor | None = None,
    ) -> RBLNFlashAttentionMetadata:
        metadata = super().build(
            common_attn_metadata,
            positions,
            batch_pad,
            is_prefill,
            skip_attn_masks=skip_attn_masks,
            back_pad=back_pad,
            is_prompt_sorted=is_prompt_sorted,
            num_accepted=num_accepted,
        )
        num_reqs = common_attn_metadata.num_reqs
        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens[:num_reqs].view(-1, 1)
        query_lens = (
            query_start_loc[1 : num_reqs + 1] - query_start_loc[:num_reqs]
        ).view(-1, 1)
        num_computed_tokens = seq_lens - query_lens
        if back_pad is not None:
            num_computed_tokens = num_computed_tokens + back_pad[:num_reqs].view(-1, 1)
        if not is_prefill:
            num_computed_tokens = rbln_utils.pad(num_computed_tokens, 0, batch_pad)
            seq_lens = rbln_utils.pad(seq_lens, 0, batch_pad)
        metadata.cache_seq_lens = self._stage(num_computed_tokens, "cache_seq_lens")
        metadata.cache_offsets = self._stage(seq_lens, "cache_offsets")
        if not isinstance(self.kv_cache_spec, MambaSpec):
            return metadata
        num_computed_tokens = num_computed_tokens[:num_reqs]
        seq_lens = seq_lens[:num_reqs]
        load_pos = num_computed_tokens - 1
        if state_load_pos is not None:
            load_pos = state_load_pos[:num_reqs].view(-1, 1)
        store_pos = seq_lens - 1
        # A fresh request (no computed token) and a padding one (0 tokens) take column
        # 0; neither reads its state.
        cols = torch.cat([load_pos, store_pos], dim=1).clamp(min=0)
        cols = cols // self.kv_cache_spec.block_size
        block_table = common_attn_metadata.block_table_tensor[:num_reqs]
        load_blocks, store_blocks = (
            block_table.gather(1, cols.long()).to(torch.int32).split(1, dim=1)
        )
        if not is_prefill:
            load_blocks = rbln_utils.pad(load_blocks, 0, batch_pad)
            store_blocks = rbln_utils.pad(store_blocks, 0, batch_pad)
        slots, slot = kda_state_slot(self.vllm_config)
        load_slots, store_slots = (
            self._stage(blocks * slots + slot, f"{name}_slots") if slots > 1 else None
            for blocks, name in ((load_blocks, "load"), (store_blocks, "store"))
        )
        load_blocks = self._stage(load_blocks, "load_blocks")
        store_blocks = self._stage(store_blocks, "store_blocks")
        return RBLNGlm5NextStateMetadata(
            **{f.name: getattr(metadata, f.name) for f in fields(metadata)},
            load_blocks=load_blocks,
            store_blocks=store_blocks,
            load_slots=load_blocks if load_slots is None else load_slots,
            store_slots=store_blocks if store_slots is None else store_slots,
        )


class RBLNGlm5NextCacheBackend(AttentionBackend):
    """A GLM-5.3 cache with no attention of its own: a KDA head group's state (one
    ``MambaSpec`` block a request, which the runner shapes from the spec), the MLA
    latent cache (one 512-wide row per token) and the block-aux caches."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["fp8", "fp8_e4m3"]
    compress_ratio: ClassVar[int] = 1

    @staticmethod
    def get_name() -> str:
        return "RBLN_GLM5_NEXT"

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        return RBLNFlashAttentionBackend.supported_kv_cache_layouts()

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNGlm5NextMetadataBuilder

    @classmethod
    def get_kv_cache_shape(
        cls,
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1
        assert block_size % cls.compress_ratio == 0
        return (num_blocks, block_size // cls.compress_ratio, head_size)


class RBLNGlm5NextIndexerBackend(RBLNGlm5NextCacheBackend):
    compress_ratio: ClassVar[int] = 4

    @staticmethod
    def get_name() -> str:
        return "RBLN_GLM5_NEXT_INDEXER"
