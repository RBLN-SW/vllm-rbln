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


@dataclass
class RBLNGlm5NextStateMetadata(RBLNFlashAttentionMetadata):
    load_blocks: torch.Tensor | None = None
    store_blocks: torch.Tensor | None = None


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
        return RBLNGlm5NextStateMetadata(
            **{f.name: getattr(metadata, f.name) for f in fields(metadata)},
            load_blocks=self._stage(load_blocks, "load_blocks"),
            store_blocks=self._stage(store_blocks, "store_blocks"),
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
