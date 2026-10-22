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
"""GLM-5.3 (glm5_next) cache backends for RBLN.

The layout is vLLM's GLM-5.3 one (``_get_kv_cache_groups_glm5_next``), region by region
(design README 2.7, SoA):

* one attention group of every DSA layer's paged MLA latent ``[N, b, 512]`` uint8 (fp8)
  and kpool indexer key ``[N, b / 4, 128]`` fp8 / scale ``[N, b / 4, 64]`` f16;
* KDA state groups: a KDA layer's heads split into g head groups, each a ``MambaSpec``
  layer of one block a request, its state R1's bytes ``[N, b, 512]`` uint8 on the
  extent of a latent layer (the same page id is never both: one block pool).

The q / k / v conv rings, the MTP draft buffer and the kpool tail ring are block-aux
caches (``RBLNBlockAuxSpec``) addressed by their owner's block ids.
"""

from dataclasses import dataclass, fields
from typing import ClassVar

import torch
from vllm.v1.attention.backend import AttentionBackend, MultipleOf
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
class RBLNGlm5StateMetadata(RBLNFlashAttentionMetadata):
    """A KDA state group's metadata: the block each request's state (and the block-aux
    caches on it) is read from and the one it is written to, ``[B, 1]`` int32."""

    load_blocks: torch.Tensor | None = None
    store_blocks: torch.Tensor | None = None


class RBLNGlm5MetadataBuilder(RBLNFlashAttentionMetadataBuilder):
    """Stages each request's cached and total token counts (``cache_seq_lens``,
    ``cache_offsets``, unclamped) for every GLM-5.3 group, not only a ring's: the KDA
    state, the conv rings and the kpool tail take the step's valid tokens from them."""

    def build(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
        skip_attn_masks: bool = False,
        back_pad: torch.Tensor | None = None,
        **kwargs,
    ) -> RBLNFlashAttentionMetadata:
        metadata = super().build(
            common_attn_metadata,
            positions,
            batch_pad,
            is_prefill,
            skip_attn_masks=skip_attn_masks,
            back_pad=back_pad,
            **kwargs,
        )
        if metadata.cache_offsets is not None:
            return metadata
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
        load_blocks, store_blocks = self._state_blocks(
            common_attn_metadata.block_table_tensor[:num_reqs],
            num_computed_tokens[:num_reqs],
            seq_lens[:num_reqs],
            None if is_prefill else kwargs.get("num_accepted"),
            is_prefill,
        )
        if not is_prefill:
            load_blocks = rbln_utils.pad(load_blocks, 0, batch_pad)
            store_blocks = rbln_utils.pad(store_blocks, 0, batch_pad)
        return RBLNGlm5StateMetadata(
            **{f.name: getattr(metadata, f.name) for f in fields(metadata)},
            load_blocks=self._stage(load_blocks, "load_blocks"),
            store_blocks=self._stage(store_blocks, "store_blocks"),
        )

    def _state_blocks(
        self,
        block_table: torch.Tensor,
        num_computed_tokens: torch.Tensor,
        seq_lens: torch.Tensor,
        num_accepted: torch.Tensor | None,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The state blocks a step reads and writes, both one block a request when the
        spec's block holds the whole sequence. A step reads where the previous one
        wrote: a verify step writes the state after token 0 (position
        ``num_computed_tokens``), so the next one reads behind its accepted drafts,
        which it replays. A prefill writes at its last token, a decode or verify step
        at token 0."""
        block_size = self.kv_cache_spec.block_size
        accepted = (
            0 if num_accepted is None else num_accepted[: len(seq_lens)].view(-1, 1)
        )
        load_pos = num_computed_tokens - 1 - accepted
        store_pos = seq_lens - 1 if is_prefill else num_computed_tokens
        # A fresh request (no computed token) and a padding one (0 tokens) take column
        # 0; neither reads its state.
        load_col = torch.div(load_pos.clamp(min=0), block_size, rounding_mode="floor")
        store_col = torch.div(store_pos.clamp(min=0), block_size, rounding_mode="floor")
        return (
            torch.gather(block_table, 1, load_col.long()).to(torch.int32),
            torch.gather(block_table, 1, store_col.long()).to(torch.int32),
        )


class _RBLNGlm5CacheBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    # KV8 only: --kv-cache-dtype fp8 sets the MLA latent cache; the other caches keep
    # their own dtypes (bf16 rings, the fp8 / f16 indexer).
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["fp8", "fp8_e4m3"]
    accept_output_buffer: bool = False
    # Entries a paged block stores per `block_size` tokens (1 for the ring caches).
    compress_ratio: ClassVar[int] = 1

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def supported_kv_cache_layouts(cls) -> tuple[KVCacheLayout, ...]:
        # As the flash-attention backend's, so pipeline stages agree on one layout
        # (MiniMax-M3's caches do the same, #1317).
        return RBLNFlashAttentionBackend.supported_kv_cache_layouts()

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNGlm5MetadataBuilder

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

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


class RBLNGlm5StateBackend(_RBLNGlm5CacheBackend):
    """A KDA head group's recurrent state, one ``MambaSpec`` block a request (the
    runner shapes it from the spec), and the block-aux caches."""

    @staticmethod
    def get_name() -> str:
        return "RBLN_GLM5_STATE"


class RBLNGlm5LatentBackend(_RBLNGlm5CacheBackend):
    """The MLA latent cache, one 512-wide row per token."""

    @staticmethod
    def get_name() -> str:
        return "RBLN_GLM5_LATENT"


class RBLNGlm5IndexerBackend(_RBLNGlm5CacheBackend):
    """The kpool indexer key / scale cache, one row per 4-token pool."""

    compress_ratio: ClassVar[int] = 4

    @staticmethod
    def get_name() -> str:
        return "RBLN_GLM5_INDEXER"
