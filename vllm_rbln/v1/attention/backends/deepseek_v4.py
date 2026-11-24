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

"""DeepSeek-V4 KV8 cache backends for CSA, HCA and SWA.

Paged caches use MLAAttentionSpec: compressed KV rows are 768 bytes, indexer keys
are 128 fp8 values, and each block holds block_size / compress_ratio rows. The
indexer scale is a block auxiliary.

The SWA cache and the compressor states are sliding-window caches
(RBLNDeepseekV4WindowSpec): rows at the token's absolute position in blocks of
their own size, on a bounded pool of their own (independent of the paged
attention IDs), managed by upstream's SlidingWindowManager. A step reads the two
W-row segments before it (W = 128 for SWA, coff * r for a compressor), so a
request keeps the blocks of its last 2 W tokens. RBLNDeepseekV4StateSpec (one
ring slot per request) is the earlier layout, kept for its prefix checkpoints.
"""

from dataclasses import dataclass
from typing import ClassVar

import torch
from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import CircularBufferSpec, SlidingWindowSpec

import vllm_rbln.utils as rbln_utils

from .flash_attention import (
    RBLNFlashAttentionMetadata,
    RBLNFlashAttentionMetadataBuilder,
)

# The packed KV8 row: 448 fp8 nope | 64 pad | 64 f16 scale lanes | 64 bf16 rope.
DSV4_KV8_ROW_BYTES = 768
DSV4_INDEX_HEAD_DIM = 128


@dataclass(frozen=True, kw_only=True)
class RBLNDeepseekV4StateSpec(CircularBufferSpec):
    """A ring on a bounded state pool, independent of paged attention IDs."""

    num_state_slots: int
    prefix_caching: bool = False
    is_draft: bool = False
    head_size_v: int = 0
    dcp_sharded: bool = False
    max_tp_shards: int | None = 1

    @property
    def prefix_cacheable(self) -> bool:
        return self.prefix_caching

    @property
    def max_requests(self) -> int:
        """Requests the pool holds at once (slot 0 is its null block)."""
        return self.num_state_slots - 1


@dataclass(frozen=True, kw_only=True)
class RBLNDeepseekV4WindowSpec(SlidingWindowSpec):
    """A sliding-window cache on a bounded pool of its own (num_state_slots blocks,
    block 0 its null block), independent of the paged attention IDs.

    sliding_window is 2 W, the rows a step reads before it (two W-row segments),
    so SlidingWindowManager keeps exactly the blocks the kernels read; the
    attention window itself (128 tokens) is the kernels'. blocks_per_request is
    what one request holds at most (its window plus the largest step)."""

    num_state_slots: int
    blocks_per_request: int
    head_size_v: int = 0
    dcp_sharded: bool = False
    max_tp_shards: int | None = 1

    @property
    def max_requests(self) -> int:
        return (self.num_state_slots - 1) // self.blocks_per_request

    def max_memory_usage_bytes(self, vllm_config) -> int:
        return self.blocks_per_request * self.page_size_bytes


# The caches on a fixed pool of their own (outside dynamic KV sizing).
DSV4_FIXED_POOL_SPECS = (RBLNDeepseekV4StateSpec, RBLNDeepseekV4WindowSpec)


class RBLNDeepseekV4MetadataBuilder(RBLNFlashAttentionMetadataBuilder):
    def build(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        positions: torch.Tensor,
        batch_pad: int,
        is_prefill: bool,
        **kwargs,
    ) -> RBLNFlashAttentionMetadata:
        metadata = super().build(
            common_attn_metadata, positions, batch_pad, is_prefill, **kwargs
        )
        num_reqs = common_attn_metadata.num_reqs
        starts = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens[:num_reqs].view(-1, 1)
        query_lens = (starts[1 : num_reqs + 1] - starts[:num_reqs]).view(-1, 1)
        num_computed_tokens = seq_lens - query_lens
        if (back_pad := kwargs.get("back_pad")) is not None:
            num_computed_tokens = num_computed_tokens + back_pad[:num_reqs].view(-1, 1)
        if not is_prefill:
            num_computed_tokens = rbln_utils.pad(num_computed_tokens, 0, batch_pad)
            seq_lens = rbln_utils.pad(seq_lens, 0, batch_pad)
        metadata.cache_seq_lens = self._stage(num_computed_tokens, "cache_seq_lens")
        metadata.cache_offsets = self._stage(seq_lens, "cache_offsets")
        return metadata


class _RBLNDeepseekV4CacheBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["fp8", "fp8_e4m3"]
    accept_output_buffer: bool = False
    # Entries a paged block stores per `block_size` tokens (1 for the ring caches).
    compress_ratio: ClassVar[int] = 1

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return []

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return True

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNDeepseekV4MetadataBuilder

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


class RBLNDeepseekV4RingBackend(_RBLNDeepseekV4CacheBackend):
    """[blocks, rows, width] row caches: the sliding-window caches (a row a token)
    and the earlier one-slot rings."""

    @staticmethod
    def get_name() -> str:
        return "RBLN_DEEPSEEK_V4_RING"

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
        return (num_blocks, block_size, head_size)


class RBLNDeepseekV4CSABackend(_RBLNDeepseekV4CacheBackend):
    """CSA compressed KV / indexer key cache (ratio 4)."""

    compress_ratio: ClassVar[int] = 4

    @staticmethod
    def get_name() -> str:
        return "RBLN_DEEPSEEK_V4_CSA"


class RBLNDeepseekV4HCABackend(_RBLNDeepseekV4CacheBackend):
    """HCA compressed KV cache (ratio 128)."""

    compress_ratio: ClassVar[int] = 128

    @staticmethod
    def get_name() -> str:
        return "RBLN_DEEPSEEK_V4_HCA"


def dsv4_paged_backend(compress_ratio: int) -> type[_RBLNDeepseekV4CacheBackend]:
    if compress_ratio == 4:
        return RBLNDeepseekV4CSABackend
    if compress_ratio == 128:
        return RBLNDeepseekV4HCABackend
    raise ValueError(f"no paged DeepSeek-V4 cache for compress_ratio={compress_ratio}")
