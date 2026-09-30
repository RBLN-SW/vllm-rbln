# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 (CSA / HCA / SWA) cache backends for RBLN. KV8 only.

Per decoder layer (``compress_ratio`` 0 / 4 / 128):

* SWA ring        every layer   ``[N, window, 768]`` u8, one slot per request
* compressed KV   CSA, HCA      paged ``[NB, block / r, 768]`` u8
* indexer key     CSA           paged ``[NB, block / 4, 128]`` fp8 + ``[.., 1]`` f16 scale
* compressor state CSA, HCA     ``[N, W, 2C]`` f32 ring, one slot per request (and a
                                second one for the CSA indexer's compressor)

The 768-byte row is ``rbln_custom_ops``' KV8 row (``sparse_attn_deepseek_v4``). The ring
caches use ``RBLNSlidingWindowSpec`` (one block per request, prefix caching off); the paged ones
``MLAAttentionSpec`` with ``compress_ratio`` (a block of ``block_size`` tokens stores
``block_size / r`` entries). All reuse ``RBLNFlashAttentionMetadataBuilder``: the kernels take its
``seq_lens`` (cache position) and ``block_tables``, and the ring ones its ``local_block_tables``
and ``cache_seq_lens`` / ``cache_offsets`` (their difference is the step's query length).
"""

from typing import ClassVar

import torch
from vllm.v1.attention.backend import AttentionBackend, MultipleOf

from .flash_attention import RBLNFlashAttentionMetadataBuilder

# The packed KV8 row: 448 fp8 nope | 64 pad | 64 f16 scale lanes | 64 bf16 rope.
DSV4_KV8_ROW_BYTES = 768
DSV4_INDEX_HEAD_DIM = 128


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
        return RBLNFlashAttentionMetadataBuilder

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
    """SWA ring and compressor state: one ``[window, width]`` slot per request.

    The ring is declared as an ``RBLNSlidingWindowSpec`` of ``head_size = width / 2`` (the spec
    budgets K + V), so a page is one ``width``-wide row per position.
    """

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
        return (num_blocks, block_size, 2 * head_size)


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
