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

Per KDA layer, one slot per request (``RBLNSlidingWindowSpec``): the recurrent state
``[H * 128, 128]`` and the q / k / v short-conv rings ``[C * W / 64, 64]``.
Per DSA layer: the paged MLA latent cache ``[NB, block, 512]``, the paged kpool indexer
key ``[NB, block / 4, 128]`` fp8 and its scale ``[.., 1]`` f16, and the indexer's tail
ring ``[4, 256]`` one slot per request. All reuse ``RBLNFlashAttentionMetadataBuilder``.
"""

from typing import ClassVar

import torch
from vllm.v1.attention.backend import AttentionBackend, MultipleOf

from .flash_attention import RBLNFlashAttentionMetadataBuilder


class _RBLNGlm5CacheBackend(AttentionBackend):
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "bfloat16"]
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


class RBLNGlm5RingBackend(_RBLNGlm5CacheBackend):
    """A ``[rows, width]`` slot per request, declared as an ``RBLNSlidingWindowSpec`` of
    ``head_size = width / 2`` (the spec budgets K + V)."""

    @staticmethod
    def get_name() -> str:
        return "RBLN_GLM5_RING"

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
