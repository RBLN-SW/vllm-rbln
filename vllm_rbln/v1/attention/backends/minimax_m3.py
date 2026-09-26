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

"""MiniMax-M3 sparse attention (MSA) backends for RBLN.

Two caches per sparse layer, mirroring the DSA (DeepSeek-V3.2) layout:

* the main paged GQA K/V cache, read by the block-sparse attention kernel
  (``rbln_custom_ops.sparse_attn_minimax_attn``), and
* the lightning-indexer's key-only side cache (one 128-wide vector per token),
  read by ``rbln_custom_ops.sparse_attn_minimax_indexer``.

Both reuse ``RBLNFlashAttentionMetadataBuilder``: the kernels take the same
``seq_lens`` (cache position) / ``block_tables`` the flash kernels take.
"""

from typing import ClassVar

import torch
from vllm.v1.attention.backend import AttentionBackend, MultipleOf

from vllm_rbln.logger import init_logger

from .flash_attention import RBLNFlashAttentionMetadataBuilder

logger = init_logger(__name__)

# The sparse block the indexer scores and picks. Fixed by the kernels.
MSA_SPARSE_BLOCK_SIZE = 128


class RBLNMiniMaxM3SparseBackend(AttentionBackend):
    """Main GQA K/V cache of a MiniMax-M3 sparse attention layer.

    The kernel expects the combined cache as ``[num_blocks, 2, H, 1, block, D]``
    (block-major so one paged block holds K then V of every kv head), unlike the
    flash backend's ``[2, num_blocks, H, 1, block, D]``.
    """

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16, torch.float16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
    accept_output_buffer: bool = False

    @staticmethod
    def get_name() -> str:
        return "RBLN_MINIMAX_M3_SPARSE"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        # The indexer scores 128-token blocks and CP shards the block 4-way
        # into 64-row chunks: a partition must hold whole sparse blocks on
        # every chiplet.
        return [MultipleOf(4 * MSA_SPARSE_BLOCK_SIZE)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        return (num_blocks, 2, num_kv_heads, 1, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3, 4, 5, 6)
        return (0, 1, 2, 3, 4, 5)


class RBLNMiniMaxM3IndexerBackend(AttentionBackend):
    """Key-only side cache of the MiniMax-M3 lightning indexer (bf16, or fp8 under KV8)."""

    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    supported_kv_cache_dtypes: ClassVar[list[str]] = ["auto", "fp8", "fp8_e4m3", "fp8_e5m2"]
    accept_output_buffer: bool = False

    @staticmethod
    def get_name() -> str:
        return "RBLN_MINIMAX_M3_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(4 * MSA_SPARSE_BLOCK_SIZE)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        return [128]

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNFlashAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        assert num_kv_heads == 1, "the indexer cache stores a single index key"
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)
