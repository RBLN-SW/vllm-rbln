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

Paged caches use MLAAttentionSpec: compressed KV rows are 768 bytes, indexer
keys are 128 fp8 values, and each block holds block_size / compress_ratio rows.
The indexer scale is a block auxiliary. SWA and compressor rings use MambaSpec
with one state per request; prefix checkpoint copies are not supported yet.
GLM-5.3 metadata supplies block tables, positions and state slots.
"""

from dataclasses import dataclass, fields
from typing import ClassVar

import torch
from vllm.v1.attention.backend import AttentionBackend, MultipleOf
from vllm.v1.kv_cache_interface import MambaSpec

import vllm_rbln.utils as rbln_utils

from .flash_attention import RBLNFlashAttentionMetadataBuilder
from .glm5_next import RBLNGlm5NextMetadataBuilder, RBLNGlm5NextStateMetadata

# The packed KV8 row: 448 fp8 nope | 64 pad | 64 f16 scale lanes | 64 bf16 rope.
DSV4_KV8_ROW_BYTES = 768
DSV4_INDEX_HEAD_DIM = 128


@dataclass(frozen=True, kw_only=True)
class RBLNDeepseekV4StateSpec(MambaSpec):
    """A ring on a bounded state pool, independent of paged attention IDs."""

    num_state_slots: int


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
        # GLM-5.3's builder: cache_seq_lens / cache_offsets for every cache, and a
        # MambaSpec
        # cache's (the rings') load / store blocks
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

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)


class RBLNDeepseekV4RingBackend(_RBLNDeepseekV4CacheBackend):
    """SWA ring and compressor state: a ``MambaSpec`` state of one ``[window,
    width]`` ring a request (the runner shapes it from the spec); the slot is the
    state's block id."""

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


@dataclass
class RBLNDeepseekV4HCAMetadata(RBLNGlm5NextStateMetadata):
    # block_tables' values as int16, a tensor of its own: the HCA history walk
    # reads it as a const buffer, which must not be the graph input the cache
    # store reads on the device (one input holds one physical layout).
    hist_block_tables: torch.Tensor | None = None


class RBLNDeepseekV4HCAMetadataBuilder(RBLNGlm5NextMetadataBuilder):
    def build(self, common_attn_metadata, positions, batch_pad, is_prefill, **kwargs):
        metadata = super().build(
            common_attn_metadata, positions, batch_pad, is_prefill, **kwargs
        )
        # block_tables as the parent builds it: a 1-D table for a prefill, the batch
        # padded for a decode.
        block_table = common_attn_metadata.block_table_tensor
        if is_prefill:
            block_table = block_table[0]
        else:
            block_table = rbln_utils.pad(block_table, 0, batch_pad)
        return RBLNDeepseekV4HCAMetadata(
            **{f.name: getattr(metadata, f.name) for f in fields(metadata)},
            hist_block_tables=self._stage(
                block_table.to(torch.int16), "hist_block_tables"
            ),
        )


class RBLNDeepseekV4HCABackend(_RBLNDeepseekV4CacheBackend):
    """HCA compressed KV cache (ratio 128)."""

    compress_ratio: ClassVar[int] = 128

    @staticmethod
    def get_name() -> str:
        return "RBLN_DEEPSEEK_V4_HCA"

    @staticmethod
    def get_builder_cls() -> type["RBLNFlashAttentionMetadataBuilder"]:
        return RBLNDeepseekV4HCAMetadataBuilder


def dsv4_paged_backend(compress_ratio: int) -> type[_RBLNDeepseekV4CacheBackend]:
    if compress_ratio == 4:
        return RBLNDeepseekV4CSABackend
    if compress_ratio == 128:
        return RBLNDeepseekV4HCABackend
    raise ValueError(f"no paged DeepSeek-V4 cache for compress_ratio={compress_ratio}")
