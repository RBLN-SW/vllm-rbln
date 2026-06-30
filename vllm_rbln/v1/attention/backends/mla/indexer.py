# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# RBLN backend for the sparse-attention lightning indexer key cache.

from typing import ClassVar

import torch
from vllm.v1.attention.backend import AttentionBackend, MultipleOf

from vllm_rbln.logger import init_logger

from ..flash_attention import RBLNFlashAttentionMetadataBuilder

logger = init_logger(__name__)


class RBLNDeepseekV32IndexerBackend(AttentionBackend):
    # TODO(kblee): check no quant, so bf16
    supported_dtypes: ClassVar[list[torch.dtype]] = [torch.bfloat16]
    accept_output_buffer: bool = False

    @staticmethod
    def get_name() -> str:
        # vLLM 0.22: AttentionBackend requires get_name (abstract).
        return "RBLN_DEEPSEEK_V32_INDEXER"

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(64)]

    @classmethod
    def get_supported_head_sizes(cls) -> list[int]:
        # TODO(kblee): check no quant in k indexer cache
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
        assert num_kv_heads == 1, "indexer cache stores a single latent vector"
        return (num_blocks, block_size, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (0, 1, 2, 3)
        return (0, 1, 2)
