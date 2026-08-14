# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Page/kernel block KV geometry. See docs/page_layout_kv_manager.md.

A page id names its own physical home -- ``page_id // pages_per_kernel_block``
is the kernel block, the remainder is the slot -- so there is no map to keep
here, only the arithmetic and the configuration that fixes it.
`KernelBlockPool` is what makes allocation respect the identity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

__all__ = [
    "ATTN_BLOCK_SIZE_KEY",
    "KernelBlockCopyOp",
    "PageLayout",
    "PageLayoutConfig",
    "kernel_block_size_from_config",
    "resolve_config",
    "validate_fragmentation",
]

# Where the compiled model publishes its kernel block size.
ATTN_BLOCK_SIZE_KEY = "attn_block_size"


# --------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PageLayout:
    page_size: int
    kernel_block_size: int

    def __post_init__(self) -> None:
        if self.page_size <= 0 or self.kernel_block_size <= 0:
            raise ValueError(
                f"sizes must be positive, got page={self.page_size} "
                f"kernel_block={self.kernel_block_size}"
            )
        if self.kernel_block_size % self.page_size != 0:
            raise ValueError(
                f"kernel_block_size ({self.kernel_block_size}) must be a multiple of "
                f"page_size ({self.page_size})"
            )

    @property
    def pages_per_kernel_block(self) -> int:
        return self.kernel_block_size // self.page_size

    @property
    def is_degenerate(self) -> bool:
        """One page per kernel block: the layer is a no-op."""
        return self.pages_per_kernel_block == 1

    def validate_chunk(self, chunk_size: int) -> None:
        """I2: a prefill step must never straddle a page boundary."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if self.page_size % chunk_size != 0:
            raise ValueError(
                f"page_size ({self.page_size}) must be a multiple of the "
                f"prefill chunk ({chunk_size}) so a prefill step never spans "
                f"two pages"
            )

    def num_kernel_blocks_for_pages(self, num_pages: int) -> int:
        return -(-num_pages // self.pages_per_kernel_block)

    def slot(self, page_index: int) -> int:
        """I3: sequential writes make the slot a function of the index alone."""
        return page_index % self.pages_per_kernel_block


# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class PageLayoutConfig:
    geometry: PageLayout
    num_kernel_blocks: int

    @property
    def enabled(self) -> bool:
        return not self.geometry.is_degenerate


def resolve_config(
    page_size: int,
    kernel_block_size: int | None,
    num_pages: int,
) -> PageLayoutConfig:
    """``kernel_block_size=None`` yields a no-op geometry."""
    geometry = PageLayout(page_size, kernel_block_size or page_size)
    return PageLayoutConfig(
        geometry, max(1, num_pages // geometry.pages_per_kernel_block)
    )


def kernel_block_size_from_config(vllm_config: VllmConfig) -> int | None:
    additional: dict[str, Any] | None = getattr(vllm_config, "additional_config", None)
    value = additional.get(ATTN_BLOCK_SIZE_KEY) if additional else None
    return int(value) if value else None


def validate_fragmentation(
    geometry: PageLayout,
    max_num_seqs: int,
    num_kernel_blocks: int,
    max_model_len: int | None = None,
) -> None:
    """A running request pins every kernel block it spans, not just one.

    Only its last block is partly filled, but all of them are held, so demand
    scales with request length. Sizing for one block per sequence under-counts
    by that factor and lets a pool through that then runs out at runtime.
    """
    if geometry.is_degenerate:
        return
    blocks_per_seq = (
        1 if not max_model_len else -(-max_model_len // geometry.kernel_block_size)
    )
    peak = max_num_seqs * blocks_per_seq
    if peak >= num_kernel_blocks:
        raise ValueError(
            f"max_num_seqs ({max_num_seqs}) x {blocks_per_seq} kernel blocks per "
            f"request at max_model_len needs {peak} kernel blocks, but the pool "
            f"holds only {num_kernel_blocks} of {geometry.kernel_block_size} "
            f"tokens; lower max_num_seqs or max_model_len, or raise the KV cache size"
        )
    if peak / num_kernel_blocks > 0.5:
        logger.warning(
            "Page layout: up to %d of %d kernel blocks can be pinned at once "
            "(%d requests x %d blocks each at max_model_len).",
            peak,
            num_kernel_blocks,
            max_num_seqs,
            blocks_per_seq,
        )


# --------------------------------------------------------------------------- #
# Copy ops
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class KernelBlockCopyOp:
    """A token range to copy before the forward pass.

    In tokens, not slots: page size is scheduler-side and the worker never
    needs it.
    """

    src_kernel_block_id: int
    dst_kernel_block_id: int
    src_start: int
    dst_start: int
    num_tokens: int
