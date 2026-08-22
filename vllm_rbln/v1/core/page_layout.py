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

"""Page/kernel block KV sizes. Design: https://github.com/RBLN-SW/vllm-rbln/issues/928

Sizes and the checks on them. A page id names its own physical home, but the
arithmetic that says so belongs to `KernelBlockPool`, which is the only thing
that can enforce it -- keeping a second copy here is how the two drift apart.
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
        """A prefill step must never straddle a page boundary."""
        if chunk_size <= 0:
            raise ValueError(f"chunk_size must be positive, got {chunk_size}")
        if self.page_size % chunk_size != 0:
            raise ValueError(
                f"page_size ({self.page_size}) must be a multiple of the "
                f"prefill chunk ({chunk_size}) so a prefill step never spans "
                f"two pages"
            )


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
    scales with request length.

    Two different things follow from that, and they need different answers:

    **No forward progress** (fatal). If the pool cannot hold *one* request at
    `max_model_len` plus a spare block, that request can never complete: it is
    admitted, grows, fails to allocate, gets preempted, and repeats forever. The
    spare is not slack -- `allocate_slots` needs one idle kernel block for the
    tail copy (`has_idle_kernel_blocks(1)`), so a pool sized to exactly one
    request deadlocks on that path.

    **Concurrency below max_num_seqs** (normal). If the pool cannot hold
    `max_num_seqs` requests *all at max_model_len simultaneously*, the scheduler
    admits what fits and preempts the rest. That is ordinary vLLM behaviour, not
    a misconfiguration: `allocate_slots` returns ``None`` and the scheduler
    preempts, exactly as it does for the upstream manager. Sizing every serving
    stack for the absolute worst case would demand a pool `max_num_seqs` times
    larger than the model's own context -- on this hardware that is several times
    the available HBM, so it rejects configurations that in fact run fine.

    The earlier version raised on the second case too. Measured counter-example
    (2026-08-22, MiniMax-M2.5 on R100): the two non-page-layout stacks ran
    `max_num_seqs=4` against a KV pool of exactly `max_model_len` -- one
    max-length request's worth, a quarter of that worst case -- and both
    completed 152/152 requests with zero failures. Preemption absorbed the
    difference. Rejecting that shape at startup would have been wrong.
    """
    if geometry.is_degenerate:
        return
    blocks_per_seq = (
        1 if not max_model_len else -(-max_model_len // geometry.kernel_block_size)
    )
    # +1: the tail-copy path needs one idle kernel block on top of the request.
    needed_for_progress = blocks_per_seq + 1
    if needed_for_progress > num_kernel_blocks:
        raise ValueError(
            f"one request at max_model_len spans {blocks_per_seq} kernel blocks "
            f"and the tail copy needs 1 more, so {needed_for_progress} are "
            f"required, but the pool holds only {num_kernel_blocks} of "
            f"{geometry.kernel_block_size} tokens. No request could ever "
            f"complete; lower max_model_len or raise the KV cache size"
        )
    peak = max_num_seqs * blocks_per_seq
    if peak >= num_kernel_blocks:
        logger.warning(
            "Page layout: %d requests x %d kernel blocks at max_model_len would "
            "need %d blocks but the pool holds %d -- concurrency will be limited "
            "by preemption below max_num_seqs=%d.",
            max_num_seqs,
            blocks_per_seq,
            peak,
            num_kernel_blocks,
            max_num_seqs,
        )
    elif peak / num_kernel_blocks > 0.5:
        logger.warning(
            "Page layout: up to %d of %d kernel blocks can be pinned at once "
            "(%d requests x %d blocks each at max_model_len).",
            peak,
            num_kernel_blocks,
            max_num_seqs,
            blocks_per_seq,
        )


# --------------------------------------------------------------------------- #
# Page layout under a KV connector (PD-disaggregation)
# --------------------------------------------------------------------------- #
#
# Page layout is developed and benchmarked without a KV connector -- see
# `benchmarks/page_layout/serve.sh`, "No KV connector: this compares vLLM-local
# mechanisms only". Running it under PD-disaggregation needed three fixes, all
# of them the same mistake in different places: a block id is a *page*, the KV
# tensors are *kernel blocks*, and every layer that names a block has to say
# which one it means.
#
#   1. `validate_fragmentation` rejected concurrency the pool cannot hold at
#      once, which is preemption's job, not a startup error (see above).
#   2. `RblnNixlConnectorWorker` registered its descriptors over pages while the
#      tensors held kernel blocks.
#   3. `RBLNScheduler.schedule` translated block ids *after* the connector had
#      already read them, so the worker got kernel blocks and the connector kept
#      pages.
#
# (3) is what made the other two hard to see: the engine starts, and short
# requests answer correctly, because nothing crosses a block boundary. Only a
# request long enough to actually move KV fails -- and it failed twice over,
# once in NIXL's descriptor prep and once in LMCache's `scatter_chunk_to_blocks`,
# which reads its block size off the tensor (kernel block) and its block *count*
# off the connector's ids (pages). Neither needed its own fix; both came right
# once the ids arrived in the same unit as the tensors.
#
# Measured 2026-08-22 (MiniMax-M2.5 / R100, 2P2D, NIXL + LMCache mp). If this
# combination breaks again, suspect another place where the two units meet
# before suspecting the transport.
#
# ⚠️ STILL NOT USABLE WITH A KV CONNECTOR. Six unit fixes later the engine runs
# and short requests answer correctly, but a decode that pulls KV from a remote
# prefill hits upstream's prefix-caching trim:
#
#     assert num_local_blocks <= len(remote_group)   # nixl/base_worker.py
#
# because the blocks a decode must fill come from
# `KVCacheBlocks.get_unhashed_block_ids_all_groups()` -- pages -- while the
# remote offers kernel blocks (measured: local=[53] remote=[7]).
#
# Converting that list is **not** a fix, and it is worth knowing why before
# trying: it was tried (2060848b, reverted in a2b0a118) and it silently
# corrupted generation -- 200 OK, incoherent tokens, on requests of any length.
# "Unhashed" means "the tail that is not cached yet", and that boundary is a
# page boundary. The kernel block holding those pages can also hold pages that
# are already cached, and upstream's trim overwrites a whole block per local
# entry, so folding to kernel blocks writes remote data over live cached pages.
#
# Prefix caching works in pages; the transfer works in kernel blocks. At the
# boundary those genuinely disagree, and no amount of unit translation settles
# it. Closing this needs a decision -- move the transfer to page granularity, or
# align prefix matching to kernel blocks -- not another conversion.
