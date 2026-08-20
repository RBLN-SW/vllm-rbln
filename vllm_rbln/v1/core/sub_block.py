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

"""Path-neutral building blocks for sub-block prefix caching.

Both KV cache managers build their sub-block caching on these: the
vLLM-native one (rbln_kv_cache_manager) and the optimum one
(optimum_kv_cache_manager). Only the pieces whose behavior must not
diverge between the paths live here — the chained hashing that decides
what counts as a hit, the hash index, and the copy-op contract carried
in the scheduler output. Orchestration (when to index, how to apply a
match) stays with each manager.

See docs/sub_block_prefix_caching.md for the design overview.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from vllm.v1.core.kv_cache_utils import (
    BlockHash,
    generate_block_hash_extra_keys,
    hash_block_tokens,
    need_extra_keys,
)

if TYPE_CHECKING:
    from vllm.v1.request import Request


@dataclass
class KVCacheCopyOp:
    """Describes a sub-block KV cache copy from a cached block to a new block."""

    group_id: int
    # NOTE: While pending, it holds a ref count of the source block to prevent eviction.
    src_block_id: int
    dst_block_id: int
    # Number of tokens to copy (= num_matched_sub_blocks * sub_block_size).
    num_tokens: int


class SubBlockHasher:
    """Computes chained sub-block hashes from token IDs.

    Uses the same ``hash_block_tokens`` as upstream, but at sub-block
    granularity.  When a *request* is provided, per-sub-block
    ``extra_keys`` (cache_salt, LoRA, multimodal, prompt_embeds) are
    mixed in, mirroring upstream full-block hashing.
    """

    def __init__(
        self,
        hash_fn: Callable,
        sub_block_size: int,
    ) -> None:
        self.hash_fn = hash_fn
        self.sub_block_size = sub_block_size

    def hash_tokens(
        self,
        token_ids: Sequence[int],
        *,
        parent_hash: BlockHash | None = None,
        num_hashed_tokens: int = 0,
        request: Request | None = None,
        start_mm_idx: int = 0,
    ) -> tuple[list[BlockHash], list[tuple[Any, ...] | None], int]:
        """Return sub-block hashes for *full* sub-blocks in ``token_ids``.

        Args:
            token_ids: Full token sequence of the request.
            parent_hash: Hash of the last sub-block before the range we
                are hashing (``None`` for the very first sub-block).
            num_hashed_tokens: Number of tokens already hashed (i.e. the
                start offset into ``token_ids``).
            request: When provided and the request carries extra hash
                keys (LoRA, cache_salt, multimodal, prompt_embeds),
                those keys are mixed into each sub-block hash.
            start_mm_idx: Starting multimodal feature index for
                incremental hashing with multimodal requests.

        Returns:
            ``(hashes, extra_keys, mm_idx)``.  ``extra_keys`` is a list
            (aligned with ``hashes``) of the per-sub-block extra-keys
            tuples actually mixed into the hash, or ``None`` for
            sub-blocks that had no extra keys — returned so callers can
            reproduce the hash when emitting KV events.
        """
        # Based on upstream request_block_hasher() in get_request_block_hasher().

        sbs = self.sub_block_size
        hashes: list[BlockHash] = []
        extra_keys_list: list[tuple[Any, ...] | None] = []
        use_extra = request is not None and need_extra_keys(request)
        # NOTE: We can't simply use `mm_idx=-1`,
        # because it means the last mm input in the entire prompt,
        # which is meant to be used during decode phase.
        mm_idx = start_mm_idx
        start = num_hashed_tokens
        for i in range(start, len(token_ids) - sbs + 1, sbs):
            extra_keys: tuple[Any, ...] | None = None
            if use_extra:
                extra_keys, mm_idx = generate_block_hash_extra_keys(
                    request, i, i + sbs, mm_idx
                )
            parent_hash = hash_block_tokens(
                self.hash_fn,
                parent_hash,
                token_ids[i : i + sbs],
                extra_keys,
            )
            hashes.append(parent_hash)
            extra_keys_list.append(extra_keys)
        return hashes, extra_keys_list, mm_idx


class SubBlockIndex:
    """Index mapping sub-block hashes to physical block IDs that contain that
    sub-block as a prefix."""

    def __init__(self) -> None:
        # sub_block_hash → set of block IDs with that prefix cached.
        self._hash_to_blocks: dict[BlockHash, set[int]] = {}
        # Reverse index: block_id → list of sub-block hashes (for removal).
        self._block_hashes: dict[int, list[BlockHash]] = {}

    def update(self, block_id: int, sub_block_hashes: list[BlockHash]) -> int:
        """Index or extend a block's sub-block hashes (idempotent).

        Returns ``first_fresh_idx``: the index in ``sub_block_hashes`` at
        which the first globally-fresh (previously absent from the whole
        index) hash was added by this call.
        ``first_fresh_idx == len(sub_block_hashes)`` means nothing was fresh.
        """
        existing = self._block_hashes.setdefault(block_id, [])
        first_fresh_idx = len(sub_block_hashes)
        for i in range(len(existing), len(sub_block_hashes)):
            h = sub_block_hashes[i]
            bucket = self._hash_to_blocks.setdefault(h, set())
            if not bucket and i < first_fresh_idx:
                first_fresh_idx = i
            existing.append(h)
            bucket.add(block_id)
        return first_fresh_idx

    def pop(self, block_id: int) -> list[BlockHash]:
        """Remove a block from the index (called on eviction).

        Returns the list of hashes whose bucket became empty as a result.
        An empty list means nothing was removed or every removed hash is still
        held by another block.
        """
        hashes = self._block_hashes.pop(block_id, None)
        if hashes is None:
            return []
        fully_removed: list[BlockHash] = []
        for h in hashes:
            s = self._hash_to_blocks.get(h)
            if s is not None:
                s.discard(block_id)
                if not s:
                    del self._hash_to_blocks[h]
                    fully_removed.append(h)
        return fully_removed

    def all_hashes(self) -> list[BlockHash]:
        """All currently-indexed hashes (order unspecified)."""
        return list(self._hash_to_blocks.keys())

    def contains(self, block_id: int) -> bool:
        """Return True if the block is indexed."""
        return block_id in self._block_hashes

    def longest_match(
        self, sub_block_hashes: Sequence[BlockHash]
    ) -> tuple[int | None, int]:
        """Find a block with the longest prefix match.

        Returns:
            ``(block_id, num_matched)`` where ``block_id`` is any block
            matching that prefix, or ``(None, 0)`` if no match.
        """
        best_block_id: int | None = None
        best_depth = 0
        for depth, h in enumerate(sub_block_hashes, start=1):
            blocks = self._hash_to_blocks.get(h)
            if not blocks:
                break
            # Pick an arbitrary block_id from the set.
            best_block_id = next(iter(blocks))
            best_depth = depth
        return best_block_id, best_depth
