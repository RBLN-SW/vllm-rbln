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

"""KV cache copies both prefix-cache managers emit, and the refs that keep
their sources alive until the worker has read them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

__all__ = ["CopyOpMixin", "CopyOpQueue", "KVCacheCopyOp"]


@dataclass(frozen=True)
class KVCacheCopyOp:
    """A token range to copy between KV blocks before the forward pass."""

    src_block_id: int
    dst_block_id: int
    num_tokens: int
    src_start: int = 0
    dst_start: int = 0


class CopyOpQueue:
    """Pending copy ops plus the source blocks each one pins.

    ``drain`` / ``release`` pair by list identity so two in-flight steps cannot
    free each other's sources. An empty drain records nothing, so a later
    ``release`` of a real batch is not stolen by a no-op step.
    """

    def __init__(self) -> None:
        self._pending_ops: list[KVCacheCopyOp] = []
        self._pending_sources: list[KVCacheBlock] = []
        self._in_flight: list[tuple[list[KVCacheCopyOp], list[KVCacheBlock]]] = []

    @property
    def pending(self) -> list[KVCacheCopyOp]:
        return self._pending_ops

    def add(self, op: KVCacheCopyOp, sources: Sequence[KVCacheBlock]) -> None:
        self._pending_ops.append(op)
        self._pending_sources.extend(sources)

    def drain(self) -> list[KVCacheCopyOp]:
        ops = self._pending_ops
        sources = self._pending_sources
        self._pending_ops = []
        self._pending_sources = []
        if ops:
            self._in_flight.append((ops, sources))
        return ops

    def release(self, ops: list[KVCacheCopyOp]) -> list[KVCacheBlock]:
        if not ops:
            return []
        for i, (held_ops, sources) in enumerate(self._in_flight):
            if held_ops is ops:
                del self._in_flight[i]
                return sources
        raise ValueError("release_copy_ops got ops that were not drained")

    def clear(self) -> None:
        self._pending_ops = []
        self._pending_sources = []
        self._in_flight = []


class CopyOpMixin:
    """``drain_pending_copy_ops`` / ``release_copy_ops`` over a ``CopyOpQueue``.

    Requires ``block_pool`` from ``KVCacheManager``.
    """

    block_pool: BlockPool
    _copy_ops: CopyOpQueue

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._copy_ops = CopyOpQueue()

    @property
    def pending_copy_ops(self) -> list[KVCacheCopyOp]:
        return self._copy_ops.pending

    def queue_copy(self, op: KVCacheCopyOp, sources: Sequence[KVCacheBlock]) -> None:
        """Pin ``sources`` until the worker has performed ``op``."""
        self._copy_ops.add(op, sources)

    def drain_pending_copy_ops(self) -> list[KVCacheCopyOp]:
        """Return this step's copy ops; source refs stay until ``release_copy_ops``."""
        return self._copy_ops.drain()

    def release_copy_ops(self, ops: list[KVCacheCopyOp]) -> None:
        """Drop the source refs held by a previously drained ``ops`` list."""
        sources = self._copy_ops.release(ops)
        if sources:
            self.block_pool.free_blocks(sources)

    def clear_copy_ops(self) -> None:
        """Drop pending and in-flight copy state without freeing refs."""
        self._copy_ops.clear()
