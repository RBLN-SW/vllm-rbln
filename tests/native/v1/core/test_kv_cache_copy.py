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

"""CopyOpQueue pairs drain with release by list identity, not FIFO order."""

import pytest

from vllm_rbln.v1.core.kv_cache_copy import CopyOpQueue, KVCacheCopyOp


def _op(src: int, dst: int) -> KVCacheCopyOp:
    return KVCacheCopyOp(src_block_id=src, dst_block_id=dst, num_tokens=4)


class TestCopyOpQueue:
    def test_empty_drain_does_not_steal_the_next_release(self):
        # A no-op drain must not enqueue a batch. Otherwise the next real
        # release would free nothing and leak the sources.
        q = CopyOpQueue()
        assert q.drain() == []
        src = object()
        q.add(_op(0, 1), [src])
        ops = q.drain()
        assert q.release(ops) == [src]

    def test_release_pairs_with_the_drained_list(self):
        # Two in-flight steps: releasing the second must not free the first.
        q = CopyOpQueue()
        first_src, second_src = object(), object()
        q.add(_op(0, 1), [first_src])
        first = q.drain()
        q.add(_op(2, 3), [second_src])
        second = q.drain()
        assert q.release(second) == [second_src]
        assert q.release(first) == [first_src]

    def test_unknown_ops_raise(self):
        q = CopyOpQueue()
        q.add(_op(0, 1), [object()])
        q.drain()
        with pytest.raises(ValueError, match="not drained"):
            q.release([_op(0, 1)])

    def test_empty_release_is_a_noop(self):
        q = CopyOpQueue()
        src = object()
        q.add(_op(0, 1), [src])
        held = q.drain()
        assert q.release([]) == []
        assert q.release(held) == [src]
