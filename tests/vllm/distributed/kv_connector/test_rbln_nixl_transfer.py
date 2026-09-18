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


# Unit coverage: which descriptors one request needs -- the ids a block list
# turns into, over a region table registration already produced.

from unittest.mock import MagicMock

import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlBaseConnectorWorker
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from tests.vllm.distributed.kv_connector.utils import (
    build_worker,
    sliding_window_spec,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)


class TestComputeDescIds:
    # Routes block ids into the Full range (offset 0) or the SWA range (offset
    # num_full_descs) by group spec, expanded across regions.
    def test_none_ratio_delegates_to_super(self, monkeypatch):
        worker = build_worker(monkeypatch)  # _sw_ratio is None
        captured = []

        def super_impl(self, block_ids, dst, ratio, phys):
            captured.append((block_ids, dst, ratio, phys))
            return "super"

        monkeypatch.setattr(NixlBaseConnectorWorker, "_compute_desc_ids", super_impl)
        out = worker._compute_desc_ids([[0]], 4, None, 1)
        assert out == "super"
        assert captured == [([[0]], 4, None, 1)]

    def test_sw_group_shifted_by_full_desc_count_across_regions(self, monkeypatch):
        # Full group -> offset 0; SWA group -> offset num_full_descs. Each id is
        # also expanded across regions as region_id * num_blocks + id.
        worker = build_worker(monkeypatch)
        worker._sw_ratio = 2
        worker.num_regions = 2
        full_spec = MagicMock()  # not a SlidingWindowSpec
        worker._group_specs = [
            full_spec,
            sliding_window_spec(block_size=64, sliding_window=32),
        ]

        # dst_num_blocks=4 -> num_full_descs = num_regions(2) * 4 = 8.
        out = worker._compute_desc_ids([[0, 1], [2]], 4, None, 1)

        # Full ids [0,1] -> r*4 + id: 0,1 then 4,5. SWA id [2] -> r*4 + 2 + 8.
        assert list(out) == [0, 1, 4, 5, 10, 14]

    def test_block_size_ratio_scales_block_span(self, monkeypatch):
        # A block_size_ratio widens the per-region block span (num_blocks *= ratio),
        # shifting both the region stride and the SWA offset.
        worker = build_worker(monkeypatch)
        worker._sw_ratio = 2
        worker.num_regions = 1
        worker._group_specs = [sliding_window_spec(block_size=64, sliding_window=32)]

        # dst_num_blocks=2, ratio=2 -> num_blocks=4, num_full_descs = 1*4 = 4.
        out = worker._compute_desc_ids([[1]], 2, 2.0, 1)
        # single region: 0*4 + 1 + offset(4) = 5.
        assert list(out) == [5]

    def test_rejects_multi_physical_blocks_per_logical(self, monkeypatch):
        # The SWA desc formula indexes physical blocks directly; the connector
        # pins one physical block per logical, so >1 is rejected.
        worker = build_worker(monkeypatch)
        worker._sw_ratio = 2
        worker.num_regions = 1
        worker._group_specs = [sliding_window_spec(block_size=64, sliding_window=32)]
        with pytest.raises(AssertionError, match="physical_blocks_per_logical"):
            worker._compute_desc_ids([[0]], 4, None, 2)

    def test_empty_groups_yield_empty(self, monkeypatch):
        worker = build_worker(monkeypatch)
        worker._sw_ratio = 2
        worker.num_regions = 1
        worker._group_specs = [MagicMock()]
        out = worker._compute_desc_ids([[]], 4, None, 1)
        assert out.size == 0


# What each layout puts in one block.
PACKED = 2  # rbln_custom_ops: (num_blocks, 2, H, 1, S, D)
SPLIT = 1  # rbln_triton_ops: (2, num_blocks, H, 1, S, D)


class TestDescIdsSpaceABlockByItsKvCount:
    """`_compute_desc_ids` indexes the lists the class above builds."""

    @staticmethod
    def _worker(kv_per_block, spec):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._sw_ratio = 2
        w._kv_per_block = kv_per_block
        w.num_regions = 2
        w._group_specs = [spec]
        return w

    def _ids(self, kv_per_block, spec):
        return self._worker(kv_per_block, spec)._compute_desc_ids(
            [[1]],
            dst_num_blocks=4,
            block_size_ratio=None,
            physical_blocks_per_logical=1,
        )

    def test_a_packed_block_names_both_of_its_halves(self):
        full = MagicMock()
        ids = self._ids(PACKED, full)
        # Region 0 block 1 -> ids 2,3; region 1 block 1 -> ids 10,11.
        assert sorted(ids) == [2, 3, 10, 11]

    def test_the_sliding_window_range_starts_past_every_full_desc(self):
        sw = MagicMock(spec=SlidingWindowSpec)
        packed = min(self._ids(PACKED, sw))
        # num_regions(2) * num_blocks(4) * kv(2) full descs come first.
        assert packed == 16 + 2
        assert min(self._ids(SPLIT, sw)) == 8 + 1
