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

# Unit coverage: descriptors over the two attention KV layouts. `rbln_triton_ops`
# keeps K and V in separate regions, `rbln_custom_ops` packs them into one block,
# and a head band is one contiguous range in the first and two in the second.
#
# Pure arithmetic over hand-built geometries -- no NIXL peer, no device.

from collections import defaultdict
from unittest.mock import MagicMock, patch

import pytest
from vllm.v1.kv_cache_interface import SlidingWindowSpec

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker import (
    RblnNixlWorkerBase,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)

# What each layout puts in one block.
PACKED = 2  # rbln_custom_ops: (num_blocks, 2, H, 1, S, D)
SPLIT = 1  # rbln_triton_ops: (2, num_blocks, H, 1, S, D)


class TestKvParts:
    """`_kv_runs`: how many ranges one piece of a block breaks into."""

    @pytest.mark.parametrize(
        "cuts_l, cuts_r",
        [
            pytest.param(4, 8, id="peer_cuts_finer"),
            pytest.param(8, 4, id="we_cut_finer"),
        ],
    )
    def test_a_packed_block_breaks_in_two_when_the_cuts_differ(self, cuts_l, cuts_r):
        # One side reads a head band out of the other's block, and under a packed
        # block that band sits once in K and once in V.
        assert RblnNixlWorkerBase._kv_runs(PACKED, cuts_l, cuts_r) == 2

    def test_a_packed_block_stays_one_range_at_equal_cuts(self):
        # Both sides name whole blocks, and K and V are adjacent inside one.
        assert RblnNixlWorkerBase._kv_runs(PACKED, 8, 8) == 1

    @pytest.mark.parametrize(
        "cuts_l, cuts_r",
        [
            pytest.param(4, 8, id="peer_cuts_finer"),
            pytest.param(8, 4, id="we_cut_finer"),
            pytest.param(8, 8, id="equal"),
        ],
    )
    def test_separate_regions_never_break(self, cuts_l, cuts_r):
        # A region is K or V alone, so a head band is contiguous however the two
        # sides cut heads. This is the arithmetic the connector shipped with.
        assert RblnNixlWorkerBase._kv_runs(SPLIT, cuts_l, cuts_r) == 1


class TestDescriptorsNameTheRightElement:
    """What a remote descriptor actually points at, not where it points.

    A descriptor test that asserts addresses passes whatever those addresses
    mean, which is how a head band came to name the wrong half. These decode
    each address back into `(K-or-V, head)` and compare it against the head the
    pairing asked for.
    """

    # One region, one area. The peer holds 4 heads per slice where we hold 2, so
    # our band starts 2 heads into its block -- the case where the offset is not
    # zero and the layout decides what it lands on.
    PEER_PAGE = 512
    LOCAL_PAGE = 256
    PEER_HEADS = 4
    PEER_BASE = 1000
    NUM_BLOCKS = 2
    WANTED_HEAD = 2

    @staticmethod
    def _worker(local_page):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.block_len_per_layer = [local_page]
        w.get_backend_aware_kv_block_len = lambda layer_idx, **_: local_page
        return w

    def _descs(self, kv_runs):
        return self._worker(self.LOCAL_PAGE)._head_matched_desc(
            region_id=0,
            logical_r=0,
            area_l=0,
            geom=(self.WANTED_HEAD, 2, 1),
            peer=(0, self.PEER_HEADS, 1, 2),
            areas_r=2,
            remote_bases=[self.PEER_BASE, 2000],
            remote_lens=[self.PEER_PAGE, self.PEER_PAGE],
            device_id=0,
            num_blocks=self.NUM_BLOCKS,
            split=1,
            kv_runs=kv_runs,
        )

    def _decode(self, addr, kv_runs):
        """(block, K-or-V, head) for an address inside the peer's region."""
        off = addr - self.PEER_BASE
        block, within_block = divmod(off, self.PEER_PAGE)
        kv_stride = self.PEER_PAGE // kv_runs
        kv, within_kv = divmod(within_block, kv_stride)
        return block, kv, within_kv // (kv_stride // self.PEER_HEADS)

    def test_a_packed_block_is_read_at_the_head_the_pairing_asked_for(self):
        descs = self._descs(kv_runs=2)
        # Two blocks, each named once in K and once in V, all at our head.
        assert [self._decode(a, 2) for a, _, _ in descs] == [
            (0, 0, self.WANTED_HEAD),
            (0, 1, self.WANTED_HEAD),
            (1, 0, self.WANTED_HEAD),
            (1, 1, self.WANTED_HEAD),
        ]

    def test_a_packed_block_read_as_one_range_names_the_wrong_element(self):
        # The arithmetic the connector shipped with, against a packed peer: one
        # descriptor per block, and the offset that used to mean "head 2" now
        # lands in V at head 0. This is the silent corruption the split avoids.
        wrong = [self._decode(a, 2) for a, _, _ in self._descs(kv_runs=1)]
        assert wrong == [(0, 1, 0), (1, 1, 0)]
        assert all(head != self.WANTED_HEAD for _, _, head in wrong)

    @pytest.mark.parametrize("kv_runs", [1, 2])
    def test_no_descriptor_runs_off_the_end_of_its_half(self, kv_runs):
        kv_stride = self.PEER_PAGE // kv_runs
        for addr, length, _ in self._descs(kv_runs):
            within_kv = (addr - self.PEER_BASE) % self.PEER_PAGE % kv_stride
            assert within_kv + length <= kv_stride

    def test_a_peer_block_that_does_not_halve_is_refused(self):
        # The length comes off the wire, so it is the peer's claim rather than
        # ours; halving it silently would move a byte short of every V.
        w = self._worker(self.LOCAL_PAGE)
        with pytest.raises(RuntimeError, match="range"):
            w._head_matched_desc(
                region_id=0,
                logical_r=0,
                area_l=0,
                geom=(self.WANTED_HEAD, 2, 1),
                peer=(0, self.PEER_HEADS, 1, 2),
                areas_r=2,
                remote_bases=[self.PEER_BASE, 2000],
                remote_lens=[self.PEER_PAGE + 1, self.PEER_PAGE],
                device_id=0,
                num_blocks=1,
                split=1,
                kv_runs=2,
            )

    def test_the_two_halves_are_a_stride_apart(self):
        addrs = [a for a, _, _ in self._descs(kv_runs=2)]
        kv_stride = self.PEER_PAGE // 2
        # K and V of the same block sit `kv_stride` apart, K first.
        assert addrs[1] - addrs[0] == kv_stride
        assert addrs[3] - addrs[2] == kv_stride


class TestTheLayoutReachesTheDescriptors:
    """`_kv_per_block` is worker state; these pin what reads it."""

    @staticmethod
    def _worker(kv_per_block, *, kv_slices=1, tp_size=1):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.use_host_buffer = False
        w._sw_ratio = None
        w._kv_areas = 1
        w._kv_slices = kv_slices
        w._kv_per_block = kv_per_block
        topo = MagicMock()
        topo.tp_size = tp_size
        topo.tp_ratio.return_value = 2
        w.transfer_topo = topo
        return w

    @pytest.mark.parametrize(
        "kv_per_block, expected",
        [pytest.param(SPLIT, 1, id="separate"), pytest.param(PACKED, 2, id="packed")],
    )
    def test_a_peer_of_a_different_width_takes_the_layout_s_kv_count(
        self, kv_per_block, expected
    ):
        w = self._worker(kv_per_block)
        meta = MagicMock()
        meta.kv_slices = 2
        assert w._peer_kv_runs(meta, remote_tp_size=1) == expected

    def test_a_peer_of_our_own_width_never_splits(self):
        # Both sides name whole blocks, so the layout does not matter.
        w = self._worker(PACKED)
        meta = MagicMock()
        meta.kv_slices = 1
        assert w._peer_kv_runs(meta, remote_tp_size=1) == 1

    def test_a_split_alone_keeps_the_handler_off_upstream_s_fast_path(self):
        # `register_local_xfer_handler` hands a whole-engine peer to upstream's
        # one handle. A packed block is not that peer even when nothing else
        # narrowed: upstream names a block once and our list names it per range.
        w = self._worker(PACKED)
        w.local_seen_layer_names = ["layer.0"]

        with patch.object(
            RblnNixlPullConnectorWorker, "_register_shard_local_xfer_handler"
        ) as shard:
            w.register_local_xfer_handler(16, kv_runs=2)

        assert shard.call_args.kwargs["kv_runs"] == 2

    def test_host_staging_never_splits(self):
        # Host staging registers whole logical buffers and is kept off every
        # head-matched path, so it reads a block as one range.
        w = self._worker(PACKED)
        w.use_host_buffer = True
        meta = MagicMock()
        meta.kv_slices = 2
        assert w._peer_kv_runs(meta, remote_tp_size=1) == 1


class TestTwoLayoutsNeverPair:
    """One process picks the layout, so the version cannot tell peers apart.

    `use_custom_kernel` is a per-process setting: two workers off the same build
    can register different numbers of regions. The layout therefore travels in
    the handshake and is checked like any other geometry.
    """

    @staticmethod
    def _worker(kv_per_block):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.use_host_buffer = False
        w._has_swa = False
        w._kv_per_block = kv_per_block
        w.block_len_per_layer = [64]
        w.num_regions = 1
        w.local_seen_layer_names = ["layer.0"]
        topo = MagicMock()
        topo.tp_ratio.return_value = 1
        w.transfer_topo = topo
        return w

    @staticmethod
    def _meta(kv_per_block):
        meta = MagicMock()
        meta.kv_per_block = kv_per_block
        meta.kv_caches_base_addr = [1000]
        meta.registered_layer_names = ["layer.0"]
        return meta

    @pytest.mark.parametrize(
        "ours, theirs",
        [
            pytest.param(SPLIT, PACKED, id="we_split_they_pack"),
            pytest.param(PACKED, SPLIT, id="we_pack_they_split"),
        ],
    )
    def test_a_peer_on_the_other_layout_is_refused(self, ours, theirs):
        with pytest.raises(RuntimeError, match="use_custom_kernel"):
            self._worker(ours)._check_d2d_region_pairing(
                self._meta(theirs), remote_tp_size=1
            )

    @pytest.mark.parametrize("kv_per_block", [SPLIT, PACKED])
    def test_a_peer_on_our_layout_pairs(self, kv_per_block):
        self._worker(kv_per_block)._check_d2d_region_pairing(
            self._meta(kv_per_block), remote_tp_size=1
        )


class TestASlidingWindowInsideAPackedBlock:
    """The SWA view is a byte prefix, so a packed block needs one per K/V.

    One prefix over the whole block would run twice as far into K and never
    reach V. Refusing the pair instead is not open to us: gpt-oss is a
    sliding-window model and the packed layout is what its kernels read.
    """

    BLOCK_LEN = 256
    NUM_BLOCKS = 2
    SW_RATIO = 2
    BASES = [0x1000, 0x2000]

    def _worker(self, kv_per_block):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._sw_ratio = self.SW_RATIO
        w._has_mamba = False
        w._kv_per_block = kv_per_block
        w.engine_id = "local"
        w.tp_rank = 0
        w.device_id = 0
        w.block_size = 64
        w.num_blocks = self.NUM_BLOCKS
        w.kv_caches_base_addr = {"local": {0: self.BASES}}
        w.block_len_per_layer = [self.BLOCK_LEN] * len(self.BASES)
        w.nixl_memory_type = "DRAM"
        w.nixl_wrapper = MagicMock()
        w.get_backend_aware_kv_block_len = lambda **_: self.BLOCK_LEN
        return w

    def _descs(self, kv_per_block):
        w = self._worker(kv_per_block)
        w.register_local_xfer_handler(w.block_size)
        return w.nixl_wrapper.get_xfer_descs.call_args[0][0]

    def test_a_window_is_taken_inside_k_and_inside_v(self):
        half = self.BLOCK_LEN // 2
        covered = {
            (addr - base) % self.BLOCK_LEN // half
            for addr, _, _ in self._descs(PACKED)
            for base in self.BASES
            if base <= addr < base + self.BLOCK_LEN * self.NUM_BLOCKS
        }
        assert covered == {0, 1}

    def test_no_descriptor_reaches_out_of_the_half_it_starts_in(self):
        half = self.BLOCK_LEN // 2
        for addr, length, _ in self._descs(PACKED):
            within = (addr - self.BASES[0]) % self.BLOCK_LEN % half
            assert within + length <= half

    def test_separate_regions_keep_the_shipped_lengths(self):
        # The layout the connector shipped with: one descriptor per block per
        # pass, Full-length then trimmed. A byte of this changing is a
        # regression, not a layout difference.
        descs = self._descs(SPLIT)
        assert len(descs) == 2 * len(self.BASES) * self.NUM_BLOCKS
        assert descs[0][1] == self.BLOCK_LEN
        assert descs[-1][1] == self.BLOCK_LEN // self.SW_RATIO

    def test_a_packed_block_doubles_the_descriptors_of_both_passes(self):
        # Both passes carry the same count so `_compute_desc_ids` can space a
        # block's ids by one number.
        assert len(self._descs(PACKED)) == 2 * len(self._descs(SPLIT))


class TestASlidingWindowOnThePeerSide(TestASlidingWindowInsideAPackedBlock):
    """The peer's list has to break the same way, or the two pair off by one.

    Inherits the geometry so both sides are read at one set of numbers; the
    local class builds our list and this one the peer's.
    """

    PEER_BASES = [0x9000, 0xA000]

    def _descs(self, kv_per_block):
        w = self._worker(kv_per_block)
        w._has_swa = True
        w._remote_agents = {}
        w.dst_num_blocks = {}
        w.dst_xfer_side_handles = defaultdict(dict)
        w.kv_caches_base_addr = defaultdict(dict)
        topo = MagicMock()
        topo.block_size_ratio.return_value = 1
        topo.tp_ratio.return_value = 1
        topo.is_kv_replicated.return_value = True
        w.transfer_topo = topo

        meta = MagicMock()
        meta.engine_id = "peer"
        meta.block_size = w.block_size
        meta.num_blocks = self.NUM_BLOCKS
        meta.kv_caches_base_addr = self.PEER_BASES
        meta.block_lens = [self.BLOCK_LEN] * len(self.PEER_BASES)
        meta.device_id = 1

        with (
            patch.object(
                RblnNixlPullConnectorWorker, "_register_remote_engine_prelude"
            ),
            patch.object(
                RblnNixlPullConnectorWorker, "_validate_remote_agent_handshake"
            ),
        ):
            w.add_remote_agent(meta)
        return w.nixl_wrapper.get_xfer_descs.call_args[0][0]

    def test_a_window_is_taken_inside_k_and_inside_v(self):
        half = self.BLOCK_LEN // 2
        covered = {
            (addr - self.PEER_BASES[0]) % self.BLOCK_LEN // half
            for addr, _, _ in self._descs(PACKED)
            if self.PEER_BASES[0] <= addr < self.PEER_BASES[1]
        }
        assert covered == {0, 1}

    def test_no_descriptor_reaches_out_of_the_half_it_starts_in(self):
        half = self.BLOCK_LEN // 2
        for addr, length, _ in self._descs(PACKED):
            within = (addr - self.PEER_BASES[0]) % self.BLOCK_LEN % half
            assert within + length <= half

    def test_separate_regions_keep_the_shipped_lengths(self):
        descs = self._descs(SPLIT)
        assert len(descs) == 2 * len(self.PEER_BASES) * self.NUM_BLOCKS
        assert descs[0][1] == self.BLOCK_LEN
        assert descs[-1][1] == self.BLOCK_LEN // self.SW_RATIO


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
