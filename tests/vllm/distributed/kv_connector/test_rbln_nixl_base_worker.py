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

# Unit coverage: what constructing the worker settles, and the Full + SWA
# descriptor layout the override upstream calls builds. Cases with a KV
# geometry use `make_worker`, where upstream's __init__ runs; the rest use
# `build_worker`, which stubs it down to what the RBLN overrides read.

from collections import defaultdict
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlBaseConnectorWorker
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from tests.vllm.distributed.kv_connector.utils import (
    KvGeometry,
    build_worker,
    decode,
    sliding_window_spec,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
    _as_descs,
)


def _merged_uniform_spec(inner):
    """One group of same-type-but-not-identical layers, which is what
    `get_kv_cache_groups` merges an MLA model with a sparse indexer into."""
    return UniformTypeKVCacheSpecs(
        block_size=inner.block_size, kv_cache_specs={"layer.0": inner}
    )


def _full_attention_spec(block_size=64):
    return FullAttentionSpec(
        block_size=block_size, num_kv_heads=1, head_size=64, dtype=torch.float16
    )


class TestRealWorkerRegistration:
    """The worker through its real __init__ over real KV tensors.

    `make_worker` builds it over a real config and real tensors, so the region
    arithmetic here runs on the tensors' own addresses and byte counts rather
    than on numbers a builder chose.
    """

    def test_the_backend_and_memory_type_follow_the_buffer_device(self, make_worker):
        w = make_worker(register=False)
        assert w.kv_buffer_device == "rbln"
        assert w.use_host_buffer is False
        assert w._use_rbln_nixl_backend is True
        assert w.nixl_memory_type == "VRAM"
        # D2D defers registration, and the topology is built inside it.
        assert w.transfer_topo is None


class TestBackendSelection:
    def test_host_bounce_with_adapter_uses_rbln_backend(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_buffer_device="cpu", nixl_available=True)
        assert worker._use_rbln_nixl_backend is True
        assert worker.use_host_buffer is True
        assert not hasattr(worker, "nixl_memory_type")  # VRAM only set for D2D

    def test_d2d_with_adapter_registers_vram(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_buffer_device="rbln", nixl_available=True)
        assert worker._use_rbln_nixl_backend is True
        assert worker.nixl_memory_type == "VRAM"
        assert worker.use_host_buffer is False

    def test_host_bounce_without_adapter_falls_back_to_upstream(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_buffer_device="cpu", nixl_available=False)
        assert worker._use_rbln_nixl_backend is False
        assert worker.use_host_buffer is True

    def test_d2d_without_adapter_is_rejected(self, monkeypatch):
        # D2D needs the RBLN backend; without nixl_rbln there is no way to
        # register device memory, so construction must fail loudly.
        with pytest.raises(RuntimeError, match="nixl-rbln"):
            build_worker(monkeypatch, kv_buffer_device="rbln", nixl_available=False)


class TestLogicalBlockPinning:
    def test_pins_logical_block_counts(self, monkeypatch):
        # num_blocks / block_size are pinned to the logical values, and the
        # physical-per-logical ratio stays 1 (no kernel-ratio multiplication).
        worker = build_worker(monkeypatch, num_blocks=128, block_size=64)
        assert worker.num_blocks == 128
        assert worker.block_size == 64
        assert worker._physical_blocks_per_logical_kv_block == 1
        assert worker._logical_num_blocks == 128
        assert worker._pending_kv_caches is None


class TestSwaWindowRatio:
    def test_opt_off_keeps_ratio_none(self, monkeypatch):
        worker = build_worker(
            monkeypatch,
            swa_window_mode=False,
            specs=[sliding_window_spec(block_size=64, sliding_window=16)],
        )
        assert worker._sw_ratio is None
        # The window is still detected -- it gates the model parallelism guards
        # whether or not window mode is on.
        assert worker._has_swa

    def test_chunk_mode_turns_the_view_on_over_the_flag(self, monkeypatch):
        # The chunk range extends this layout and has nowhere else to sit:
        # `_compute_desc_ids` hands the whole list to upstream where the ratio
        # is None, and upstream's list carries no second range.
        worker = build_worker(
            monkeypatch,
            kv_buffer_device="rbln",  # chunk mode is the direct path's
            swa_window_mode=False,
            chunk_mode=True,
            specs=[sliding_window_spec(block_size=64, sliding_window=16)],
        )
        assert worker._sw_ratio == 4

    def test_chunk_mode_invents_no_ratio_without_a_window(self, monkeypatch):
        # The override rides on the window, not on the knob: an engine with no
        # sliding window has nothing to view.
        worker = build_worker(
            monkeypatch,
            kv_buffer_device="rbln",
            swa_window_mode=False,
            chunk_mode=True,
            specs=[MagicMock()],
        )
        assert worker._sw_ratio is None

    def test_pure_full_attention_keeps_ratio_none(self, monkeypatch):
        # A non-sliding-window group contributes no ratio.
        worker = build_worker(monkeypatch, swa_window_mode=True, specs=[MagicMock()])
        assert worker._sw_ratio is None

    def test_sliding_window_derives_block_over_window_ratio(self, monkeypatch):
        worker = build_worker(
            monkeypatch,
            swa_window_mode=True,
            specs=[sliding_window_spec(block_size=64, sliding_window=16)],
        )
        assert worker._sw_ratio == 4

    def test_chunk_mode_turning_window_mode_on_says_so(self, monkeypatch, caplog):
        # The operator asked for no view and got one. This log is the only place
        # that says so, and a chunk range has nowhere else to sit.
        with caplog.at_level("WARNING"):
            worker = build_worker(
                monkeypatch,
                kv_buffer_device="rbln",
                swa_window_mode=False,
                chunk_mode=True,
                specs=[sliding_window_spec(block_size=64, sliding_window=16)],
            )

        assert worker._sw_ratio == 4
        assert [
            r.getMessage()
            for r in caplog.records
            if "turned SWA window mode on" in r.getMessage()
        ]

    def test_window_equal_to_block_collapses_to_none(self, monkeypatch):
        # ratio 1 means the window equals the full block -> no trimming.
        worker = build_worker(
            monkeypatch,
            swa_window_mode=True,
            specs=[sliding_window_spec(block_size=64, sliding_window=64)],
        )
        assert worker._sw_ratio is None

    def test_full_attention_groups_are_skipped(self, monkeypatch):
        # The hybrid shape: a model interleaves full-attention and sliding-window
        # layers, so the ratio has to come from the windowed groups alone.
        worker = build_worker(
            monkeypatch,
            swa_window_mode=True,
            specs=[MagicMock(), sliding_window_spec(block_size=64, sliding_window=16)],
        )
        assert worker._sw_ratio == 4

    def test_consistent_ratio_across_groups(self, monkeypatch):
        worker = build_worker(
            monkeypatch,
            swa_window_mode=True,
            specs=[
                sliding_window_spec(block_size=64, sliding_window=16),
                sliding_window_spec(block_size=64, sliding_window=16),
            ],
        )
        assert worker._sw_ratio == 4

    def test_mismatched_ratios_are_rejected(self, monkeypatch):
        with pytest.raises(AssertionError, match="single SWA ratio"):
            build_worker(
                monkeypatch,
                swa_window_mode=True,
                specs=[
                    sliding_window_spec(block_size=64, sliding_window=16),
                    sliding_window_spec(block_size=64, sliding_window=32),
                ],
            )

    def test_window_not_dividing_block_is_rejected(self, monkeypatch):
        with pytest.raises(AssertionError):
            build_worker(
                monkeypatch,
                swa_window_mode=True,
                specs=[sliding_window_spec(block_size=64, sliding_window=15)],
            )

    def test_mla_with_window_mode_is_rejected_at_startup(self, monkeypatch):
        # The dual desc range and a key-only latent have not been combined,
        # so fail at construction rather than at the first handshake.
        with pytest.raises(RuntimeError, match="sliding-window MLA"):
            build_worker(
                monkeypatch,
                swa_window_mode=True,
                use_mla=True,
                specs=[sliding_window_spec(block_size=64, sliding_window=16)],
            )


class TestSwaWindowDelegation:
    # Both collapse to the upstream Full-only implementation when _sw_ratio is
    # None; the SWA dual-range paths are exercised in the Swa classes below.
    def test_register_local_xfer_handler_delegates_when_no_swa(self, monkeypatch):
        worker = build_worker(monkeypatch)  # _sw_ratio is None
        calls: list = []

        def super_handler(self, block_size):
            calls.append(block_size)
            return "super"

        monkeypatch.setattr(
            NixlBaseConnectorWorker, "register_local_xfer_handler", super_handler
        )
        assert worker.register_local_xfer_handler(64) == "super"
        assert calls == [64]

    def test_a_fanned_out_peer_does_not_take_the_whole_engine_path(self, monkeypatch):
        # One handle covers every region once, which cannot express a slice the
        # peer holds on several of its chiplets.
        worker = build_worker(monkeypatch)  # _sw_ratio is None

        monkeypatch.setattr(
            NixlBaseConnectorWorker,
            "register_local_xfer_handler",
            lambda self, block_size: "super",
        )
        monkeypatch.setattr(
            type(worker),
            "_register_shard_local_xfer_handler",
            lambda self, *args, **kwargs: "shard",
        )

        assert worker.register_local_xfer_handler(64, replica_fanout=2) == "shard"

    def test_add_remote_agent_delegates_when_no_swa(self, monkeypatch):
        worker = build_worker(monkeypatch)  # _sw_ratio is None
        calls: list = []

        def super_agent(self, meta, rank=0, size=1):
            calls.append((rank, size))
            return "agent"

        monkeypatch.setattr(NixlBaseConnectorWorker, "add_remote_agent", super_agent)
        assert worker.add_remote_agent(MagicMock(engine_id="peer"), 2, 4) == "agent"
        assert calls == [(2, 4)]

    def test_add_remote_agent_is_idempotent_on_rehandshake(self, monkeypatch):
        # With SWA active, a remote already handshaked returns its cached name
        # without re-registering (no super() / topology work).
        worker = build_worker(
            monkeypatch,
            swa_window_mode=True,
            specs=[sliding_window_spec(block_size=64, sliding_window=16)],
        )
        # Flat rank 1 of a TP2 peer is its (pp 0, tp 1): no pipelining, which
        # this branch refuses, and not a palindrome, which would pass reversed.
        worker._remote_agents = {"peer": {(0, 1): "cached-name"}}
        super_calls = []
        monkeypatch.setattr(
            NixlBaseConnectorWorker,
            "add_remote_agent",
            lambda self, *a, **k: super_calls.append(1),
        )
        result = worker.add_remote_agent(MagicMock(engine_id="peer"), 1, 2)
        assert result == "cached-name"
        assert super_calls == []


# Heavy D2D paths: the deferred registration body and the SWA dual-range descs.
# The real method bodies run; only the external NIXL pieces are faked.


class TestRegisterLocalXferHandlerSwa:
    # With a sliding-window group and window mode on, register_local_xfer_handler
    # emits a dual desc range: Full, then the `_sw_ratio` granules that tile a
    # block, over the same addresses.
    def test_swa_builds_dual_desc_ranges(self, make_worker):
        geo = KvGeometry(spec="swa", sliding_window=512, block_size=1024, num_blocks=4)
        w = make_worker(kv_cache=geo, swa_window_mode=True)
        assert (w._has_swa, w._sw_ratio) == (True, 2)

        blocks_data = w.src_blocks_data
        full_len = w.block_len_per_layer[0]
        # A whole block is one descriptor whatever it packs; the window range
        # cuts that same block into the granules that tile it.
        whole_descs = w.num_regions * w.num_blocks
        assert len(blocks_data) == whole_descs * (1 + w._sw_ratio)
        full, swa = blocks_data[:whole_descs], blocks_data[whole_descs:]
        # The order is a contract, not a detail: a transfer turns (region, block)
        # into a desc id as region * num_blocks + block, so entry i of the list
        # has to BE that pair. Asserting only that SWA repeats Full would pass a
        # reordering applied to both passes.
        bases = w.kv_caches_base_addr[w.engine_id][w.tp_rank]
        decoded = dict(
            bases=bases, block_lens=w.block_len_per_layer, num_blocks=w.num_blocks
        )
        assert decode(full, **decoded) == [
            (region, block, 0)
            for region in range(w.num_regions)
            for block in range(w.num_blocks)
        ]
        assert decode(swa, **decoded) == [
            (region, block, unit)
            for region in range(w.num_regions)
            for block in range(w.num_blocks)
            for unit in range(w._sw_ratio)
        ]
        assert {desc_len for _, desc_len, _ in full} == {full_len}
        assert {desc_len for _, desc_len, _ in swa} == {full_len // w._sw_ratio}

    def test_a_chunk_grid_appends_a_third_range(self, monkeypatch):
        # A grid of (2 runs, 2 chunks) turns each region-block's one Full
        # descriptor into four quarter-length ones, appended after BOTH
        # existing ranges -- the window range keeps its index space and the
        # transfer picks a range by offset.
        worker = build_worker(monkeypatch, num_blocks=4, block_size=64)
        worker._sw_ratio = 2
        worker._has_mamba = False
        worker.tp_rank = 0
        worker.device_id = 0
        worker.transfer_topo = MagicMock(is_kv_layout_blocks_first=False)
        worker.kv_caches_base_addr = {worker.engine_id: {0: [0x1000, 0x2000]}}
        worker.block_len_per_layer = [256, 256]
        worker.nixl_memory_type = "DRAM"
        worker.nixl_wrapper = MagicMock()

        with (
            patch.object(worker, "get_backend_aware_kv_block_len", return_value=256),
            patch.object(type(worker), "_shard_chunk_grid", return_value=(2, 2)),
        ):
            worker.register_local_xfer_handler(64)

        blocks_data = worker.nixl_wrapper.get_xfer_descs.call_args[0][0]
        # 8 whole and 8 x sw_ratio window, then 2 regions x 4 blocks x 2 runs
        # x 2 chunks.
        assert len(blocks_data) == 24 + 32
        # Region 0, block 0: two runs of two chunks, quarter length each. A run
        # is a head's stretch of the block, so the second run starts halfway.
        assert np.array_equal(
            blocks_data[24:28],
            _as_descs(
                [
                    (0x1000, 64, 0),
                    (0x1040, 64, 0),
                    (0x1080, 64, 0),
                    (0x10C0, 64, 0),
                ]
            ),
        )

    def test_a_packed_block_is_chunked_across_both_halves(self, monkeypatch):
        # The third range cuts what the first one names, which is the whole
        # block. Derived for K alone it would tile half of it and leave V
        # unwritten, and the count the two sides compare would still match.
        worker = build_worker(monkeypatch, num_blocks=4, block_size=64)
        worker._sw_ratio = 2
        worker._has_mamba = False
        worker._kv_per_block = 2
        worker.tp_rank = 0
        worker.device_id = 0
        worker.transfer_topo = MagicMock(is_kv_layout_blocks_first=False)
        worker.kv_caches_base_addr = {worker.engine_id: {0: [0x1000, 0x2000]}}
        worker.block_len_per_layer = [256, 256]
        worker.nixl_memory_type = "DRAM"
        worker.nixl_wrapper = MagicMock()

        with (
            patch.object(worker, "get_backend_aware_kv_block_len", return_value=256),
            patch.object(type(worker), "_shard_chunk_grid", return_value=(4, 2)),
        ):
            worker.register_local_xfer_handler(64)

        blocks_data = worker.nixl_wrapper.get_xfer_descs.call_args[0][0]
        # 8 whole and 16 window, then 2 regions x 4 blocks x 4 runs x 2 chunks.
        assert len(blocks_data) == 24 + 64
        chunks = blocks_data[24:32]
        assert [int(addr) for addr, _, _ in chunks] == list(range(0x1000, 0x1100, 32))
        assert {int(length) for _, length, _ in chunks} == {32}

    def test_no_chunk_grid_leaves_the_two_ranges_alone(self, monkeypatch):
        # Off the knob the list must not grow: a longer dlist is memory every
        # peer pays for.
        worker = build_worker(monkeypatch, num_blocks=4, block_size=64)
        worker._sw_ratio = 2
        worker._has_mamba = False
        worker.tp_rank = 0
        worker.device_id = 0
        worker.transfer_topo = MagicMock(is_kv_layout_blocks_first=False)
        worker.kv_caches_base_addr = {worker.engine_id: {0: [0x1000, 0x2000]}}
        worker.block_len_per_layer = [256, 256]
        worker.nixl_memory_type = "DRAM"
        worker.nixl_wrapper = MagicMock()

        with (
            patch.object(worker, "get_backend_aware_kv_block_len", return_value=256),
            patch.object(type(worker), "_shard_chunk_grid", return_value=None),
        ):
            worker.register_local_xfer_handler(64)

        assert len(worker.nixl_wrapper.get_xfer_descs.call_args[0][0]) == 24


class TestHmaRefusalSuppression:
    """The override suppresses the hybrid-KV-manager flag across
    `super().__init__()` for one merged group of full attention specs, and
    puts it back. These build that layout and the ones it must leave alone.
    """

    @staticmethod
    def _pp_worker(monkeypatch, specs, **kwargs):
        return build_worker(monkeypatch, specs=specs, pp_size=4, **kwargs)

    def test_the_flag_is_off_while_upstream_looks_at_it(self, monkeypatch):
        # The refusal reads the flag inside `super().__init__()`, so that call
        # is the only point where the suppression is observable at all.
        specs = [_merged_uniform_spec(_full_attention_spec())]
        worker = self._pp_worker(monkeypatch, specs)

        assert worker.hma_flag_seen_by_upstream_init is True

    def test_the_suppression_is_undone(self, monkeypatch):
        # Both halves matter. `_is_hma_required` still gates the block-size
        # and permute guards downstream, and the config object is shared, so a
        # flag left flipped would outlive this constructor.
        specs = [_merged_uniform_spec(_full_attention_spec())]
        worker = self._pp_worker(monkeypatch, specs)

        assert worker._is_hma_required is True
        scheduler_config = worker.vllm_config.scheduler_config
        assert scheduler_config.disable_hybrid_kv_cache_manager is False

    def test_hma_switched_off_by_the_operator_stays_off(self, monkeypatch):
        # The suppression is the manager's own flag, so for someone who turned
        # the manager off there is nothing to suppress and nothing to require.
        specs = [_merged_uniform_spec(_full_attention_spec())]
        worker = self._pp_worker(monkeypatch, specs, hma_disabled=True)

        assert worker.hma_flag_seen_by_upstream_init is True
        assert worker._is_hma_required is False

    @pytest.mark.parametrize("hma_disabled", [False, True])
    @pytest.mark.parametrize(
        "specs",
        [
            pytest.param(
                [
                    _merged_uniform_spec(
                        MambaSpec(
                            block_size=64, shapes=((1, 1),), dtypes=(torch.float16,)
                        )
                    )
                ],
                id="merged-mamba",
            ),
            pytest.param(
                [
                    _merged_uniform_spec(
                        SlidingWindowSpec(
                            block_size=64,
                            num_kv_heads=1,
                            head_size=64,
                            dtype=torch.float16,
                            sliding_window=128,
                        )
                    )
                ],
                id="merged-swa",
            ),
            pytest.param([_full_attention_spec()], id="unmerged"),
            pytest.param(
                [_merged_uniform_spec(_full_attention_spec())] * 2, id="two-groups"
            ),
        ],
    )
    def test_a_layout_upstream_judges_right_is_left_alone(
        self, monkeypatch, specs, hma_disabled
    ):
        # Every later net -- upstream's `_has_mamba`, our `_has_swa` and
        # `_check_pp_constraints` -- reads the group spec, not what a merged
        # group wraps, so a merged Mamba or SWA group suppressed here would
        # reach the PP region slicing with nothing left to refuse it.
        worker = self._pp_worker(monkeypatch, specs, hma_disabled=hma_disabled)

        assert worker.hma_flag_seen_by_upstream_init is hma_disabled
        assert not hasattr(worker, "_is_hma_required")


# What each layout puts in one block.
PACKED = 2  # rbln_custom_ops: (num_blocks, 2, H, 1, S, D)
SPLIT = 1  # rbln_triton_ops: (2, num_blocks, H, 1, S, D)


class TestTheWindowRangeTilesABlock:
    """The window range cuts a block into the blocks the kernel addresses it in.

    `sw_ratio` of them tile it exactly and one is a contiguous run, so unlike
    the byte prefix this replaced, what a block packs does not enter.
    """

    BLOCK_LEN = 256
    NUM_BLOCKS = 2
    SW_RATIO = 2
    BASES = [0x1000, 0x2000]
    #: Which side's addresses the descriptors are read against.
    DECODE_BASES = BASES

    def _worker(self, kv_per_block):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._sw_ratio = self.SW_RATIO
        w._has_mamba = False
        w._chunk_mode = False
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

    def _window_pieces(self, kv_per_block):
        """The distinct (offset in its block, length) the window range names."""
        return sorted(
            {
                (int(addr - base) % self.BLOCK_LEN, int(length))
                for addr, length, _ in self._descs(kv_per_block)
                if length < self.BLOCK_LEN
                for base in self.DECODE_BASES
                if base <= addr < base + self.BLOCK_LEN * self.NUM_BLOCKS
            }
        )

    def test_the_window_descriptors_tile_the_block(self):
        # A prefix of K and a prefix of V covers a fraction of the block and
        # leaves the window's own bytes behind, at a count the two sides still
        # agree on.
        unit = self.BLOCK_LEN // self.SW_RATIO
        assert self._window_pieces(PACKED) == [
            (unit * i, unit) for i in range(self.SW_RATIO)
        ]

    def test_a_whole_packed_block_stays_one_descriptor(self):
        # What the packing buys: the Full pass names the block once, since K
        # and V are adjacent inside it.
        descs = self._descs(PACKED)
        whole = [d for d in descs if d[1] == self.BLOCK_LEN]
        assert len(whole) == len(self.DECODE_BASES) * self.NUM_BLOCKS
        assert len(descs) == len(whole) * (1 + self.SW_RATIO)


class TestASlidingWindowOnThePeerSide(TestTheWindowRangeTilesABlock):
    """The peer's list has to break the same way, or the two pair off by one.

    Inherits the geometry and the assertions so both sides are read at one set
    of numbers; the base class builds our list and this one the peer's.
    """

    PEER_BASES = [0x9000, 0xA000]
    DECODE_BASES = PEER_BASES

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
