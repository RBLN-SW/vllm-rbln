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


class TestLinkDownExitRole:
    # A consumer serves its running requests on recompute while its links are
    # down; only a producer may exit to be recycled.

    @pytest.mark.parametrize("kv_role", ["kv_consumer", "kv_both"])
    def test_a_consumer_rejects_the_exit(self, monkeypatch, kv_role):
        with pytest.raises(RuntimeError, match="producer"):
            build_worker(monkeypatch, kv_role=kv_role, link_down_exit_s=30)

    def test_a_producer_takes_it(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_role="kv_producer", link_down_exit_s=30)
        assert worker._link_down_exit_s == 30


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


class TestSwaViewRatio:
    def test_opt_off_keeps_ratio_none(self, monkeypatch):
        worker = build_worker(
            monkeypatch,
            swa_view_opt=False,
            specs=[sliding_window_spec(block_size=64, sliding_window=16)],
        )
        assert worker._sw_ratio is None
        # The window is still detected -- it gates the model parallelism guards
        # whether or not the view-opt is on.
        assert worker._has_swa

    def test_pure_full_attention_keeps_ratio_none(self, monkeypatch):
        # A non-sliding-window group contributes no ratio.
        worker = build_worker(monkeypatch, swa_view_opt=True, specs=[MagicMock()])
        assert worker._sw_ratio is None

    def test_sliding_window_derives_block_over_window_ratio(self, monkeypatch):
        worker = build_worker(
            monkeypatch,
            swa_view_opt=True,
            specs=[sliding_window_spec(block_size=64, sliding_window=16)],
        )
        assert worker._sw_ratio == 4

    def test_window_equal_to_block_collapses_to_none(self, monkeypatch):
        # ratio 1 means the SWA view equals the full block -> no trimming.
        worker = build_worker(
            monkeypatch,
            swa_view_opt=True,
            specs=[sliding_window_spec(block_size=64, sliding_window=64)],
        )
        assert worker._sw_ratio is None

    def test_full_attention_groups_are_skipped(self, monkeypatch):
        # The hybrid shape: a model interleaves full-attention and sliding-window
        # layers, so the ratio has to come from the windowed groups alone.
        worker = build_worker(
            monkeypatch,
            swa_view_opt=True,
            specs=[MagicMock(), sliding_window_spec(block_size=64, sliding_window=16)],
        )
        assert worker._sw_ratio == 4

    def test_consistent_ratio_across_groups(self, monkeypatch):
        worker = build_worker(
            monkeypatch,
            swa_view_opt=True,
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
                swa_view_opt=True,
                specs=[
                    sliding_window_spec(block_size=64, sliding_window=16),
                    sliding_window_spec(block_size=64, sliding_window=32),
                ],
            )

    def test_window_not_dividing_block_is_rejected(self, monkeypatch):
        with pytest.raises(AssertionError):
            build_worker(
                monkeypatch,
                swa_view_opt=True,
                specs=[sliding_window_spec(block_size=64, sliding_window=15)],
            )

    def test_mla_with_view_opt_is_rejected_at_startup(self, monkeypatch):
        # The dual desc range and a key-only latent have not been combined,
        # so fail at construction rather than at the first handshake.
        with pytest.raises(RuntimeError, match="SWA_VIEW_OPT"):
            build_worker(
                monkeypatch,
                swa_view_opt=True,
                use_mla=True,
                specs=[sliding_window_spec(block_size=64, sliding_window=16)],
            )


class TestSwaViewDelegation:
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
            swa_view_opt=True,
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
    # With a sliding-window group and the view-opt on, register_local_xfer_handler
    # emits a dual desc range: Full then SWA over the same addresses, trimmed by
    # _sw_ratio.
    def test_swa_builds_dual_desc_ranges(self, make_worker):
        geo = KvGeometry(spec="swa", sliding_window=512, block_size=1024, num_blocks=4)
        w = make_worker(kv_cache=geo, swa_view_opt=True)
        assert (w._has_swa, w._sw_ratio) == (True, 2)

        blocks_data = w.src_blocks_data
        # A packed block is named once in K and once in V, so a region's block
        # is `_kv_per_block` descriptors wide before the two passes double it.
        full_len = w.block_len_per_layer[0] // w._kv_per_block
        # Two passes over every (region, block): a Full-only pass would be half
        # this, so the count alone tells the dual range apart.
        assert len(blocks_data) == w.num_regions * w.num_blocks * w._kv_per_block * 2
        cut = len(blocks_data) // 2
        full, swa = blocks_data[:cut], blocks_data[cut:]
        # The order is a contract, not a detail: a transfer turns (region, block)
        # into a desc id as region * num_blocks + block, so entry i of the list
        # has to BE that pair. Asserting only that SWA repeats Full would pass a
        # reordering applied to both passes.
        bases = w.kv_caches_base_addr[w.engine_id][w.tp_rank]
        order = [
            (region, block, kv)
            for region in range(w.num_regions)
            for block in range(w.num_blocks)
            for kv in range(w._kv_per_block)
        ]
        assert (
            decode(
                full,
                bases=bases,
                block_lens=w.block_len_per_layer,
                num_blocks=w.num_blocks,
            )
            == order
        )
        # The SWA pass repeats those addresses at the trimmed length, which the
        # decoder reads as the same (region, block) at a shorter piece size.
        assert [addr for addr, _, _ in swa] == [addr for addr, _, _ in full]
        assert {desc_len for _, desc_len, _ in full} == {full_len}
        assert {desc_len for _, desc_len, _ in swa} == {full_len // w._sw_ratio}


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
