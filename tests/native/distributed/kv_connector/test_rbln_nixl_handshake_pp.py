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

# Unit coverage: RBLN NIXL PP-aware consumer handshake fan-out.
#
# Exercises RblnNixlConnectorWorker._nixl_handshake with the ZMQ side-channel and
# add_remote_agent mocked, so it needs neither a live NIXL peer nor nixl-rbln.

import queue
from collections import defaultdict
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import msgspec
import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlAgentMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlHandshakePayload,
)

import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.worker as W
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.connector import (
    RblnNixlConnector,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    RblnNixlAgentMetadata,
    rbln_pp_compat_hash,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.worker import (
    RblnNixlConnectorWorker,
)


def _encode_payload(
    pp_rank,
    pp_size,
    *,
    engine_id="eng",
    compat="HASH",
    layers_per_stage=1,
    layer_names=None,
):
    if layer_names is None:
        layer_names = [
            f"layer.{pp_rank * layers_per_stage + j}" for j in range(layers_per_stage)
        ]
    meta = RblnNixlAgentMetadata(
        engine_id=engine_id,
        agent_metadata=b"agent",
        kv_caches_base_addr=[0x1000],
        device_id=0,
        num_blocks=4,
        block_lens=[8192],
        kv_cache_layout="HND",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="RBLN",
        physical_blocks_per_logical_kv_block=1,
        pp_rank=pp_rank,
        pp_size=pp_size,
        registered_layer_names=list(layer_names),
    )
    payload = NixlHandshakePayload(
        compatibility_hash=compat,
        agent_metadata_bytes=msgspec.msgpack.Encoder().encode(meta),
    )
    return msgspec.msgpack.Encoder().encode(payload)


class _FakeSock:
    # ZMQ REQ stand-in: replies to (GET_META_MSG, rank) with rank's payload.
    #
    # TP is 1 in these tests, so global_rank == pp_rank.

    def __init__(
        self,
        pp_size,
        *,
        engine_id="eng",
        compat="HASH",
        layers_per_stage=1,
        stage_layers=None,
    ):
        self.pp_size = pp_size
        self.engine_id = engine_id
        self.compat = compat
        self.layers_per_stage = layers_per_stage
        # Optional per-stage layer-name lists (for uneven splits); indexed by
        # global_rank. When None, stages advertise a uniform layers_per_stage.
        self.stage_layers = stage_layers
        self.queried = []
        self._last = None

    def setsockopt(self, *a):
        pass

    def send(self, msg):
        _, rank = msgspec.msgpack.decode(msg)
        self.queried.append(rank)
        self._last = rank

    def recv(self):
        return _encode_payload(
            self._last,
            self.pp_size,
            engine_id=self.engine_id,
            compat=self.compat,
            layers_per_stage=self.layers_per_stage,
            layer_names=(
                self.stage_layers[self._last] if self.stage_layers is not None else None
            ),
        )


def _make_worker(*, tp_target_ranks=(0,), sw_ratio=None, compat="HASH"):
    w = object.__new__(RblnNixlConnectorWorker)
    w.use_host_buffer = True  # skip current_platform.set_device
    w.transfer_topo = MagicMock()
    w.transfer_topo.handshake_target_ranks.return_value = list(tp_target_ranks)
    w.compat_hash = compat
    w.enforce_compat_hash = True
    w._sw_ratio = sw_ratio
    w._remote_shard_layer_names = defaultdict(dict)
    w._remote_pp_size = {}
    w._overlapping_ranks = defaultdict(list)
    # Full-model consumer: owns every producer stage's layer, so every stage
    # overlaps (the fan-out default). _FakeSock advertises stage i as "layer.i".
    w.local_seen_layer_names = ["layer.0", "layer.1", "layer.2"]
    w.add_remote_agent = MagicMock(side_effect=lambda meta, rank, tps: f"agent-{rank}")
    # Stubbed so the fan-out tests stay on stage enumeration; the body runs for
    # real in TestShardLocalRegions::test_register_shard_read_state_keys_the_stage.
    w._register_shard_read_state = MagicMock()
    return w


@contextmanager
def _patched_socket(sock):
    @contextmanager
    def fake_zmq_ctx(_type, _path):
        yield sock

    with (
        patch.object(W, "zmq_ctx", fake_zmq_ctx),
        patch.object(W, "make_zmq_path", lambda *a: "tcp://x"),
        patch.object(W, "current_platform", MagicMock()),
    ):
        yield


def _handshake(worker, sock, *, remote_tp_size=1, engine_id="eng"):
    with _patched_socket(sock):
        return worker._nixl_handshake("h", 1234, remote_tp_size, engine_id)


class TestPpHandshakeFanout:
    def test_single_stage_matches_base_shape(self):
        # pp_size == 1: one shard queried, keyed by tp_rank (global_rank).
        w = _make_worker()
        sock = _FakeSock(pp_size=1)
        result = _handshake(w, sock)
        assert result == {0: "agent-0"}
        assert sock.queried == [0]  # bootstrap only
        assert w.add_remote_agent.call_count == 1
        assert dict(w._remote_shard_layer_names["eng"]) == {0: ("layer.0",)}

    def test_pp_fans_out_over_stages(self):
        # pp_size == 2: both stages queried and registered by global_rank.
        w = _make_worker()
        sock = _FakeSock(pp_size=2)
        result = _handshake(w, sock)
        assert result == {0: "agent-0", 1: "agent-1"}
        # rank 0 bootstrap (reused), rank 1 queried once.
        assert sock.queried == [0, 1]
        ranks = sorted(c.args[1] for c in w.add_remote_agent.call_args_list)
        assert ranks == [0, 1]
        assert dict(w._remote_shard_layer_names["eng"]) == {
            0: ("layer.0",),
            1: ("layer.1",),
        }
        assert w._remote_pp_size["eng"] == 2

    def test_bootstrap_reuses_first_query(self):
        # The pp_rank-0 shard is queried exactly once (bootstrap reused).
        w = _make_worker()
        sock = _FakeSock(pp_size=3)
        _handshake(w, sock)
        assert sock.queried == [0, 1, 2]
        assert sock.queried.count(0) == 1

    def test_handshake_registers_only_overlapping_stages(self):
        # Decode-PP: this rank owns only layer.1, so of the two producer stages
        # only stage 1 overlaps -> only it is handshaked (add_remote_agent) and
        # registered for reading. The non-overlapping stage 0 is skipped BEFORE
        # add_remote_agent so its base FA-remote build never runs (it would index
        # the local block_len_per_layer out of range under an uneven split);
        # its layer names are still recorded during enumeration.
        w = _make_worker()
        w.local_seen_layer_names = ["layer.1"]  # this decode rank's band
        sock = _FakeSock(pp_size=2)
        result = _handshake(w, sock)
        # Only the overlapping stage (1) is handshaked and read.
        assert result == {1: "agent-1"}
        assert [c.args[1] for c in w.add_remote_agent.call_args_list] == [1]
        assert set(w._remote_shard_layer_names["eng"]) == {0, 1}  # both enumerated
        assert w._overlapping_ranks["eng"] == [1]
        assert w._register_shard_read_state.call_count == 1

    def test_multiple_ratio_registers_k_stages(self):
        # prefill_pp=4, decode_pp=2 (k=2): this decode rank owns 2 producer
        # stages' layers, so only those 2 are handshaked/registered; the other
        # two non-overlapping stages are skipped before add_remote_agent.
        w = _make_worker()
        w.local_seen_layer_names = ["layer.0", "layer.1"]  # this rank's band
        sock = _FakeSock(pp_size=4)  # stages advertise layer.0..layer.3
        _handshake(w, sock)
        assert sorted(c.args[1] for c in w.add_remote_agent.call_args_list) == [0, 1]
        assert w._overlapping_ranks["eng"] == [0, 1]  # only owned stages read
        assert w._register_shard_read_state.call_count == 2

    def test_symmetric_uneven_skips_larger_nonoverlapping_peer(self):
        # Symmetric PP with an uneven layer split (e.g. 5 layers / 2 = [3, 2]):
        # this decode rank is the *smaller* last stage and owns only its own
        # layers. The peer producer stage is *larger* and does not overlap, so it
        # must be skipped entirely -- add_remote_agent (and hence the base
        # FA-remote build, which indexes the local block_len_per_layer by the
        # remote region position) never runs on it. Regression for the
        # symmetric-uneven PP4-PP4 handshake out-of-range crash.
        w = _make_worker()
        # stage 0 = 3 layers (larger), stage 1 = 2 layers (this rank, smaller).
        stage_layers = [["layer.0", "layer.1", "layer.2"], ["layer.3", "layer.4"]]
        w.local_seen_layer_names = ["layer.3", "layer.4"]  # this rank's own stage
        sock = _FakeSock(pp_size=2, stage_layers=stage_layers)
        result = _handshake(w, sock)
        # Only this rank's own (overlapping) stage is handshaked/registered; the
        # larger non-overlapping peer (stage 0) is skipped, not crashed.
        assert result == {1: "agent-1"}
        assert [c.args[1] for c in w.add_remote_agent.call_args_list] == [1]
        assert w._overlapping_ranks["eng"] == [1]
        assert w._register_shard_read_state.call_count == 1

    def test_partial_overlap_raises(self):
        # Prefill pipeline size not an integer multiple of the decode pipeline
        # size: a producer stage's layer band straddles this decode rank's band
        # (partial overlap) -> reject.
        w = _make_worker()
        # Producer: 2 stages x 2 layers -> stage0=[layer.0,layer.1],
        # stage1=[layer.2,layer.3]. This decode rank owns a misaligned band
        # [layer.1, layer.2], so stage0 partially overlaps (owns layer.1 only).
        w.local_seen_layer_names = ["layer.1", "layer.2"]
        sock = _FakeSock(pp_size=2, layers_per_stage=2)
        with pytest.raises(RuntimeError, match="partially overlaps"):
            _handshake(w, sock)

    def test_swa_plus_pp_raises(self):
        # The consumer's own guard, hit when it discovers a PP producer while it
        # runs the SWA view-opt. TestPpConstraints covers the producer-side check
        # in _check_pp_constraints; deleting either leaves its guard uncovered.
        w = _make_worker(sw_ratio=0.5)
        sock = _FakeSock(pp_size=2)
        with pytest.raises(RuntimeError, match="sliding-window"):
            _handshake(w, sock)

    def test_tp_plus_pp_raises(self):
        w = _make_worker(tp_target_ranks=(0,))
        sock = _FakeSock(pp_size=2)
        with pytest.raises(RuntimeError, match="tensor parallel"):
            _handshake(w, sock, remote_tp_size=2)

    def test_compat_hash_mismatch_raises(self):
        w = _make_worker(compat="LOCAL")
        sock = _FakeSock(pp_size=1, compat="REMOTE")
        with pytest.raises(RuntimeError, match="compatibility hash"):
            _handshake(w, sock)

    def test_engine_id_mismatch_raises(self):
        w = _make_worker()
        sock = _FakeSock(pp_size=1, engine_id="other")
        with pytest.raises(RuntimeError, match="engine ID"):
            _handshake(w, sock, engine_id="eng")


class TestLocalRegionIndicesForLayerNames:
    # Name-based matching of a producer shard's layers to local region
    # indices.

    @staticmethod
    def _worker(local_names):
        w = object.__new__(RblnNixlConnectorWorker)
        w.local_seen_layer_names = list(local_names)
        return w

    def test_contiguous_shard(self):
        w = self._worker(["l0", "l1", "l2", "l3"])
        assert w._local_region_indices_for_layer_names(["l2", "l3"]) == [2, 3]
        assert w._local_region_indices_for_layer_names(["l0", "l1"]) == [0, 1]

    def test_full_model_consumer_maps_all(self):
        names = [f"l{i}" for i in range(4)]
        w = self._worker(names)
        assert w._local_region_indices_for_layer_names(names) == [0, 1, 2, 3]

    def test_repeated_names_resolved_by_occurrence(self):
        # HMA pools can register a name more than once; match by occurrence.
        w = self._worker(["a", "a", "b"])
        assert w._local_region_indices_for_layer_names(["a", "b", "a"]) == [0, 2, 1]

    def test_zero_overlap_returns_empty(self):
        # A producer stage entirely outside this rank's band -> empty: the
        # stage is read by whichever rank owns it, not here.
        w = self._worker(["l0", "l1"])
        assert w._local_region_indices_for_layer_names(["l2"]) == []

    def test_decode_shard_maps_only_owned_band(self):
        # Decode-PP rank owns layers [l4..l7]: producer stages outside its band
        # map empty; stages inside map to this rank's local indices.
        w = self._worker(["l4", "l5", "l6", "l7"])
        assert w._local_region_indices_for_layer_names(["l0", "l1"]) == []
        assert w._local_region_indices_for_layer_names(["l4", "l5"]) == [0, 1]
        assert w._local_region_indices_for_layer_names(["l6", "l7"]) == [2, 3]


class TestShardLocalRegions:
    # Layer-name -> local region-index expansion and the per-shard local
    # xfer handle (region subset).

    @staticmethod
    def _worker(local_names, num_regions):
        w = object.__new__(RblnNixlConnectorWorker)
        w.local_seen_layer_names = list(local_names)
        w.num_regions = num_regions
        return w

    def test_regions_per_layer(self):
        w = self._worker(["l0", "l1", "l2", "l3"], num_regions=8)
        assert w._regions_per_layer() == 2  # K/V split
        w2 = self._worker(["l0", "l1"], num_regions=2)
        assert w2._regions_per_layer() == 1

    def test_regions_per_layer_non_divisible_raises(self):
        w = self._worker(["l0", "l1", "l2"], num_regions=8)
        with pytest.raises(AssertionError, match="not divisible"):
            w._regions_per_layer()

    def test_shard_local_region_ids_kv_split(self):
        # rpl=2: layer L -> regions [2L, 2L+1] (layer-major).
        w = self._worker(["l0", "l1", "l2", "l3"], num_regions=8)
        assert w._shard_local_region_ids(("l2", "l3")) == [4, 5, 6, 7]
        assert w._shard_local_region_ids(("l0",)) == [0, 1]

    def test_shard_local_region_ids_no_split(self):
        w = self._worker(["l0", "l1", "l2", "l3"], num_regions=4)
        assert w._shard_local_region_ids(("l1", "l2")) == [1, 2]

    @classmethod
    def _wired_worker(cls):
        # Enough of the worker for the handle building to run for real: 4 layers
        # over 8 regions (rpl=2), base addrs 1000 apart, prep_xfer_dlist -> 42.
        w = cls._worker(["l0", "l1", "l2", "l3"], num_regions=8)
        w.block_size = 16
        w.num_blocks = 4
        w.device_id = 0
        w.engine_id = "eng"
        w.tp_rank = 0
        w._has_mamba = False
        w.nixl_memory_type = "DRAM"
        w.kv_caches_base_addr = {"eng": {0: [1000 * i for i in range(8)]}}
        w.block_len_per_layer = [64] * 8
        w.transfer_topo = MagicMock()
        w.transfer_topo.is_kv_layout_blocks_first = False
        w.get_backend_aware_kv_block_len = MagicMock(return_value=64)
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.prep_xfer_dlist.return_value = 42
        return w

    def test_register_shard_local_xfer_handler_covers_subset(self):
        w = self._wired_worker()

        handle, blocks = w._register_shard_local_xfer_handler(16, ("l2", "l3"))

        assert handle == 42
        # shard regions [4,5,6,7] x num_blocks 4 = 16 descriptors.
        assert len(blocks) == 16
        # first desc: region 4 base addr (4000), block 0.
        assert blocks[0] == (4000, 64, 0)
        # block 1 of region 4: base + 1*stride(64).
        assert blocks[1] == (4000 + 64, 64, 0)
        # regions used are exactly the shard's (no addr below 4000).
        assert min(a for a, _, _ in blocks) == 4000

    def test_register_local_xfer_handler_routes_to_the_shard_path(self):
        # The dispatch that puts a PP stage on the shard path: no SWA view opt,
        # layer names present. Miss it and a stage registers the whole model's
        # regions, so the descriptor math addresses layers it does not own. The
        # no-names branch is upstream's and is covered by the non-PP tests.
        w = self._wired_worker()
        w._sw_ratio = None

        with patch.object(
            RblnNixlConnectorWorker, "_register_shard_local_xfer_handler"
        ) as shard:
            w.register_local_xfer_handler(16, registered_layer_names=("l2", "l3"))

        shard.assert_called_once_with(16, ("l2", "l3"))

    def test_register_shard_read_state_keys_the_stage(self):
        # The read and cleanup paths look these two maps up by exactly these
        # keys, and the fan-out tests reach this function as a stub -- so the
        # real key shape is pinned here or nowhere: a swapped key order or a
        # group-id tuple of the wrong length would leave both sides agreeing
        # with their own fixtures.
        w = self._wired_worker()
        w.kv_cache_config = MagicMock(kv_cache_groups=[object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}

        w._register_shard_read_state("eng", 2, 16, ("l2", "l3"))

        assert w.src_xfer_handles_by_remote == {("eng", 2, 16): 42}
        # One group id per region of the shard: layers l2,l3 x rpl 2 = 4.
        assert w._shard_region_group_ids == {("eng", 2): (0, 0, 0, 0)}

    def test_register_shard_read_state_rejects_multiple_groups(self):
        # The single-group assumption is what makes the all-zero tuple above
        # right; more than one group has to fail rather than mislabel regions.
        w = self._wired_worker()
        w.kv_cache_config = MagicMock(kv_cache_groups=[object(), object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}

        with pytest.raises(AssertionError, match="single KV-cache group"):
            w._register_shard_read_state("eng", 2, 16, ("l2", "l3"))


class TestShardReadPath:
    # Per-stage read loop + shard descriptor ids.

    def test_get_block_descs_ids_for_shard(self):
        # 0-based descs over the shard's regions; group_id picks the block
        # group. Single group -> region_id * num_blocks + block_ids[0].
        w = object.__new__(RblnNixlConnectorWorker)
        w._shard_region_group_ids = {("eng", 1): (0, 0, 0, 0)}  # 4 regions, group 0
        descs = w._get_block_descs_ids_for_shard(
            "eng", 1, num_blocks=10, block_ids=[[2, 5]]
        )
        # region r contributes r*10 + [2,5]: [2,5, 12,15, 22,25, 32,35]
        assert list(descs) == [2, 5, 12, 15, 22, 25, 32, 35]

    def test_get_block_descs_ids_for_shard_empty_group(self):
        w = object.__new__(RblnNixlConnectorWorker)
        w._shard_region_group_ids = {("eng", 0): (0, 0)}
        descs = w._get_block_descs_ids_for_shard("eng", 0, num_blocks=4, block_ids=[[]])
        assert descs.size == 0

    @staticmethod
    def _read_worker(pp_size):
        w = object.__new__(RblnNixlConnectorWorker)
        w._remote_pp_size = {"eng": pp_size}
        # Full-model decode: every stage overlaps. Decode-PP subsets this (see
        # test_reads_only_overlapping_stages).
        w._overlapping_ranks = {"eng": list(range(pp_size))}
        w._has_mamba = False  # non-Mamba scope: _apply_prefix_caching end-trims
        w.world_size = 1
        w.num_blocks = 8
        w.dst_num_blocks = {"eng": 8}
        w._recving_transfers = defaultdict(list)
        w._engine_last_active = {}
        # What the base failure path reads: it logs with the engine id, looks the
        # request's metadata up, queues the failure and the invalidated blocks.
        w.engine_id = "local"
        w._recving_metadata = {}
        w._invalid_block_ids = queue.Queue()
        w._failed_recv_reqs = queue.Queue()
        w._is_hma_required = False
        w.xfer_stats = MagicMock()
        # single group, 2 regions per shard
        w.kv_cache_config = MagicMock(kv_cache_groups=[0])
        w._shard_region_group_ids = {("eng", r): (0, 0) for r in range(pp_size)}
        w.src_xfer_handles_by_remote = {("eng", r, 16): 100 + r for r in range(pp_size)}
        w.dst_xfer_side_handles = {"eng": {r: 200 + r for r in range(pp_size)}}
        w._remote_agents = {"eng": {r: f"agent{r}" for r in range(pp_size)}}
        topo = MagicMock()
        topo.get_engine_info.return_value = MagicMock(
            remote_tp_size=1, remote_block_size=16, remote_physical_blocks_per_logical=1
        )
        topo.tp_ratio.return_value = 1
        topo.block_size_ratio.return_value = 1
        w.transfer_topo = topo
        w._logical_to_remote_kernel_block_ids = lambda ids, _n: ids
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.make_prepped_xfer.side_effect = lambda *a, **k: object()
        return w

    def _meta(self, local_ids, remote_ids):
        remote = MagicMock()
        remote.engine_id = "eng"
        remote.request_id = "r0"
        remote.block_ids = remote_ids
        meta = MagicMock()
        meta.remote = remote
        meta.local_physical_block_ids = local_ids
        return meta

    def test_reads_every_stage(self):
        # pp_size=2 -> one prepped READ per stage, each with its own shard
        # handles; the request accrues one transfer handle per stage.
        w = self._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", self._meta([[1, 2]], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        # stage 0 uses local handle 100 / remote 200; stage 1 -> 101 / 201.
        calls = w.nixl_wrapper.make_prepped_xfer.call_args_list
        assert calls[0].args[1] == 100 and calls[0].args[3] == 200
        assert calls[1].args[1] == 101 and calls[1].args[3] == 201
        # completion: one handle per stage -> req done only when both finish.
        assert len(w._recving_transfers["r0"]) == 2

    def test_a_failed_stage_leaves_no_handles_behind(self):
        # A stage that fails takes the request with it, so nothing may stay in
        # flight for it: get_finished() reports a failed request and drops its
        # metadata, and a leftover handle completing later would report the same
        # request a second time -- against metadata that is already gone.
        w = self._read_worker(pp_size=3)
        first = object()
        w.nixl_wrapper.make_prepped_xfer.side_effect = [first, RuntimeError("boom")]

        w._read_blocks_for_req("r0", self._meta([[1, 2]], [[3, 4]]))

        assert w._recving_transfers["r0"] == []
        # The stage that already submitted is released, and the third is never
        # submitted -- the request is failed, not partially read.
        w.nixl_wrapper.release_xfer_handle.assert_called_once_with(first)
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        # Reported failed exactly once, which is what the engine counts.
        assert w._failed_recv_reqs.qsize() == 1
        assert w._failed_recv_reqs.get_nowait() == "r0"

    def test_prefix_hit_notifies_each_stage_no_read(self):
        # Full prefix hit (empty local list): no read, one notif per stage.
        w = self._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", self._meta([], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 0
        assert w.nixl_wrapper.send_notif.call_count == 2
        assert len(w._recving_transfers["r0"]) == 0

    def test_partial_prefix_hit_trims_remote_per_stage(self):
        # Partial hit: D allocated only the uncached suffix (1 block) while the
        # remote prompt is 3 blocks, so the remote is end-trimmed to the last
        # block -> local/remote desc counts match per stage and the read fires.
        w = self._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", self._meta([[7]], [[3, 4, 7]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        for c in w.nixl_wrapper.make_prepped_xfer.call_args_list:
            local_descs, remote_descs = c.args[2], c.args[4]
            assert len(local_descs) == len(remote_descs)
            # 2 regions per shard x 1 (trimmed) block = 2 descriptors.
            assert len(remote_descs) == 2
        assert len(w._recving_transfers["r0"]) == 2

    def test_single_stage_read_delegates_to_base(self):
        # A producer that advertised pp_size 1 must read through the upstream
        # path untouched -- the per-stage loop below would key handles the
        # non-PP registration never wrote. Liveness is still stamped first,
        # since TTL eviction must not consider this producer idle.
        w = object.__new__(RblnNixlConnectorWorker)
        w._engine_last_active = {}
        w._remote_pp_size = {}  # unknown engine defaults to a single stage
        w.transfer_topo = MagicMock()
        meta = MagicMock()
        meta.remote.engine_id = "eng"
        base = RblnNixlConnectorWorker.__bases__[0]

        with patch.object(base, "_read_blocks_for_req") as base_read:
            w._read_blocks_for_req("r0", meta)

        base_read.assert_called_once_with("r0", meta)
        assert "eng" in w._engine_last_active

    def test_reads_only_overlapping_stages(self):
        # Decode-PP (m=4, n=2): this rank owns 2 of the 4 producer stages, so
        # it READs only its overlapping stages; the other two are neither read
        # nor notified (another decode rank owns and notifies them).
        w = self._read_worker(pp_size=4)
        w._overlapping_ranks = {"eng": [0, 1]}  # this rank's band = stages 0,1
        w._read_blocks_for_req("r0", self._meta([[1, 2]], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        # only stages 0,1 local handles used (100, 101); never 102/103.
        used = {c.args[1] for c in w.nixl_wrapper.make_prepped_xfer.call_args_list}
        assert used == {100, 101}
        assert w.nixl_wrapper.send_notif.call_count == 0
        assert len(w._recving_transfers["r0"]) == 2

    def test_prefix_hit_notifies_only_overlapping_stages(self):
        # Full prefix hit under decode-PP: notify only this rank's stages.
        w = self._read_worker(pp_size=4)
        w._overlapping_ranks = {"eng": [0, 1]}
        w._read_blocks_for_req("r0", self._meta([], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 0
        assert w.nixl_wrapper.send_notif.call_count == 2  # only stages 0,1
        notified = {c.args[0] for c in w.nixl_wrapper.send_notif.call_args_list}
        assert notified == {"agent0", "agent1"}


class TestValidateRemoteAgentHandshake:
    # PP-aware handshake validation. The base
    # ``_validate_remote_agent_handshake`` asserts matching P/D region counts
    # (`len(remote.kv_caches_base_addr) == len(self.block_len_per_layer)`), which
    # a layer-sharded PP producer necessarily violates against a full-model
    # consumer. Regression for that end-to-end AssertionError.

    @staticmethod
    def _consumer(*, num_layers=28, dst_num_blocks=8):
        # Full-model host-bounce consumer: num_regions = num_layers * 2 (K/V),
        # so regions-per-layer = 2.
        w = object.__new__(RblnNixlConnectorWorker)
        w.local_seen_layer_names = [f"l{i}" for i in range(num_layers)]
        w.num_regions = num_layers * 2
        w.block_len_per_layer = [64] * (num_layers * 2)
        w.dst_num_blocks = {"eng": dst_num_blocks}
        topo = MagicMock()
        topo.get_engine_info.return_value = MagicMock(remote_tp_size=1)
        topo.block_size_ratio.return_value = 1
        w.transfer_topo = topo
        return w

    @staticmethod
    def _meta(*, pp_size, n_regions, num_blocks=8, block_size=16):
        m = MagicMock()
        m.pp_size = pp_size
        m.engine_id = "eng"
        m.kv_caches_base_addr = [1000 * i for i in range(n_regions)]
        m.num_blocks = num_blocks
        m.block_size = block_size
        return m

    def test_pp_shard_wellformed_passes(self):
        # 28-layer model, PP2 -> each stage owns 14 layers = 28 regions,
        # a valid sub-multiple of the local 56. Base assert would fire (28!=56).
        w = self._consumer(num_layers=28)
        w._validate_remote_agent_handshake(
            self._meta(pp_size=2, n_regions=28), remote_tp_size=1
        )  # no raise

    def test_pp_shard_region_count_not_multiple_of_rpl_raises(self):
        w = self._consumer(num_layers=28)  # regions-per-layer = 2
        with pytest.raises(AssertionError, match="sub-multiple"):
            w._validate_remote_agent_handshake(
                self._meta(pp_size=2, n_regions=27), remote_tp_size=1
            )

    def test_pp_shard_larger_than_full_model_raises(self):
        w = self._consumer(num_layers=28)  # full model = 56 regions
        with pytest.raises(AssertionError, match="sub-multiple"):
            w._validate_remote_agent_handshake(
                self._meta(pp_size=2, n_regions=58), remote_tp_size=1
            )

    def test_pp_with_tp_raises(self):
        w = self._consumer()
        w.transfer_topo.get_engine_info.return_value = MagicMock(remote_tp_size=2)
        with pytest.raises(AssertionError, match="TP=1"):
            w._validate_remote_agent_handshake(
                self._meta(pp_size=2, n_regions=28), remote_tp_size=2
            )

    def test_pp_block_size_mismatch_raises(self):
        w = self._consumer()
        w.transfer_topo.block_size_ratio.return_value = 2
        with pytest.raises(AssertionError, match="block sizes"):
            w._validate_remote_agent_handshake(
                self._meta(pp_size=2, n_regions=28), remote_tp_size=1
            )

    def test_pp_num_blocks_mismatch_raises(self):
        w = self._consumer(dst_num_blocks=8)
        with pytest.raises(AssertionError):
            w._validate_remote_agent_handshake(
                self._meta(pp_size=2, n_regions=28, num_blocks=16), remote_tp_size=1
            )

    def test_non_pp_delegates_to_base(self):
        # pp_size == 1 must fall through to the upstream validation untouched.
        w = self._consumer()
        base = RblnNixlConnectorWorker.__bases__[0]
        with patch.object(base, "_validate_remote_agent_handshake") as base_val:
            w._validate_remote_agent_handshake(
                self._meta(pp_size=1, n_regions=56), remote_tp_size=1
            )
        base_val.assert_called_once()


class TestPpConstraints:
    # Reject PP + unsupported features early.

    @staticmethod
    def _worker(*, pp_size, cross_layers=False, has_mamba=False, sw_ratio=None):
        w = object.__new__(RblnNixlConnectorWorker)
        w.vllm_config = MagicMock()
        w.vllm_config.parallel_config.pipeline_parallel_size = pp_size
        w.transfer_topo = MagicMock()
        w.transfer_topo.cross_layers_blocks = cross_layers
        w._has_mamba = has_mamba
        w._sw_ratio = sw_ratio
        return w

    def test_no_pp_is_noop(self):
        # pp_size == 1: even with otherwise-unsupported features, no raise.
        self._worker(
            pp_size=1, cross_layers=True, has_mamba=True, sw_ratio=2
        )._check_pp_constraints()

    def test_plain_pp_ok(self):
        self._worker(pp_size=2)._check_pp_constraints()

    def test_cross_layers_pp_raises(self):
        with pytest.raises(RuntimeError, match="cross-layer-blocks"):
            self._worker(pp_size=2, cross_layers=True)._check_pp_constraints()

    def test_mamba_pp_raises(self):
        with pytest.raises(RuntimeError, match="Mamba"):
            self._worker(pp_size=2, has_mamba=True)._check_pp_constraints()

    def test_swa_pp_raises(self):
        with pytest.raises(RuntimeError, match="sliding-window"):
            self._worker(pp_size=2, sw_ratio=2)._check_pp_constraints()


class TestPublishPpHandshakeMetadata:
    # The shared producer-side helper used by BOTH the D2D and the
    # host-bounce paths, so PP metadata is advertised regardless of transport.

    @staticmethod
    def _base_meta():
        return NixlAgentMetadata(
            engine_id="eng",
            agent_metadata=b"agent",
            kv_caches_base_addr=[0x1000, 0x2000],
            device_id=0,
            num_blocks=4,
            block_lens=[8192, 8192],
            kv_cache_layout="HND",
            block_size=16,
            ssm_sizes=(0, 0),
            attn_backend_name="RBLN",
            physical_blocks_per_logical_kv_block=1,
        )

    def _publish(self, *, pp_rank, pp_size, layer_names):
        w = object.__new__(RblnNixlConnectorWorker)
        w.compat_hash = "BASE"
        # _check_pp_constraints reads these; a plain PP producer passes.
        w.vllm_config = MagicMock()
        w.vllm_config.parallel_config.pipeline_parallel_size = pp_size
        w.transfer_topo = MagicMock()
        w.transfer_topo.cross_layers_blocks = False
        w._has_mamba = False
        w._sw_ratio = None
        pp_group = MagicMock()
        pp_group.rank_in_group = pp_rank
        pp_group.world_size = pp_size
        with patch.object(W, "get_pp_group", return_value=pp_group):
            w._publish_pp_handshake_metadata(self._base_meta(), layer_names)
        return w

    def test_wraps_base_and_folds_compat(self):
        w = self._publish(pp_rank=1, pp_size=2, layer_names=["l7", "l8"])
        # compat hash folded with the RBLN PP version and mirrored into payload.
        assert w.compat_hash == rbln_pp_compat_hash("BASE")
        assert w.xfer_handshake_metadata.compatibility_hash == w.compat_hash
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        # base fields preserved ...
        assert decoded.engine_id == "eng"
        assert decoded.block_lens == [8192, 8192]
        assert decoded.kv_caches_base_addr == [0x1000, 0x2000]
        # ... and PP fields populated.
        assert (decoded.pp_rank, decoded.pp_size) == (1, 2)
        assert decoded.registered_layer_names == ["l7", "l8"]

    def test_register_kv_caches_wires_the_publish(self):
        # The helper above is only useful if the registration path reaches it.
        # Host-bounce is the path that does so directly (D2D defers to
        # finalize_kv_cache_registration), and the ordered layer names it
        # captures are what the consumer matches regions by.
        w = object.__new__(RblnNixlConnectorWorker)
        w.kv_buffer_device = "cpu"
        w._use_rbln_nixl_backend = False
        w.xfer_handshake_metadata = MagicMock(
            agent_metadata_bytes=msgspec.msgpack.encode(self._base_meta())
        )
        w._publish_pp_handshake_metadata = MagicMock()
        base = RblnNixlConnectorWorker.__bases__[0]
        kv_caches = {"l0": MagicMock(), "l1": MagicMock()}

        with patch.object(base, "register_kv_caches"):
            w.register_kv_caches(kv_caches)

        assert w.local_seen_layer_names == ["l0", "l1"]
        w._publish_pp_handshake_metadata.assert_called_once()
        published_meta, published_names = w._publish_pp_handshake_metadata.call_args[0]
        # The base metadata is handed over decoded, not as bytes.
        assert published_meta.engine_id == "eng"
        assert list(published_names) == ["l0", "l1"]

    def test_single_stage_defaults(self):
        # pp_size == 1 still folds compat but advertises no-PP layer fields.
        w = self._publish(pp_rank=0, pp_size=1, layer_names=["l0"])
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert (decoded.pp_rank, decoded.pp_size) == (0, 1)


class TestSetXferHandshakeMetadataPpAware:
    # Producer side: every (pp_rank, tp_rank) shard must reach the side channel.
    #
    # EngineCore hands the merged worker dicts to
    # ``set_xfer_handshake_metadata_pp_aware``. The base implementation rejects
    # pp_rank > 0 and keys by tp_rank alone; this connector flattens the pair into
    # the rank a consumer asks for in ``_nixl_handshake``.
    #

    @staticmethod
    def _connector(tp_size):
        c = object.__new__(RblnNixlConnector)
        vllm_config = MagicMock()
        vllm_config.parallel_config.tensor_parallel_size = tp_size
        c._vllm_config = vllm_config
        return c

    def _flatten(self, metadata, *, tp_size):
        c = self._connector(tp_size)
        with patch.object(RblnNixlConnector, "set_xfer_handshake_metadata") as forward:
            c.set_xfer_handshake_metadata_pp_aware(metadata)
        forward.assert_called_once()
        return forward.call_args[0][0]

    def test_single_stage_reduces_to_tp_rank(self):
        # pp_size == 1: flat rank == tp_rank, i.e. the base behavior.
        assert self._flatten({(0, 0): "m0", (0, 1): "m1"}, tp_size=2) == {
            0: "m0",
            1: "m1",
        }

    def test_pp_stages_get_distinct_flat_ranks(self):
        assert self._flatten({(0, 0): "s0", (1, 0): "s1"}, tp_size=1) == {
            0: "s0",
            1: "s1",
        }
        assert self._flatten(
            {(0, 0): "a", (0, 1): "b", (1, 0): "c", (1, 1): "d"}, tp_size=2
        ) == {0: "a", 1: "b", 2: "c", 3: "d"}

    def test_pp_rank_gt_zero_is_accepted(self):
        # The base would raise here; a PP-aware connector must not.
        assert self._flatten({(3, 0): "s3"}, tp_size=1) == {3: "s3"}

    def test_collision_is_rejected(self):
        # A tp_size disagreeing with the reported ranks would silently drop a
        # shard; fail loudly instead.
        with pytest.raises(ValueError, match="Duplicate handshake metadata"):
            self._flatten({(0, 1): "a", (1, 0): "b"}, tp_size=1)


class TestEngineLivenessAndCleanup:
    # TTL eviction (base, engine_ttl=3600s by default) tears a remote engine's
    # state down once its timestamp goes stale, so the PP read path must refresh
    # it and cleanup must cover the per-stage state this worker adds.

    def test_read_marks_remote_active(self):
        w = TestShardReadPath._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", TestShardReadPath()._meta([[1, 2]], [[3, 4]]))
        assert "eng" in w._engine_last_active

    def test_prefix_hit_read_marks_remote_active(self):
        # A full prefix hit sends notifs only -- still activity.
        w = TestShardReadPath._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", TestShardReadPath()._meta([], [[3, 4]]))
        assert "eng" in w._engine_last_active

    def test_cleanup_purges_per_stage_state_and_releases_handles(self):
        w = object.__new__(RblnNixlConnectorWorker)
        w.nixl_wrapper = MagicMock()
        w.src_xfer_handles_by_remote = {
            ("eng", 0, 16): 100,
            ("eng", 1, 16): 101,
            ("other", 0, 16): 300,
        }
        w._shard_region_group_ids = {
            ("eng", 0): (0,),
            ("eng", 1): (0,),
            ("other", 0): (0,),
        }
        w._remote_shard_layer_names = defaultdict(dict, {"eng": {0: ("l0",)}})
        w._overlapping_ranks = defaultdict(list, {"eng": [0, 1], "other": [0]})
        w._remote_pp_size = {"eng": 2, "other": 1}

        with patch.object(W.NixlPullConnectorWorker, "_cleanup_remote_engine") as base:
            w._cleanup_remote_engine("eng")
        base.assert_called_once_with("eng", log_eviction=True)

        # This engine's stages are gone -- a re-handshake must not double-read.
        assert "eng" not in w._overlapping_ranks
        assert "eng" not in w._remote_pp_size
        assert "eng" not in w._remote_shard_layer_names
        assert [k for k in w.src_xfer_handles_by_remote if k[0] == "eng"] == []
        assert [k for k in w._shard_region_group_ids if k[0] == "eng"] == []
        # Local dlist handles are ours to release; one per stage.
        assert sorted(
            c.args[0] for c in w.nixl_wrapper.release_dlist_handle.call_args_list
        ) == [100, 101]
        # Other engines untouched.
        assert w._overlapping_ranks["other"] == [0]
        assert ("other", 0, 16) in w.src_xfer_handles_by_remote
