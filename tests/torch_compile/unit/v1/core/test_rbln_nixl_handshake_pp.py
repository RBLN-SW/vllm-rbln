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

"""Unit coverage: RBLN NIXL PP-aware consumer handshake fan-out.

Exercises RblnNixlConnectorWorker._nixl_handshake with the ZMQ side-channel and
add_remote_agent mocked, so it needs neither a live NIXL peer nor nixl-rbln.
"""

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
    """ZMQ REQ stand-in: replies to (GET_META_MSG, rank) with rank's payload.

    TP is 1 in these tests, so global_rank == pp_rank."""

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


def _make_worker(*, tp_target_ranks=(0,), sw_ratio=None, compat="HASH", tp_ratio=1):
    w = object.__new__(RblnNixlConnectorWorker)
    w.use_host_buffer = True  # skip current_platform.set_device
    w.transfer_topo = MagicMock()
    w.transfer_topo.handshake_target_ranks.return_value = list(tp_target_ranks)
    # Equal P/D TP unless a test says otherwise: the handshake now consults
    # tp_ratio to decide between positional and head-band region pairing.
    w.transfer_topo.tp_ratio.return_value = tp_ratio
    w.transfer_topo.tp_size = 1
    w.vllm_config = MagicMock()
    w.vllm_config.parallel_config.pipeline_parallel_size = 1
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
    # Per-shard local handle building is exercised in TestShardLocalRegions;
    # stub it here so the fan-out tests stay focused on stage enumeration.
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
        """pp_size == 1: one shard queried, keyed by tp_rank (global_rank)."""
        w = _make_worker()
        sock = _FakeSock(pp_size=1)
        result = _handshake(w, sock)
        assert result == {0: "agent-0"}
        assert sock.queried == [0]  # bootstrap only
        assert w.add_remote_agent.call_count == 1
        assert dict(w._remote_shard_layer_names["eng"]) == {0: ("layer.0",)}

    def test_pp_fans_out_over_stages(self):
        """pp_size == 2: both stages queried and registered by global_rank."""
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
        """The pp_rank-0 shard is queried exactly once (bootstrap reused)."""
        w = _make_worker()
        sock = _FakeSock(pp_size=3)
        _handshake(w, sock)
        assert sock.queried == [0, 1, 2]
        assert sock.queried.count(0) == 1

    def test_handshake_registers_only_overlapping_stages(self):
        """Decode-PP: this rank owns only layer.1, so of the two producer stages
        only stage 1 overlaps -> only it is handshaked (add_remote_agent) and
        registered for reading. The non-overlapping stage 0 is skipped BEFORE
        add_remote_agent so its base FA-remote build never runs (it would index
        the local block_len_per_layer out of range under an uneven split);
        its layer names are still recorded during enumeration."""
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
        """prefill_pp=4, decode_pp=2 (k=2): this decode rank owns 2 producer
        stages' layers, so only those 2 are handshaked/registered; the other
        two non-overlapping stages are skipped before add_remote_agent."""
        w = _make_worker()
        w.local_seen_layer_names = ["layer.0", "layer.1"]  # this rank's band
        sock = _FakeSock(pp_size=4)  # stages advertise layer.0..layer.3
        _handshake(w, sock)
        assert sorted(c.args[1] for c in w.add_remote_agent.call_args_list) == [0, 1]
        assert w._overlapping_ranks["eng"] == [0, 1]  # only owned stages read
        assert w._register_shard_read_state.call_count == 2

    def test_symmetric_uneven_skips_larger_nonoverlapping_peer(self):
        """Symmetric PP with an uneven layer split (e.g. 5 layers / 2 = [3, 2]):
        this decode rank is the *smaller* last stage and owns only its own
        layers. The peer producer stage is *larger* and does not overlap, so it
        must be skipped entirely -- add_remote_agent (and hence the base
        FA-remote build, which indexes the local block_len_per_layer by the
        remote region position) never runs on it. Regression for the
        symmetric-uneven PP4-PP4 handshake out-of-range crash."""
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
        """Prefill pipeline size not an integer multiple of the decode pipeline
        size: a producer stage's layer band straddles this decode rank's band
        (partial overlap) -> reject."""
        w = _make_worker()
        # Producer: 2 stages x 2 layers -> stage0=[layer.0,layer.1],
        # stage1=[layer.2,layer.3]. This decode rank owns a misaligned band
        # [layer.1, layer.2], so stage0 partially overlaps (owns layer.1 only).
        w.local_seen_layer_names = ["layer.1", "layer.2"]
        sock = _FakeSock(pp_size=2, layers_per_stage=2)
        with pytest.raises(RuntimeError, match="partially overlaps"):
            _handshake(w, sock)

    def test_swa_plus_pp_raises(self):
        w = _make_worker(sw_ratio=0.5)
        sock = _FakeSock(pp_size=2)
        with pytest.raises(RuntimeError, match="sliding-window"):
            _handshake(w, sock)

    def test_pp_with_larger_peer_tp_raises(self):
        """A PP producer with MORE TP ranks than us: each of our regions would
        span several of theirs, which no path builds."""
        w = _make_worker(tp_target_ranks=(0,), tp_ratio=-2)
        sock = _FakeSock(pp_size=2)
        with pytest.raises(RuntimeError, match="larger tensor-parallel size"):
            _handshake(w, sock, remote_tp_size=4)

    def test_pp_on_both_sides_with_heterogeneous_tp_raises(self):
        """Layers and heads are each handled, but splitting both axes on both
        sides at once is untested and stays out."""
        w = _make_worker(tp_target_ranks=(0,), tp_ratio=2)
        w.vllm_config.parallel_config.pipeline_parallel_size = 2
        sock = _FakeSock(pp_size=2)
        with pytest.raises(RuntimeError, match="BOTH sides"):
            _handshake(w, sock, remote_tp_size=1)

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
    """Name-based matching of a producer shard's layers to local region
    indices."""

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
        """HMA pools can register a name more than once; match by occurrence."""
        w = self._worker(["a", "a", "b"])
        assert w._local_region_indices_for_layer_names(["a", "b", "a"]) == [0, 2, 1]

    def test_zero_overlap_returns_empty(self):
        """A producer stage entirely outside this rank's band -> empty: the
        stage is read by whichever rank owns it, not here."""
        w = self._worker(["l0", "l1"])
        assert w._local_region_indices_for_layer_names(["l2"]) == []

    def test_decode_shard_maps_only_owned_band(self):
        """Decode-PP rank owns layers [l4..l7]: producer stages outside its band
        map empty; stages inside map to this rank's local indices."""
        w = self._worker(["l4", "l5", "l6", "l7"])
        assert w._local_region_indices_for_layer_names(["l0", "l1"]) == []
        assert w._local_region_indices_for_layer_names(["l4", "l5"]) == [0, 1]
        assert w._local_region_indices_for_layer_names(["l6", "l7"]) == [2, 3]


class TestShardLocalRegions:
    """Layer-name -> local region-index expansion and the per-shard local
    xfer handle (region subset)."""

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
        """rpl=2: layer L -> regions [2L, 2L+1] (layer-major)."""
        w = self._worker(["l0", "l1", "l2", "l3"], num_regions=8)
        assert w._shard_local_region_ids(("l2", "l3")) == [4, 5, 6, 7]
        assert w._shard_local_region_ids(("l0",)) == [0, 1]

    def test_shard_local_region_ids_no_split(self):
        w = self._worker(["l0", "l1", "l2", "l3"], num_regions=4)
        assert w._shard_local_region_ids(("l1", "l2")) == [1, 2]

    def test_register_shard_local_xfer_handler_covers_subset(self):
        w = self._worker(["l0", "l1", "l2", "l3"], num_regions=8)  # rpl=2
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


class TestShardReadPath:
    """Per-stage read loop + shard descriptor ids."""

    def test_get_block_descs_ids_for_shard(self):
        """0-based descs over the shard's regions; group_id picks the block
        group. Single group -> region_id * num_blocks + block_ids[0]."""
        w = object.__new__(RblnNixlConnectorWorker)
        w._shard_region_group_ids = {("eng", 1): (0, 0, 0, 0)}  # 4 regions, group 0
        w._shard_desc_split = {}
        descs = w._get_block_descs_ids_for_shard(
            "eng", 1, num_blocks=10, block_ids=[[2, 5]]
        )
        # region r contributes r*10 + [2,5]: [2,5, 12,15, 22,25, 32,35]
        assert list(descs) == [2, 5, 12, 15, 22, 25, 32, 35]

    def test_get_block_descs_ids_for_shard_with_split(self):
        """split=2: each block becomes two consecutive descriptors, because
        both dlists are laid out region-major, then block, then piece."""
        w = object.__new__(RblnNixlConnectorWorker)
        w._shard_region_group_ids = {("eng", 1): (0, 0)}  # 2 regions
        w._shard_desc_split = {("eng", 1): 2}
        descs = w._get_block_descs_ids_for_shard(
            "eng", 1, num_blocks=10, block_ids=[[2, 5]]
        )
        # region 0: blocks 2,5 -> descs (2*2,+1) and (5*2,+1); region 1 adds 10*2.
        assert list(descs) == [4, 5, 10, 11, 24, 25, 30, 31]

    def test_get_block_descs_ids_for_shard_empty_group(self):
        w = object.__new__(RblnNixlConnectorWorker)
        w._shard_region_group_ids = {("eng", 0): (0, 0)}
        w._shard_desc_split = {}
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
        # single group, 2 regions per shard
        w._shard_region_group_ids = {("eng", r): (0, 0) for r in range(pp_size)}
        w._shard_desc_split = {}
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
        """pp_size=2 -> one prepped READ per stage, each with its own shard
        handles; the request accrues one transfer handle per stage."""
        w = self._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", self._meta([[1, 2]], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        # stage 0 uses local handle 100 / remote 200; stage 1 -> 101 / 201.
        calls = w.nixl_wrapper.make_prepped_xfer.call_args_list
        assert calls[0].args[1] == 100 and calls[0].args[3] == 200
        assert calls[1].args[1] == 101 and calls[1].args[3] == 201
        # completion: one handle per stage -> req done only when both finish.
        assert len(w._recving_transfers["r0"]) == 2

    def test_prefix_hit_notifies_each_stage_no_read(self):
        """Full prefix hit (empty local list): no read, one notif per stage."""
        w = self._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", self._meta([], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 0
        assert w.nixl_wrapper.send_notif.call_count == 2
        assert len(w._recving_transfers["r0"]) == 0

    def test_partial_prefix_hit_trims_remote_per_stage(self):
        """Partial hit: D allocated only the uncached suffix (1 block) while the
        remote prompt is 3 blocks, so the remote is end-trimmed to the last
        block -> local/remote desc counts match per stage and the read fires."""
        w = self._read_worker(pp_size=2)
        w._read_blocks_for_req("r0", self._meta([[7]], [[3, 4, 7]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 2
        for c in w.nixl_wrapper.make_prepped_xfer.call_args_list:
            local_descs, remote_descs = c.args[2], c.args[4]
            assert len(local_descs) == len(remote_descs)
            # 2 regions per shard x 1 (trimmed) block = 2 descriptors.
            assert len(remote_descs) == 2
        assert len(w._recving_transfers["r0"]) == 2

    def test_reads_only_overlapping_stages(self):
        """Decode-PP (m=4, n=2): this rank owns 2 of the 4 producer stages, so
        it READs only its overlapping stages; the other two are neither read
        nor notified (another decode rank owns and notifies them)."""
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
        """Full prefix hit under decode-PP: notify only this rank's stages."""
        w = self._read_worker(pp_size=4)
        w._overlapping_ranks = {"eng": [0, 1]}
        w._read_blocks_for_req("r0", self._meta([], [[3, 4]]))
        assert w.nixl_wrapper.make_prepped_xfer.call_count == 0
        assert w.nixl_wrapper.send_notif.call_count == 2  # only stages 0,1
        notified = {c.args[0] for c in w.nixl_wrapper.send_notif.call_args_list}
        assert notified == {"agent0", "agent1"}


class TestValidateRemoteAgentHandshake:
    """PP-aware handshake validation. The base
    ``_validate_remote_agent_handshake`` asserts matching P/D region counts
    (`len(remote.kv_caches_base_addr) == len(self.block_len_per_layer)`), which
    a layer-sharded PP producer necessarily violates against a full-model
    consumer. Regression for that end-to-end AssertionError."""

    @staticmethod
    def _consumer(*, num_layers=28, dst_num_blocks=8):
        # Full-model host-bounce consumer: num_regions = num_layers * 2 (K/V),
        # so regions-per-layer = 2. host-bounce registers one logical region per
        # layer (never the per-chiplet list), so the D2D region-pairing guard is
        # a no-op here -- see TestD2DRegionPairing for that path.
        w = object.__new__(RblnNixlConnectorWorker)
        w.use_host_buffer = True
        w.local_seen_layer_names = [f"l{i}" for i in range(num_layers)]
        w.num_regions = num_layers * 2
        w.block_len_per_layer = [64] * (num_layers * 2)
        w.dst_num_blocks = {"eng": dst_num_blocks}
        topo = MagicMock()
        topo.get_engine_info.return_value = MagicMock(remote_tp_size=1)
        topo.block_size_ratio.return_value = 1
        topo.tp_ratio.return_value = 1   # equal P/D TP unless a test overrides
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

    def test_pp_with_larger_peer_tp_raises(self):
        """PP no longer bars TP outright — a producer with FEWER TP ranks is
        head-matched. Only the other direction is impossible."""
        w = self._consumer()
        w.transfer_topo.get_engine_info.return_value = MagicMock(remote_tp_size=2)
        w.transfer_topo.tp_ratio.return_value = -2
        with pytest.raises(AssertionError, match="larger TP size"):
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


class TestHeadBandMatching:
    """Pairing local and remote chiplet regions by KV head range.

    On D2D a region is one chiplet area, so the heads are area-major and a peer
    with a different TP degree lays them out differently. Position is therefore
    the wrong key; these pin the two ways it goes wrong in practice.
    """

    @staticmethod
    def _worker(*, tp_rank, tp_size, areas, slices, n_logical, block_len):
        w = object.__new__(RblnNixlConnectorWorker)
        w.tp_rank = tp_rank
        w._kv_areas = areas
        w._kv_slices = slices
        w.block_len_per_layer = [block_len] * (n_logical * areas)
        topo = MagicMock()
        topo.tp_size = tp_size
        topo.total_num_kv_heads = 8
        w.transfer_topo = topo
        w.get_backend_aware_kv_block_len = lambda layer_idx, **_: block_len
        return w

    @staticmethod
    def _meta(*, areas, slices, n_logical, block_len, num_blocks=2, base=1000):
        m = MagicMock()
        m.kv_areas, m.kv_slices = areas, slices
        m.num_blocks, m.device_id = num_blocks, 0
        n = n_logical * areas
        # Distinct, easily-read bases: region i starts at base * (i + 1).
        m.kv_caches_base_addr = [base * (i + 1) for i in range(n)]
        m.block_lens = [block_len] * n
        return m

    def test_slice_head_bounds(self):
        # 8 KV heads on 4 chiplets. TP1: 2 heads per area, no replication.
        # TP4: 2 heads per rank -> 1 per area, each held by 2 areas.
        f = RblnNixlConnectorWorker._slice_head_bounds
        assert f(0, 1, 8, 4, 4) == (0, 2)
        assert f(0, 2, 8, 4, 4) == (0, 1)
        assert f(1, 2, 8, 4, 4) == (4, 1)
        assert f(0, 4, 8, 4, 2) == (0, 1)
        assert f(2, 4, 8, 4, 2) == (4, 1)

    def test_offset_into_coarser_remote_area(self):
        """P TP1 -> D TP4: the peer's area holds 2 heads, we want one of them,
        so half the descriptors start halfway into the remote area."""
        w = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=512)
        out = w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=1)
        # local areas [h0, h0, h1, h1] -> remote area 0 (h0,h1) throughout;
        # h1 sits at +256B inside it. 2 blocks each, stride = remote page 512.
        assert [(a, ln) for a, ln, _ in out] == [
            (1000, 256), (1512, 256),      # area0 -> remote area0 + 0
            (1000, 256), (1512, 256),      # area1 (replica of h0) -> same
            (1256, 256), (1768, 256),      # area2 -> remote area0 + 256
            (1256, 256), (1768, 256),      # area3 (replica of h1) -> same
        ]

    def test_area_index_permutation_zero_offset(self):
        """P TP2 -> D TP4: head widths match so the offset is 0, but local area
        2 carries head 1, which is the peer's area 1 — not its area 2."""
        w = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=256)
        out = w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=2)
        addrs = [a for a, _, _ in out]
        # remote region bases are 1000/2000/3000/4000; block stride 256.
        assert addrs == [
            1000, 1256,   # local area0 (h0) -> remote area0
            1000, 1256,   # local area1 (h0 replica) -> remote area0
            2000, 2256,   # local area2 (h1) -> remote area1, NOT area2
            2000, 2256,   # local area3 (h1 replica) -> remote area1
        ]

    def test_second_rank_reads_its_own_head_band(self):
        """D TP4 rank 2 owns heads 4,5; against a TP2 peer those live on the
        peer's rank 1, whose local slice numbering restarts at head 4."""
        w = self._worker(
            tp_rank=2, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=256)
        out = w._build_head_matched_remote(meta, remote_tp_rank=1, remote_tp_size=2)
        assert [a for a, _, _ in out] == [1000, 1256, 1000, 1256,
                                          2000, 2256, 2000, 2256]

    def test_multiple_logical_regions_stay_layer_major(self):
        """K and V of the same layer are separate logical regions; the mapping
        must stay inside each one."""
        w = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=2, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=2, block_len=256, num_blocks=1)
        out = w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=2)
        # logical region 0 -> remote regions 0..3, logical region 1 -> 4..7.
        assert [a for a, _, _ in out] == [1000, 1000, 2000, 2000,
                                          5000, 5000, 6000, 6000]

    def test_peer_with_narrower_slice_splits_each_region(self):
        """P TP2 -> D TP1: our area holds 2 heads, each of the peer's holds 1,
        so every region is read in two half-length pieces from two different
        remote regions -- block-major, piece-minor."""
        w = self._worker(
            tp_rank=0, tp_size=1, areas=4, slices=4, n_logical=1, block_len=512
        )
        meta = self._meta(
            areas=4, slices=4, n_logical=1, block_len=256, num_blocks=2, base=1000
        )
        out = w._build_head_matched_remote(
            meta, remote_tp_rank=0, remote_tp_size=2, peer_areas=[0, 1]
        )
        # 2 areas x 2 blocks x 2 pieces, every piece half of our 512B region.
        assert len(out) == 8
        assert {ln for _, ln, _ in out} == {256}
        # Local area 0 = heads {0,1} -> peer regions 0 and 1 (bases 1000, 2000);
        # area 1 = heads {2,3} -> peer regions 2 and 3 (3000, 4000). Remote page
        # is 256B, so block 1 sits one page on.
        assert [a for a, _, _ in out] == [
            1000, 2000, 1256, 2256,   # area 0, blocks 0 and 1
            3000, 4000, 3256, 4256,   # area 1
        ]

    def test_incommensurate_slices_raise(self):
        # 3 heads per area against 2 does not divide; refuse rather than
        # transfer a partial head.
        with pytest.raises(RuntimeError, match="does not divide it"):
            RblnNixlConnectorWorker._head_split(3, 2)

    def test_head_split_is_one_unless_we_are_coarser(self):
        f = RblnNixlConnectorWorker._head_split
        assert f(1, 1) == 1  # equal granularity
        assert f(1, 2) == 1  # peer coarser -> offset, not split
        assert f(2, 1) == 2  # we are coarser -> two pieces
        assert f(4, 1) == 4


class TestFanInAreaPartition:
    """Splitting our chiplet areas across a peer that has MORE TP ranks.

    Our head band then lives on several producer ranks, so a transfer to any
    one of them must carry exactly the areas whose heads that rank owns --
    every area on exactly one peer. Reading an area from the wrong peer is
    silent corruption, not an error, so these pin the partition itself.
    """

    @staticmethod
    def _worker(*, tp_rank, tp_size, areas, slices, host_buffer=False):
        w = object.__new__(RblnNixlConnectorWorker)
        w.tp_rank = tp_rank
        w._kv_areas = areas
        w._kv_slices = slices
        w.use_host_buffer = host_buffer
        topo = MagicMock()
        topo.tp_size = tp_size
        topo.total_num_kv_heads = 8
        topo.tp_ratio = lambda remote: (
            tp_size // remote if tp_size >= remote else -(remote // tp_size)
        )
        w.transfer_topo = topo
        return w

    def test_peer_with_fewer_or_equal_tp_is_not_partitioned(self):
        # tp_ratio > 0: the peer holds our whole band, so every area takes part
        # and callers get None rather than an explicit list.
        w = self._worker(tp_rank=0, tp_size=4, areas=4, slices=2)
        assert w._fan_in_peer_areas(0, remote_tp_size=1) is None
        assert w._fan_in_peer_areas(0, remote_tp_size=4) is None

    def test_p4_to_d2_splits_areas_in_half(self):
        """D TP2 rank 0 owns heads 0-3, one per area. P TP4 rank 0 owns heads
        0-1, rank 1 owns 2-3 -- so areas {0,1} and {2,3}."""
        w = self._worker(tp_rank=0, tp_size=2, areas=4, slices=4)
        assert w._fan_in_peer_areas(0, remote_tp_size=4) == [0, 1]
        assert w._fan_in_peer_areas(1, remote_tp_size=4) == [2, 3]

    def test_p4_to_d2_second_decode_rank_reads_the_upper_peers(self):
        # D TP2 rank 1 owns heads 4-7, which live on P TP4 ranks 2 and 3.
        w = self._worker(tp_rank=1, tp_size=2, areas=4, slices=4)
        assert w._fan_in_peer_areas(2, remote_tp_size=4) == [0, 1]
        assert w._fan_in_peer_areas(3, remote_tp_size=4) == [2, 3]
        # ...and nothing from the peers holding the other half.
        assert w._fan_in_peer_areas(0, remote_tp_size=4) == []

    def test_replicated_areas_follow_their_slice(self):
        """D TP4 holds 2 heads over 2 slices, each replicated across 2 areas.
        Both replicas of a slice must go to the same peer."""
        w = self._worker(tp_rank=0, tp_size=4, areas=4, slices=2)
        assert w._fan_in_peer_areas(0, remote_tp_size=8) == [0, 1]
        assert w._fan_in_peer_areas(1, remote_tp_size=8) == [2, 3]

    @pytest.mark.parametrize(
        "tp_size,slices,remote_tp_size",
        [(2, 4, 4), (1, 4, 2), (1, 4, 4), (4, 2, 8)],
    )
    def test_every_area_lands_on_exactly_one_peer(
        self, tp_size, slices, remote_tp_size
    ):
        # The completeness property the descriptor lists depend on: no area
        # read twice (last write wins, silently) and none dropped (stale KV).
        w = self._worker(tp_rank=0, tp_size=tp_size, areas=4, slices=slices)
        seen = [
            area
            for peer in range(remote_tp_size)
            for area in (w._fan_in_peer_areas(peer, remote_tp_size) or [])
        ]
        assert sorted(seen) == [0, 1, 2, 3]

    def test_area_straddling_two_peers_raises(self):
        """P TP8 -> D TP1: an area holds heads {2a, 2a+1} but each peer owns a
        single head, so the area would have to be split across two agents."""
        w = self._worker(tp_rank=0, tp_size=1, areas=4, slices=4)
        with pytest.raises(RuntimeError, match="straddle several"):
            w._fan_in_peer_areas(0, remote_tp_size=8)

    def test_host_bounce_never_fans_in(self):
        # Host-bounce registers one logical full-shape buffer per layer, so
        # upstream's model holds and base handles the whole engine.
        w = self._worker(tp_rank=0, tp_size=2, areas=4, slices=4, host_buffer=True)
        assert w._is_fan_in_peer(remote_tp_size=4) is False


class TestShardRegionAreaFilter:
    """`_shard_local_region_ids` narrows on two independent axes: a pipeline
    stage's layers and a fan-in peer's chiplet areas."""

    @staticmethod
    def _worker(*, areas, n_layers):
        w = object.__new__(RblnNixlConnectorWorker)
        w._kv_areas = areas
        # One K and one V region per layer, each expanded over `areas`.
        w.num_regions = n_layers * 2 * areas
        w.local_seen_layer_names = [f"l{i}" for i in range(n_layers)]
        return w

    def test_no_filter_keeps_every_region(self):
        w = self._worker(areas=4, n_layers=2)
        assert w._shard_local_region_ids(("l0", "l1")) == list(range(16))

    def test_area_filter_keeps_those_areas_of_every_logical_region(self):
        # Region ids are logical-major, area-minor: (layer * K/V) * areas + area.
        # Keeping areas {0,1} keeps K and V of both layers at those offsets.
        w = self._worker(areas=4, n_layers=2)
        assert w._shard_local_region_ids(("l0", "l1"), peer_areas=[0, 1]) == [
            0, 1, 4, 5, 8, 9, 12, 13
        ]

    def test_both_axes_compose(self):
        # One pipeline stage (layer 1 only) AND one fan-in peer (areas {2,3}).
        w = self._worker(areas=4, n_layers=2)
        assert w._shard_local_region_ids(("l1",), peer_areas=[2, 3]) == [
            10, 11, 14, 15
        ]


class TestD2DRegionPairing:
    """D2D publishes one region PER CHIPLET AREA and pairs local region i with
    remote region i positionally, so both sides must expand identically.

    Two topologies break that silently and are rejected at handshake:
    heterogeneous TP (upstream's rank_offset assumes the remote holds this
    rank's heads contiguously in one region -- after chiplet expansion they are
    area-major, so it reads the wrong heads while every numeric assert still
    passes) and differing region counts. Host-bounce registers logical
    full-shape buffers instead and is explicitly exempt."""

    @staticmethod
    def _worker(*, host_buffer, tp_ratio, n_local, tp_size=1, sw_ratio=None):
        w = object.__new__(RblnNixlConnectorWorker)
        w.use_host_buffer = host_buffer
        w.block_len_per_layer = [64] * n_local
        w._sw_ratio = sw_ratio
        topo = MagicMock()
        topo.tp_ratio.return_value = tp_ratio
        topo.tp_size = tp_size
        topo.get_engine_info.return_value = MagicMock(remote_tp_size=1)
        topo.block_size_ratio.return_value = 1
        w.transfer_topo = topo
        return w

    @staticmethod
    def _meta(n_regions):
        m = MagicMock()
        m.pp_size = 1
        m.engine_id = "eng"
        m.kv_caches_base_addr = [1000 * i for i in range(n_regions)]
        return m

    def test_symmetric_tp_passes(self):
        w = self._worker(host_buffer=False, tp_ratio=1, n_local=56)
        w._check_d2d_region_pairing(self._meta(56), remote_tp_size=1)  # no raise

    @pytest.mark.parametrize("tp_ratio", [2, 4])
    def test_peer_with_fewer_tp_ranks_allowed(self, tp_ratio):
        # Handled by _add_remote_agent_head_matched: the peer's regions carry
        # wider head bands, matched by head range rather than by position.
        w = self._worker(host_buffer=False, tp_ratio=tp_ratio, n_local=56, tp_size=4)
        w._check_d2d_region_pairing(self._meta(56), remote_tp_size=1)  # no raise

    def test_peer_with_more_tp_ranks_allowed(self):
        # Fan-in: our head band is spread over several of the peer's ranks and
        # _fan_in_peer_areas splits our chiplet areas between them. Whether an
        # area straddles two peers depends on the measured chiplet geometry, so
        # that bound is enforced there, not in this region-count check.
        w = self._worker(host_buffer=False, tp_ratio=-2, n_local=56, tp_size=2)
        w._check_d2d_region_pairing(self._meta(56), remote_tp_size=4)  # no raise

    def test_heterogeneous_tp_with_swa_view_opt_raises(self):
        # The SWA two-descriptor-range layout and head matching are not combined.
        w = self._worker(
            host_buffer=False, tp_ratio=2, n_local=56, tp_size=2, sw_ratio=4
        )
        with pytest.raises(RuntimeError, match="sliding-window view-opt"):
            w._check_d2d_region_pairing(self._meta(56), remote_tp_size=1)

    def test_region_count_mismatch_raises(self):
        w = self._worker(host_buffer=False, tp_ratio=1, n_local=56)
        with pytest.raises(RuntimeError, match="advertises 28 KV regions"):
            w._check_d2d_region_pairing(self._meta(28), remote_tp_size=1)

    @pytest.mark.parametrize("tp_ratio,n_remote", [(2, 56), (1, 28)])
    def test_host_bounce_exempt(self, tp_ratio, n_remote):
        # Verified working over kv_buffer_device=cpu for both shapes.
        w = self._worker(host_buffer=True, tp_ratio=tp_ratio, n_local=56)
        w._check_d2d_region_pairing(self._meta(n_remote), remote_tp_size=1)


class TestPpConstraints:
    """Reject PP + unsupported features early."""

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
    """The shared producer-side helper used by BOTH the D2D and the
    host-bounce paths, so PP metadata is advertised regardless of transport."""

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

    def _publish(self, *, pp_rank, pp_size, layer_names, areas=1, slices=1):
        w = object.__new__(RblnNixlConnectorWorker)
        w.compat_hash = "BASE"
        # _check_pp_constraints reads these; a plain PP producer passes.
        w.vllm_config = MagicMock()
        w.vllm_config.parallel_config.pipeline_parallel_size = pp_size
        w.transfer_topo = MagicMock()
        w.transfer_topo.cross_layers_blocks = False
        w._has_mamba = False
        w._sw_ratio = None
        # Chiplet geometry travels with the metadata so a consumer with a
        # different TP degree can match head bands. Defaults are host-bounce's
        # permanent values (one logical region, never expanded per area).
        w._kv_areas = areas
        w._kv_slices = slices
        pp_group = MagicMock()
        pp_group.rank_in_group = pp_rank
        pp_group.world_size = pp_size
        with patch.object(W, "get_pp_group", return_value=pp_group):
            w._publish_pp_handshake_metadata(self._base_meta(), layer_names)
        return w

    def test_advertises_chiplet_geometry(self):
        """Head-band matching on the consumer needs the producer's areas/slices;
        they cannot be derived from the address list without assuming exactly
        two regions per layer."""
        w = self._publish(
            pp_rank=0, pp_size=1, layer_names=["l0"], areas=4, slices=2
        )
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert (decoded.kv_areas, decoded.kv_slices) == (4, 2)

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

    def test_single_stage_defaults(self):
        """pp_size == 1 still folds compat but advertises no-PP layer fields."""
        w = self._publish(pp_rank=0, pp_size=1, layer_names=["l0"])
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert (decoded.pp_rank, decoded.pp_size) == (0, 1)
