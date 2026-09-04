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

# Unit coverage: how a consumer pairs with the peers it handshakes.
#
# The handshake fan-out over a peer's shards, and the two axes each shard is
# paired on -- the layers it owns and what its chiplet areas hold, which is KV
# heads only when the cache was cut on that axis. The ZMQ side-channel and
# add_remote_agent are mocked, so none of it needs a live NIXL peer or nixl-rbln.

import time
from collections import defaultdict
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import msgspec
import pytest
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlBaseConnectorWorker,
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlHandshakePayload,
)

import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker as W
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
    RblnNixlAgentMetadata,
    rbln_compat_hash,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_worker import (
    RblnNixlPushConnectorWorker,
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


def _agent_meta(**overrides):
    """A minimal RblnNixlAgentMetadata for the region-slicing cases."""
    fields = dict(
        engine_id="eng",
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
        registered_layer_names=[f"layer.{i}" for i in range(4)],
    )
    fields.update(overrides)
    return RblnNixlAgentMetadata(**fields)


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


def _make_worker(
    *,
    tp_target_ranks=(0,),
    sw_ratio=None,
    compat="HASH",
    tp_ratio=1,
    host_buffer=True,
    has_swa=None,
):
    w = object.__new__(RblnNixlPullConnectorWorker)
    # Host staging closes the D2D-only paths -- head matching and fan-in -- so
    # it is the default and a D2D shape has to ask for the device buffer.
    w.use_host_buffer = host_buffer
    w.device_id = 0
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
    w._has_swa = (sw_ratio is not None) if has_swa is None else has_swa
    w._remote_shard_layer_names = defaultdict(dict)
    w._remote_pp_size = {}
    w._overlapping_ranks = defaultdict(list)
    # Full-model consumer: owns every producer stage's layer, so every stage
    # overlaps (the fan-out default). _FakeSock advertises stage i as "layer.i".
    w.local_seen_layer_names = ["layer.0", "layer.1", "layer.2"]
    w.add_remote_agent = MagicMock(side_effect=lambda meta, rank, tps: f"agent-{rank}")
    # Stubbed so the fan-out tests stay on stage enumeration; the body runs for
    # real in TestShardLocalRegions::test_register_shard_xfer_state_keys_the_stage.
    w._register_shard_xfer_state = MagicMock()
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
    @pytest.mark.parametrize("pp_size", [1, 2, 3])
    def test_every_stage_is_queried_once_and_keyed_by_global_rank(self, pp_size):
        # The fan-out from one stage up. pp_size comes from the pp_rank-0 shard,
        # so that shard must not be queried twice; at pp_size 1 the whole thing
        # reduces to upstream's single-shard shape, keyed by tp_rank.
        w = _make_worker()
        sock = _FakeSock(pp_size=pp_size)

        result = _handshake(w, sock)

        assert result == {r: f"agent-{r}" for r in range(pp_size)}
        assert sock.queried == list(range(pp_size))
        assert sock.queried.count(0) == 1  # bootstrap reused, not re-queried
        assert w.add_remote_agent.call_count == pp_size
        assert dict(w._remote_shard_layer_names["eng"]) == {
            r: (f"layer.{r}",) for r in range(pp_size)
        }
        assert w._remote_pp_size["eng"] == pp_size

    @pytest.mark.parametrize(
        ("local_band", "stage_layers"),
        [
            # Even split: this decode rank owns the second stage's single layer.
            (["layer.1"], None),
            # Uneven split (5 layers / 2 -> [3, 2]): this rank is the smaller last
            # stage, so the peer stage is the LARGER one. Regression for the
            # symmetric-uneven handshake crash: add_remote_agent indexes the local
            # block_len_per_layer by the remote region position, so a larger
            # non-overlapping peer must be skipped before it runs, not after.
            (
                ["layer.3", "layer.4"],
                [["layer.0", "layer.1", "layer.2"], ["layer.3", "layer.4"]],
            ),
        ],
        ids=["even", "uneven_larger_peer"],
    )
    def test_handshake_registers_only_overlapping_stages(
        self, local_band, stage_layers
    ):
        # Of the two producer stages only the one this rank's band overlaps is
        # handshaked and registered for reading. The other is skipped BEFORE
        # add_remote_agent; its layer names are still recorded during enumeration.
        w = _make_worker()
        w.local_seen_layer_names = local_band
        sock = _FakeSock(pp_size=2, stage_layers=stage_layers)

        result = _handshake(w, sock)

        assert result == {1: "agent-1"}
        assert [c.args[1] for c in w.add_remote_agent.call_args_list] == [1]
        assert set(w._remote_shard_layer_names["eng"]) == {0, 1}  # both enumerated
        assert w._overlapping_ranks["eng"] == [1]
        assert w._register_shard_xfer_state.call_count == 1

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
        assert w._register_shard_xfer_state.call_count == 2

    def test_a_retry_after_a_partial_handshake_does_not_double_the_stages(self):
        # A handshake that raises partway has already registered the stages
        # before the failure. Nothing cleans up after it -- upstream only clears
        # an engine it finished -- so the retry has to start from empty, or the
        # read path walks the surviving stages twice and moves every block twice.
        w = _make_worker()
        w._register_shard_xfer_state.side_effect = [None, RuntimeError("boom")]
        with pytest.raises(RuntimeError, match="boom"):
            _handshake(w, _FakeSock(pp_size=2))
        assert "eng" not in w._overlapping_ranks

        w._register_shard_xfer_state.side_effect = None
        _handshake(w, _FakeSock(pp_size=2))
        assert w._overlapping_ranks["eng"] == [0, 1]

    def test_partial_overlap_is_read_per_shard(self):
        # Producer: 2 stages x 2 layers -> stage0=[layer.0,layer.1],
        # stage1=[layer.2,layer.3]. This decode rank owns [layer.1, layer.2], so
        # it reads part of each stage and shares both with another decode rank.
        w = _make_worker()
        w.local_seen_layer_names = ["layer.1", "layer.2"]
        w.num_regions = 2
        w.add_remote_agent = lambda meta, rank, size: "agent"
        w._register_shard_xfer_state = lambda *a, **k: None
        _handshake(w, _FakeSock(pp_size=2, layers_per_stage=2))
        assert w._overlapping_ranks["eng"] == [0, 1]

    def test_trim_agent_meta_to_layers_trims_to_the_owned_layers(self):
        # A stage wider than our band is presented to upstream as just our
        # slice, so its region i pairs with our region i again.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.local_seen_layer_names = ["layer.2"]
        w.num_regions = 1  # one region per layer
        meta = _agent_meta(
            kv_caches_base_addr=[0xA, 0xB, 0xC, 0xD], block_lens=[10, 20, 30, 40]
        )
        # We own the peer's third layer only.
        sliced = w._trim_agent_meta_to_layers(meta, [(2, 0)])
        assert list(sliced.kv_caches_base_addr) == [0xC]
        assert list(sliced.block_lens) == [30]
        # The layer list moves with the regions, or regions-per-layer -- which
        # the handshake check divides out -- comes back wrong.
        assert list(sliced.registered_layer_names) == ["layer.2"]

    def test_trim_agent_meta_to_layers_rejects_a_non_contiguous_span(self):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.local_seen_layer_names = ["layer.0", "layer.2"]
        w.num_regions = 2
        meta = _agent_meta(kv_caches_base_addr=[0xA, 0xB, 0xC], block_lens=[1, 2, 3])
        with pytest.raises(RuntimeError, match="non-contiguous"):
            w._trim_agent_meta_to_layers(meta, [(0, 0), (2, 1)])

    def test_non_pp_producer_wider_than_our_band_is_narrowed(self):
        # The producer runs no pipeline parallelism, but ours is finer, so it
        # holds more layers than this rank owns: it must still be registered
        # per shard rather than as a whole engine.
        w = _make_worker()
        w.local_seen_layer_names = ["layer.1"]
        w.num_regions = 1
        _handshake(w, _FakeSock(pp_size=1, layers_per_stage=3))
        assert w._overlapping_ranks["eng"] == [0]
        assert w._register_shard_xfer_state.call_count == 1

    def test_non_pp_producer_matching_our_band_stays_whole_engine(self):
        # The same path must leave the ordinary case alone: nothing narrowed,
        # so no per-shard state and the read path delegates.
        w = _make_worker()
        _handshake(w, _FakeSock(pp_size=1))
        assert w._overlapping_ranks["eng"] == []
        assert w._register_shard_xfer_state.call_count == 0

    @pytest.mark.parametrize("sw_ratio", [0.5, None])
    def test_swa_plus_pp_raises(self, sw_ratio):
        # The consumer's own guard, hit when it discovers a PP producer while
        # it has a sliding window; _check_pp_constraints is the separate
        # producer-side check. sw_ratio=None is the same model, view-opt off.
        w = _make_worker(sw_ratio=sw_ratio, has_swa=True)
        sock = _FakeSock(pp_size=2)
        with pytest.raises(RuntimeError, match="sliding-window"):
            _handshake(w, sock)

    def test_pp_with_larger_peer_tp_raises(self):
        """A PP producer with MORE TP ranks than us: host staging would then
        borrow upstream's split, which needs a full region list a stage lacks."""
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

    def test_swa_plus_local_pp_raises(self):
        # The peer runs no pipeline, ours does: the guard has to key on either
        # side, not just the peer's.
        w = _make_worker(sw_ratio=0.5)
        w.vllm_config.parallel_config.pipeline_parallel_size = 2
        with pytest.raises(RuntimeError, match="sliding-window"):
            _handshake(w, _FakeSock(pp_size=1))

    @pytest.mark.parametrize(("peer_pp", "local_pp"), [(3, 2), (2, 3), (4, 3)])
    def test_pipelines_that_do_not_divide_raise(self, peer_pp, local_pp):
        # Neither side's stages tile the other's, so a stage's layers straddle
        # two of ours and there is no whole band to pair.
        w = _make_worker()
        w.vllm_config.parallel_config.pipeline_parallel_size = local_pp
        with pytest.raises(RuntimeError, match="multiple of the other"):
            _handshake(w, _FakeSock(pp_size=peer_pp))

    def test_larger_peer_tp_is_allowed_without_peer_pp(self):
        # The reverse shape: a peer with more TP ranks and no pipeline of its
        # own is a target, not a rejection -- only a pipelined peer with more
        # TP ranks stays out. Unequal TP also decides how the peer is paired,
        # so the registration must go by head band, not by position.
        w = _make_worker(tp_target_ranks=(0,), tp_ratio=-2, host_buffer=False)
        w.vllm_config.parallel_config.pipeline_parallel_size = 2
        w.local_seen_layer_names = ["layer.1"]
        w.num_regions = 1
        w._add_remote_agent_head_matched = MagicMock(return_value="agent")
        # Chiplet geometry is exercised in TestFanInAreaPartition and
        # TestHeadBandMatching; stub it so this stays about the guard and the
        # registration decision.
        w._peer_head_split = lambda *a, **k: 1
        w._fan_in_peer_areas = lambda *a, **k: [0]
        _handshake(w, _FakeSock(pp_size=1, layers_per_stage=3), remote_tp_size=4)
        assert w._overlapping_ranks["eng"] == [0]
        assert w._add_remote_agent_head_matched.call_args.args[2] == 4
        w.add_remote_agent.assert_not_called()

    def test_a_finer_peer_alone_forces_the_per_shard_path(self):
        # Nothing else narrows here -- one stage, our band is exactly its
        # layers, one piece per region -- so being cut finer than us is the only
        # reason to leave upstream's whole-engine handle, which would otherwise
        # describe every producer rank's band at once.
        w = _make_worker(tp_ratio=-2, host_buffer=False)
        w.local_seen_layer_names = ["layer.0"]
        w.num_regions = 1
        w._add_remote_agent_head_matched = lambda *a, **k: "agent"
        w._peer_head_split = lambda *a, **k: 1
        w._fan_in_peer_areas = lambda *a, **k: [0]
        _handshake(w, _FakeSock(pp_size=1, layers_per_stage=1), remote_tp_size=4)
        assert w._register_shard_xfer_state.call_count == 1
        assert w._overlapping_ranks["eng"] == [0]

    def test_each_stage_is_registered_with_its_own_geometry(self):
        # The loop hands three things per stage, and the two functions it calls
        # are stubbed everywhere else in this class, so this is where they are
        # pinned: the peer's own layer names, the head geometry of that peer,
        # and its TP rank -- which is NOT the flat rank the stages are keyed by.
        w = _make_worker(tp_ratio=2, host_buffer=False)
        w._add_remote_agent_head_matched = MagicMock(return_value="agent")
        w._peer_head_split = lambda *a, **k: 2
        w._fan_in_peer_areas = MagicMock(return_value=[0])

        _handshake(w, _FakeSock(pp_size=2), remote_tp_size=2)

        # The last stage is global rank 2 of a TP2 peer, i.e. its tp_rank 0,
        # and it advertises layer.2.
        assert w._fan_in_peer_areas.call_args.args == (0, 2)
        assert w._add_remote_agent_head_matched.call_args.kwargs[
            "registered_layer_names"
        ] == ("layer.2",)
        kwargs = w._register_shard_xfer_state.call_args.kwargs
        assert (kwargs["split"], kwargs["replica_fanout"]) == (2, 1)

    def test_a_split_region_alone_forces_the_per_shard_path(self):
        # Companion to test_a_finer_peer_alone_forces_the_per_shard_path: here
        # nothing narrows either, but our region spans several of the peer's
        # slices, so it is read in pieces the whole-engine handle cannot name.
        w = _make_worker(tp_ratio=2, host_buffer=False)
        w.local_seen_layer_names = ["layer.0"]
        w.num_regions = 1
        w._add_remote_agent_head_matched = lambda *a, **k: "agent"
        w._peer_head_split = lambda *a, **k: 2
        w._fan_in_peer_areas = lambda *a, **k: None

        _handshake(w, _FakeSock(pp_size=1, layers_per_stage=1), remote_tp_size=1)

        assert w._register_shard_xfer_state.call_count == 1
        assert w._overlapping_ranks["eng"] == [0]

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


class TestLayerOverlap:
    # Name-based matching of a producer shard's layers to ours. Only the local
    # index is read here; the peer position it is paired with is what
    # TestHeadBandMatching's layer-offset cases exercise.

    @staticmethod
    def _worker(local_names):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.local_seen_layer_names = list(local_names)
        return w

    @staticmethod
    def _local(w, names):
        return [local for _, local in w._layer_overlap(names)]

    def test_contiguous_shard(self):
        w = self._worker(["l0", "l1", "l2", "l3"])
        assert self._local(w, ["l2", "l3"]) == [2, 3]
        assert self._local(w, ["l0", "l1"]) == [0, 1]

    def test_full_model_consumer_maps_all(self):
        names = [f"l{i}" for i in range(4)]
        w = self._worker(names)
        assert self._local(w, names) == [0, 1, 2, 3]

    def test_repeated_names_resolved_by_occurrence(self):
        # HMA pools can register a name more than once; match by occurrence.
        w = self._worker(["a", "a", "b"])
        assert self._local(w, ["a", "b", "a"]) == [0, 2, 1]

    def test_zero_overlap_returns_empty(self):
        # A producer stage entirely outside this rank's band -> empty: the
        # stage is read by whichever rank owns it, not here.
        w = self._worker(["l0", "l1"])
        assert self._local(w, ["l2"]) == []

    def test_decode_shard_maps_only_owned_band(self):
        # Decode-PP rank owns layers [l4..l7]: producer stages outside its band
        # map empty; stages inside map to this rank's local indices.
        w = self._worker(["l4", "l5", "l6", "l7"])
        assert self._local(w, ["l0", "l1"]) == []
        assert self._local(w, ["l4", "l5"]) == [0, 1]
        assert self._local(w, ["l6", "l7"]) == [2, 3]


class TestShardLocalRegions:
    # Layer-name -> local region-index expansion and the per-shard local
    # xfer handle (region subset).

    @staticmethod
    def _worker(local_names, num_regions):
        w = object.__new__(RblnNixlPullConnectorWorker)
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

    @pytest.mark.parametrize(
        ("num_regions", "names", "expected"),
        [
            # rpl=2 (K/V split): layer L -> regions [2L, 2L+1], layer-major.
            (8, ("l2", "l3"), [4, 5, 6, 7]),
            (8, ("l0",), [0, 1]),
            # rpl=1: the region index is the layer index.
            (4, ("l1", "l2"), [1, 2]),
        ],
    )
    def test_shard_local_region_ids(self, num_regions, names, expected):
        w = self._worker(["l0", "l1", "l2", "l3"], num_regions=num_regions)
        assert w._shard_local_region_ids(names) == expected

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
        w._sw_ratio = None  # shard registration goes through the SWA dispatch
        w.use_host_buffer = False  # D2D: narrowing comes from chiplet areas
        w._shard_descs_per_block = {}
        w._borrowed_src_handles = set()
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
        # Dispatch to the shard path: no SWA view opt, layer names present. Miss
        # it and a stage registers the whole model's regions, so the descriptor
        # math addresses layers it does not own.
        w = self._wired_worker()
        w._sw_ratio = None
        w._has_swa = False

        with patch.object(
            RblnNixlPullConnectorWorker, "_register_shard_local_xfer_handler"
        ) as shard:
            w.register_local_xfer_handler(16, registered_layer_names=("l2", "l3"))

        shard.assert_called_once_with(
            16,
            ("l2", "l3"),
            peer_areas=None,
            split=1,
            region_ids=None,
            replica_fanout=1,
        )

    def test_a_borrowed_handle_is_recorded_as_borrowed(self):
        # Host staging has no chiplet areas to narrow by, so the split comes
        # from upstream's own per-producer handles rather than a fresh one of
        # ours. Recording that is what keeps cleanup from releasing a handle
        # every peer at this tp ratio shares (test_cleanup_keeps_a_borrowed_
        # handle_alive covers the other end).
        w = self._wired_worker()
        w.kv_cache_config = MagicMock(kv_cache_groups=[object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}

        with (
            patch.object(
                RblnNixlPullConnectorWorker, "_base_fan_in_handle", return_value=7
            ),
            patch.object(
                RblnNixlPullConnectorWorker, "register_local_xfer_handler"
            ) as own_handle,
        ):
            w._register_shard_xfer_state("eng", 2, 16, ("l2", "l3"))

        assert w._borrowed_src_handles == {("eng", 2, 16)}
        assert w.src_xfer_handles_by_remote[("eng", 2, 16)] == 7
        own_handle.assert_not_called()

    def test_register_shard_xfer_state_keys_the_stage(self):
        # Pins the key shape of the two maps the read and cleanup paths use. The
        # fan-out tests stub this function, so a swapped key order or a
        # wrong-length group-id tuple would go unnoticed on both sides.
        w = self._wired_worker()
        w.kv_cache_config = MagicMock(kv_cache_groups=[object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}

        w._register_shard_xfer_state("eng", 2, 16, ("l2", "l3"))

        assert w.src_xfer_handles_by_remote == {("eng", 2, 16): 42}
        # One group id per region of the shard: layers l2,l3 x rpl 2 = 4.
        assert w._shard_region_group_ids == {("eng", 2): (0, 0, 0, 0)}

    def test_register_shard_xfer_state_rejects_multiple_groups(self):
        # The single-group assumption is what makes the all-zero tuple above
        # right; more than one group has to fail rather than mislabel regions.
        w = self._wired_worker()
        w.kv_cache_config = MagicMock(kv_cache_groups=[object(), object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}

        with pytest.raises(AssertionError, match="single KV-cache group"):
            w._register_shard_xfer_state("eng", 2, 16, ("l2", "l3"))


class TestBaseFanInHandle:
    # Host staging has no chiplet areas to narrow with, so a shard reading from
    # a finer-grained producer borrows upstream's per-producer split instead of
    # emitting descriptors that span every producer's head band.

    @staticmethod
    def _worker(*, host_buffer=True, tp_ratio=-4):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.use_host_buffer = host_buffer
        w.block_size = 16
        w.num_regions = 4
        w.transfer_topo = MagicMock()
        w.transfer_topo.tp_ratio.return_value = tp_ratio
        w.src_xfer_handles_by_tp_ratio = {-4: [70, 71, 72, 73]}
        w.tp_mappings = {"eng": MagicMock(all_source_ranks=(0, 1, 2, 3))}
        return w

    def test_picks_the_split_for_this_producer(self):
        w = self._worker()
        assert w._base_fan_in_handle("eng", 2, 16, [0, 1, 2, 3], 4) == 72

    def test_no_narrowing_for_device_transfers(self):
        # D2D narrows by area, so borrowing would double-narrow.
        w = self._worker(host_buffer=False)
        assert w._base_fan_in_handle("eng", 2, 16, [0, 1, 2, 3], 4) is None

    def test_no_narrowing_when_the_producer_is_not_finer(self):
        w = self._worker(tp_ratio=2)
        assert w._base_fan_in_handle("eng", 2, 16, [0, 1, 2, 3], 2) is None

    def test_no_narrowing_for_a_smaller_remote_block(self):
        # Upstream's split describes local blocks; a re-registration at the
        # remote block size is a different descriptor list.
        w = self._worker()
        assert w._base_fan_in_handle("eng", 2, 8, [0, 1, 2, 3], 4) is None

    def test_rejects_a_region_subset(self):
        # Borrowing holds only while upstream's list covers the same regions
        # in the same order; a peer narrowing ours further breaks it.
        w = self._worker()
        with pytest.raises(AssertionError, match="same regions"):
            w._base_fan_in_handle("eng", 2, 16, [2, 3], 4)


class TestValidateRemoteAgentHandshake:
    # PP-aware handshake validation. Upstream
    # ``_validate_remote_agent_handshake`` asserts matching P/D region counts
    # (`len(remote.kv_caches_base_addr) == len(self.block_len_per_layer)`), which
    # a layer-sharded PP producer necessarily violates against a full-model
    # consumer. Regression for that end-to-end AssertionError.

    @staticmethod
    def _consumer(
        *, num_layers=28, dst_num_blocks=8, use_mla=False, host_buffer=True, areas=1
    ):
        # Full-model consumer: num_regions = num_layers * 2 (K/V) * areas, so
        # regions-per-layer = 2 * areas. host-bounce registers one logical region
        # per layer (never the per-chiplet list), so the D2D region-pairing guard
        # is a no-op there -- see TestD2DRegionPairing for that path.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.use_host_buffer = host_buffer
        w.use_mla = use_mla
        w.local_seen_layer_names = [f"l{i}" for i in range(num_layers)]
        w.num_regions = num_layers * 2 * areas
        w.block_len_per_layer = [64] * (num_layers * 2 * areas)
        w.dst_num_blocks = {"eng": dst_num_blocks}
        w.vllm_config = MagicMock()
        w.vllm_config.parallel_config.pipeline_parallel_size = 1
        w._kv_areas = 1
        w._kv_slices = 1
        w._kv_split_axis = KVSplitAxis.HEAD
        w._sw_ratio = None
        w._has_swa = False
        topo = MagicMock()
        topo.get_engine_info.return_value = MagicMock(remote_tp_size=1)
        topo.block_size_ratio.return_value = 1
        topo.tp_ratio.return_value = 1  # equal P/D TP unless a test overrides
        topo.tp_size = 1
        w.transfer_topo = topo
        return w

    @staticmethod
    def _meta(
        *,
        pp_size,
        n_regions,
        num_blocks=8,
        block_size=16,
        kv_areas=1,
        kv_slices=1,
        kv_split_axis=KVSplitAxis.HEAD,
        n_layers=0,
    ):
        return _agent_meta(
            pp_size=pp_size,
            kv_caches_base_addr=[1000 * i for i in range(n_regions)],
            num_blocks=num_blocks,
            block_size=block_size,
            kv_areas=kv_areas,
            kv_slices=kv_slices,
            kv_split_axis=kv_split_axis,
            registered_layer_names=[f"l{i}" for i in range(n_layers)],
        )

    def test_a_peer_on_another_axis_is_refused_through_the_entry_point(self):
        # The guard's own cases call it directly, which says nothing about the
        # entry point still calling it: deleting that one line kept the suite
        # green. Equal TP, so no other refusal can account for the raise.
        w = self._consumer(host_buffer=False)
        with pytest.raises(RuntimeError, match="cut its KV cache on the NON_HEAD"):
            w._validate_remote_agent_handshake(
                self._meta(pp_size=1, n_regions=56, kv_split_axis=KVSplitAxis.NON_HEAD),
                remote_tp_size=1,
            )

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

    def test_pipelined_peer_with_fewer_chiplet_areas_raises(self):
        # D2D pairs region i with region i, so a pipelined peer that expanded a
        # layer into half our chiplet areas moves the wrong bytes -- and the PP
        # sub-multiple assert passes it, 56 being a clean multiple of our 8.
        w = self._consumer(num_layers=28, host_buffer=False, areas=4)
        with pytest.raises(RuntimeError, match="per layer"):
            w._validate_remote_agent_handshake(
                self._meta(pp_size=2, n_regions=56, n_layers=14), remote_tp_size=1
            )

    def test_pipelined_peer_with_matching_chiplet_areas_passes(self):
        # Guard: the pairing check now runs on the PP path, where a stage's
        # totals legitimately differ from ours while regions per layer match.
        w = self._consumer(num_layers=28, host_buffer=False, areas=4)
        w._validate_remote_agent_handshake(
            self._meta(pp_size=2, n_regions=112, n_layers=14), remote_tp_size=1
        )  # no raise

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

    def test_upstream_add_remote_agent_reaches_the_validation(self):
        # Upstream calls _validate_remote_agent_handshake from inside its own
        # add_remote_agent; if that call moves, every topology refusal silently
        # stops firing. The peer below is one upstream accepts and we must not.
        w = self._consumer(num_layers=28, host_buffer=False, areas=4)
        w.tp_rank = 0
        w.use_mla = False
        w.nixl_wrapper = MagicMock()
        w._remote_agents = defaultdict(dict)
        w._group_spec_types = ()
        w.tp_mappings = MagicMock()
        w.kv_caches_base_addr = defaultdict(dict)
        w.dst_num_blocks = {}
        w.transfer_topo.register_remote_engine = MagicMock()
        w.transfer_topo.block_size_ratio.return_value = 1

        with pytest.raises(RuntimeError, match="KV regions over"):
            NixlBaseConnectorWorker.add_remote_agent(
                w,
                _agent_meta(
                    kv_caches_base_addr=[0] * 57,  # not a whole number per layer
                    registered_layer_names=[f"l{i}" for i in range(28)],
                ),
                0,
                1,
            )

    def test_non_pp_delegates_to_upstream(self):
        # pp_size == 1 must fall through to the upstream validation untouched.
        w = self._consumer()
        with patch.object(
            NixlBaseConnectorWorker, "_validate_remote_agent_handshake"
        ) as base_val:
            w._validate_remote_agent_handshake(
                self._meta(pp_size=1, n_regions=56), remote_tp_size=1
            )
        base_val.assert_called_once()

    # Both head-matched routes below are closed to a host-bounce consumer
    # (_is_head_matched_peer), which is why the cases above never reach them.

    def test_non_pp_head_matched_peer_takes_the_head_geometry_check(self):
        w = self._consumer(host_buffer=False)
        w.transfer_topo.tp_ratio.return_value = 2
        with (
            patch.object(
                RblnNixlPullConnectorWorker, "_validate_head_matched_handshake"
            ) as head_val,
            patch.object(
                NixlBaseConnectorWorker, "_validate_remote_agent_handshake"
            ) as base_val,
        ):
            w._validate_remote_agent_handshake(
                _agent_meta(kv_caches_base_addr=[0] * 56, registered_layer_names=[]),
                remote_tp_size=1,
            )
        head_val.assert_called_once()
        # Upstream's check scales a region by heads per RANK, which is what the
        # head-matched check exists to replace, so it must not also run.
        base_val.assert_not_called()

    @pytest.mark.parametrize("host_buffer", [False, True])
    def test_pipelined_head_matched_peer_takes_the_same_check(self, host_buffer):
        # PP and unequal TP compose: the layer axis picks the regions, the head
        # axis where inside each one to read, so the head invariant still holds.
        # Host staging is parametrized because it reaches this only here -- the
        # non-PP path gates head matching off, and the PP branch never calls
        # upstream's own check, so skipping it would leave no byte check at all.
        w = self._consumer(host_buffer=host_buffer)
        w.transfer_topo.tp_ratio.return_value = 2
        w.transfer_topo.get_engine_info.return_value = MagicMock(remote_tp_size=2)
        with patch.object(
            RblnNixlPullConnectorWorker, "_validate_head_matched_handshake"
        ) as head_val:
            w._validate_remote_agent_handshake(
                _agent_meta(
                    pp_size=2,
                    kv_caches_base_addr=[0] * 28,
                    num_blocks=8,
                    registered_layer_names=[f"l{i}" for i in range(14)],
                ),
                remote_tp_size=2,
            )
        head_val.assert_called_once()

    # MLA is REPLICATE across TP ranks, so it has no head band to match and no
    # layer-band path of its own; the topologies that would need either are
    # rejected here rather than paired on arithmetic that does not apply.

    def test_mla_symmetric_tp_without_pp_passes(self):
        w = self._consumer(use_mla=True)
        with patch.object(
            NixlBaseConnectorWorker, "_validate_remote_agent_handshake"
        ) as base_val:
            w._validate_remote_agent_handshake(
                self._meta(pp_size=1, n_regions=56), remote_tp_size=1
            )  # no raise
        # Upstream's positional pairing already expresses a replicated region,
        # so the MLA checks must let it through rather than take it over.
        base_val.assert_called_once()

    @pytest.mark.parametrize("tp_ratio", [2, -2])
    def test_mla_heterogeneous_tp_raises(self, tp_ratio):
        w = self._consumer(use_mla=True, host_buffer=False)
        w.transfer_topo.tp_ratio.return_value = tp_ratio
        with pytest.raises(RuntimeError, match="heterogeneous tensor"):
            w._validate_remote_agent_handshake(
                self._meta(pp_size=1, n_regions=56), remote_tp_size=2
            )

    def test_mla_heterogeneous_tp_on_host_staging_is_allowed(self):
        # The refusal exists because head-band matching would size bands from a
        # head count MLA does not have. Host staging never head-matches, so the
        # same topology is upstream's to handle.
        w = self._consumer(use_mla=True, host_buffer=True)
        w.transfer_topo.tp_ratio.return_value = 2
        with patch.object(
            NixlBaseConnectorWorker, "_validate_remote_agent_handshake"
        ) as base_val:
            w._validate_remote_agent_handshake(
                self._meta(pp_size=1, n_regions=56), remote_tp_size=2
            )
        base_val.assert_called_once()

    def test_mla_with_pipelined_peer_is_allowed(self):
        # A pipelined producer is paired by layer name, which does not depend on
        # whether the cache is head-sharded.
        w = self._consumer(use_mla=True)
        w._validate_remote_agent_handshake(
            self._meta(pp_size=2, n_regions=28), remote_tp_size=1
        )  # no raise

    def test_mla_geometry_mismatch_raises(self):
        # Positional pairing needs both sides to expand a logical region the
        # same way; disagreeing means the block stride differs.
        w = self._consumer(use_mla=True)
        with pytest.raises(RuntimeError, match="chiplet geometry"):
            w._validate_remote_agent_handshake(
                self._meta(pp_size=1, n_regions=56, kv_areas=4, kv_slices=1),
                remote_tp_size=1,
            )


class TestHeadBandMatching:
    """Pairing local and remote chiplet regions by KV head range.

    On D2D a region is one chiplet area, so the heads are area-major and a peer
    with a different TP degree lays them out differently. Position is therefore
    the wrong key; these pin the two ways it goes wrong in practice.
    """

    @staticmethod
    def _worker(*, tp_rank, tp_size, areas, slices, n_logical, block_len):
        w = object.__new__(RblnNixlPullConnectorWorker)
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
        n = n_logical * areas
        return _agent_meta(
            kv_areas=areas,
            kv_slices=slices,
            num_blocks=num_blocks,
            device_id=0,
            # Distinct, easily-read bases: region i starts at base * (i + 1).
            kv_caches_base_addr=[base * (i + 1) for i in range(n)],
            block_lens=[block_len] * n,
        )

    def test_slice_head_bounds(self):
        # 8 KV heads on 4 chiplets. TP1: 2 heads per area, no replication.
        # TP4: 2 heads per rank -> 1 per area, each held by 2 areas.
        def f(*args):
            return RblnNixlPullConnectorWorker._slice_head_bounds(*args, side="local")

        assert f(0, 1, 8, 4, 4) == (0, 2)
        assert f(0, 2, 8, 4, 4) == (0, 1)
        assert f(1, 2, 8, 4, 4) == (4, 1)
        assert f(0, 4, 8, 4, 2) == (0, 1)
        assert f(2, 4, 8, 4, 2) == (4, 1)

    @pytest.mark.parametrize(
        "args,side,message",
        [
            # Upstream serves TP > num_kv_heads by replicating one head across
            # ranks (tp_mapping's `tp_size > total_num_kv_heads` branch); a band
            # is then a fraction of a head, which no descriptor names.
            ((0, 8, 4, 4, 4), "peer", "does not divide the model's 4 KV heads"),
            ((0, 2, 8, 4, 3), "local", "cut into 3 logical slice"),
            ((0, 2, 8, 6, 4), "peer", "6 chiplet area"),
            # A peer that advertises no slices at all: without the `slices <= 0`
            # half of that guard the modulo beside it divides by zero.
            ((0, 2, 8, 4, 0), "peer", "cut into 0 logical slice"),
        ],
    )
    def test_a_geometry_that_cannot_be_banded_is_refused(self, args, side, message):
        # The three refusals run on a peer's advertised numbers as well as our
        # own, so they refuse a pairing across the handshake rather than assert
        # an invariant -- and the message has to name whose numbers failed.
        with pytest.raises(RuntimeError, match=message) as e:
            RblnNixlPullConnectorWorker._slice_head_bounds(*args, side=side)
        assert side in str(e.value)

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
            (1000, 256),
            (1512, 256),  # area0 -> remote area0 + 0
            (1000, 256),
            (1512, 256),  # area1 (replica of h0) -> same
            (1256, 256),
            (1768, 256),  # area2 -> remote area0 + 256
            (1256, 256),
            (1768, 256),  # area3 (replica of h1) -> same
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
            1000,
            1256,  # local area0 (h0) -> remote area0
            1000,
            1256,  # local area1 (h0 replica) -> remote area0
            2000,
            2256,  # local area2 (h1) -> remote area1, NOT area2
            2000,
            2256,  # local area3 (h1 replica) -> remote area1
        ]

    def test_second_rank_reads_its_own_head_band(self):
        """D TP4 rank 2 owns heads 4,5; against a TP2 peer those live on the
        peer's rank 1, whose local slice numbering restarts at head 4."""
        w = self._worker(
            tp_rank=2, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=256)
        out = w._build_head_matched_remote(meta, remote_tp_rank=1, remote_tp_size=2)
        assert [a for a, _, _ in out] == [
            1000,
            1256,
            1000,
            1256,
            2000,
            2256,
            2000,
            2256,
        ]

    def test_multiple_logical_regions_stay_layer_major(self):
        """K and V of the same layer are separate logical regions; the mapping
        must stay inside each one."""
        w = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=2, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=2, block_len=256, num_blocks=1)
        out = w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=2)
        # logical region 0 -> remote regions 0..3, logical region 1 -> 4..7.
        assert [a for a, _, _ in out] == [
            1000,
            1000,
            2000,
            2000,
            5000,
            5000,
            6000,
            6000,
        ]

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
            1000,
            2000,
            1256,
            2256,  # area 0, blocks 0 and 1
            3000,
            4000,
            3256,
            4256,  # area 1
        ]

    def test_a_replicating_peer_is_read_past_its_replica_areas(self):
        """P TP4 -> D TP1: the peer's rank holds 2 of the 8 heads and repeats
        each over 2 of its 4 areas, so its second head begins at its area 2.
        Pairing by slice index would read the replica of the first one."""
        w = self._worker(
            tp_rank=0, tp_size=1, areas=4, slices=4, n_logical=1, block_len=512
        )
        meta = self._meta(
            areas=4, slices=2, n_logical=1, block_len=256, num_blocks=2, base=1000
        )
        out = w._build_head_matched_remote(
            meta, remote_tp_rank=0, remote_tp_size=4, peer_areas=[0]
        )
        # Our area 0 = heads {0,1}; the peer lays them out [h0, h0, h1, h1], so
        # the two pieces come from its regions 0 and 2. Remote page is 256B.
        assert {ln for _, ln, _ in out} == {256}
        assert [a for a, _, _ in out] == [1000, 3000, 1256, 3256]

    def test_layer_offset_composes_with_the_area_filter(self):
        """The reverse-pipeline shape: the peer holds more layers than we own AND
        more TP ranks, so the layer axis takes an offset into its region list
        while the head axis keeps only the areas this peer serves."""
        # We own one layer (2 logical regions, K and V) and all 8 heads on 4
        # areas; the peer is TP2, so each of its areas carries half our heads
        # and we take areas 0,1 from this one.
        w = self._worker(
            tp_rank=0, tp_size=1, areas=4, slices=4, n_logical=2, block_len=512
        )
        w.local_seen_layer_names = ["layer.2"]
        w.num_regions = 2 * 4  # logical regions x areas, all on one layer
        meta = self._meta(
            areas=4, slices=4, n_logical=6, block_len=256, num_blocks=1, base=100
        )
        # The peer advertises three layers; we own its middle one, so our two
        # logical regions land on its regions 2..3 -> remote regions 8..15.
        out = w._build_head_matched_remote(
            meta,
            remote_tp_rank=0,
            remote_tp_size=2,
            registered_layer_names=("layer.1", "layer.2", "layer.3"),
            peer_areas=[0, 1],
        )
        # 2 logical regions x 2 kept areas x 1 block x 2 pieces.
        assert len(out) == 8
        assert {ln for _, ln, _ in out} == {256}
        # Peer region i starts at 100 * (i + 1). Our logical region 0 is the
        # peer's region index 2 -> areas 8,9,10,11 -> bases 900,1000,1100,1200;
        # logical region 1 is index 3 -> 1300,1400,1500,1600.
        assert [a for a, _, _ in out] == [
            900,
            1000,
            1100,
            1200,  # layer.2 K: areas 0,1 -> two pieces each
            1300,
            1400,
            1500,
            1600,  # layer.2 V
        ]

    def test_layer_offset_alone_keeps_every_area(self):
        """Same layer offset with matching TP: no area filter, no piece split."""
        w = self._worker(
            tp_rank=0, tp_size=1, areas=4, slices=4, n_logical=1, block_len=256
        )
        w.local_seen_layer_names = ["layer.2"]
        w.num_regions = 1 * 4  # logical regions x areas
        meta = self._meta(
            areas=4, slices=4, n_logical=3, block_len=256, num_blocks=1, base=100
        )
        out = w._build_head_matched_remote(
            meta,
            remote_tp_rank=0,
            remote_tp_size=1,
            registered_layer_names=("layer.1", "layer.2", "layer.3"),
        )
        # Our single logical region is the peer's index 1 -> remote regions 4..7.
        assert [a for a, _, _ in out] == [500, 600, 700, 800]

    def test_incommensurate_slices_raise(self):
        # 3 heads per area against 2 does not divide; refuse rather than
        # transfer a partial head.
        with pytest.raises(RuntimeError, match="does not divide it"):
            RblnNixlPullConnectorWorker._head_split(3, 2)

    def test_head_split_is_one_unless_we_are_coarser(self):
        f = RblnNixlPullConnectorWorker._head_split
        assert f(1, 1) == 1  # equal granularity
        assert f(1, 2) == 1  # peer coarser -> offset, not split
        assert f(2, 1) == 2  # we are coarser -> two pieces
        assert f(4, 1) == 4

    @pytest.mark.parametrize(
        ("tp_size", "local", "peer", "remote_tp_size", "tp_ratio", "expected"),
        [
            # Peer TP4 holds 2 of the 8 heads and replicates each over 2 areas,
            # so its slice is finer than ours and our region splits.
            (1, (4, 4), (4, 2), 4, -4, 2),
            # The same pair from the other end: we are the finer side, so the
            # peer's slice covers ours whole and an offset suffices.
            (4, (4, 2), (4, 4), 1, 4, 1),
            # Equal TP keeps positional pairing even where the geometries alone
            # would have split.
            (2, (4, 2), (4, 4), 2, 1, 1),
        ],
    )
    def test_peer_head_split_reads_both_geometries(
        self, tp_size, local, peer, remote_tp_size, tp_ratio, expected
    ):
        w = self._worker(
            tp_rank=0,
            tp_size=tp_size,
            areas=local[0],
            slices=local[1],
            n_logical=1,
            block_len=256,
        )
        w.use_host_buffer = False
        w._sw_ratio = None
        w._has_swa = False
        w.transfer_topo.tp_ratio.return_value = tp_ratio
        meta = self._meta(areas=peer[0], slices=peer[1], n_logical=1, block_len=256)
        assert w._peer_head_split(meta, remote_tp_size) == expected

    # The three guards below sit between the head arithmetic and the emitted
    # descriptor, which is the last point at which a wrong address is still
    # cheap: past it the transfer reads whatever bytes the peer has there.

    def test_an_area_routed_to_a_peer_without_its_heads_raises(self):
        # P TP2 -> D TP1: our areas are split across the two producer ranks
        # (_fan_in_peer_areas), and area 0 carries heads 0..1, which live on the
        # peer's rank 0. Handing it to rank 1 asks for heads it does not own.
        w = self._worker(
            tp_rank=0, tp_size=1, areas=4, slices=4, n_logical=1, block_len=512
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=256)
        with pytest.raises(RuntimeError, match=r"outside the peer's range"):
            w._build_head_matched_remote(
                meta, remote_tp_rank=1, remote_tp_size=2, peer_areas=[0]
            )

    def test_a_peer_region_not_divisible_by_its_heads_raises(self):
        # The offset into a coarser peer area is derived as page // heads per
        # area, so a length that does not divide would silently floor and point
        # partway into a head.
        w = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=1, block_len=64
        )
        meta = self._meta(areas=4, slices=2, n_logical=1, block_len=250)
        with pytest.raises(RuntimeError, match="does not split into 4 heads"):
            w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=1)

    def test_a_region_longer_than_the_peer_leaves_at_that_offset_raises(self):
        # Our rank holds heads 2..3, which start halfway into the peer's area,
        # so only half of it is ours to read; a longer local region would run
        # past its end. _validate_head_matched_handshake rejects the same
        # geometry from the advertised lengths, before any descriptor is built.
        w = self._worker(
            tp_rank=1, tp_size=4, areas=4, slices=2, n_logical=1, block_len=384
        )
        meta = self._meta(areas=4, slices=2, n_logical=1, block_len=512)
        with pytest.raises(RuntimeError, match=r"wants 384B at \+256B"):
            w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=1)


class TestFanInAreaPartition:
    """Splitting our chiplet areas across a peer that has MORE TP ranks.

    Our head band then lives on several producer ranks, so a transfer to any
    one of them must carry exactly the areas whose heads that rank owns --
    every area on exactly one peer. Reading an area from the wrong peer is
    silent corruption, not an error, so these pin the partition itself.
    """

    @staticmethod
    def _worker(*, tp_rank, tp_size, areas, slices, host_buffer=False):
        w = object.__new__(RblnNixlPullConnectorWorker)
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

    def test_host_bounce_has_no_areas_to_partition(self):
        # Same exemption on the partition itself, including the geometry whose
        # chiplet bound rejects a D2D transfer: with no areas registered there
        # is nothing for that bound to be about.
        w = self._worker(tp_rank=0, tp_size=1, areas=4, slices=4, host_buffer=True)
        assert w._fan_in_peer_areas(0, remote_tp_size=4) is None
        assert w._fan_in_peer_areas(0, remote_tp_size=8) is None


class TestShardRegionAreaFilter:
    """`_shard_local_region_ids` narrows on two independent axes: a pipeline
    stage's layers and a fan-in peer's chiplet areas."""

    @staticmethod
    def _worker(*, areas, n_layers):
        w = object.__new__(RblnNixlPullConnectorWorker)
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
            0,
            1,
            4,
            5,
            8,
            9,
            12,
            13,
        ]

    def test_both_axes_compose(self):
        # One pipeline stage (layer 1 only) AND one fan-in peer (areas {2,3}).
        w = self._worker(areas=4, n_layers=2)
        assert w._shard_local_region_ids(("l1",), peer_areas=[2, 3]) == [10, 11, 14, 15]


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
    def _worker(
        *,
        host_buffer,
        tp_ratio,
        n_local,
        tp_size=1,
        sw_ratio=None,
        has_swa=None,
        n_layers=7,
    ):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.use_host_buffer = host_buffer
        w.block_len_per_layer = [64] * n_local
        # The check compares regions PER LAYER, so it needs both figures.
        w.num_regions = n_local
        w.local_seen_layer_names = [f"layer.{i}" for i in range(n_layers)]
        w._sw_ratio = sw_ratio
        w._has_swa = (sw_ratio is not None) if has_swa is None else has_swa
        topo = MagicMock()
        topo.tp_ratio.return_value = tp_ratio
        topo.tp_size = tp_size
        topo.get_engine_info.return_value = MagicMock(remote_tp_size=1)
        topo.block_size_ratio.return_value = 1
        w.transfer_topo = topo
        return w

    @staticmethod
    def _meta(n_regions, n_layers=7):
        return _agent_meta(
            kv_caches_base_addr=[1000 * i for i in range(n_regions)],
            registered_layer_names=[f"layer.{i}" for i in range(n_layers)],
        )

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
        # Fan-in: our band spreads over several peer ranks and
        # _fan_in_peer_areas splits our areas between them. Whether an area
        # straddles two peers needs the measured geometry, so it is checked there.
        w = self._worker(host_buffer=False, tp_ratio=-2, n_local=56, tp_size=2)
        w._check_d2d_region_pairing(self._meta(56), remote_tp_size=4)  # no raise

    @pytest.mark.parametrize("sw_ratio", [4, None])
    def test_heterogeneous_tp_with_swa_raises(self, sw_ratio):
        # A sliding window is refused with model parallelism whether or not the
        # view-opt is on: `sw_ratio=None` is the model with the flag off, which
        # keys on `_has_swa` alone.
        w = self._worker(
            host_buffer=False,
            tp_ratio=2,
            n_local=56,
            tp_size=2,
            sw_ratio=sw_ratio,
            has_swa=True,
        )
        with pytest.raises(RuntimeError, match="sliding-window attention"):
            w._check_d2d_region_pairing(self._meta(56), remote_tp_size=1)

    def test_regions_per_layer_mismatch_raises(self):
        # Half the regions over the same layers: the peer expanded to fewer
        # chiplets, so nothing pairs.
        w = self._worker(host_buffer=False, tp_ratio=1, n_local=56)
        with pytest.raises(RuntimeError, match="per layer"):
            w._check_d2d_region_pairing(self._meta(28), remote_tp_size=1)

    def test_peer_holding_more_layers_passes(self):
        # The reverse pipeline shape: the peer publishes every layer while we
        # own a quarter of them. Totals differ by design; per-layer matches.
        w = self._worker(host_buffer=False, tp_ratio=1, n_local=56, n_layers=7)
        w._check_d2d_region_pairing(
            self._meta(224, n_layers=28), remote_tp_size=1
        )  # no raise

    @pytest.mark.parametrize("tp_ratio,n_remote", [(2, 56), (1, 28)])
    def test_host_bounce_exempt(self, tp_ratio, n_remote):
        # Host-bounce has no per-area list to pair, so the check is skipped.
        w = self._worker(host_buffer=True, tp_ratio=tp_ratio, n_local=56)
        w._check_d2d_region_pairing(self._meta(n_remote), remote_tp_size=1)  # no raise


class TestPpConstraints:
    # Reject PP + unsupported features early.

    @staticmethod
    def _worker(
        *,
        pp_size,
        cross_layers=False,
        has_mamba=False,
        sw_ratio=None,
        has_swa=None,
        use_mla=False,
    ):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.vllm_config = MagicMock()
        w.vllm_config.parallel_config.pipeline_parallel_size = pp_size
        w.transfer_topo = MagicMock()
        w.transfer_topo.cross_layers_blocks = cross_layers
        w._has_mamba = has_mamba
        w._sw_ratio = sw_ratio
        w._has_swa = (sw_ratio is not None) if has_swa is None else has_swa
        w.use_mla = use_mla
        return w

    def test_no_pp_is_noop(self):
        # pp_size == 1: even with otherwise-unsupported features, no raise.
        self._worker(
            pp_size=1, cross_layers=True, has_mamba=True, sw_ratio=2, use_mla=True
        )._check_pp_constraints()

    def test_plain_pp_ok(self):
        self._worker(pp_size=2)._check_pp_constraints()  # no raise

    def test_cross_layers_pp_raises(self):
        with pytest.raises(RuntimeError, match="cross-layer-blocks"):
            self._worker(pp_size=2, cross_layers=True)._check_pp_constraints()

    def test_mamba_pp_raises(self):
        with pytest.raises(RuntimeError, match="Mamba"):
            self._worker(pp_size=2, has_mamba=True)._check_pp_constraints()

    @pytest.mark.parametrize("sw_ratio", [2, None])
    def test_swa_pp_raises(self, sw_ratio):
        # `sw_ratio=None` is the model with the view-opt off: a sliding window
        # bars pipelining on its own, which is what `_has_swa` exists for.
        with pytest.raises(RuntimeError, match="sliding-window attention"):
            self._worker(
                pp_size=2, sw_ratio=sw_ratio, has_swa=True
            )._check_pp_constraints()

    def test_mla_pp_is_allowed(self):
        # MLA is replicated on the head axis only; the layer axis is derived
        # from the registration, so pipelining composes with it.
        self._worker(pp_size=2, use_mla=True)._check_pp_constraints()  # no raise


class TestPublishHandshakeMetadata:
    # The shared producer-side helper both the D2D and the host-bounce paths
    # reach, so what a peer pairs on is advertised regardless of transport.

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

    def _publish(
        self,
        *,
        pp_rank,
        pp_size,
        layer_names,
        areas=1,
        slices=1,
        axis=KVSplitAxis.HEAD,
        cls=None,
    ):
        w = object.__new__(cls or RblnNixlPullConnectorWorker)
        # __init__ never ran, so the writer state shutdown() reaches through
        # __del__ is absent; silence it rather than leak an unraisable at GC.
        w.shutdown = lambda: None
        w.compat_hash = "BASE"
        # _check_pp_constraints reads these; a plain PP producer passes.
        w.vllm_config = MagicMock()
        w.vllm_config.parallel_config.pipeline_parallel_size = pp_size
        w.transfer_topo = MagicMock()
        w.transfer_topo.cross_layers_blocks = False
        w._has_mamba = False
        w._sw_ratio = None
        w._has_swa = False
        w.use_mla = False
        # Chiplet geometry travels with the metadata so a consumer with a
        # different TP degree can match head bands. Defaults are host-bounce's
        # permanent values (one logical region, never expanded per area).
        w._kv_areas = areas
        w._kv_slices = slices
        w._kv_split_axis = axis
        pp_group = MagicMock()
        pp_group.rank_in_group = pp_rank
        pp_group.world_size = pp_size
        with patch.object(W, "get_pp_group", return_value=pp_group):
            w._publish_handshake_metadata(self._base_meta(), layer_names)
        return w

    def test_a_writer_publishes_the_write_path_hash(self):
        # The direction is a class fact, and the hash has to come from the class
        # that is running: a producer that writes must not present the hash a
        # reader would accept.
        w = self._publish(
            pp_rank=0,
            pp_size=1,
            layer_names=["l0"],
            cls=RblnNixlPushConnectorWorker,
        )
        assert w.compat_hash == rbln_compat_hash("BASE", writes_into_peer=True)

    def test_advertises_the_split_axis(self):
        # A consumer cannot derive it: the areas and slices it also receives are
        # the same numbers under either axis (see TestSplitAxisConstraints).
        w = self._publish(
            pp_rank=0,
            pp_size=1,
            layer_names=["l0"],
            areas=4,
            slices=4,
            axis=KVSplitAxis.NON_HEAD,
        )
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert decoded.kv_split_axis is KVSplitAxis.NON_HEAD

    def test_advertises_chiplet_geometry(self):
        """Head-band matching on the consumer needs the producer's areas/slices;
        they cannot be derived from the address list without assuming exactly
        two regions per layer."""
        w = self._publish(pp_rank=0, pp_size=1, layer_names=["l0"], areas=4, slices=2)
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert (decoded.kv_areas, decoded.kv_slices) == (4, 2)

    def test_wraps_upstream_and_folds_compat(self):
        w = self._publish(pp_rank=1, pp_size=2, layer_names=["l7", "l8"])
        # compat hash folded with our version and direction, mirrored into the
        # payload. A read-path worker must publish the read-path hash.
        assert w.compat_hash == rbln_compat_hash("BASE", writes_into_peer=False)
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
        # The helper above is only useful if registration reaches it. Host-bounce
        # does so directly (D2D defers to finalize_kv_cache_registration), and
        # the layer names it captures are what the consumer matches regions by.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.kv_buffer_device = "cpu"
        w._use_rbln_nixl_backend = False
        w.xfer_handshake_metadata = MagicMock(
            agent_metadata_bytes=msgspec.msgpack.encode(self._base_meta())
        )
        w._publish_handshake_metadata = MagicMock()
        kv_caches = {"l0": MagicMock(), "l1": MagicMock()}

        with patch.object(NixlBaseConnectorWorker, "register_kv_caches"):
            w.register_kv_caches(kv_caches)

        assert w.local_seen_layer_names == ["l0", "l1"]
        w._publish_handshake_metadata.assert_called_once()
        published_meta, published_names = w._publish_handshake_metadata.call_args[0]
        # Upstream's metadata is handed over decoded, not as bytes.
        assert published_meta.engine_id == "eng"
        assert list(published_names) == ["l0", "l1"]

    def test_single_stage_defaults(self):
        # pp_size == 1 still folds compat but advertises no-PP layer fields.
        w = self._publish(pp_rank=0, pp_size=1, layer_names=["l0"])
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert (decoded.pp_rank, decoded.pp_size) == (0, 1)


class TestCleanupRemoteEngine:
    # Cleanup has to cover the per-stage state this worker adds on top of
    # upstream's, whether the teardown comes from TTL eviction or a re-handshake.

    def test_eviction_reaches_the_override(self):
        # Upstream routes eviction through _cleanup_remote_engine; if it stops,
        # the per-stage state survives a TTL sweep and a re-handshake reads a
        # stage this engine no longer serves.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._engine_ttl = 1.0
        w._engine_last_active = {"eng": time.perf_counter() - 10.0}
        w.nixl_wrapper = MagicMock()
        w.src_xfer_handles_by_remote = {("eng", 0, 16): 100}
        w._shard_region_group_ids = {("eng", 0): (0,)}
        w._shard_descs_per_block = {("eng", 0): 1}
        w._borrowed_src_handles = set()
        w._remote_shard_layer_names = defaultdict(dict, {"eng": {0: ("l0",)}})
        w._overlapping_ranks = defaultdict(list, {"eng": [0]})
        w._remote_pp_size = {"eng": 1}

        with patch.object(NixlBaseConnectorWorker, "_cleanup_remote_engine"):
            NixlBaseConnectorWorker._evict_stale_engines(w)

        # The real override ran: our per-stage state is gone, not just upstream's.
        assert "eng" not in w._overlapping_ranks
        assert [k for k in w.src_xfer_handles_by_remote if k[0] == "eng"] == []
        w.nixl_wrapper.release_dlist_handle.assert_called_once_with(100)

    def test_cleanup_purges_per_stage_state_and_releases_handles(self):
        w = object.__new__(RblnNixlPullConnectorWorker)
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
        w._shard_descs_per_block = {("eng", 0): 1, ("eng", 1): 2, ("other", 0): 1}
        w._borrowed_src_handles = set()
        w._remote_shard_layer_names = defaultdict(dict, {"eng": {0: ("l0",)}})
        w._overlapping_ranks = defaultdict(list, {"eng": [0, 1], "other": [0]})
        w._remote_pp_size = {"eng": 2, "other": 1}

        with patch.object(NixlPullConnectorWorker, "_cleanup_remote_engine") as base:
            w._cleanup_remote_engine("eng")
        base.assert_called_once_with("eng", log_eviction=True)

        # This engine's stages are gone -- a re-handshake must not double-read.
        assert "eng" not in w._overlapping_ranks
        assert "eng" not in w._remote_pp_size
        assert "eng" not in w._remote_shard_layer_names
        assert [k for k in w.src_xfer_handles_by_remote if k[0] == "eng"] == []
        assert [k for k in w._shard_region_group_ids if k[0] == "eng"] == []
        assert [k for k in w._shard_descs_per_block if k[0] == "eng"] == []
        # Local dlist handles are ours to release; one per stage.
        assert sorted(
            c.args[0] for c in w.nixl_wrapper.release_dlist_handle.call_args_list
        ) == [100, 101]
        # Other engines untouched.
        assert w._overlapping_ranks["other"] == [0]
        assert ("other", 0, 16) in w.src_xfer_handles_by_remote

    def test_cleanup_keeps_a_borrowed_handle_alive(self):
        # A borrowed handle is upstream's, shared by every peer at that tp
        # ratio: releasing it here would tear down transfers still in use.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.nixl_wrapper = MagicMock()
        w.src_xfer_handles_by_remote = {("eng", 0, 16): 100, ("eng", 1, 16): 101}
        w._borrowed_src_handles = {("eng", 1, 16)}
        w._shard_region_group_ids = {}
        w._shard_descs_per_block = {}
        w._remote_shard_layer_names = defaultdict(dict)
        w._overlapping_ranks = defaultdict(list)
        w._remote_pp_size = {}

        with patch.object(NixlPullConnectorWorker, "_cleanup_remote_engine"):
            w._cleanup_remote_engine("eng")

        assert [
            c.args[0] for c in w.nixl_wrapper.release_dlist_handle.call_args_list
        ] == [100]
        assert w._borrowed_src_handles == set()


class TestHeadMatchedHandshakeChecks:
    # The checks that replace upstream's length assert once both sides expand a
    # region per chiplet area. Every other test in this file pairs at equal TP or
    # over host-bounce, so this branch is unreachable from them.

    @staticmethod
    def _worker(*, local_len=128, block_size_ratio=1, layout="NHD"):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.tp_rank = 1
        w.kv_cache_layout = layout
        w.block_len_per_layer = [local_len]
        # 8 heads over TP4 is 2 per rank, cut into 2 slices -> 1 head per area.
        w._kv_areas, w._kv_slices = 4, 2
        topo = MagicMock()
        topo.tp_size = 4
        topo.total_num_kv_heads = 8
        topo.block_size_ratio.return_value = block_size_ratio
        w.transfer_topo = topo
        return w

    @staticmethod
    def _meta(*, remote_len, layout="NHD"):
        # A TP1 peer keeps all 8 heads, cut into 4 slices -> 2 heads per area,
        # so its per-area block length must be twice ours for a head to cost the
        # same on both sides.
        return _agent_meta(
            block_size=16,
            kv_cache_layout=layout,
            kv_areas=4,
            kv_slices=4,
            block_lens=[remote_len],
        )

    def test_equal_bytes_per_head_passes(self):
        w = self._worker(local_len=128)
        w._validate_head_matched_handshake(
            self._meta(remote_len=256), remote_tp_size=1
        )  # no raise

    def test_unequal_bytes_per_head_raises(self):
        # The invariant that replaces upstream's heads-per-RANK ratio: what has
        # to match is the bytes one head costs per block, so a peer whose area
        # is not twice ours means a differing block_size, head_dim or dtype.
        w = self._worker(local_len=128)
        with pytest.raises(RuntimeError, match="a KV head occupies"):
            w._validate_head_matched_handshake(
                self._meta(remote_len=300), remote_tp_size=1
            )

    def test_a_peer_geometry_is_reported_as_the_peer_s(self):
        # `_slice_head_bounds` sees one side's numbers at a time and is told
        # which; a call site labelling the peer's geometry as ours would send an
        # operator to the wrong engine.
        w = self._worker()
        meta = _agent_meta(
            block_size=16,
            kv_cache_layout="NHD",
            kv_areas=4,
            kv_slices=4,
            block_lens=[256],
        )
        with pytest.raises(RuntimeError, match="peer tensor-parallel size 16") as e:
            w._validate_head_matched_handshake(meta, remote_tp_size=16)
        assert "local" not in str(e.value)

    def test_our_own_geometry_is_reported_as_ours(self):
        # The mirror of the case above: this rank's numbers are read first, so a
        # local call site wearing the peer's label would blame the wrong engine.
        w = self._worker()
        w.transfer_topo.tp_size = 16
        with pytest.raises(RuntimeError, match="local tensor-parallel size 16") as e:
            w._validate_head_matched_handshake(
                self._meta(remote_len=256), remote_tp_size=1
            )
        assert "peer" not in str(e.value)

    def test_unequal_block_size_raises(self):
        w = self._worker(block_size_ratio=2)
        with pytest.raises(RuntimeError, match="equal P/D block sizes"):
            w._validate_head_matched_handshake(
                self._meta(remote_len=256), remote_tp_size=1
            )

    def test_layout_mismatch_raises(self):
        w = self._worker(layout="NHD")
        with pytest.raises(RuntimeError, match="peer KV layout"):
            w._validate_head_matched_handshake(
                self._meta(remote_len=256, layout="HND"), remote_tp_size=1
            )


class TestHeadMatchedAgentRegistration:
    # Assembling a head-matched peer. The parts have their own tests; what is
    # only pinned here is the rank the head band is computed from, since the
    # caller keys shards by the flat global rank under PP.

    def test_the_head_band_comes_from_the_tp_part_of_the_rank(self):
        # global rank 5 is pp_rank 2 of a TP2 peer, i.e. its tp_rank 1. Feeding
        # 5 through would place the band three ranks too far along, and the flat
        # rank equals the tp_rank whenever the peer runs no pipeline -- which is
        # every shape without PP, so nothing else notices. The layer names ride
        # along untouched: dropping them pairs a stage as if it held every layer.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._remote_agents = {}
        w.dst_num_blocks = {}
        w.kv_caches_base_addr = {"eng": {}}
        w.dst_xfer_side_handles = {"eng": {}}
        w.nixl_memory_type = "VRAM"
        w._kv_areas, w._kv_slices = 4, 2
        w.transfer_topo = MagicMock()
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.add_remote_agent.return_value = "agent"
        meta = MagicMock()
        meta.engine_id = "eng"

        with (
            patch.object(
                RblnNixlPullConnectorWorker, "_register_remote_engine_prelude"
            ),
            patch.object(
                RblnNixlPullConnectorWorker, "_validate_remote_agent_handshake"
            ),
            patch.object(
                RblnNixlPullConnectorWorker, "_fan_in_peer_areas", return_value=None
            ) as areas,
            patch.object(
                RblnNixlPullConnectorWorker,
                "_build_head_matched_remote",
                return_value=[],
            ) as build,
        ):
            w._add_remote_agent_head_matched(
                meta, 5, 2, registered_layer_names=("l1", "l2")
            )

        assert build.call_args.args[1] == 1
        assert build.call_args.kwargs["registered_layer_names"] == ("l1", "l2")
        assert areas.call_args.args[0] == 1

    def test_a_shard_already_registered_is_not_registered_again(self):
        # This path returns before upstream's own idempotence guard, so it
        # carries its own. Without it a re-handshake -- which the retry after a
        # partial one is -- hands NIXL a second agent for a rank it already has.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._remote_agents = {"eng": {3: "already"}}
        w.nixl_wrapper = MagicMock()
        meta = MagicMock()
        meta.engine_id = "eng"

        assert w._add_remote_agent_head_matched(meta, 3, 2) == "already"
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    def test_a_peer_with_a_different_tp_degree_is_routed_to_head_matching(self):
        # The fork every head-matched path hangs off: unequal TP means position
        # is the wrong key, so upstream's add_remote_agent must not be reached.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._sw_ratio = None
        w._has_swa = False
        w.use_host_buffer = False
        w.transfer_topo = MagicMock()
        w.transfer_topo.tp_ratio.return_value = 2
        meta = _agent_meta()

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_add_remote_agent_head_matched",
                return_value="head-matched",
            ) as head_matched,
            patch.object(NixlBaseConnectorWorker, "add_remote_agent") as base,
        ):
            assert w.add_remote_agent(meta, 1, 2) == "head-matched"

        assert head_matched.call_args.args == (meta, 1, 2)
        base.assert_not_called()


class TestSplitAxisConstraints:
    # The guard that reads the advertised axis. Every other check here compares
    # counts; two peers can agree on every count and still mean different axes
    # by them, which is the one thing a count cannot say.

    @staticmethod
    def _worker(*, axis, tp_ratio=1, host_buffer=False):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.use_host_buffer = host_buffer
        w._kv_split_axis = axis
        w._sw_ratio = None
        topo = MagicMock()
        topo.tp_size = 2
        topo.tp_ratio.return_value = tp_ratio
        w.transfer_topo = topo
        return w

    def test_a_peer_that_cut_another_axis_is_rejected(self):
        w = self._worker(axis=KVSplitAxis.HEAD)
        meta = _agent_meta(kv_areas=4, kv_slices=4, kv_split_axis=KVSplitAxis.NON_HEAD)
        with pytest.raises(RuntimeError, match="cut its KV cache on the NON_HEAD"):
            w._check_split_axis_constraints(meta, 2)

    def test_a_context_cut_is_rejected_with_unequal_tp(self):
        # The silent case: the head bands this peer would be matched by name
        # ranges that no chiplet area holds, while every byte count still fits.
        w = self._worker(axis=KVSplitAxis.NON_HEAD, tp_ratio=2)
        meta = _agent_meta(kv_areas=4, kv_slices=4, kv_split_axis=KVSplitAxis.NON_HEAD)
        with pytest.raises(RuntimeError, match="heterogeneous tensor parallelism"):
            w._check_split_axis_constraints(meta, 1)

    def test_a_context_cut_at_equal_tp_passes(self):
        # Area k pairs with area k and no band is consulted, so the bytes are
        # right; rejecting this would refuse the only shape that does work.
        w = self._worker(axis=KVSplitAxis.NON_HEAD)
        meta = _agent_meta(kv_areas=4, kv_slices=4, kv_split_axis=KVSplitAxis.NON_HEAD)
        w._check_split_axis_constraints(meta, 2)  # no raise

    def test_host_bounce_is_exempt(self):
        # Host staging registers one logical buffer per layer and never expands
        # per area, so its HEAD default describes a peer of any axis.
        w = self._worker(axis=KVSplitAxis.HEAD, tp_ratio=2, host_buffer=True)
        meta = _agent_meta(kv_split_axis=KVSplitAxis.NON_HEAD)
        w._check_split_axis_constraints(meta, 1)  # no raise
