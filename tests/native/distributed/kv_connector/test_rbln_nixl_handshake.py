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

# Unit coverage: how a consumer pairs with the peers it handshakes -- the
# fan-out over a peer's shards, and the two axes each is paired on: the layers
# it owns, and what its chiplet areas hold (KV heads only when the cache was
# cut on that axis). The side channel is a real ZMQ pair, so the compat-hash
# and engine-id gates are behaviour here; nixl-rbln is the only stand-in.

import collections
import time
from collections import Counter, defaultdict
from contextlib import contextmanager
from unittest.mock import MagicMock, patch

import msgspec
import pytest
from vllm.distributed.kv_transfer.kv_connector.utils import EngineTransferInfo
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlBaseConnectorWorker,
    NixlPullConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    NixlHandshakePayload,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec

from tests.native.distributed.kv_connector.utils import (
    build_worker,
    patched_in_package,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
    RblnNixlAgentMetadata,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)


def _encode_payload(
    pp_rank,
    pp_size,
    *,
    engine_id="eng",
    compat="HASH",
    layers_per_stage=1,
    layer_names=None,
    kv_slices=1,
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
        kv_slices=kv_slices,
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
        kv_slices=1,
    ):
        self.pp_size = pp_size
        self.engine_id = engine_id
        self.compat = compat
        self.layers_per_stage = layers_per_stage
        # Optional per-stage layer-name lists (for uneven splits); indexed by
        # global_rank. When None, stages advertise a uniform layers_per_stage.
        self.stage_layers = stage_layers
        self.kv_slices = kv_slices
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
            kv_slices=self.kv_slices,
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
    # No speculative decoding: the compat hash then folds what it always did.
    w.vllm_config.speculative_config = None
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
        patched_in_package("zmq_ctx", fake_zmq_ctx),
        patched_in_package("make_zmq_path", lambda *a: "tcp://x"),
        patched_in_package("current_platform", MagicMock()),
    ):
        yield


def _handshake(worker, sock, *, remote_tp_size=1, engine_id="eng"):
    with _patched_socket(sock):
        return worker._nixl_handshake("h", 1234, remote_tp_size, engine_id)


def test_peer_meta_block_length_mirrors_the_local_table():
    # peer_meta is the peer's view of the same geometry, so its per-region block
    # length has to be the one KvGeometry reports locally -- halve it and the
    # address stride stops being wider than a region, which is what lets decode()
    # attribute an address to one region.
    from tests.native.distributed.kv_connector.utils import KvGeometry, peer_meta

    decoder = msgspec.msgpack.Decoder(RblnNixlAgentMetadata)
    cases = [
        KvGeometry(spec="full"),
        KvGeometry(spec="mla"),
        # A draft layer owns fewer heads, so the regions are not all one length.
        KvGeometry(per_layer_heads={"l1": 4}),
    ]
    for geo in cases:
        local = geo.xfer_tables(geo.kv_caches()).block_lens
        agent = decoder.decode(peer_meta(geo).agent_metadata_bytes)
        assert list(agent.block_lens) == list(local), geo


def test_a_replicated_head_band_is_refused_rather_than_mismodelled():
    # Regression guard for the refusal in KvGeometry.__post_init__, which owns
    # the reason the replicated case is not modelled.
    from tests.native.distributed.kv_connector.utils import KvGeometry

    with pytest.raises(AssertionError, match="replicated head band"):
        KvGeometry(areas=4, slices=2)


class TestSideChannelOverARealSocket:
    """The metadata query against a real ZMQ peer.

    _query_agent_meta takes the socket, so nothing has to be substituted for
    the transport: the compat-hash gate and the engine-id check run as
    behaviour, and our own metadata fields cross a real msgspec round trip
    rather than being handed back by a mock.
    """

    @staticmethod
    def _sock(port):
        import zmq

        ctx = zmq.Context()
        sock = ctx.socket(zmq.REQ)
        sock.setsockopt(zmq.RCVTIMEO, 5000)
        sock.connect(f"tcp://127.0.0.1:{port}")
        return ctx, sock

    def test_a_matching_peer_round_trips_our_own_metadata_fields(
        self, make_worker, peer_listener
    ):
        from tests.native.distributed.kv_connector.utils import KvGeometry, peer_meta

        geo = KvGeometry(areas=4, slices=4)
        w = make_worker(kv_cache=geo)
        port = peer_listener(
            lambda rank: peer_meta(geo, engine_id="peer", compat_hash=w.compat_hash)
        )
        ctx, sock = self._sock(port)
        try:
            meta = w._query_agent_meta(sock, 0, "peer")
        finally:
            sock.close()
            ctx.term()
        # The fields upstream's NixlAgentMetadata does not have: a peer that
        # dropped them would decode into upstream's type and read as areas 1.
        assert (meta.kv_areas, meta.kv_slices) == (4, 4)
        assert meta.registered_layer_names == list(geo.layers)

    def test_a_peer_whose_hash_differs_is_refused(self, make_worker, peer_listener):
        from tests.native.distributed.kv_connector.utils import KvGeometry, peer_meta

        geo = KvGeometry()
        w = make_worker(kv_cache=geo)
        port = peer_listener(
            lambda rank: peer_meta(geo, engine_id="peer", compat_hash="not-ours")
        )
        ctx, sock = self._sock(port)
        try:
            with pytest.raises(RuntimeError, match="compatibility hash mismatch"):
                w._query_agent_meta(sock, 0, "peer")
        finally:
            sock.close()
            ctx.term()

    def test_a_peer_serving_another_engine_is_refused(self, make_worker, peer_listener):
        from tests.native.distributed.kv_connector.utils import KvGeometry, peer_meta

        geo = KvGeometry()
        w = make_worker(kv_cache=geo)
        port = peer_listener(
            lambda rank: peer_meta(
                geo, engine_id="someone-else", compat_hash=w.compat_hash
            )
        )
        ctx, sock = self._sock(port)
        try:
            with pytest.raises(RuntimeError, match="engine ID mismatch"):
                w._query_agent_meta(sock, 0, "expected-peer")
        finally:
            sock.close()
            ctx.term()


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

    def test_a_fanned_out_peer_alone_forces_the_per_shard_path(self):
        # Third companion: nothing narrows and no region is split, but the peer
        # replicates each of its head slices across chiplet areas, so a write
        # has to reach every copy. The remote list then carries one descriptor
        # per copy while upstream's whole-engine handle carries one per block,
        # and the two are indexed by the same desc ids.
        w = _make_worker(tp_ratio=2, host_buffer=False)
        w.local_seen_layer_names = ["layer.0"]
        w.num_regions = 1
        w._add_remote_agent_head_matched = lambda *a, **k: "agent"
        w._peer_head_split = lambda *a, **k: 1
        w._peer_replica_fanout = lambda *a, **k: 2
        w._fan_in_peer_areas = lambda *a, **k: None

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


class TestPeerRegionView:
    # Runs the real upstream loop, so a release that stops reading a region length
    # through `get_backend_aware_kv_block_len` shows up here as wrong lengths.

    # A consumer holding every layer, the last of which is a speculative draft
    # whose KV region is 6x the target's. rpl = 2 (K/V), so region 2L..2L+1
    # belong to layer L, and the draft's are the last two.
    N_LAYERS = 7
    RPL = 2
    TARGET_LEN = 2048
    DRAFT_LEN = TARGET_LEN * 6

    @classmethod
    def _consumer(cls, *, mla_tail=0, uniform_lens=False):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.local_seen_layer_names = [f"l{i}" for i in range(cls.N_LAYERS)]
        w.num_regions = cls.N_LAYERS * cls.RPL
        w.block_len_per_layer = [cls.TARGET_LEN] * w.num_regions
        if not uniform_lens:
            w.block_len_per_layer[-cls.RPL :] = [cls.DRAFT_LEN] * cls.RPL
        w._region_is_mla = [False] * (w.num_regions - mla_tail) + [True] * mla_tail
        w._kv_areas = 1
        w.device_id = 0
        w.transfer_topo = MagicMock()
        w.transfer_topo.virtually_split_kv_in_blocks = False
        w._mamba_ssm_size = (0, 0)
        w._group_spec_types = [FullAttentionSpec]
        return w

    PEER_BASE = 0x10000

    @classmethod
    def _peer(cls, *, layer_names, block_lens, num_blocks=2, base=PEER_BASE):
        n = len(block_lens)
        return _agent_meta(
            engine_id="p",
            # Distinct, easily-read bases: region i starts at base * (i + 1).
            kv_caches_base_addr=[base * (i + 1) for i in range(n)],
            block_lens=list(block_lens),
            num_blocks=num_blocks,
            registered_layer_names=list(layer_names),
            pp_size=4,
        )

    @staticmethod
    def _plan():
        plan = MagicMock()
        plan.source_ranks_per_group = [(0,)]  # one source rank -> num_reads 1
        plan.rank_offset_factor = 0
        return plan

    def test_a_stage_holding_our_tail_reads_its_own_lengths(self):
        # The failing shape: the peer is the LAST pipeline stage, so its region
        # positions are 0..3 while ours are 10..13. Untranslated, our layers 0-1
        # lengths land on the peer's draft regions -- the length mismatch NIXL
        # rejects at transfer setup.
        w = self._consumer()
        peer = self._peer(
            layer_names=["l5", "l6"],
            block_lens=[self.TARGET_LEN] * 2 + [self.DRAFT_LEN] * 2,
        )

        out = w._build_fa_remote(self._plan(), peer, block_size_ratio=1)

        # 4 regions x 2 blocks, region-major.
        assert [ln for _, ln, _ in out] == [
            self.TARGET_LEN,
            self.TARGET_LEN,  # peer region 0 = our region 10 (layer 5, K)
            self.TARGET_LEN,
            self.TARGET_LEN,  # region 1 = our 11 (layer 5, V)
            self.DRAFT_LEN,
            self.DRAFT_LEN,  # region 2 = our 12 (draft, K)
            self.DRAFT_LEN,
            self.DRAFT_LEN,  # region 3 = our 13 (draft, V)
        ]

    def test_without_the_view_the_tail_stage_reads_the_wrong_lengths(self):
        # The translation suppressed -- what upstream does on its own. Kept so the
        # fix cannot regress into a no-op: the draft regions come back target-sized.
        w = self._consumer()
        peer = self._peer(
            layer_names=["l5", "l6"],
            block_lens=[self.TARGET_LEN] * 2 + [self.DRAFT_LEN] * 2,
        )

        with patch.object(
            RblnNixlPullConnectorWorker, "_peer_region_ids", return_value=None
        ):
            out = w._build_fa_remote(self._plan(), peer, block_size_ratio=1)

        assert [ln for _, ln, _ in out] == [self.TARGET_LEN] * 8

    def test_a_stage_starting_at_our_first_layer_needs_no_translation(self):
        # The first stage's positions already are our region ids, which is why a
        # pipelined consumer never hit this: its band always starts at 0.
        w = self._consumer()
        peer = self._peer(layer_names=["l0", "l1"], block_lens=[self.TARGET_LEN] * 4)

        assert w._peer_region_ids(peer) is None

    def test_a_peer_that_advertises_no_layers_is_left_positional(self):
        # Nothing to match on, so the positions stay upstream's -- which is what
        # every peer that does not publish its layers gets.
        w = self._consumer()
        peer = self._peer(layer_names=[], block_lens=[self.TARGET_LEN] * 4)

        assert w._peer_region_ids(peer) is None

    # Whether a region is REPLICATE is read through the view too, and it decides
    # how many remote ranks a block is gathered from. Showing that needs a plan
    # with more than one source rank and a non-zero offset: with one rank and no
    # offset the answer is inert, and a case built that way asserts nothing.
    SPLIT_PLAN_RANKS = (0, 1)
    OFFSET_FACTOR = 1

    @classmethod
    def _split_plan(cls):
        plan = MagicMock()
        plan.source_ranks_per_group = [cls.SPLIT_PLAN_RANKS]
        plan.rank_offset_factor = cls.OFFSET_FACTOR
        return plan

    def test_a_stage_holding_our_replicated_tail_reads_each_block_once(self):
        # Our last two layers are key-only REPLICATE, and the peer is the stage
        # that holds them: positions 0..3 are our regions 10..13. Read through the
        # view they are replicated, so each block is one whole read at offset 0.
        w = self._consumer(mla_tail=self.RPL * 2, uniform_lens=True)
        peer = self._peer(layer_names=["l5", "l6"], block_lens=[self.TARGET_LEN] * 4)

        out = w._build_fa_remote(self._split_plan(), peer, block_size_ratio=1)

        assert [ln for _, ln, _ in out] == [self.TARGET_LEN] * 8
        bases = [self.PEER_BASE * (i + 1) for i in range(4)]
        assert [addr for addr, _, _ in out] == [
            base + block * self.TARGET_LEN for base in bases for block in range(2)
        ]

    def test_without_the_view_a_replicated_tail_is_gathered_as_split(self):
        # The translation suppressed: positions 0..3 read our regions 0..3, which
        # are SPLIT, so every block comes back half-length at a rank offset --
        # bytes from the wrong half of a region that has no second half.
        w = self._consumer(mla_tail=self.RPL * 2, uniform_lens=True)
        peer = self._peer(layer_names=["l5", "l6"], block_lens=[self.TARGET_LEN] * 4)

        with patch.object(
            RblnNixlPullConnectorWorker, "_peer_region_ids", return_value=None
        ):
            out = w._build_fa_remote(self._split_plan(), peer, block_size_ratio=1)

        reads = len(self.SPLIT_PLAN_RANKS)
        assert [ln for _, ln, _ in out] == [self.TARGET_LEN // reads] * 8
        # And every address is shifted by the per-rank offset a replicated
        # region is read without -- asserted as the addresses themselves, since
        # the offset here is one region span and a modulo cannot see it.
        offset = self.OFFSET_FACTOR * self.TARGET_LEN
        bases = [self.PEER_BASE * (i + 1) for i in range(4)]
        assert [addr for addr, _, _ in out] == [
            base + offset + block * self.TARGET_LEN
            for base in bases
            for block in range(2)
        ]

    def test_a_peer_publishing_other_regions_than_we_own_is_refused(self):
        # A peer whose region count disagrees would pair by position and differ
        # in length.
        w = self._consumer()
        peer = self._peer(layer_names=["l5", "l6"], block_lens=[self.TARGET_LEN] * 3)

        with pytest.raises(RuntimeError, match="the peer publishes"):
            w._peer_region_ids(peer)


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

        from tests.native.distributed.kv_connector.utils import decode

        assert handle == 42
        assert len(blocks) == 16
        assert {ln for _, ln, _ in blocks} == {64}
        # Exactly the shard's own four regions, every block of each, one piece:
        # register the whole model's regions instead and regions 0..3 appear.
        assert Counter(
            decode(
                blocks,
                bases=w.kv_caches_base_addr["eng"][0],
                block_lens=w.block_len_per_layer,
                num_blocks=w.num_blocks,
            )
        ) == Counter(
            {(region, block, 0): 1 for region in (4, 5, 6, 7) for block in range(4)}
        )

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
        w.vllm_config.speculative_config = None
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
    def _worker(*, tp_rank, tp_size, areas, slices, n_logical, block_len, kv_heads=8):
        # `block_len` and `kv_heads` take a list to give each logical region its
        # own geometry; a scalar applies to every region.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.tp_rank = tp_rank
        w._kv_areas = areas
        w._kv_slices = slices
        block_lens = (
            list(block_len) if isinstance(block_len, list) else [block_len] * n_logical
        )
        w.block_len_per_layer = [ln for ln in block_lens for _ in range(areas)]
        w._logical_region_kv_heads = (
            list(kv_heads) if isinstance(kv_heads, list) else [kv_heads] * n_logical
        )
        topo = MagicMock()
        topo.tp_size = tp_size
        topo.total_num_kv_heads = w._logical_region_kv_heads[0]
        w.transfer_topo = topo
        w.get_backend_aware_kv_block_len = lambda layer_idx, **_: w.block_len_per_layer[
            layer_idx
        ]
        return w

    @staticmethod
    def _meta(*, areas, slices, n_logical, block_len, num_blocks=2):
        n = n_logical * areas
        block_lens = (
            [ln for ln in block_len for _ in range(areas)]
            if isinstance(block_len, list)
            else [block_len] * n
        )
        # Region i starts at stride * (i + 1), and the stride is exactly one
        # region's span -- so the regions abut and an address always says which
        # one it is in, which is what _decoded reads it for. A hand-picked base
        # does not: 1000 with a 512-byte region over 2 blocks overlaps.
        stride = max(block_lens) * num_blocks
        return _agent_meta(
            kv_areas=areas,
            kv_slices=slices,
            num_blocks=num_blocks,
            device_id=0,
            kv_caches_base_addr=[stride * (i + 1) for i in range(n)],
            block_lens=block_lens,
        )

    @staticmethod
    def _decoded(out, meta):
        """``out`` as (peer region, block, piece), counted.

        A descriptor is asserted by what it means, not by the order the loops
        emit it in; TestDescriptorOrderContract owns the one ordering that does
        carry meaning here. `piece` is the index inside the peer's block in
        units of the descriptor's own length, which is the head band it names.
        """
        from tests.native.distributed.kv_connector.utils import decode

        return Counter(
            decode(
                out,
                bases=meta.kv_caches_base_addr,
                block_lens=meta.block_lens,
                num_blocks=meta.num_blocks,
            )
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

    def test_a_peer_advertising_no_slices_is_refused_on_arrival(self):
        # The per-region refusal that names the side runs after the divisions,
        # so without this the operator gets a ZeroDivisionError instead.
        w = _make_worker()
        with pytest.raises(RuntimeError, match="advertises 0 logical slice"):
            _handshake(w, _FakeSock(pp_size=1, kv_slices=0))

    def test_offset_into_coarser_remote_area(self):
        """P TP1 -> D TP4: the peer's area holds 2 heads, we want one of them,
        so half the descriptors start halfway into the remote area."""
        w = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=512)
        out = w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=1)
        # Our areas are [h0, h0, h1, h1] and the peer holds both heads in its
        # area 0, so every descriptor reads region 0: h0 from its first piece
        # and h1 from its second, each twice because two of our areas replicate
        # it, at both blocks.
        assert {ln for _, ln, _ in out} == {256}
        assert self._decoded(out, meta) == Counter(
            {(0, 0, 0): 2, (0, 1, 0): 2, (0, 0, 1): 2, (0, 1, 1): 2}
        )

    def test_area_index_permutation_zero_offset(self):
        """P TP2 -> D TP4: head widths match so the offset is 0, but local area
        2 carries head 1, which is the peer's area 1 — not its area 2."""
        w = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=256)
        out = w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=2)
        decoded = self._decoded(out, meta)
        # The head widths match, so no piece is split -- what the mapping has
        # to get right is WHICH peer region. Our area 2 carries head 1, which
        # is the peer's area 1; pairing by position would read its area 2.
        assert {region for region, _, _ in decoded} == {0, 1}
        assert decoded == Counter(
            {(0, 0, 0): 2, (0, 1, 0): 2, (1, 0, 0): 2, (1, 1, 0): 2}
        )

    def test_second_rank_reads_its_own_head_band(self):
        """D TP4 rank 2 owns heads 4,5; against a TP2 peer those live on the
        peer's rank 1, whose local slice numbering restarts at head 4."""
        w = self._worker(
            tp_rank=2, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=256)
        out = w._build_head_matched_remote(meta, remote_tp_rank=1, remote_tp_size=2)
        # Heads 4 and 5 are that peer rank's own first two, so they land on
        # the regions it advertises first -- not four regions further along.
        assert self._decoded(out, meta) == Counter(
            {(0, 0, 0): 2, (0, 1, 0): 2, (1, 0, 0): 2, (1, 1, 0): 2}
        )

    def test_multiple_logical_regions_stay_layer_major(self):
        """K and V of the same layer are separate logical regions; the mapping
        must stay inside each one."""
        w = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=2, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=2, block_len=256, num_blocks=1)
        out = w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=2)
        # Logical region 0 owns the peer's regions 0..3 and logical region 1
        # its 4..7, so no descriptor may cross from one band into the other.
        decoded = self._decoded(out, meta)
        assert {region for region, _, _ in decoded} == {0, 1, 4, 5}
        assert decoded == Counter(
            {(0, 0, 0): 2, (1, 0, 0): 2, (4, 0, 0): 2, (5, 0, 0): 2}
        )

    def test_peer_with_narrower_slice_splits_each_region(self):
        """P TP2 -> D TP1: our area holds 2 heads, each of the peer's holds 1,
        so every region is read in two half-length pieces from two different
        remote regions -- block-major, piece-minor."""
        w = self._worker(
            tp_rank=0, tp_size=1, areas=4, slices=4, n_logical=1, block_len=512
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=256, num_blocks=2)
        out = w._build_head_matched_remote(
            meta, remote_tp_rank=0, remote_tp_size=2, peer_areas=[0, 1]
        )
        # 2 areas x 2 blocks x 2 pieces, every piece half of our 512B region.
        assert len(out) == 8
        assert {ln for _, ln, _ in out} == {256}
        # Our area 0 holds heads {0,1} and area 1 holds {2,3}; each of the
        # peer's regions holds one head, so all four are read at both blocks
        # and nothing is replicated.
        assert self._decoded(out, meta) == Counter(
            {(region, block, 0): 1 for region in range(4) for block in range(2)}
        )

    def test_a_draft_region_is_banded_by_its_own_head_count(self):
        # The target's 8 heads cut cleanly over TP2 x 4 areas while the draft's 4
        # do not, so the draft's region has no band and the divisibility guard is
        # what has to say so. A model-config count takes the target's for both
        # and computes one anyway.
        w = self._worker(
            tp_rank=0,
            tp_size=2,
            areas=4,
            slices=4,
            n_logical=2,
            block_len=[256, 128],
            kv_heads=[8, 4],
        )
        meta = self._meta(
            areas=4, slices=4, n_logical=2, block_len=[512, 256], num_blocks=2
        )
        with pytest.raises(RuntimeError, match="cut into 4 logical slice"):
            w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=1)

    def test_a_replicating_peer_is_read_past_its_replica_areas(self):
        """P TP4 -> D TP1: the peer's rank holds 2 of the 8 heads and repeats
        each over 2 of its 4 areas, so its second head begins at its area 2.
        Pairing by slice index would read the replica of the first one."""
        w = self._worker(
            tp_rank=0, tp_size=1, areas=4, slices=4, n_logical=1, block_len=512
        )
        meta = self._meta(areas=4, slices=2, n_logical=1, block_len=256, num_blocks=2)
        out = w._build_head_matched_remote(
            meta, remote_tp_rank=0, remote_tp_size=4, peer_areas=[0]
        )
        # Our area 0 = heads {0,1}; the peer lays them out [h0, h0, h1, h1], so
        # the two pieces come from its regions 0 and 2. Remote page is 256B.
        assert {ln for _, ln, _ in out} == {256}
        decoded = self._decoded(out, meta)
        # The peer lays our heads {0,1} out as [h0, h0, h1, h1], so its head 1
        # begins at its region 2. Pairing by slice index would read region 1,
        # which is a replica of head 0.
        assert {region for region, _, _ in decoded} == {0, 2}
        assert decoded == Counter(
            {(0, 0, 0): 1, (2, 0, 0): 1, (0, 1, 0): 1, (2, 1, 0): 1}
        )

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
        meta = self._meta(areas=4, slices=4, n_logical=6, block_len=256, num_blocks=1)
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
        # The layer offset puts our logical region 0 at the peer's region index
        # 2 (its areas 8..11) and logical region 1 at index 3 (areas 12..15).
        # Both axes have to hold at once: without the offset the regions start
        # at 0, and without the area filter areas 2 and 3 appear as well.
        assert self._decoded(out, meta) == Counter(
            {(region, 0, 0): 1 for region in range(8, 16)}
        )

    def test_layer_offset_alone_keeps_every_area(self):
        """Same layer offset with matching TP: no area filter, no piece split."""
        w = self._worker(
            tp_rank=0, tp_size=1, areas=4, slices=4, n_logical=1, block_len=256
        )
        w.local_seen_layer_names = ["layer.2"]
        w.num_regions = 1 * 4  # logical regions x areas
        meta = self._meta(areas=4, slices=4, n_logical=3, block_len=256, num_blocks=1)
        out = w._build_head_matched_remote(
            meta,
            remote_tp_rank=0,
            remote_tp_size=1,
            registered_layer_names=("layer.1", "layer.2", "layer.3"),
        )
        # Our single logical region is the peer's index 1, so every one of its
        # four areas is read and none of index 0's or 2's is.
        assert self._decoded(out, meta) == Counter(
            {(region, 0, 0): 1 for region in range(4, 8)}
        )

    def test_incommensurate_slices_raise(self):
        # The peer cuts its heads 3 ways against our 2, so its slice is not a
        # whole fraction of ours; refuse rather than transfer a partial head.
        with pytest.raises(RuntimeError, match="does not divide it"):
            RblnNixlPullConnectorWorker._head_split(2, 3)

    def test_head_split_is_one_unless_we_are_coarser(self):
        # More cuts means a finer slice; our area splits only when the peer is finer.
        f = RblnNixlPullConnectorWorker._head_split
        assert f(1, 1) == 1  # equal granularity
        assert f(2, 1) == 1  # peer coarser -> offset, not split
        assert f(1, 2) == 2  # peer finer -> two pieces
        assert f(1, 4) == 4

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


class TestDescriptorOrderContract:
    """The two descriptor lists are paired by index, so their orders must agree.

    Everything else here is asserted as a set, on purpose: what a transfer
    means is which local piece is read into which peer piece, not the order the
    loops emit them in. That leaves one thing unasserted -- a transfer pairs the
    two desc-id lists position by position, and an id is a position in its
    dlist, so reordering ONE dlist moves the right bytes to the wrong place.
    This class owns that contract. Where a group sits inside one list is a
    separate one, since desc ids offset into it, and is pinned where that group
    is emitted.
    """

    AREAS = 4

    @staticmethod
    def _worker(n_logical=1):
        # Each layer over 4 chiplet areas, all 8 heads, TP1: our area holds 2
        # heads. Wired for the local dlist as well as the remote one, since the
        # contract here is that the two are emitted in the same order.
        areas = TestDescriptorOrderContract.AREAS
        w = TestHeadBandMatching._worker(
            tp_rank=0,
            tp_size=1,
            areas=areas,
            slices=areas,
            n_logical=n_logical,
            block_len=512,
        )
        w.engine_id = "eng"
        w.block_size = 16
        w.num_blocks = 2
        w.device_id = 0
        w._has_mamba = False
        w.nixl_memory_type = "DRAM"
        w.local_seen_layer_names = [f"l{i}" for i in range(n_logical)]
        w.num_regions = n_logical * areas
        w.kv_caches_base_addr = {
            "eng": {0: [0x10000 * (i + 1) for i in range(w.num_regions)]}
        }
        w.transfer_topo.is_kv_layout_blocks_first = False
        w.nixl_wrapper = MagicMock()
        w._shard_descs_per_block = {}
        w._borrowed_src_handles = set()
        return w

    def test_local_and_remote_descriptors_pair_up_by_index(self):
        from tests.native.distributed.kv_connector.utils import decode

        w = self._worker()
        # A peer at TP2 cuts its heads one per area, so each of our two-head
        # areas is read in two pieces from two of its regions.
        meta = TestHeadBandMatching._meta(
            areas=4, slices=4, n_logical=1, block_len=256, num_blocks=2
        )
        remote = w._build_head_matched_remote(
            meta, remote_tp_rank=0, remote_tp_size=2, peer_areas=[0, 1]
        )
        _, local = w._register_shard_local_xfer_handler(
            w.block_size, ("l0",), peer_areas=[0, 1], split=2, replica_fanout=1
        )
        assert len(local) == len(remote) == 8

        pairs = list(
            zip(
                decode(
                    local,
                    bases=w.kv_caches_base_addr["eng"][0],
                    block_lens=w.block_len_per_layer,
                    num_blocks=w.num_blocks,
                ),
                decode(
                    remote,
                    bases=meta.kv_caches_base_addr,
                    block_lens=meta.block_lens,
                    num_blocks=meta.num_blocks,
                ),
            )
        )
        # Our area 0 holds heads 0,1 and area 1 holds heads 2,3; the peer holds
        # one head per region, so head h is its region h. Read as
        # (local region, block, piece) -> (peer region, block, piece).
        assert set(pairs) == {
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 1), (1, 0, 0)),
            ((0, 1, 0), (0, 1, 0)),
            ((0, 1, 1), (1, 1, 0)),
            ((1, 0, 0), (2, 0, 0)),
            ((1, 0, 1), (3, 0, 0)),
            ((1, 1, 0), (2, 1, 0)),
            ((1, 1, 1), (3, 1, 0)),
        }
        # And the pairing is a bijection: no local piece feeds two peer pieces.
        assert len(set(pairs)) == len(pairs)

    def test_two_layers_pair_within_their_own_layer(self):
        # Two layers, so the layer axis can be reordered at all: reversing
        # either side's layer loop then sends layer 0's pieces at layer 1's
        # regions and the pairing changes.
        from tests.native.distributed.kv_connector.utils import decode

        w = self._worker(n_logical=2)
        meta = TestHeadBandMatching._meta(
            areas=self.AREAS,
            slices=self.AREAS,
            n_logical=2,
            block_len=256,
            num_blocks=2,
        )
        remote = w._build_head_matched_remote(
            meta, remote_tp_rank=0, remote_tp_size=2, peer_areas=[0, 1]
        )
        _, local = w._register_shard_local_xfer_handler(
            w.block_size, ("l0", "l1"), peer_areas=[0, 1], split=2, replica_fanout=1
        )
        assert len(local) == len(remote) == 16

        pairs = list(
            zip(
                decode(
                    local,
                    bases=w.kv_caches_base_addr["eng"][0],
                    block_lens=w.block_len_per_layer,
                    num_blocks=w.num_blocks,
                ),
                decode(
                    remote,
                    bases=meta.kv_caches_base_addr,
                    block_lens=meta.block_lens,
                    num_blocks=meta.num_blocks,
                ),
            )
        )
        # Layer L owns our regions 4L..4L+3 and the peer's 4L..4L+3. Our area 0
        # holds heads 0,1 and area 1 heads 2,3; the peer holds one head per
        # region, so our (area a, piece j) reads its region 4L + 2a + j.
        assert set(pairs) == {
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 1), (1, 0, 0)),
            ((0, 1, 0), (0, 1, 0)),
            ((0, 1, 1), (1, 1, 0)),
            ((1, 0, 0), (2, 0, 0)),
            ((1, 0, 1), (3, 0, 0)),
            ((1, 1, 0), (2, 1, 0)),
            ((1, 1, 1), (3, 1, 0)),
            ((4, 0, 0), (4, 0, 0)),
            ((4, 0, 1), (5, 0, 0)),
            ((4, 1, 0), (4, 1, 0)),
            ((4, 1, 1), (5, 1, 0)),
            ((5, 0, 0), (6, 0, 0)),
            ((5, 0, 1), (7, 0, 0)),
            ((5, 1, 0), (6, 1, 0)),
            ((5, 1, 1), (7, 1, 0)),
        }
        assert len(set(pairs)) == len(pairs)


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
    def _worker(*, local_len=128, block_size_ratio=1, layout="NHD", kv_heads=8):
        # `local_len` and `kv_heads` take a list for one entry per logical region;
        # the transfer table repeats each across the 4 chiplet areas.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.tp_rank = 1
        w.kv_cache_layout = layout
        lens = list(local_len) if isinstance(local_len, list) else [local_len]
        # 8 heads over TP4 is 2 per rank, cut into 2 slices -> 1 head per area.
        w._kv_areas, w._kv_slices = 4, 2
        w.block_len_per_layer = [ln for ln in lens for _ in range(w._kv_areas)]
        w._logical_region_kv_heads = (
            list(kv_heads) if isinstance(kv_heads, list) else [kv_heads] * len(lens)
        )
        topo = MagicMock()
        topo.tp_size = 4
        topo.total_num_kv_heads = w._logical_region_kv_heads[0]
        topo.block_size_ratio.return_value = block_size_ratio
        w.transfer_topo = topo
        return w

    @staticmethod
    def _meta(*, remote_len, layout="NHD"):
        # A TP1 peer keeps all 8 heads, cut into 4 slices -> 2 heads per area,
        # so its per-area block length must be twice ours for a head to cost the
        # same on both sides.
        lens = list(remote_len) if isinstance(remote_len, list) else [remote_len]
        return _agent_meta(
            block_size=16,
            kv_cache_layout=layout,
            kv_areas=4,
            kv_slices=4,
            block_lens=[ln for ln in lens for _ in range(4)],
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

    def test_a_draft_region_with_the_wrong_bytes_per_head_is_rejected(self):
        # Region 0 is consistent (128 x 2 == 256 x 1) while the peer's draft
        # region is 2048B where a head costing the same demands 1024B -- a peer
        # compiled against a different draft. The descriptors land inside the
        # wrong bytes without failing.
        w = self._worker(local_len=[128, 512], kv_heads=[8, 32])
        with pytest.raises(RuntimeError, match="logical region 1"):
            w._validate_head_matched_handshake(
                self._meta(remote_len=[256, 2048]), remote_tp_size=1
            )

    def test_a_consistent_heterogeneous_pair_passes(self):
        # The same two regions with the draft's peer width corrected: a head costs
        # 128B per block on both sides in both regions.
        w = self._worker(local_len=[128, 512], kv_heads=[8, 32])
        w._validate_head_matched_handshake(
            self._meta(remote_len=[256, 1024]), remote_tp_size=1
        )  # no raise

    def test_a_draft_region_that_cannot_be_banded_is_refused(self):
        # Our shard cuts the draft's 4 heads into 2 slices, which does not divide,
        # while the target's 8 do -- and a model-config count reports the
        # target's for both. The byte invariant cannot say so.
        w = self._worker(local_len=[128, 64], kv_heads=[8, 4])

        with pytest.raises(RuntimeError, match="cut into 2 logical slice"):
            w._validate_head_matched_handshake(
                self._meta(remote_len=[256, 128]), remote_tp_size=1
            )

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
        w._logical_region_slices = [2, 2]
        w._logical_region_kv_heads = [8, 8]
        w.transfer_topo = MagicMock()
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.add_remote_agent.return_value = "agent"
        meta = MagicMock()
        meta.engine_id = "eng"

        with (
            patch.object(
                RblnNixlPullConnectorWorker, "_register_remote_engine_prelude"
            ) as prelude,
            patch.object(
                RblnNixlPullConnectorWorker, "_validate_remote_agent_handshake"
            ) as validate,
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
        # This path returns before super(), so registering the peer engine and
        # running the geometry guards are its own. Both are stubbed above, which
        # makes these two assertions the only hold on those call sites.
        assert prelude.call_args.args == (meta, 2)
        assert validate.call_args.args == (meta, 2)

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

    @staticmethod
    def _slice_worker(*, slices, kv_heads):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._remote_agents = {}
        w.nixl_wrapper = MagicMock()
        w._logical_region_slices = list(slices)
        w._logical_region_kv_heads = list(kv_heads)
        return w

    def test_uneven_region_slice_counts_are_refused_before_an_agent_is_added(self):
        # Silent without the refusal: the smaller count divides the larger side's
        # heads per rank, so a wrong band transfers with nothing objecting.
        w = self._slice_worker(slices=[4, 4, 2, 2], kv_heads=[8, 8, 32, 32])
        meta = MagicMock()
        meta.engine_id = "eng"

        with pytest.raises(RuntimeError, match="different numbers of chiplet"):
            w._add_remote_agent_head_matched(meta, 1, 2)

        # Refused at the entry, so no peer state was taken on.
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    def test_the_loud_direction_is_refused_by_the_same_check(self):
        # The mirror case already fails, but inside `_slice_head_bounds` as
        # "owns 2 KV heads cut into 4 logical slice(s)" -- which names the
        # arithmetic rather than the cause. One condition covers both directions.
        w = self._slice_worker(slices=[2, 2, 4, 4], kv_heads=[8, 8, 32, 32])

        with pytest.raises(RuntimeError, match=r"chiplet slices \[2, 4\]"):
            w._reject_uneven_region_slices(2)

    def test_uniform_slice_counts_are_not_refused(self):
        # A differing KV geometry is supported; only a differing chiplet CUT is
        # not. Refusing this would close the case the head bands exist for.
        w = self._slice_worker(slices=[4, 4, 4, 4], kv_heads=[8, 8, 32, 32])

        w._reject_uneven_region_slices(2)

    def test_a_region_without_a_head_axis_is_not_counted(self):
        # An SSM state is not cut by KV heads, so its slice count says nothing
        # about the head bands -- and the refusal that does name Mamba sits
        # further in, past this one.
        w = self._slice_worker(slices=[4, 4, 1, 1], kv_heads=[8, 8, None, None])

        w._reject_uneven_region_slices(2)

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


def _remote_agent_meta():
    meta = MagicMock()
    meta.engine_id = "remote-eng"
    meta.block_size = 64
    meta.block_lens = [256, 256]
    meta.physical_blocks_per_logical_kv_block = 1
    meta.num_blocks = 8
    meta.kv_caches_base_addr = [0x5000, 0x6000]
    meta.device_id = 1
    meta.agent_metadata = b"x"
    return meta


class TestAddRemoteAgentSwa:
    # The remote engine must be registered and its TPMapping built before any
    # topology lookup, or get_engine_info() KeyErrors.
    def test_registers_remote_engine_before_topology_lookups(self, monkeypatch):
        worker = build_worker(monkeypatch, num_blocks=4, block_size=64)
        worker._sw_ratio = 2
        worker._has_mamba = False
        worker.use_mla = False
        worker.tp_rank = 0
        worker._group_spec_types = ()
        worker.nixl_memory_type = "DRAM"

        topo = MagicMock(is_kv_layout_blocks_first=False)
        topo.block_size_ratio.return_value = 1
        topo.tp_ratio.return_value = 1
        topo.is_kv_replicated.return_value = True
        worker.transfer_topo = topo

        worker.tp_mappings = {}
        worker.dst_num_blocks = {}
        worker._remote_agents = {}
        worker.kv_caches_base_addr = collections.defaultdict(dict)
        worker.dst_xfer_side_handles = collections.defaultdict(dict)
        worker.src_xfer_handles_by_block_size = {}
        worker.src_blocks_data = []
        worker.nixl_wrapper = MagicMock()
        worker.nixl_wrapper.add_remote_agent.return_value = "remote-agent-name"

        meta = _remote_agent_meta()
        mapping_sentinel = MagicMock(name="tp_mapping")
        with (
            patched_in_package(
                "compute_tp_mapping", MagicMock(return_value=mapping_sentinel)
            ) as ctm,
            patch.object(worker, "_validate_remote_agent_handshake") as validate,
            patch.object(worker, "get_backend_aware_kv_block_len", return_value=256),
        ):
            out = worker.add_remote_agent(meta, 0, 1)

        # add_remote_agent has to reach _validate_remote_agent_handshake with the
        # peer's own metadata; it is stubbed here, so this is what says so.
        assert validate.call_args.args == (meta, 1)

        # Prelude: remote engine registered with an EngineTransferInfo from meta.
        topo.register_remote_engine.assert_called_once()
        eng_id_arg, eti = topo.register_remote_engine.call_args[0]
        assert eng_id_arg == "remote-eng"
        assert isinstance(eti, EngineTransferInfo)
        assert eti.remote_tp_size == 1
        assert eti.remote_block_size == 64
        assert eti.remote_block_len == 256
        assert eti.remote_physical_blocks_per_logical == 1

        # TPMapping built from the topology and stashed under the engine id.
        ctm.assert_called_once_with(
            transfer_topology=topo, remote_tp_size=1, group_spec_types=()
        )
        assert worker.tp_mappings["remote-eng"] is mapping_sentinel

        # The topology methods are consulted for the desc math.
        topo.block_size_ratio.assert_called_once_with(64)
        topo.tp_ratio.assert_called_once_with(1)
        topo.is_kv_replicated.assert_called_once_with("remote-eng")

        # ORDERING GUARD: register_remote_engine must precede any topology lookup;
        # flipping the order would (with a real topo) KeyError in get_engine_info.
        names = [c[0] for c in topo.mock_calls if c[0]]
        assert names.index("register_remote_engine") < names.index("block_size_ratio")

        assert out == "remote-agent-name"

    def test_a_smaller_remote_block_shortens_descs_and_adds_a_local_handle(
        self, monkeypatch
    ):
        # A peer with a smaller block holds less per block than we do, so a desc
        # can only span the peer's block length -- on both sides, which is why a
        # second local handler keyed by the peer's block size is registered.
        worker = build_worker(monkeypatch, num_blocks=4, block_size=64)
        worker._sw_ratio = 2
        worker._has_mamba = False
        worker.use_mla = False
        worker.tp_rank = 0
        worker._group_spec_types = ()
        worker.nixl_memory_type = "DRAM"

        topo = MagicMock(is_kv_layout_blocks_first=False)
        topo.block_size_ratio.return_value = 2
        topo.tp_ratio.return_value = 1
        topo.is_kv_replicated.return_value = True
        worker.transfer_topo = topo

        worker.tp_mappings = {}
        worker.dst_num_blocks = {}
        worker._remote_agents = {}
        worker.kv_caches_base_addr = collections.defaultdict(dict)
        worker.dst_xfer_side_handles = collections.defaultdict(dict)
        worker.src_xfer_handles_by_block_size = {}
        worker.src_blocks_data = []
        worker.nixl_wrapper = MagicMock()
        worker.nixl_wrapper.add_remote_agent.return_value = "remote-agent-name"

        meta = _remote_agent_meta()
        meta.block_size = 32  # half of ours, hence the ratio of 2

        with (
            patched_in_package(
                "compute_tp_mapping", MagicMock(return_value=MagicMock())
            ),
            patch.object(worker, "_validate_remote_agent_handshake"),
            patch.object(worker, "get_backend_aware_kv_block_len", return_value=256),
            patch.object(
                worker,
                "register_local_xfer_handler",
                return_value=("peer-sized-handle", []),
            ) as local,
        ):
            worker.add_remote_agent(meta, 0, 1)

        # Full pass then SWA pass: the peer's 128B block, then half of it.
        blocks_data = worker.nixl_wrapper.get_xfer_descs.call_args.args[0]
        assert {desc_len for _, desc_len, _ in blocks_data} == {128, 64}
        # Keyed by the PEER's block size, not ours: the read path picks the
        # handler by what the peer advertised.
        assert worker.src_xfer_handles_by_block_size == {32: "peer-sized-handle"}
        assert local.call_args.args == (32,)
