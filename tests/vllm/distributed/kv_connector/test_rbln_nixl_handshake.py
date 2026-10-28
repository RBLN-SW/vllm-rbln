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
# cut on that axis) -- plus what else the handshake establishes: upstream's
# calling convention and the clock offset it estimates. The side channel is a
# real ZMQ pair, so the compat-hash and engine-id gates are behaviour here;
# nixl-rbln is the only stand-in.

import collections
import inspect
import itertools
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from types import SimpleNamespace
from typing import NamedTuple
from unittest.mock import MagicMock, call, patch

import msgspec
import numpy as np
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

from tests.vllm.distributed.kv_connector.utils import (
    area_slices,
    build_worker,
    fill_region_geometry,
    mock_vllm_config,
    patched_in_package,
    set_shape,
    shape,
    uniform_grid,
    window_mode,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    ChipletSplitAxis,
    RblnNixlAgentMetadata,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_worker import (
    RblnNixlPushConnectorWorker,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
    AreaPlace,
    AreaRect,
    ChunkGrid,
    Piece,
    RblnNixlWorkerState,
    kv_chunk_tokens,
    pair_area,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.transfer import (
    _chunk_desc_ids,
)


def _encode_payload(
    pp_rank,
    pp_size,
    *,
    engine_id="eng",
    compat="HASH",
    layers_per_stage=1,
    layer_names=None,
    kv_area_slices=None,
    kv_split_axes=None,
    regions_per_layer=1,
    region_mem_types=None,
):
    if layer_names is None:
        layer_names = [
            f"layer.{pp_rank * layers_per_stage + j}" for j in range(layers_per_stage)
        ]
    n_regions = len(layer_names) * regions_per_layer
    meta = RblnNixlAgentMetadata(
        engine_id=engine_id,
        agent_metadata=b"agent",
        kv_caches_base_addr=[0x1000 * (i + 1) for i in range(n_regions)],
        device_id=0,
        num_blocks=4,
        block_lens=[8192] * n_regions,
        block_strides=[8192] * n_regions,
        kv_cache_layout="LBHNC",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="RBLN",
        physical_blocks_per_logical_kv_block=1,
        pp_rank=pp_rank,
        pp_size=pp_size,
        region_names=list(layer_names),
        region_mem_types=region_mem_types,
        kv_area_slices=(
            [[0]] * n_regions if kv_area_slices is None else kv_area_slices
        ),
        kv_split_axes=(
            [ChipletSplitAxis.HEAD] * n_regions
            if kv_split_axes is None
            else kv_split_axes
        ),
    )
    payload = NixlHandshakePayload(
        compatibility_hash=compat,
        agent_metadata_bytes=msgspec.msgpack.Encoder().encode(meta),
    )
    return msgspec.msgpack.Encoder().encode(payload)


def _agent_meta(**overrides):
    """A minimal RblnNixlAgentMetadata for the region-slicing cases.

    `kv_areas` and `kv_slices` describe one geometry for every logical region,
    published as `kv_area_slices` -- one region per axis named, else per
    `kv_areas` block lengths."""
    # RBLN registers a region's blocks contiguously, so a caller that gives
    # lengths and no strides means the two are the same list.
    block_lens = list(overrides.pop("block_lens", [8192]))
    areas = overrides.pop("kv_areas", 1)
    slices = overrides.pop("kv_slices", areas)
    regions = len(overrides.get("kv_split_axes", ())) or max(
        1, len(block_lens) // areas
    )
    overrides.setdefault("kv_area_slices", area_slices(areas, slices, regions))
    overrides.setdefault(
        "kv_split_axes", [ChipletSplitAxis.HEAD] * len(overrides["kv_area_slices"])
    )
    fields = dict(
        engine_id="eng",
        agent_metadata=b"agent",
        kv_caches_base_addr=[0x1000],
        device_id=0,
        num_blocks=4,
        block_lens=block_lens,
        block_strides=list(block_lens),
        kv_cache_layout="LBHNC",
        block_size=16,
        ssm_sizes=(0, 0),
        attn_backend_name="RBLN",
        physical_blocks_per_logical_kv_block=1,
        region_names=[f"layer.{i}" for i in range(4)],
    )
    fields.update(overrides)
    return RblnNixlAgentMetadata(**fields)


class _FakeClock:
    """Scripted ``perf_counter``: the handshake reads it twice per query."""

    def __init__(self, ticks):
        self._ticks = list(ticks)

    def perf_counter(self):
        return self._ticks.pop(0)


class _FakeSock:
    # Stands in for the listener at nixl/base_scheduler.py. An unpublished pair
    # takes the real one down -- it indexes its metadata dict with no guard --
    # and the caller then sees only a timeout; this raises in the caller instead,
    # so the bad pair surfaces where it was asked for.

    def __init__(
        self,
        pp_size,
        *,
        engine_id="eng",
        compat="HASH",
        layers_per_stage=1,
        stage_layers=None,
        tp_size=1,
        peer_stamps=None,
        kv_area_slices=None,
        kv_split_axes=None,
        regions_per_layer=1,
        region_mem_types=None,
    ):
        self.pp_size = pp_size
        self.tp_size = tp_size
        # What the peer echoes from ITS process clock, in query order. A real
        # peer's origin is unrelated to ours; that unknown is what the midpoint
        # and the lowest-RTT rule exist to absorb.
        self.peer_stamps = list(peer_stamps) if peer_stamps is not None else None
        self.engine_id = engine_id
        self.compat = compat
        self.layers_per_stage = layers_per_stage
        self.regions_per_layer = regions_per_layer
        # Optional per-stage layer-name lists (for uneven splits); indexed by
        # pp_rank, since layer ownership is a function of the stage alone.
        self.stage_layers = stage_layers
        self.region_mem_types = region_mem_types
        self.kv_area_slices = kv_area_slices
        self.kv_split_axes = kv_split_axes
        self.queried = []
        self._last = None

    def setsockopt(self, *a):
        pass

    def send(self, msg):
        _, pp_rank, tp_rank = msgspec.msgpack.decode(msg)
        if not (0 <= pp_rank < self.pp_size and 0 <= tp_rank < self.tp_size):
            raise KeyError((pp_rank, tp_rank))
        self.queried.append(pp_rank * self.tp_size + tp_rank)
        self._last = pp_rank

    def recv_multipart(self):
        payload = _encode_payload(
            self._last,
            self.pp_size,
            engine_id=self.engine_id,
            compat=self.compat,
            layers_per_stage=self.layers_per_stage,
            layer_names=(
                self.stage_layers[self._last] if self.stage_layers is not None else None
            ),
            kv_area_slices=self.kv_area_slices,
            kv_split_axes=self.kv_split_axes,
            regions_per_layer=self.regions_per_layer,
            region_mem_types=self.region_mem_types,
        )
        stamp = (
            self.peer_stamps.pop(0)
            if self.peer_stamps is not None
            else time.perf_counter()
        )
        return [payload, msgspec.msgpack.encode(stamp)]


def _stub_plan(pieces: int, region_ids=(0,)):
    """A `_head_matched_plan` answer of `pieces` pieces a block per region,
    for a test about what the handshake does with the count."""
    piece = Piece(0, 0, 0, 1, 1)
    return list(region_ids), [[piece] * pieces for _ in region_ids]


def _split_plan(region_ids, block_len, split=1, copies=1, rows=1):
    """A plan cutting each region's block into `split` equal pieces, each sent
    to `copies` peer copies, in `pair_area`'s order -- for tests about what
    the lists and the indices do with a plan rather than how one is derived."""
    sub = block_len // split
    return list(region_ids), [
        [Piece(j * sub, 0, 0, sub, rows) for j in range(split) for _ in range(copies)]
        for _ in region_ids
    ]


def _make_worker(
    *,
    tp_target_ranks=(0,),
    sw_ratio=None,
    compat="HASH",
    tp_ratio=1,
    host_buffer=True,
    has_swa=None,
    cls=RblnNixlPullConnectorWorker,
):
    w = object.__new__(cls)
    # Bypassing __init__ skips the layout probe, and a split cache and host
    # staging both register one K/V per block.
    w._kv_per_block = 1
    # Host staging closes the D2D-only paths -- head matching and fan-in -- so
    # it is the default and a D2D shape has to ask for the device buffer.
    w.use_host_buffer = host_buffer
    w.kv_buffer_device = "cpu" if host_buffer else "rbln"
    w.nixl_memory_type = "DRAM" if host_buffer else "VRAM"
    w.device_id = 0
    w.transfer_topo = MagicMock()
    w.transfer_topo.handshake_target_ranks.return_value = list(tp_target_ranks)
    # Equal P/D TP unless a test says otherwise: the handshake now consults
    # tp_ratio to decide between positional and head-band region pairing.
    w.transfer_topo.tp_ratio.return_value = tp_ratio
    w.transfer_topo.tp_size = 1
    w.vllm_config = mock_vllm_config()
    w.vllm_config.parallel_config.pipeline_parallel_size = 1
    # No speculative decoding: the compat hash then folds what it always did.
    w.vllm_config.speculative_config = None
    w.compat_hash = compat
    w.enforce_compat_hash = True
    w.use_mla = False
    window_mode(
        w, sw_ratio, has_swa=(sw_ratio is not None) if has_swa is None else has_swa
    )
    set_shape(w, chunk_mode=False)
    w._remote_shard_layer_names = defaultdict(dict)
    w._remote_pp_size = {}
    w._overlapping_ranks = defaultdict(list)
    # Full-model consumer: owns every producer stage's layer, so every stage
    # overlaps (the fan-out default). _FakeSock advertises stage i as "layer.i".
    w.local_seen_layer_names = ["layer.0", "layer.1", "layer.2"]
    w.add_remote_agent = MagicMock(
        # Strict arity: a caller that stops forwarding the peer's dcp_size
        # must fail here, not be absorbed by a default.
        side_effect=lambda meta, rank, tps, dcp: f"agent-{rank}"
    )
    # Stubbed so the fan-out tests stay on stage enumeration; the body runs for
    # real in TestShardLocalRegions::test_register_shard_xfer_state_keys_the_stage.
    w._register_shard_xfer_state = MagicMock()
    w._logical_region_kv_heads = [8] * 3
    return fill_region_geometry(w, areas=1, axes=[ChipletSplitAxis.HEAD] * 3)


@contextmanager
def _patched_socket(sock, clock=None):
    @contextmanager
    def fake_zmq_ctx(_type, _path):
        yield sock

    with (
        patched_in_package("zmq_ctx", fake_zmq_ctx),
        patched_in_package("make_zmq_path", lambda *a: "tcp://x"),
        patched_in_package("current_platform", MagicMock()),
        patched_in_package("time", clock if clock is not None else time),
    ):
        yield


def _handshake(worker, sock, *, remote_tp_size=1, engine_id="eng"):
    with _patched_socket(sock):
        return worker._nixl_handshake("h", 1234, remote_tp_size, engine_id)


class TestUpstreamReachesTheHandshake:
    # The other handshake tests here reach _nixl_handshake through a helper
    # that calls it directly, which cannot see the calling convention. 0.26
    # submits it to its own executor with six positional arguments and unpacks
    # an (agents, clock_offset) pair.

    @staticmethod
    def _worker(cls=RblnNixlPullConnectorWorker):
        w = _make_worker(cls=cls)
        w._evict_stale_engines = lambda: None
        w._handshake_lock = threading.RLock()
        w._handshake_futures = {}
        w._handshake_initiation_executor = ThreadPoolExecutor(max_workers=1)
        w._remote_agents = {}
        w._engine_clock_offset = {}
        w._engine_last_active = {}
        w._engine_by_address = {}
        w._log_failure = MagicMock()
        # This stand-in holds no NIXL state, and __del__ runs shutdown().
        w.shutdown = w._handshake_initiation_executor.shutdown
        return w

    def test_ensure_handshake_publishes_the_agents_and_the_offset_midpoint(self):
        # The offset is the peer's stamp minus the MIDPOINT of the round trip:
        # the reply was stamped somewhere inside it and the midpoint is the
        # least-wrong guess. Stage 0 spans 0..10 locally, so a peer stamp of
        # 105 means the peer's clock reads 100 ahead of ours.
        sock = _FakeSock(pp_size=2, peer_stamps=[105.0, 1005.0])
        clock = _FakeClock([0.0, 10.0, 1000.0, 1010.0])
        w = self._worker()

        with _patched_socket(sock, clock):
            fut = NixlBaseConnectorWorker._ensure_handshake(w, "eng", "h", 1234, 1)
            fut.result()

        assert w._remote_agents["eng"] == {(0, 0): "agent-0", (1, 0): "agent-1"}
        assert w._engine_clock_offset["eng"] == pytest.approx(100.0)
        w._log_failure.assert_not_called()

    def test_the_lowest_rtt_sample_decides_the_offset(self):
        # Two stages disagree on the offset because their round trips differ.
        # The shorter trip pins the peer's stamp more tightly, so it wins --
        # taking the longer one would import that trip's whole hop cost.
        sock = _FakeSock(pp_size=2, peer_stamps=[150.0, 1002.0])
        # Stage 0: 0..100 (slow, midpoint 50 -> offset 100).
        # Stage 1: 1000..1002 (fast, midpoint 1001 -> offset 1).
        clock = _FakeClock([0.0, 100.0, 1000.0, 1002.0])
        w = self._worker()

        with _patched_socket(sock, clock):
            fut = NixlBaseConnectorWorker._ensure_handshake(w, "eng", "h", 1234, 1)
            fut.result()

        assert w._engine_clock_offset["eng"] == pytest.approx(1.0)

    def test_the_push_writer_reaches_it_and_unpacks_the_pair(self):
        # P's first write to a decode engine goes through the handshake, which
        # unpacks (agents, clock_offset). A dict of two shards, as here, unpacks
        # into its own keys instead of raising, so the engine would store an int
        # where an agent map belongs. vllm 0.30.0 folded push's own blocking
        # _ensure_d_handshake into the base _ensure_handshake.
        sock = _FakeSock(pp_size=2)
        w = self._worker(cls=RblnNixlPushConnectorWorker)

        with _patched_socket(sock):
            fut = NixlBaseConnectorWorker._ensure_handshake(w, "eng", "h", 1234, 1)
            assert fut is not None
            fut.result(timeout=5)

        assert w._remote_agents["eng"] == {(0, 0): "agent-0", (1, 0): "agent-1"}

    def test_a_peer_whose_pipeline_size_contradicts_the_caller_is_refused(self):
        # Push mode's decode side is told the producer's pipeline size by the
        # router. If the producer then reports a different one, one of the two
        # is describing another engine and pairing them would read from a stage
        # that does not exist.
        sock = _FakeSock(pp_size=2)
        w = self._worker()

        with _patched_socket(sock):
            fut = NixlBaseConnectorWorker._ensure_handshake(
                w, "eng", "h", 1234, 1, pp_size=3, notif_agents_only=True
            )
            with pytest.raises(RuntimeError, match="pipeline size 3, peer reports 2"):
                fut.result()

        assert "eng" not in w._remote_agents

    def test_a_notif_only_peer_skips_descriptor_setup(self):
        # Push mode's decode side never addresses the producer's memory, so
        # upstream asks for agents alone; registering descriptors for a stage
        # this rank cannot read would pair lists of different lengths.
        sock = _FakeSock(pp_size=2)
        w = self._worker()
        w._add_notif_only_remote_agent = MagicMock(side_effect=["n0", "n1"])

        with _patched_socket(sock):
            fut = NixlBaseConnectorWorker._ensure_handshake(
                w, "eng", "h", 1234, 2, pp_size=2, notif_agents_only=True
            )
            fut.result()

        assert w._remote_agents["eng"] == {(0, 0): "n0", (1, 0): "n1"}
        # Upstream builds the engine's EngineTransferInfo out of this
        # argument, so what reaches it has to be the peer's tensor-parallel
        # size and not ours.
        assert [c.args[1] for c in w._add_notif_only_remote_agent.call_args_list] == [
            2,
            2,
        ]
        w.add_remote_agent.assert_not_called()
        w._register_shard_xfer_state.assert_not_called()
        assert w._overlapping_ranks == {"eng": []}


def test_peer_meta_block_length_mirrors_the_local_table():
    # peer_meta is the peer's view of the same geometry, so its per-region block
    # length has to be the one KvGeometry reports locally -- halve it and the
    # address stride stops being wider than a region, which is what lets decode()
    # attribute an address to one region.
    from vllm.config import set_current_vllm_config

    from tests.vllm.distributed.kv_connector.utils import (
        KvGeometry,
        engine_config,
        peer_meta,
    )

    decoder = msgspec.msgpack.Decoder(RblnNixlAgentMetadata)
    cases = [
        KvGeometry(spec="full"),
        KvGeometry(spec="mla"),
        # A draft layer owns fewer heads, so the regions are not all one length.
        KvGeometry(per_layer_heads={"l1": 4}),
    ]
    for geo in cases:
        # `kv_caches` asks the attention backend which axis carries the blocks,
        # and that answer comes off the RBLN config.
        with set_current_vllm_config(engine_config(block_size=geo.block_size)):
            # A blocks-first entry is one region, a K/V-first one two; each
            # region spreads over the areas.
            local = [
                entry.numel()
                * entry.element_size()
                // (regions * geo.areas)
                // geo.num_blocks
                for entry in geo.kv_caches().values()
                for regions in [1 if entry.shape[0] == geo.num_blocks else 2]
                for _ in range(regions * geo.areas)
            ]
            agent = decoder.decode(peer_meta(geo).agent_metadata_bytes)
        assert list(agent.block_lens) == list(local), geo


def test_peer_meta_names_every_region_the_way_a_producer_does():
    # A producer publishes the four region lists one entry per REGION, and
    # `registered_layer_names` folds the names back to layers. At one chiplet
    # area the per-layer and per-region forms read alike, which is what let the
    # peer double drift to the per-layer one.
    from vllm.config import set_current_vllm_config

    from tests.vllm.distributed.kv_connector.utils import (
        KvGeometry,
        engine_config,
        peer_meta,
    )

    decoder = msgspec.msgpack.Decoder(RblnNixlAgentMetadata)
    geo = KvGeometry(areas=4, slices=4)

    with set_current_vllm_config(engine_config(block_size=geo.block_size)):
        agent = decoder.decode(peer_meta(geo).agent_metadata_bytes)

    n_regions = len(agent.kv_caches_base_addr)
    assert n_regions > len(geo.layers)  # or the two forms cannot be told apart
    for field in (
        "region_names",
        "region_num_blocks",
        "region_group_ids",
        "region_mem_types",
    ):
        assert len(getattr(agent, field)) == n_regions, field
    assert agent.registered_layer_names == list(geo.layers)


def test_a_partly_replicated_band_is_refused_rather_than_mismodelled():
    # Regression guard for the refusal in KvGeometry.__post_init__, which owns
    # the reason a band held by some of the areas is not modelled; a whole
    # entry on every area is, and the impl tests cover the case between.
    from tests.vllm.distributed.kv_connector.utils import KvGeometry

    with pytest.raises(AssertionError, match="replica only of a whole entry"):
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
        from tests.vllm.distributed.kv_connector.utils import KvGeometry, peer_meta

        geo = KvGeometry(areas=4, slices=4)
        w = make_worker(kv_cache=geo)
        port = peer_listener(
            lambda rank: peer_meta(geo, engine_id="peer", compat_hash=w.compat_hash)
        )
        ctx, sock = self._sock(port)
        try:
            meta, _rtt, _offset = w._query_agent_meta(sock, 0, 0, "peer")
        finally:
            sock.close()
            ctx.term()
        # The fields upstream's NixlAgentMetadata does not have: a peer that
        # dropped them would decode into upstream's type and carry no areas.
        assert {(len(ids), len(set(ids))) for ids in meta.kv_area_slices} == {(4, 4)}
        assert meta.registered_layer_names == list(geo.layers)

    def test_a_peer_whose_hash_differs_is_refused(self, make_worker, peer_listener):
        from tests.vllm.distributed.kv_connector.utils import KvGeometry, peer_meta

        geo = KvGeometry()
        w = make_worker(kv_cache=geo)
        port = peer_listener(
            lambda rank: peer_meta(geo, engine_id="peer", compat_hash="not-ours")
        )
        ctx, sock = self._sock(port)
        try:
            with pytest.raises(RuntimeError, match="compatibility hash mismatch"):
                w._query_agent_meta(sock, 0, 0, "peer")
        finally:
            sock.close()
            ctx.term()

    def test_a_peer_serving_another_engine_is_refused(self, make_worker, peer_listener):
        from tests.vllm.distributed.kv_connector.utils import KvGeometry, peer_meta

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
                w._query_agent_meta(sock, 0, 0, "expected-peer")
        finally:
            sock.close()
            ctx.term()


class TestPpHandshakeFanout:
    @pytest.mark.parametrize("pp_size", [1, 2, 3])
    def test_every_stage_is_queried_once_and_keyed_by_its_pair(self, pp_size):
        # The fan-out from one stage up. pp_size comes from the pp_rank-0 shard,
        # so that shard must not be queried twice; at pp_size 1 the whole thing
        # reduces to upstream's single-shard shape, whose pair is (0, tp_rank).
        w = _make_worker()
        sock = _FakeSock(pp_size=pp_size)

        result = _handshake(w, sock)

        agents, _ = result
        assert agents == {(r, 0): f"agent-{r}" for r in range(pp_size)}
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

        assert result[0] == {(1, 0): "agent-1"}
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
        w.add_remote_agent = lambda meta, rank, size, dcp: "agent"
        w._register_shard_xfer_state = lambda *a, **k: None
        _handshake(w, _FakeSock(pp_size=2, layers_per_stage=2))
        assert w._overlapping_ranks["eng"] == [0, 1]

    def test_trim_agent_meta_to_layers_trims_to_the_owned_layers(self):
        # A stage wider than our band is presented to upstream as just our
        # slice, so its region i pairs with our region i again.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
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

    def test_trim_agent_meta_to_layers_scales_by_regions_per_layer(self):
        # With one region per layer the trim's scaling is the identity, so the
        # case above cannot tell it apart from slicing by layer index. On D2D a
        # layer is K/V times the chiplet count, and the peer's band starts that
        # many regions in.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.local_seen_layer_names = ["layer.2"]
        w.num_regions = 2  # K and V: two regions per layer
        meta = _agent_meta(
            kv_caches_base_addr=[0x10 * i for i in range(8)],
            block_lens=[10 * i for i in range(8)],
            # One name per REGION, so a layer with K and V appears twice.
            region_names=[f"layer.{i}" for i in range(4) for _ in range(2)],
        )
        sliced = w._trim_agent_meta_to_layers(meta, [(2, 0)])
        assert list(sliced.kv_caches_base_addr) == [0x40, 0x50]
        assert list(sliced.block_lens) == [40, 50]
        assert list(sliced.registered_layer_names) == ["layer.2"]

    def test_trim_keeps_the_axes_of_the_owned_layers(self):
        # One axis a logical region, not one a region: cut by the region
        # window, a two-area peer's axes would come back twice as long and
        # starting at the wrong layer.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.local_seen_layer_names = ["layer.2"]
        w.num_regions = 4  # K and V over two areas
        token, head = ChipletSplitAxis.TOKEN, ChipletSplitAxis.HEAD
        meta = _agent_meta(
            kv_caches_base_addr=[0x10 * i for i in range(16)],
            block_lens=[10] * 16,
            region_names=[f"layer.{i}" for i in range(4) for _ in range(4)],
            kv_split_axes=[token, token, token, token, head, token, token, token],
        )

        sliced = w._trim_agent_meta_to_layers(meta, [(2, 0)])

        assert sliced.kv_split_axes == [head, token]

    def test_trim_keeps_the_area_slices_of_the_owned_layers(self):
        # Indexed like the axes, one entry a logical region; the pairing reads
        # the copies of a slice off them, so a stale window pairs wrong areas.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.local_seen_layer_names = ["layer.2"]
        w.num_regions = 4  # K and V over two areas
        meta = _agent_meta(
            kv_caches_base_addr=[0x10 * i for i in range(16)],
            block_lens=[10] * 16,
            region_names=[f"layer.{i}" for i in range(4) for _ in range(4)],
            kv_area_slices=[[0, 1]] * 4 + [[1, 0]] + [[0, 1]] * 3,
        )

        sliced = w._trim_agent_meta_to_layers(meta, [(2, 0)])

        assert sliced.kv_area_slices == [[1, 0], [0, 1]]

    def test_trim_agent_meta_to_layers_rejects_a_non_contiguous_span(self):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
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

    def test_a_side_that_moves_part_of_a_request_asks_even_a_matching_peer(
        self, monkeypatch
    ):
        # The same peer as above. Nothing is narrowed, so the only reason to
        # build per-shard state is that a transfer covering part of a request
        # needs a notification saying which part -- which upstream's
        # whole-engine handle has no room for.
        w = _make_worker()
        set_shape(w, streams_prefix=True)

        _handshake(w, _FakeSock(pp_size=1))

        assert w._overlapping_ranks["eng"] == [0]
        assert w._register_shard_xfer_state.call_count == 1

    def test_a_side_that_trims_a_last_block_asks_even_a_matching_peer(self):
        # Same peer again, and the same reason in a different shape: the trim
        # narrows a block rather than a peer, and only the per-shard ids can
        # leave part of one out.
        w = _make_worker()
        set_shape(w, chunk_mode=True)

        _handshake(w, _FakeSock(pp_size=1))

        assert w._overlapping_ranks["eng"] == [0]
        assert w._register_shard_xfer_state.call_count == 1

    @pytest.mark.parametrize("sw_ratio", [2, None])
    def test_swa_plus_pp_raises(self, sw_ratio):
        # The consumer's own guard, hit when it discovers a PP producer while
        # it has a sliding window; _check_pp_constraints is the separate
        # producer-side check. sw_ratio=None is the same model, window mode off.
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
        w = _make_worker(sw_ratio=2)
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
        w._head_matched_plan = lambda *a, **k: _stub_plan(1)
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
        w._head_matched_plan = lambda *a, **k: _stub_plan(1)
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
        plan = _stub_plan(2)
        w._head_matched_plan = MagicMock(return_value=plan)
        w._fan_in_peer_areas = MagicMock(return_value=[0])

        _handshake(w, _FakeSock(pp_size=2, tp_size=2), remote_tp_size=2)

        # Both stages are the peer's tp_rank 0 -- the head band comes from that,
        # not from the flat rank they are keyed by -- and stage i owns layer.i.
        assert w._fan_in_peer_areas.call_args_list == [call(0, 2), call(0, 2)]
        assert [
            c.kwargs["registered_layer_names"]
            for c in w._add_remote_agent_head_matched.call_args_list
        ] == [("layer.0",), ("layer.1",)]
        # The plan is asked for the same peer rank and areas, and what it
        # answers is what registers the stage's local list.
        assert [
            (c.args[1:3], c.kwargs["peer_areas"])
            for c in w._head_matched_plan.call_args_list
        ] == [((0, 2), [0]), ((0, 2), [0])]
        assert w._register_shard_xfer_state.call_args.kwargs["plan"] is plan

    def test_the_head_band_follows_the_tp_rank_across_a_wider_peer(self):
        # Every other test in this class talks to tp_rank 0 alone, which cannot
        # tell the peer's TP rank from the constant 0. Here the consumer reads
        # from both of a TP2 peer's ranks, so the two bands must differ.
        w = _make_worker(tp_ratio=-2, tp_target_ranks=(0, 1), host_buffer=False)
        w.local_seen_layer_names = ["layer.0"]
        w.num_regions = 1
        w._add_remote_agent_head_matched = MagicMock(side_effect=["a0", "a1"])
        w._head_matched_plan = lambda *a, **k: _stub_plan(1)
        w._fan_in_peer_areas = MagicMock(return_value=[0])

        agents, _ = _handshake(w, _FakeSock(pp_size=1, tp_size=2), remote_tp_size=2)

        assert w._fan_in_peer_areas.call_args_list == [call(0, 2), call(1, 2)]
        # The head-matched branch publishes under the pair too, and this is
        # the only shape here whose TP component is non-zero.
        assert agents == {(0, 0): "a0", (0, 1): "a1"}

    def test_a_notif_only_peer_is_still_refused_a_wider_pipelined_tp(self):
        # The guard sits above the notif_agents_only branch, so it holds on the
        # one shape where its reasoning is about work this rank will not do.
        # (That it reads the peer's report rather than the caller's hint is not
        # visible here -- the two agree on this branch -- but from the default
        # hint in test_pp_with_larger_peer_tp_raises.)
        w = _make_worker(tp_target_ranks=(0,), tp_ratio=-2)

        with (
            _patched_socket(_FakeSock(pp_size=2)),
            pytest.raises(RuntimeError, match="larger tensor-parallel size"),
        ):
            w._nixl_handshake(
                "h", 1234, 4, "eng", remote_pp_size=2, notif_agents_only=True
            )

    def test_both_sides_pipelined_is_allowed_on_the_pull_side(self):
        # The shape the push-side guard below must not take with it: reading,
        # this rank counts its own stages, so a pipeline on each side pairs
        # stage for stage and both are registered.
        w = _make_worker()
        w.vllm_config.parallel_config.pipeline_parallel_size = 2

        agents, _ = _handshake(w, _FakeSock(pp_size=2))

        assert agents == {(0, 0): "agent-0", (1, 0): "agent-1"}
        assert w._overlapping_ranks["eng"] == [0, 1]

    def test_a_pipelined_consumer_on_the_push_side_is_refused(self):
        # Upstream's completion count is the producer's pp times the producers
        # per consumer rank; nothing in it stands for our stages, so a consumer
        # split into them waits for notifications that never come. Upstream
        # refuses this too, but only under kv_role=kv_consumer, and the
        # launcher runs kv_both.
        w = _make_worker()
        w.vllm_config.parallel_config.pipeline_parallel_size = 2

        with (
            _patched_socket(_FakeSock(pp_size=2)),
            pytest.raises(RuntimeError, match="pipelined consumer"),
        ):
            w._nixl_handshake(
                "h", 1234, 1, "eng", remote_pp_size=2, notif_agents_only=True
            )

    def test_a_pipelined_consumer_is_fine_when_the_producer_is_not(self):
        # The count is right whenever the producer has one stage: every rank of
        # ours is written by that stage and waits for exactly one notification.
        w = _make_worker()
        w.vllm_config.parallel_config.pipeline_parallel_size = 2
        w._add_notif_only_remote_agent = MagicMock(return_value="n0")

        with _patched_socket(_FakeSock(pp_size=1)):
            agents, _ = w._nixl_handshake(
                "h", 1234, 1, "eng", remote_pp_size=1, notif_agents_only=True
            )

        assert agents == {(0, 0): "n0"}

    def test_a_split_region_alone_forces_the_per_shard_path(self):
        # Companion to test_a_finer_peer_alone_forces_the_per_shard_path: here
        # nothing narrows either, but our region spans several of the peer's
        # slices, so it is read in pieces the whole-engine handle cannot name.
        w = _make_worker(tp_ratio=2, host_buffer=False)
        w.local_seen_layer_names = ["layer.0"]
        w.num_regions = 1
        w._add_remote_agent_head_matched = lambda *a, **k: "agent"
        w._head_matched_plan = lambda *a, **k: _stub_plan(2)
        w._fan_in_peer_areas = lambda *a, **k: None

        _handshake(w, _FakeSock(pp_size=1, layers_per_stage=1), remote_tp_size=1)

        assert w._register_shard_xfer_state.call_count == 1
        assert w._overlapping_ranks["eng"] == [0]

    def test_a_wider_peer_geometry_is_refused_before_the_trim(self):
        # The peer expands a logical region into more chiplet areas than we do,
        # so nothing pairs. The trim that presents a wider stage as our own band
        # slices by OUR regions per layer, which makes the pairing check's
        # division an identity -- so it has to see what the peer published.
        w = _make_worker(tp_ratio=1, host_buffer=False)
        w.local_seen_layer_names = ["layer.0"]
        w.num_regions = 2  # K and V: two regions per layer
        sock = _FakeSock(pp_size=1, layers_per_stage=2, regions_per_layer=4)

        with pytest.raises(RuntimeError, match="per layer"):
            _handshake(w, sock, remote_tp_size=1)

    @pytest.mark.parametrize("notif_only", [False, True], ids=["full", "notif_only"])
    @pytest.mark.parametrize(
        "host_buffer, peer_mem_type, peer_device",
        [
            pytest.param(False, "DRAM", "cpu", id="we_d2d_they_host"),
            pytest.param(True, "VRAM", "rbln", id="we_host_they_d2d"),
        ],
    )
    def test_a_peer_on_the_other_buffer_device_is_refused_on_arrival(
        self, host_buffer, peer_mem_type, peer_device, notif_only
    ):
        # The peer publishes one region per layer against our two, so in the
        # D2D case the pairing check refuses it under another name if this runs
        # too late. A notif-only handshake builds nothing a later check sees.
        w = _make_worker(tp_ratio=1, host_buffer=host_buffer)
        w.local_seen_layer_names = ["layer.0"]
        w.num_regions = 2
        w._add_notif_only_remote_agent = MagicMock(return_value="n0")

        with (
            _patched_socket(_FakeSock(pp_size=1, region_mem_types=[peer_mem_type])),
            pytest.raises(
                RuntimeError, match=f"the peer runs kv_buffer_device='{peer_device}'"
            ),
        ):
            w._nixl_handshake("h", 1234, 1, "eng", notif_agents_only=notif_only)

    def test_a_host_staging_peer_advertising_its_type_pairs(self):
        w = _make_worker(tp_ratio=1, host_buffer=True)

        agents, _ = _handshake(w, _FakeSock(pp_size=1, region_mem_types=["DRAM"]))

        assert agents == {(0, 0): "agent-0"}

    @pytest.mark.parametrize(
        "region_ids, own_lists",
        [((0,), 1), ((0, 1), 0)],
        ids=["left_out", "whole"],
    )
    def test_a_plan_leaving_a_region_out_alone_forces_the_per_shard_path(
        self, region_ids, own_lists
    ):
        # One piece an entry: a plan leaving a region out, as one leaving a
        # latent to another rank of the side sending it does, needs our own
        # lists, since upstream's handle would still name it, and a plan naming
        # every region once does not. The latent only labels the case.
        w = _make_worker(tp_ratio=2, host_buffer=False)
        fill_region_geometry(
            w,
            areas=1,
            axes=[ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN],
            latent=[False, True],
        )
        w.local_seen_layer_names = ["layer.0", "layer.1"]
        w.num_regions = 2
        w._add_remote_agent_head_matched = lambda *a, **k: "agent"
        w._head_matched_plan = lambda *a, **k: _stub_plan(1, region_ids=region_ids)
        w._fan_in_peer_areas = lambda *a, **k: None

        _handshake(w, _FakeSock(pp_size=1, layers_per_stage=2), remote_tp_size=1)

        assert w._register_shard_xfer_state.call_count == own_lists

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
        # One piece, written to two copies: two entries a block.
        plan = _stub_plan(2)
        w._head_matched_plan = lambda *a, **k: plan
        w._fan_in_peer_areas = lambda *a, **k: None

        _handshake(w, _FakeSock(pp_size=1, layers_per_stage=1), remote_tp_size=1)

        assert w._register_shard_xfer_state.call_count == 1
        assert w._overlapping_ranks["eng"] == [0]
        assert w._register_shard_xfer_state.call_args.kwargs["plan"] is plan

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

    def test_a_peer_with_every_link_down_refuses_the_handshake(self, monkeypatch):
        # The refusal is an empty frame; it must not read as a version mismatch.
        w = _make_worker()
        sock = _FakeSock(pp_size=1)
        monkeypatch.setattr(
            sock, "recv_multipart", lambda: (b"", msgspec.msgpack.encode(0.0))
        )
        with pytest.raises(RuntimeError, match="every RDMA link down"):
            _handshake(w, sock)


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
        w._shape = shape()
        w._kv_per_block = 1
        w.local_seen_layer_names = [f"l{i}" for i in range(cls.N_LAYERS)]
        w.num_regions = cls.N_LAYERS * cls.RPL
        w.block_len_per_layer = [cls.TARGET_LEN] * w.num_regions
        if not uniform_lens:
            w.block_len_per_layer[-cls.RPL :] = [cls.DRAFT_LEN] * cls.RPL
        w._region_is_mla = [False] * (w.num_regions - mla_tail) + [True] * mla_tail
        # Empty as upstream's __init__ leaves it: regions pair by index.
        w._transfer_layer_region_indices = ()
        w._kv_areas = 1
        set_shape(w, chunk_mode=False)
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
            region_names=list(layer_names),
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

    def test_the_peer_list_carries_the_chunk_range_at_equal_tp(self):
        """A peer with our TP degree is served by the list upstream builds,
        while ours carries the chunk range like any other shard's. A prepared
        transfer pairs the two by position, so an index into a range only one
        side has names a descriptor the peer never published -- which the
        device rejects at transfer time, not at registration."""
        w = self._consumer()
        peer = self._peer(layer_names=["l5", "l6"], block_lens=[self.TARGET_LEN] * 4)

        plain = w._build_fa_remote(self._plan(), peer, block_size_ratio=1)
        with patch.object(
            RblnNixlPullConnectorWorker,
            "_shard_chunk_grid",
            return_value=uniform_grid(1, 2, regions=4),
        ) as grid:
            both = w._build_fa_remote(self._plan(), peer, block_size_ratio=1)

        # The peer's block size, not ours: its addresses are what is being
        # cut. And this builder serves the peers this rank does not band by
        # head, so every region is one piece and no plan cuts the runs.
        assert grid.call_args.kwargs == {"block_size": peer.block_size}
        # Upstream hands back its own descriptor shape, and the chunk range is
        # CONCATENATED onto it. `+` there is arithmetic on the very rows this
        # is meant to follow, and a descriptor merely wrong reaches the device.
        assert both.dtype == np.uint64 and both.shape[1] == 3
        assert np.array_equal(both[: len(plain)], plain)
        assert len(both) == len(plain) * 3
        addr, span, dev = (int(x) for x in plain[0])
        assert np.array_equal(
            both[len(plain) : len(plain) + 2],
            [(addr, span // 2, dev), (addr + span // 2, span // 2, dev)],
        )

    def test_more_than_one_run_walks_the_piece(self):
        # A head cut gives a span one run per head the region holds, so the
        # run axis has to advance inside the piece. At one run the stride is
        # unasserted -- the second chunk of the first run and the first chunk
        # of the second land on the same address.
        w = self._consumer()
        peer = self._peer(layer_names=["l5", "l6"], block_lens=[self.TARGET_LEN] * 4)

        plain = w._build_fa_remote(self._plan(), peer, block_size_ratio=1)
        with patch.object(
            RblnNixlPullConnectorWorker,
            "_shard_chunk_grid",
            return_value=uniform_grid(2, 2, regions=4),
        ):
            both = w._build_fa_remote(self._plan(), peer, block_size_ratio=1)

        assert len(both) == len(plain) * 5
        addr, span, dev = (int(x) for x in plain[0])
        quarter = span // 4
        # run 0's two chunks, then run 1's -- the second run starts halfway.
        assert np.array_equal(
            both[len(plain) : len(plain) + 4],
            [
                (addr, quarter, dev),
                (addr + quarter, quarter, dev),
                (addr + 2 * quarter, quarter, dev),
                (addr + 3 * quarter, quarter, dev),
            ],
        )

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
        w._kv_per_block = 1
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
        w._kv_per_block = 1
        w.local_seen_layer_names = list(local_names)
        w.num_regions = num_regions
        set_shape(w, chunk_mode=False)
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
        w.get_backend_aware_kv_block_len = MagicMock(return_value=64)
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.prep_xfer_dlist.return_value = 42
        window_mode(w, None)
        w.use_host_buffer = False  # D2D: narrowing comes from chiplet areas
        w._shard_descs_per_block = {}
        w._shard_chunk_grids = {}
        w._shard_region_spans = {}
        fill_region_geometry(w, areas=1, axes=[ChipletSplitAxis.HEAD] * 8)
        w._borrowed_src_handles = set()
        return w

    def test_register_shard_local_xfer_handler_covers_subset(self):
        w = self._wired_worker()

        handle, blocks = w._register_shard_local_xfer_handler(16, ("l2", "l3"))

        from tests.vllm.distributed.kv_connector.utils import decode

        assert handle == 42
        # shard regions [4,5,6,7] x num_blocks 4 = 16 descriptors, as the Nx3
        # uint64 array upstream's hetero-TP split reads back.
        assert (blocks.shape, blocks.dtype) == ((16, 3), np.uint64)
        # first desc: region 4 base addr (4000), block 0.
        assert blocks[0].tolist() == [4000, 64, 0]
        # block 1 of region 4: base + 1*stride(64).
        assert blocks[1].tolist() == [4000 + 64, 64, 0]
        # regions used are exactly the shard's (no addr below 4000).
        assert blocks[:, 0].min() == 4000
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

    def test_a_chunk_grid_appends_a_second_range_over_the_same_bytes(self):
        # A token chunk is one run per axis the block spreads it over -- on a
        # head cut, one per head. So a grid of (2 runs, 2 chunks) turns each
        # block's one descriptor into four, appended after the whole-block
        # range rather than replacing it -- both ranges stay selectable.
        w = self._wired_worker()

        _handle, plain = w._register_shard_local_xfer_handler(16, ("l2", "l3"))
        _handle, both = w._register_shard_local_xfer_handler(
            16, ("l2", "l3"), chunk_grid=uniform_grid(2, 2, regions=4)
        )

        assert np.array_equal(both[: len(plain)], plain)
        assert len(both) == len(plain) * 5  # 1 whole + 2 runs x 2 chunks
        # One block of one region: the whole piece, then its four quarters at
        # head- and chunk-strided offsets inside it.
        start, span, _ = (int(x) for x in plain[0])
        assert [
            (int(a) - start, int(ln)) for a, ln, _ in both[len(plain) : len(plain) + 4]
        ] == [
            (0, span // 4),
            (span // 4, span // 4),
            (span // 2, span // 4),
            (span // 2 + span // 4, span // 4),
        ]

    # A peer served one piece per block, and one cut into two pieces that each
    # go to two copies. The second is what says the per-block factor is in the
    # index: at one descriptor a block it cancels and any factor would do.
    @pytest.mark.parametrize(("split", "fanout"), [(1, 1), (2, 2)])
    def test_the_chunk_indices_point_at_the_chunks_the_builder_made(
        self, split, fanout
    ):
        # The one check that ties the two halves together: an index computed
        # from the layout has to land on the descriptor the builder put there.
        w = self._wired_worker()
        regions = [4, 5, 6, 7]
        region_ids, pieces = _split_plan(regions, 64, split, fanout)
        _handle, blocks = w._register_shard_local_xfer_handler(
            16,
            ("l2", "l3"),
            region_ids=region_ids,
            pieces=pieces,
            chunk_grid=uniform_grid(2, 2, regions=4),
        )
        per_block = split * fanout
        key = ("eng", 0)
        w._shard_region_group_ids = {key: (0,) * len(regions)}
        w._shard_descs_per_block[key] = (per_block,) * len(regions)
        w._shard_chunk_grids[key] = uniform_grid(2, 2, regions=4)

        ids = w._chunk_descs_ids_for_shard("eng", 0, w.num_blocks, 1, (0, 1))

        # Chunk 0 of block 1, for every piece the block is cut into and every
        # run inside it.
        assert len(ids) == len(regions) * per_block * 2
        whole = len(regions) * w.num_blocks * per_block
        picked = [tuple(int(x) for x in blocks[i]) for i in ids]
        for n in range(len(regions)):
            for p in range(per_block):
                piece = tuple(
                    int(x) for x in blocks[(n * w.num_blocks + 1) * per_block + p]
                )
                span = piece[1]
                at = (n * per_block + p) * 2
                assert picked[at] == (piece[0], span // 4, piece[2])
                assert picked[at + 1] == (piece[0] + span // 2, span // 4, piece[2])
        assert all(i >= whole for i in ids)

    def test_a_chunk_range_outside_the_block_is_refused(self):
        # An index that lands anywhere else lands on a descriptor for other
        # bytes, and a write that reaches them is not told.
        w = self._wired_worker()
        key = ("eng", 0)
        w._shard_region_group_ids = {key: (0,) * 4}
        w._shard_descs_per_block[key] = (1,) * 4
        w._shard_chunk_grids[key] = uniform_grid(2, 2, regions=4)

        with pytest.raises(RuntimeError, match="outside the 2 chunk"):
            w._chunk_descs_ids_for_shard("eng", 0, w.num_blocks, 0, (1, 3))

    def test_no_chunk_range_selects_nothing(self):
        # A peer the grid was not derived for: the lists hold no such range, so
        # an index into it would address past their end.
        w = self._wired_worker()
        key = ("eng", 0)
        w._shard_region_group_ids = {key: (0,) * 4}
        w._shard_descs_per_block[key] = (1,) * 4
        w._shard_chunk_grids[key] = None

        assert w._chunk_descs_ids_for_shard("eng", 0, w.num_blocks, 0, (0, 1)).size == 0

    def test_no_chunk_grid_leaves_the_list_as_it_was(self):
        # Guard: the grid is what the knob turns on, and off it must leave
        # the list as upstream built it. Measured against the grid on: `None`
        # is also the default, so comparing the two spellings of off compares
        # a call with itself.
        w = self._wired_worker()

        _handle, off = w._register_shard_local_xfer_handler(
            16, ("l2", "l3"), chunk_grid=None
        )
        _handle, on = w._register_shard_local_xfer_handler(
            16, ("l2", "l3"), chunk_grid=uniform_grid(2, 2, regions=4)
        )

        assert len(off) < len(on)
        assert np.array_equal(on[: len(off)], off)

    def test_a_windowed_engine_carries_the_chunk_range_on_the_whole_engine_list(
        self,
    ):
        # A sliding window keeps a request on the whole-engine lists -- the one
        # pair that can name two KV groups -- so the chunk range has to follow
        # the window's range there. The per-shard builder never sees it.
        w = self._wired_worker()
        window_mode(w, 2, has_swa=True)
        set_shape(w, chunk_mode=True)
        runs, chunks = 1, 2

        with patch.object(
            RblnNixlPullConnectorWorker,
            "_shard_chunk_grid",
            return_value=uniform_grid(runs, chunks, regions=w.num_regions),
        ):
            _handle, blocks = w.register_local_xfer_handler(16)

        whole = w.num_regions * w.num_blocks
        # Whole blocks, then the granules that tile them, then the chunk range.
        window_end = whole * (1 + w._shape.window_ratio)
        assert len(blocks) == window_end + whole * runs * chunks
        full_len = w.block_len_per_layer[0]
        assert {ln for _, ln, _ in blocks[:whole]} == {full_len}
        assert {ln for _, ln, _ in blocks[whole:window_end]} == {
            full_len // w._shape.window_ratio
        }
        assert {ln for _, ln, _ in blocks[window_end:]} == {full_len // (runs * chunks)}

    def test_register_local_xfer_handler_routes_to_the_shard_path(self):
        # Dispatch to the shard path: no SWA window mode, layer names present. Miss
        # it and a stage registers the whole model's regions, so the descriptor
        # math addresses layers it does not own.
        w = self._wired_worker()
        window_mode(w, None, has_swa=False)

        with patch.object(
            RblnNixlPullConnectorWorker, "_register_shard_local_xfer_handler"
        ) as shard:
            w.register_local_xfer_handler(16, registered_layer_names=("l2", "l3"))

        shard.assert_called_once_with(
            16,
            ("l2", "l3"),
            peer_areas=None,
            region_ids=None,
            pieces=None,
            chunk_grid=None,
        )

    def test_a_borrowed_handle_is_recorded_as_borrowed(self):
        # Host staging has no chiplet areas to narrow by, so the split comes
        # from upstream's own per-producer handles rather than a fresh one of
        # ours. Recording that is what keeps cleanup from releasing a handle
        # every peer at this tp ratio shares (test_cleanup_keeps_a_borrowed_
        # handle_alive covers the other end).
        w = self._wired_worker()
        w.kv_cache_config = MagicMock(transfer_groups=[object()])
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
        w.kv_cache_config = MagicMock(transfer_groups=[object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}

        w._register_shard_xfer_state("eng", 2, 16, ("l2", "l3"))

        assert w.src_xfer_handles_by_remote == {("eng", 2, 16): 42}
        # One group id per region of the shard: layers l2,l3 x rpl 2 = 4.
        assert w._shard_region_group_ids == {("eng", 2): (0, 0, 0, 0)}

    def test_register_shard_xfer_state_records_each_regions_pieces(self):
        # `_shard_descs_per_block` is what the read path multiplies a (region,
        # block) pair by to reach its descriptor ids, so it has to be every
        # entry the plan gives THAT region -- the pieces a head is cut into
        # times the peer copies each is written to. Two pieces and three copies
        # for two regions, one for the other two.
        w = self._wired_worker()
        w.kv_cache_config = MagicMock(transfer_groups=[object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}
        w._shard_descs_per_block = {}

        # Regions 4 and 5 in two pieces sent to three copies each, 6 and 7 whole.
        ids, cut = _split_plan([4, 5], 64, 2, 3)
        more, whole = _split_plan([6, 7], 64)
        w._register_shard_xfer_state(
            "eng", 2, 16, ("l2", "l3"), plan=(ids + more, cut + whole)
        )

        assert w._shard_descs_per_block == {("eng", 2): (6, 6, 1, 1)}

    @pytest.mark.parametrize(
        "sw_ratio, needs_own",
        [
            # A sliding window put a second range on the whole-engine list,
            # which is the one list a third can follow and the only one that
            # can name two KV groups sharing every region.
            (8, False),
            (None, True),
        ],
    )
    def test_a_chunked_engine_asks_for_its_own_descriptors_unless_windowed(
        self, sw_ratio, needs_own
    ):
        w = self._wired_worker()
        window_mode(w, sw_ratio, chunk_mode=True)

        assert (
            w._needs_own_descriptors(
                pp_size=1, partial=False, fan_in=False, pieces=False
            )
            is needs_own
        )

    @pytest.mark.parametrize("sw_ratio, needs_own", [(8, False), (None, True)])
    def test_a_streamed_engine_asks_for_its_own_descriptors_unless_windowed(
        self, sw_ratio, needs_own
    ):
        # Streaming asks for per-shard ids so a batch can be named on the
        # wire, and gives way to a window for the same reason chunk mode
        # does: a hybrid's two groups fit on no other list.
        w = self._wired_worker()
        w._shape = shape(chunk_mode=False, window_ratio=sw_ratio, streams_prefix=True)

        assert (
            w._needs_own_descriptors(
                pp_size=1, partial=False, fan_in=False, pieces=False
            )
            is needs_own
        )

    def test_register_shard_xfer_state_hands_the_derived_grid_on(self):
        # Derived once here and used for the local list; left out, the local
        # list holds whole blocks while every peer's carries the range.
        w = self._wired_worker()
        w.kv_cache_config = MagicMock(transfer_groups=[object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}
        w._shard_descs_per_block = {}

        w._register_shard_xfer_state(
            "eng", 2, 16, ("l2", "l3"), plan=_split_plan([4, 5, 6, 7], 64, 2, 3)
        )

        assert w._shard_descs_per_block == {("eng", 2): (6,) * 4}

        grid = uniform_grid(1, 2, regions=4)
        with (
            patch.object(
                RblnNixlPullConnectorWorker, "_shard_chunk_grid", return_value=grid
            ),
            patch.object(
                RblnNixlPullConnectorWorker,
                "register_local_xfer_handler",
                return_value=(42, []),
            ) as register,
        ):
            w._register_shard_xfer_state("eng", 2, 16, ("l2", "l3"))

        assert register.call_args.kwargs["chunk_grid"] == grid

    @pytest.mark.parametrize(
        ("rows", "kv_per_block", "expected"),
        [
            (None, 1, (2, 2)),
            # A piece holding one of the region's two heads is one run.
            (1, 1, (1, 2)),
            # A peer that reads K and V apart leaves the piece one of them, so
            # the runs are its heads alone. Sized against the whole block
            # instead, a chunk comes out the span and there is no grid at all
            # -- while the peer's list still carries one.
            (2, 2, (2, 2)),
        ],
    )
    def test_the_local_grid_is_derived_for_the_piece_this_shard_sends(
        self, rows, kv_per_block, expected
    ):
        # The runs follow the plan's pieces where there is one. The peer's list
        # is cut from the same plan (`_build_head_matched_remote`); cut from a
        # different one, the two carry different grids and a transfer pairs
        # them by position anyway.
        w = self._wired_worker()
        w.kv_cache_config = MagicMock(transfer_groups=[object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}
        w._shard_descs_per_block = {}
        # The geometry the grid is read off: 8 heads over 4 areas is 2 a
        # region, and 512B over a 16-token block is 16B a token.
        set_shape(w, chunk_mode=True)
        fill_region_geometry(
            w, areas=4, axes=[ChipletSplitAxis.HEAD] * 2, slices=[4] * 2
        )
        w._logical_region_local_heads = [8] * 2
        w.block_len_per_layer = [512] * 8
        w.tp_rank = 0
        w.transfer_topo.tp_size = 1
        w.vllm_config = mock_vllm_config()
        w.vllm_config.scheduler_config.max_num_batched_tokens = 8
        w._kv_per_block = kv_per_block

        with patch.object(
            RblnNixlPullConnectorWorker,
            "register_local_xfer_handler",
            return_value=(42, []),
        ) as register:
            w._register_shard_xfer_state(
                "eng",
                2,
                16,
                ("l2", "l3"),
                plan=None
                if rows is None
                else _split_plan([4, 5, 6, 7], 512, 2, rows=rows),
            )

        grid = uniform_grid(*expected, regions=4)
        assert register.call_args.kwargs["chunk_grid"] == grid
        assert w._shard_chunk_grids[("eng", 2)] == grid

    def test_a_head_cut_chunk_engine_takes_a_narrowed_shard(self):
        # A head cut pairs with a peer of any TP, so its shard can be narrowed
        # -- every region holds every token of its own heads.
        w = self._wired_worker()
        set_shape(w, chunk_mode=True)
        fill_region_geometry(w, areas=2, axes=[ChipletSplitAxis.HEAD] * 4, slices=2)
        w.kv_cache_config = MagicMock(transfer_groups=[object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}

        with patch.object(
            RblnNixlPullConnectorWorker, "_shard_chunk_grid", return_value=None
        ):
            w._register_shard_xfer_state(
                "eng", 2, 16, ("l2", "l3"), plan=_split_plan([4, 5, 6, 7], 64, 2)
            )

        assert w.src_xfer_handles_by_remote == {("eng", 2, 16): 42}

    def test_a_context_cut_shard_reads_each_regions_span_off_its_slice(self):
        # Which token range a region holds is the slice its area holds, not
        # where the area sits: a layout naming them out of order still names
        # each region's own range.
        w = self._wired_worker()
        fill_region_geometry(w, areas=2, axes=[ChipletSplitAxis.TOKEN] * 4, slices=2)
        w._logical_region_geometry = [
            g._replace(areas=g.areas[::-1]) for g in w._logical_region_geometry
        ]
        w.kv_cache_config = MagicMock(transfer_groups=[object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}

        w._register_shard_xfer_state("eng", 2, 16, ("l2", "l3"))

        assert w._shard_region_spans[("eng", 2)] == (1, 0, 1, 0)

    def test_register_shard_xfer_state_rejects_multiple_groups(self):
        # The single-group assumption is what makes the all-zero tuple above
        # right; more than one group has to fail rather than mislabel regions.
        w = self._wired_worker()
        w.kv_cache_config = MagicMock(transfer_groups=[object(), object()])
        w.src_xfer_handles_by_remote = {}
        w._shard_region_group_ids = {}

        with pytest.raises(AssertionError, match="single KV-cache group"):
            w._register_shard_xfer_state("eng", 2, 16, ("l2", "l3"))


@dataclass(frozen=True)
class _Cut:
    """One engine's chiplet geometry, layer by layer, for the `_CUTS` cases.

    A layer is `(name, axis, KV heads the rank holds, slices)`; a head cut
    gives each area `heads // slices` of them, a token cut every one over
    `block // areas` tokens. `peer_layers` makes the peer a pipeline stage
    holding only those.
    """

    layers: tuple[tuple[str, str, int, int], ...]
    areas: int = 4
    packed: bool = True
    step: int = 2
    peer_layers: tuple[str, ...] | None = None


# A target cut on tokens beside a draft cut on heads, and the shapes around it
# that the same arithmetic has to survive: the draft replicated, partly
# replicated, or first; K and V in regions of their own; two areas; a peer
# holding part of the layers; both on heads at different widths.
_CUTS = {
    "token-target-head-draft": _Cut(
        (("t0", "T", 2, 4), ("t1", "T", 2, 4), ("d0", "H", 12, 4))
    ),
    # Both on the head axis, the draft six heads an area to the target's one:
    # the regions still take their own runs.
    "both-on-heads-draft-wider": _Cut(
        (("t0", "H", 4, 4), ("t1", "H", 4, 4), ("d0", "H", 24, 4))
    ),
    "draft-replicated": _Cut((("t0", "T", 2, 4), ("d0", "H", 3, 1))),
    "draft-partly-replicated": _Cut((("t0", "T", 2, 4), ("d0", "H", 4, 2))),
    "draft-first": _Cut((("d0", "H", 12, 4), ("t0", "T", 2, 4))),
    "draft-between": _Cut((("t0", "T", 2, 4), ("d0", "H", 12, 4), ("t1", "T", 2, 4))),
    "one-head-target": _Cut((("t0", "T", 1, 4), ("d0", "H", 8, 4))),
    "three-head-target": _Cut((("t0", "T", 3, 4), ("d0", "H", 12, 4))),
    "split-kv": _Cut((("t0", "T", 2, 4), ("d0", "H", 12, 4)), packed=False),
    "two-areas": _Cut((("t0", "T", 2, 2), ("d0", "H", 6, 2)), areas=2),
    "step-at-a-span": _Cut((("t0", "T", 2, 4), ("d0", "H", 12, 4)), step=4),
    "step-past-a-span": _Cut((("t0", "T", 2, 4), ("d0", "H", 12, 4)), step=8),
    "step-at-the-block": _Cut((("t0", "T", 2, 4), ("d0", "H", 12, 4)), step=16),
    # No head cut to need a range: a chunk of two spans is two areas' whole
    # blocks, so it pairs with no chunk descriptor at all.
    "token-cut-only": _Cut((("t0", "T", 2, 4), ("t1", "T", 2, 4)), step=8),
    "peer-stage-with-the-draft": _Cut(
        (("t0", "T", 2, 4), ("t1", "T", 2, 4), ("d0", "H", 12, 4)),
        peer_layers=("t1", "d0"),
    ),
    "peer-stage-without-the-draft": _Cut(
        (("t0", "T", 2, 4), ("t1", "T", 2, 4), ("d0", "H", 12, 4)),
        peer_layers=("t0", "t1"),
    ),
}


class TestAnEngineCutOnTwoAxes:
    """A speculative draft keeps its own head count under the target's TP, so
    the compiler can cut the target's KV on the token axis and the draft's on
    the head axis in one engine. Both lists then cut a block's chunks into a
    different number of runs per region, and they pair element-wise -- an
    index that strides by one region's count lands the other's chunks on
    bytes nothing reports.

    Every case is checked exhaustively: each fill of a request's last block,
    and each chunk range a streamed piece can name. The expected bytes come
    off the layout alone -- K then V, each head's tokens contiguous, a token
    cut's area holding its span of every head -- so nothing is shared with
    the arithmetic under test.
    """

    BLOCK, TOKEN_BYTES = 16, 4
    LOCAL_BASE, PEER_BASE = 0x100000, 0x9000000
    LOCAL_BLOCKS, PEER_BLOCKS = 3, 2

    @classmethod
    def _regions(cls, cut):
        """Per local region, in id order: (layer, axis, heads it holds, first
        token it holds, tokens it holds)."""
        out = []
        for name, axis, heads, slices in cut.layers:
            for _ in range(1 if cut.packed else 2):
                for area in range(cut.areas):
                    if axis == "T":
                        span = cls.BLOCK // cut.areas
                        out.append((name, axis, heads, area * span, span))
                    else:
                        out.append((name, axis, heads // slices, 0, cls.BLOCK))
        return out

    @classmethod
    def _block_len(cls, cut, region):
        _, _, heads, _, width = cls._regions(cut)[region]
        return (2 if cut.packed else 1) * heads * width * cls.TOKEN_BYTES

    @classmethod
    def _bases(cls, base, lens, num_blocks):
        stride = max(lens) * num_blocks * 2
        return [base + i * stride for i in range(len(lens))]

    @classmethod
    def _worker(cls, cut):
        w = object.__new__(RblnNixlPullConnectorWorker)
        set_shape(w, chunk_mode=True)
        w.vllm_config = mock_vllm_config()
        w.vllm_config.scheduler_config.max_num_batched_tokens = cut.step
        w.block_size = cls.BLOCK
        w.num_blocks = cls.LOCAL_BLOCKS
        w.engine_id = "local"
        w.tp_rank = 0
        w.device_id = 0
        w._has_mamba = False
        w.nixl_memory_type = "VRAM"
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.get_xfer_descs.side_effect = lambda data, _t: data
        w.local_seen_layer_names = [name for name, *_ in cut.layers]
        regions = cls._regions(cut)
        w.num_regions = len(regions)
        w.block_len_per_layer = [cls._block_len(cut, r) for r in range(len(regions))]
        w.kv_caches_base_addr = {
            "local": {
                0: cls._bases(cls.LOCAL_BASE, w.block_len_per_layer, cls.LOCAL_BLOCKS)
            }
        }
        w._region_is_mla = [False] * len(regions)
        # Empty as upstream's __init__ leaves it: regions pair by index.
        w._transfer_layer_region_indices = ()
        w._kv_per_block = 2 if cut.packed else 1
        per_layer = 1 if cut.packed else 2
        token = ChipletSplitAxis.TOKEN
        w._logical_region_local_heads = [
            heads for _, _, heads, _ in cut.layers for _ in range(per_layer)
        ]
        w.transfer_topo = MagicMock(virtually_split_kv_in_blocks=False)
        w._mamba_ssm_size = (0, 0)
        w._group_spec_types = [FullAttentionSpec]
        return fill_region_geometry(
            w,
            areas=cut.areas,
            axes=[
                token if axis == "T" else ChipletSplitAxis.HEAD
                for _, axis, _, _ in cut.layers
                for _ in range(per_layer)
            ],
            slices=[
                cut.areas if axis == "T" else slices
                for _, axis, _, slices in cut.layers
                for _ in range(per_layer)
            ],
        )

    @classmethod
    def _lists(cls, cut):
        """Both lists for an equal-TP peer, our region ids in the peer's
        order, and the worker indexing them."""
        w = cls._worker(cut)
        names = cut.peer_layers or tuple(w.local_seen_layer_names)
        region_ids = w._shard_local_region_ids(names)
        grid = w._shard_chunk_grid(region_ids, block_size=cls.BLOCK)
        assert grid is not None, "every case here cuts a block on both axes"
        _handle, local = w._register_shard_local_xfer_handler(
            cls.BLOCK, names, chunk_grid=grid
        )
        lens = [w.block_len_per_layer[r] for r in region_ids]
        per_layer = len(region_ids) // len(names)
        peer = _agent_meta(
            engine_id="peer",
            kv_caches_base_addr=cls._bases(cls.PEER_BASE, lens, cls.PEER_BLOCKS),
            block_lens=lens,
            num_blocks=cls.PEER_BLOCKS,
            block_size=cls.BLOCK,
            region_names=[name for name in names for _ in range(per_layer)],
        )
        plan = MagicMock(source_ranks_per_group=[(0,)], rank_offset_factor=0)
        remote = w._build_fa_remote(plan, peer, block_size_ratio=1)
        key = ("peer", 0)
        w._shard_region_group_ids = {key: (0,) * len(region_ids)}
        w._shard_descs_per_block = {key: (1,) * len(region_ids)}
        w._shard_chunk_grids = {key: grid}
        local = [tuple(int(x) for x in d) for d in local]
        return w, grid, region_ids, local, remote.tolist()

    @staticmethod
    def _bytes(descs, bases, lens, num_blocks, region_ids):
        """(our region id, block, byte in block) every descriptor covers."""
        out = []
        for addr, length, _dev in descs:
            (pos,) = [
                p for p, b in enumerate(bases) if b <= addr < b + lens[p] * num_blocks
            ]
            block, offset = divmod(addr - bases[pos], lens[pos])
            assert offset + length <= lens[pos], "a descriptor crosses a block"
            out.append({(region_ids[pos], block, offset + i) for i in range(length)})
        return out

    @classmethod
    def _expected(cls, cut, region_ids, block, tokens):
        """The bytes of tokens [lo, hi) of one block, in the given regions."""
        lo, hi = tokens
        regions = cls._regions(cut)
        out = set()
        for rid in region_ids:
            _, _, heads, first, width = regions[rid]
            for t in range(max(lo, first), min(hi, first + width)):
                for run in range((2 if cut.packed else 1) * heads):
                    start = (run * width + t - first) * cls.TOKEN_BYTES
                    out |= {(rid, block, start + i) for i in range(cls.TOKEN_BYTES)}
        return out

    def _paired(self, cut, region_ids, local, remote, ids, peer_block):
        """The two picks pair element-wise, and what they cover once each."""
        local_ids, remote_ids = ids
        assert len(local_ids) == len(remote_ids)
        w = self._worker(cut)
        local_lens = w.block_len_per_layer
        mine = self._bytes(
            [local[i] for i in local_ids],
            w.kv_caches_base_addr["local"][0],
            local_lens,
            self.LOCAL_BLOCKS,
            range(len(local_lens)),
        )
        peer_lens = [local_lens[r] for r in region_ids]
        theirs = self._bytes(
            [tuple(remote[i]) for i in remote_ids],
            self._bases(self.PEER_BASE, peer_lens, self.PEER_BLOCKS),
            peer_lens,
            self.PEER_BLOCKS,
            region_ids,
        )
        for a, b in zip(mine, theirs):
            assert {(r, peer_block[blk], o) for r, blk, o in a} == b
        covered = set().union(*mine) if mine else set()
        # Each byte once: a region sent whole and again in chunks pairs and
        # covers the same set, and is still twice the bytes on the fabric.
        assert sum(len(a) for a in mine) == len(covered), "a byte went out twice"
        return covered

    @pytest.mark.parametrize("cut", _CUTS.values(), ids=_CUTS.keys())
    def test_a_request_s_last_block_goes_out_as_the_tokens_it_holds(self, cut):
        w, grid, region_ids, local, remote = self._lists(cut)
        chunk = self.BLOCK // grid.chunks_per_block
        # The peer numbered its blocks from zero, in the order the list below
        # names ours.
        peer_block = {2: 0, 1: 1}
        for tail in range(1, self.BLOCK + 1):
            valid = self.BLOCK + tail
            ids = (
                w._shard_descs_for_tokens(
                    "peer",
                    0,
                    self.LOCAL_BLOCKS,
                    [[2, 1]],
                    num_valid_tokens=valid,
                    num_prompt_blocks=2,
                ),
                w._shard_descs_for_tokens(
                    "peer",
                    0,
                    self.PEER_BLOCKS,
                    [[0, 1]],
                    num_valid_tokens=valid,
                    num_prompt_blocks=2,
                ),
            )
            covered = self._paired(cut, region_ids, local, remote, ids, peer_block)
            # Rounded up to a chunk, every byte of which is the request's block.
            held = -(-tail // chunk) * chunk
            assert covered == self._expected(
                cut, region_ids, 2, (0, self.BLOCK)
            ) | self._expected(cut, region_ids, 1, (0, held)), f"tail {tail}"

    @pytest.mark.parametrize("cut", _CUTS.values(), ids=_CUTS.keys())
    def test_a_streamed_piece_reaches_every_region_holding_its_tokens(self, cut):
        # A piece counts chunks of the block, and a range crossing a target
        # area is one piece here: which regions hold it is the index's to say.
        w, grid, region_ids, local, remote = self._lists(cut)
        cpb = grid.chunks_per_block
        chunk = self.BLOCK // cpb
        for lo in range(cpb):
            for hi in range(lo + 1, cpb + 1):
                ids = (
                    w._chunk_descs_ids_for_shard(
                        "peer", 0, self.LOCAL_BLOCKS, 2, (lo, hi)
                    ),
                    w._chunk_descs_ids_for_shard(
                        "peer", 0, self.PEER_BLOCKS, 1, (lo, hi)
                    ),
                )
                covered = self._paired(cut, region_ids, local, remote, ids, {2: 1})
                assert covered == self._expected(
                    cut, region_ids, 2, (lo * chunk, hi * chunk)
                ), f"chunks [{lo}, {hi})"


class TestChunkSizing:
    """What a chunk descriptor covers, and the grid a shard's lists take.

    Sized in tokens rather than bytes, so the answer does not depend on which
    regions a rank registered; floored by the prefill chunk because that is
    the unit a write covers; and rounded to a size that cuts every span or
    takes several whole, because a descriptor may not reach past one.
    """

    @staticmethod
    def _sized(*, span_tokens, block=8192, prefill=512, tokens=0):
        return kv_chunk_tokens(
            block_tokens=block,
            span_tokens=span_tokens,
            prefill_step_tokens=prefill,
            chunk_tokens=tokens,
        )

    @pytest.mark.parametrize(
        "tokens, span_tokens, prefill, expected",
        [
            # Unset takes the prefill step, which tiles an 8k block cut into
            # four areas.
            (0, [2048], 512, 512),
            # The knob is taken over the floor where it is above it.
            (1024, [2048], 512, 1024),
            # And it tiles the span as well -- 700 would leave a descriptor
            # reaching into the next span.
            (700, [2048], 512, 1024),
            # A span one prefill step fills leaves the span uncut.
            (0, [512], 512, 512),
            (0, [2048], 2048, 2048),
            # A step wider than the span takes two spans whole rather than
            # falling under the floor.
            (0, [2048], 4096, 4096),
            (4096, [2048], 512, 4096),
            # 3000 would end inside a span; the next width that does not is two
            # spans.
            (3000, [2048], 512, 4096),
            # A step as wide as the block leaves nothing to chunk.
            (0, [2048], 8192, 8192),
            # A head cut beside a context cut: one width serves both, cutting
            # the head cut's block and taking context spans whole.
            (0, [2048, 8192], 4096, 4096),
            # Two spans tile a span multiple but not the block; the next width
            # that does is the block.
            (0, [2048], 5000, 8192),
        ],
    )
    def test_the_chunk_is_the_smallest_tiling_size_the_floors_allow(
        self, tokens, span_tokens, prefill, expected
    ):
        assert self._sized(span_tokens=span_tokens, prefill=prefill, tokens=tokens) == (
            expected
        )

    def test_a_block_no_power_of_two_still_tiles_its_spans(self):
        # Every width above a 3072-token block's quarter tiles the block, so
        # only the span rule rules any of them out: 1024 divides the block and
        # ends inside a 768-token span, and the next width that does not is
        # two spans.
        assert self._sized(span_tokens=[768], block=3072, prefill=800) == 1536

    def test_a_block_whose_spans_a_step_overfills_keeps_the_step(self):
        # A 1k block over four areas is 256 tokens an area against a 512-token
        # step: the chunk is two areas, not one.
        assert self._sized(span_tokens=[256], block=1024) == 512

    def test_a_token_knob_under_one_prefill_step_is_refused(self):
        # Rounding it up instead would ignore the size an operator asked for,
        # and a step closing several chunks is not something the write path
        # can count in.
        with pytest.raises(RuntimeError, match="prefill step"):
            self._sized(span_tokens=[2048], tokens=256)

    @staticmethod
    def _grid_worker(
        *, block_len, areas=4, prefill=512, cls=None, local_heads=1, kv_per_block=1
    ):
        w = object.__new__(cls or RblnNixlPullConnectorWorker)
        w._shape = shape()
        set_shape(w, chunk_mode=True)
        # One head a rank is the latent cache; a tensor-parallel GQA rank
        # brings more.
        w._logical_region_local_heads = [local_heads] * 8
        w._kv_per_block = kv_per_block
        w.block_len_per_layer = [block_len] * 8
        w.vllm_config = mock_vllm_config()
        w.vllm_config.scheduler_config.max_num_batched_tokens = prefill
        # __init__ never ran, so the state a push worker's shutdown()
        # reaches through __del__ is absent; silence it rather than leak
        # an unraisable at GC.
        w.shutdown = lambda: None
        return fill_region_geometry(
            w, areas=areas, axes=[ChipletSplitAxis.TOKEN] * 8, slices=[areas] * 8
        )

    @staticmethod
    def _grid(w, *, block_size=8192):
        """The grid over the first logical region's four areas."""
        return w._shard_chunk_grid(range(4), block_size=block_size)

    def test_a_context_cut_of_one_head_spreads_a_chunk_over_one_run(self, monkeypatch):
        # An 8k block over four areas: each area holds 2048 tokens, and a
        # 512-token prefill step cuts that into four. One run because the rank
        # holds one head -- the area is a token range of every head it has.
        w = self._grid_worker(block_len=2048 * 256)

        assert self._grid(w) == uniform_grid(1, 4, regions=4, spans=4)

    def test_a_context_cut_is_a_run_per_head_the_rank_holds(self, monkeypatch):
        # The same geometry with the shape a tensor-parallel GQA model brings:
        # too few heads a rank for the compiler to split them across chiplets,
        # so it cuts the tokens and leaves every head on every area. A token
        # range of that area is then one run of bytes per head, because the
        # heads are the outer axis inside a block.
        w = self._grid_worker(block_len=2048 * 256, local_heads=2)

        assert self._grid(w) == uniform_grid(2, 4, regions=4, spans=4)

    def test_a_packed_block_runs_once_in_each_half(self, monkeypatch):
        # K and V sharing a block puts a second copy of every head after the
        # first, so a token range of one head sits once in each -- the layout
        # the run count already carried, and the only thing a context cut over
        # a packed block ever needed.
        w = self._grid_worker(block_len=2048 * 256, local_heads=2, kv_per_block=2)

        assert self._grid(w) == uniform_grid(4, 4, regions=4, spans=4)

    def test_regions_disagreeing_on_the_rank_band_take_their_own_runs(self):
        # A draft beside its target holds its own head count, so its regions
        # are a run per head of that count. One chunk width still serves both,
        # which is what keeps a block the same number of chunks in each.
        w = self._grid_worker(block_len=2048 * 256)
        w._logical_region_local_heads = [1, 2] * 4

        grid = w._shard_chunk_grid(range(8), block_size=8192)

        assert grid == ChunkGrid(
            16, tuple((area * 4, runs, 4) for runs in (1, 2) for area in range(4))
        )

    def test_a_step_wider_than_an_area_takes_areas_whole(self):
        # A 1k block over four areas against a 512-token step: a chunk is two
        # areas, each area's whole-block descriptor half of one, so the range
        # carries nothing.
        w = self._grid_worker(block_len=256 * 256)

        assert self._grid(w, block_size=1024) == ChunkGrid(
            2, ((0, 0, 1), (0, 0, 1), (1, 0, 1), (1, 0, 1))
        )

    def test_a_head_cut_beside_takes_the_same_wider_chunk(self):
        # The draft's head cut holds the whole block, so the two-area chunk
        # cuts it in two -- where a chunk of one area would have cut it in four.
        w = self._grid_worker(block_len=256 * 256)
        w._logical_region_local_heads = [1, 4] * 4
        fill_region_geometry(
            w,
            areas=4,
            axes=[ChipletSplitAxis.TOKEN, ChipletSplitAxis.HEAD] * 4,
            slices=[4] * 8,
        )

        grid = w._shard_chunk_grid(range(8), block_size=1024)

        assert grid == ChunkGrid(
            2, ((0, 0, 1), (0, 0, 1), (1, 0, 1), (1, 0, 1)) + ((0, 1, 2),) * 4
        )

    def test_an_area_exactly_one_chunk_wide_is_named_by_its_whole_block(self):
        # A 256-token step over a 1k block cut on tokens into four areas: each
        # area is one chunk, so its whole-block descriptor names it and the
        # range carries none -- while the head-cut draft beside is cut in four.
        w = self._grid_worker(block_len=256 * 256, prefill=256)
        w._logical_region_local_heads = [1, 4] * 4
        fill_region_geometry(
            w,
            areas=4,
            axes=[ChipletSplitAxis.TOKEN, ChipletSplitAxis.HEAD] * 4,
            slices=[4] * 8,
        )

        grid = w._shard_chunk_grid(range(8), block_size=1024)

        assert grid == ChunkGrid(
            4, tuple((a, 0, 1) for a in range(4)) + ((0, 1, 4),) * 4
        )

    @pytest.mark.parametrize(
        "kwargs, block_size",
        [
            # A 2k block over four areas is 512 tokens an area, which one
            # prefill step fills: the area is already the smallest unit, and
            # that is what chunk mode drops today.
            ({"block_len": 512 * 256}, 2048),
            # A prefill chunk as wide as an area, at any block size.
            ({"block_len": 2048 * 256, "prefill": 2048}, 8192),
        ],
    )
    def test_a_span_a_chunk_cannot_cut_gets_no_grid(self, kwargs, block_size):
        w = self._grid_worker(**kwargs)

        assert self._grid(w, block_size=block_size) is None

    def test_off_the_knob_there_is_no_grid(self):
        # A block length the shape could cut, so only the knob answers None.
        w = self._grid_worker(block_len=2048 * 256)
        set_shape(w, chunk_mode=False)

        assert self._grid(w) is None

    def test_a_side_that_streams_alone_gets_no_grid(self):
        # Streaming writes whole blocks: the ones the prefill closed, and the
        # last one whole at the handover. Only chunk mode cuts a block.
        w = self._grid_worker(block_len=2048 * 256, cls=RblnNixlPushConnectorWorker)
        w._shape = shape(chunk_mode=False, streams_prefix=True)

        assert self._grid(w) is None


class TestBaseFanInHandle:
    # Host staging has no chiplet areas to narrow with, so a shard reading from
    # a finer-grained producer borrows upstream's per-producer split instead of
    # emitting descriptors that span every producer's head band.

    @staticmethod
    def _worker(
        *, host_buffer=True, tp_ratio=-4, use_mla=False, source_ranks=(0, 1, 2, 3)
    ):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
        w.use_host_buffer = host_buffer
        w.block_size = 16
        w.num_regions = 4
        # Read by upstream's `_needs_split_local_xfer_handles`, which decides
        # whether a split was built at all.
        w.use_mla = use_mla
        w.transfer_topo = MagicMock()
        w.transfer_topo.tp_ratio.return_value = tp_ratio
        # Keyed as upstream keys it: one ratio can describe peers whose block
        # sizes differ, so the block size is part of the key.
        w.src_xfer_handles_by_tp_ratio = {(-4, 16): [70, 71, 72, 73]}
        w.tp_mappings = {"eng": MagicMock(all_source_ranks=source_ranks)}
        return w

    def test_a_replicated_mla_peer_has_no_split_to_borrow(self):
        # Upstream builds the split for everything but a pure-MLA mapping, whose
        # one source rank writes the whole region. Reading the map on the ratio
        # alone indexes a key it never wrote.
        w = self._worker(use_mla=True, source_ranks=(0,))

        assert w._base_fan_in_handle("eng", 0, 16, [0, 1, 2, 3], 4) is None

    def test_picks_the_split_for_this_producer(self):
        w = self._worker()
        assert w._base_fan_in_handle("eng", 2, 16, [0, 1, 2, 3], 4) == 72

    def test_picks_by_position_in_the_producer_list_not_by_rank(self):
        # `all_source_ranks` is the identity only for local rank 0: upstream
        # builds it from `tp_rank * abs(tp_ratio)`, so rank 1 at a ratio of -4
        # reads producers 4..7. Indexing the borrowed handles by the global rank
        # would then run off the end, or take another rank's split.
        w = self._worker()
        w.tp_mappings = {"eng": MagicMock(all_source_ranks=(4, 5, 6, 7))}
        assert w._base_fan_in_handle("eng", 6, 16, [0, 1, 2, 3], 4) == 72

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
        w._kv_per_block = 1
        w.use_host_buffer = host_buffer
        w.use_mla = use_mla
        w.local_seen_layer_names = [f"l{i}" for i in range(num_layers)]
        w.num_regions = num_layers * 2 * areas
        w.block_len_per_layer = [64] * (num_layers * 2 * areas)
        w.dst_num_blocks = {"eng": dst_num_blocks}
        # 0.30.0 keeps the peer's per-region block counts alongside its
        # total; every region here holds the whole count.
        w.dst_region_num_blocks = {"eng": [dst_num_blocks] * (num_layers * 2 * areas)}
        w.region_group_ids = [0] * (num_layers * 2 * areas)
        w.dst_region_group_ids = {"eng": [0] * (num_layers * 2 * areas)}
        w.dst_uses_region_group_mapping = {"eng": False}
        w._uses_region_group_mapping = False
        w._transfer_layer_region_indices = ()
        w.nixl_memory_type = "VRAM"
        w.region_mem_types = ["VRAM"] * (num_layers * 2 * areas)
        w.dst_region_mem_types = {"eng": ["VRAM"] * (num_layers * 2 * areas)}
        w.vllm_config = mock_vllm_config()
        w.vllm_config.parallel_config.pipeline_parallel_size = 1
        # add_remote_agent reads the cached copy __init__ makes of it.
        w.pp_size = 1
        w.vllm_config.speculative_config = None
        fill_region_geometry(
            w, areas=areas, axes=[ChipletSplitAxis.HEAD] * (num_layers * 2)
        )
        window_mode(w, None, chunk_mode=False, has_swa=False)
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
        axis=ChipletSplitAxis.HEAD,
        n_layers=0,
    ):
        # K and V of every layer the peer names, or of the consumer's 28
        # where it names none.
        return _agent_meta(
            pp_size=pp_size,
            kv_caches_base_addr=[1000 * i for i in range(n_regions)],
            num_blocks=num_blocks,
            block_size=block_size,
            kv_areas=kv_areas,
            kv_slices=kv_slices,
            kv_split_axes=[axis] * (2 * (n_layers or 28)),
            region_names=[f"l{i}" for i in range(n_layers)],
        )

    def test_a_peer_on_another_axis_is_refused_through_the_entry_point(self):
        # The guard's own cases call it directly, which says nothing about the
        # entry point still calling it: deleting that one line kept the suite
        # green. Equal TP, so no other refusal can account for the raise.
        w = self._consumer(host_buffer=False)
        with pytest.raises(RuntimeError, match=r"as \[\('TOKEN'"):
            w._validate_remote_agent_handshake(
                self._meta(pp_size=1, n_regions=56, axis=ChipletSplitAxis.TOKEN),
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
            self._meta(pp_size=2, n_regions=112, n_layers=14, kv_areas=4, kv_slices=1),
            remote_tp_size=1,
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
                    region_names=[f"l{i}" for i in range(28)],
                    kv_split_axes=[ChipletSplitAxis.HEAD] * 56,
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
                _agent_meta(
                    kv_caches_base_addr=[0] * 56,
                    region_names=[],
                    kv_split_axes=[ChipletSplitAxis.HEAD] * 56,
                ),
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
                    region_names=[f"l{i}" for i in range(14)],
                    kv_split_axes=[ChipletSplitAxis.HEAD] * 28,
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

    def test_mla_heterogeneous_tp_is_refused_as_mla_past_the_bands(self):
        # Registering a peer reads both head bands before it validates the
        # pairing, and an MLA latent bands no region by head -- so the bands
        # must not refuse it first under a message about slice layouts.
        w = self._consumer(use_mla=True, host_buffer=False)
        w._logical_region_kv_heads = [None] * len(w._logical_region_geometry)
        w.transfer_topo.tp_ratio.return_value = 2
        meta = self._meta(pp_size=1, n_regions=56)

        w._local_band(2)
        w._peer_band(meta, 2, None)
        with pytest.raises(RuntimeError, match="MLA is not supported"):
            w._validate_remote_agent_handshake(meta, remote_tp_size=2)

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

    @pytest.mark.parametrize("host_buffer", [True, False])
    def test_mla_with_pipelined_peer_is_allowed(self, host_buffer):
        # A pipelined producer is paired by layer name, which does not depend on
        # whether the cache is head-sharded -- and its geometry is compared on
        # the layers it holds, not against every layer of ours.
        w = self._consumer(use_mla=True, host_buffer=host_buffer)
        w._validate_remote_agent_handshake(
            self._meta(pp_size=2, n_regions=28, n_layers=14), remote_tp_size=1
        )  # no raise

    def test_a_peer_laying_its_slices_out_otherwise_is_refused(self):
        # Equal TP pairs area k with area k, so both sides have to lay each
        # region's slices over their areas alike; otherwise the block stride
        # differs and the wrong bytes move.
        w = self._consumer(use_mla=True, host_buffer=False)
        with pytest.raises(RuntimeError, match="an area means something else"):
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
        w._shape = shape()
        w._kv_per_block = 1
        w.tp_rank = tp_rank
        w._kv_areas = areas
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
        w._logical_region_local_heads = [
            heads // tp_size for heads in w._logical_region_kv_heads
        ]
        fill_region_geometry(
            w,
            areas=areas,
            axes=[ChipletSplitAxis.HEAD] * len(w._logical_region_kv_heads),
            slices=slices,
        )
        set_shape(w, chunk_mode=False)  # the grid tests turn it on
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
        from tests.vllm.distributed.kv_connector.utils import decode

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

        assert f(0, 1, 8, 4) == (0, 2)
        assert f(0, 2, 8, 4) == (0, 1)
        assert f(1, 2, 8, 4) == (4, 1)
        assert f(0, 4, 8, 2) == (0, 1)
        assert f(2, 4, 8, 2) == (4, 1)

    @pytest.mark.parametrize(
        "args,side,message",
        [
            # Upstream serves TP > num_kv_heads by replicating one head across
            # ranks (tp_mapping's `tp_size > total_num_kv_heads` branch); a band
            # is then a fraction of a head, which no descriptor names.
            ((0, 8, 4, 4), "peer", "does not divide the model's 4 KV heads"),
            ((0, 2, 8, 3), "local", "cut into 3 logical slice"),
        ],
    )
    def test_a_geometry_that_cannot_be_banded_is_refused(self, args, side, message):
        # The two refusals run on a peer's advertised numbers as well as our
        # own, so they refuse a pairing across the handshake rather than assert
        # an invariant -- and the message has to name whose numbers failed.
        with pytest.raises(RuntimeError, match=message) as e:
            RblnNixlPullConnectorWorker._slice_head_bounds(*args, side=side)
        assert side in str(e.value)

    @pytest.mark.parametrize("ids", [[], [0, 2], [0, 0, 1]])
    def test_a_peer_whose_areas_are_not_whole_slices_is_refused_on_arrival(self, ids):
        # The descriptor arithmetic divides a region's areas by its slices, so
        # a region with none, a missing slice, or unevenly replicated ones
        # would otherwise surface as a division or a wrong pairing far in.
        w = _make_worker()
        with pytest.raises(RuntimeError, match="each held by the same number"):
            _handshake(w, _FakeSock(pp_size=1, kv_area_slices=[ids]))

    @pytest.mark.parametrize(
        "geometry",
        [
            {"kv_split_axes": []},
            {"kv_area_slices": [[0, 1]]},
            # Three regions the two lists name, so only the area counts disagree.
            {
                "kv_area_slices": [[0], [0, 1]],
                "kv_split_axes": [ChipletSplitAxis.HEAD] * 2,
                "regions_per_layer": 3,
            },
        ],
        ids=["no-axes", "areas-over-one-region", "uneven-area-counts"],
    )
    def test_a_peer_whose_geometry_does_not_describe_its_regions_is_refused(
        self, geometry
    ):
        # Every reader indexes a peer region as `logical * areas + area`, so
        # the lists have to agree with each other and with the regions.
        w = _make_worker()
        with pytest.raises(RuntimeError, match="one region an area"):
            _handshake(w, _FakeSock(pp_size=1, **geometry))

    def test_each_peer_region_is_read_where_its_own_slices_sit(self):
        # A peer's regions only have to agree on how many areas and slices they
        # have; each region's descriptors follow its own slice ids. Here the
        # peer's second region holds slice 0 in areas 1 and 3, not 0 and 1.
        w = self._worker(
            tp_rank=0, tp_size=2, areas=2, slices=2, n_logical=2, block_len=256
        )
        meta = self._meta(areas=4, slices=2, n_logical=2, block_len=512)
        meta.kv_area_slices = [[0, 0, 1, 1], [1, 0, 1, 0]]

        assert w._peer_band(meta, 1, None) == (4, 2)
        out = w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=1)
        # A read takes a slice's first holder: area 0 of region 0, area 1 of
        # region 1.
        assert {region for region, _, _ in self._decoded(out, meta)} == {0, 4 + 1}

    def test_a_peer_stage_region_is_read_where_its_own_slices_sit(self):
        # The same under PP: our layer l1 is the stage's first region, so the
        # slice ids are the ones at that position of the stage's list, not at
        # ours.
        w = self._worker(
            tp_rank=0, tp_size=2, areas=2, slices=2, n_logical=2, block_len=256
        )
        w.local_seen_layer_names = ["l0", "l1"]
        meta = self._meta(areas=4, slices=2, n_logical=2, block_len=512)
        meta.kv_area_slices = [[1, 0, 1, 0], [0, 0, 1, 1]]

        out = w._build_head_matched_remote(
            meta,
            remote_tp_rank=0,
            remote_tp_size=1,
            registered_layer_names=["l1", "l2"],
        )
        assert {region for region, _, _ in self._decoded(out, meta)} == {1}

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

    def test_a_packed_block_is_named_once_in_k_and_once_in_v(self):
        # The count is the wiring: `pair_area` takes how many ranges a block
        # breaks into, and only this builder hands it on. Testing that function
        # directly leaves the hand-off unpinned, and a peer then reads a whole
        # packed block where it wanted K.
        split = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=512)
        packed = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        packed._kv_per_block = 2

        one = split._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=1)
        two = packed._build_head_matched_remote(
            meta, remote_tp_rank=0, remote_tp_size=1
        )

        # Unequal cuts, so a packed block is read once inside K and once inside
        # V where a split one is read whole.
        assert len(two) == len(one) * 2
        assert {ln for _, ln, _ in two} == {128}

    def test_the_peer_list_derives_its_grid_for_the_halves_it_reads(self):
        # The grid is derived twice, once per list, and a transfer pairs the
        # two by position: a tail selecting chunks of one names a different
        # range of the other. Unequal cuts leave the piece one half here, so
        # sizing a chunk against the whole block drops the range entirely.
        w = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        w._kv_per_block = 2
        set_shape(w, chunk_mode=True)
        w.vllm_config = mock_vllm_config()
        w.vllm_config.scheduler_config.max_num_batched_tokens = 8
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=512)

        out = w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=1)

        # 4 areas x 2 blocks x 2 halves, each a whole piece and a 1-run,
        # 2-chunk grid over it.
        assert len(out) == 4 * 2 * 2 * (1 + 1 * 2)

    def _grid_worker(self, *, monkeypatch, kv_heads=8, prefill=8):
        w = self._worker(
            tp_rank=0,
            tp_size=1,
            areas=4,
            slices=4,
            n_logical=1,
            block_len=512,
            kv_heads=kv_heads,
        )
        # A 16-token block whose 2-head region costs 16B a token, so a 128B
        # descriptor target is 8 tokens: two chunks of a block.
        set_shape(w, chunk_mode=True)
        w.vllm_config = mock_vllm_config()
        w.vllm_config.scheduler_config.max_num_batched_tokens = prefill
        return w

    @pytest.mark.parametrize(
        "rows, expected",
        [
            # 8 heads over 4 areas is 2 per area, and a block is two chunks.
            (None, (2, 2)),
            # A piece holding half the area's heads is half the runs.
            (1, (1, 2)),
        ],
    )
    def test_a_head_cut_spreads_a_chunk_over_one_run_per_head(
        self, monkeypatch, rows, expected
    ):
        w = self._grid_worker(monkeypatch=monkeypatch)

        assert w._shard_chunk_grid(
            range(4), block_size=16, rows=None if rows is None else [rows] * 4
        ) == uniform_grid(*expected, regions=4)

    @pytest.mark.parametrize(
        "kv_per_block, rows, expected",
        [
            # A piece that is K or V alone: one run per head, as before.
            (1, None, (2, 8)),
            (2, 2, (2, 8)),
            # A piece holding both: the same token range sits once in each, so
            # it takes twice the runs over the same chunks.
            (2, None, (4, 8)),
        ],
    )
    def test_a_piece_holding_both_halves_takes_a_run_in_each(
        self, monkeypatch, kv_per_block, rows, expected
    ):
        w = self._grid_worker(monkeypatch=monkeypatch, prefill=2)
        w._kv_per_block = kv_per_block

        assert w._shard_chunk_grid(
            range(4), block_size=16, rows=None if rows is None else [rows] * 4
        ) == uniform_grid(*expected, regions=4)

    def test_regions_that_disagree_on_their_band_take_their_own_runs(self, monkeypatch):
        # A draft's regions hold their own head band, so a chunk of one is
        # fewer runs than a chunk of the target's -- over the same chunks.
        w = self._grid_worker(monkeypatch=monkeypatch, kv_heads=[8, 4])

        assert w._shard_chunk_grid(range(8), block_size=16) == ChunkGrid(
            2, ((0, 2, 2),) * 4 + ((0, 1, 2),) * 4
        )

    @pytest.mark.parametrize(
        ("tp_l", "rank", "tp_r", "heads", "packed"),
        [
            (2, 0, 1, (8, 4), False),
            # A rank past the first reads into the middle of the peer's slice.
            (2, 1, 1, (8, 4), False),
            # A packed block read by head band is a run in K and one in V.
            (2, 0, 1, (8, 4), True),
            (4, 2, 1, (16, 8), True),
            # The peer holds two of our ranks' heads, not all of them.
            (4, 3, 2, (16, 8), False),
        ],
    )
    def test_bands_that_differ_pair_head_for_head_across_tp(
        self, tp_l, rank, tp_r, heads, packed
    ):
        """A draft beside its target at unequal TP: the target region holds
        twice the draft's heads an area, so a chunk is twice the runs in one.
        The lists pair element-wise, so each entry has to name the same heads,
        the same half and the same tokens on both sides."""
        token_bytes, block, areas = 4, 16, 2
        kv = 2 if packed else 1
        local_hpa = [h // tp_l // areas for h in heads]
        peer_hpa = [h // tp_r // areas for h in heads]
        w = self._worker(
            tp_rank=rank,
            tp_size=tp_l,
            areas=areas,
            slices=areas,
            n_logical=2,
            block_len=[kv * h * block * token_bytes for h in local_hpa],
            kv_heads=list(heads),
        )
        w._kv_per_block = kv
        set_shape(w, chunk_mode=True)
        w.vllm_config = mock_vllm_config()
        w.vllm_config.scheduler_config.max_num_batched_tokens = 4
        w.block_size, w.num_blocks, w.num_regions = block, 2, 4
        w.engine_id, w.device_id, w._has_mamba = "local", 0, False
        w.nixl_memory_type = "VRAM"
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.get_xfer_descs.side_effect = lambda data, _t: data
        stride = max(w.block_len_per_layer) * w.num_blocks
        bases = [stride * (i + 1) for i in range(4)]
        w.kv_caches_base_addr = {"local": {rank: bases}}
        meta = self._meta(
            areas=areas,
            slices=areas,
            n_logical=2,
            block_len=[kv * h * block * token_bytes for h in peer_hpa],
        )
        remote_rank = rank // (tp_l // tp_r)

        remote = w._build_head_matched_remote(
            meta, remote_tp_rank=remote_rank, remote_tp_size=tp_r
        )
        region_ids, pieces = w._head_matched_plan(
            meta, remote_tp_rank=remote_rank, remote_tp_size=tp_r
        )
        assert region_ids == [0, 1, 2, 3]
        grid = w._shard_chunk_grid(
            region_ids, block_size=block, rows=[p[0].rows for p in pieces]
        )
        _handle, local = w._register_shard_local_xfer_handler(
            block,
            (),
            region_ids=region_ids,
            pieces=pieces,
            chunk_grid=grid,
        )

        def meaning(descs, bases, lens, hpa, tp, owner):
            """(logical region, block, K or V, head, token) each covers."""
            out = []
            for addr, length, _dev in (tuple(int(x) for x in d) for d in descs):
                (region,) = [
                    r
                    for r, b in enumerate(bases)
                    if b <= addr < b + lens[r] * w.num_blocks
                ]
                logical, area = divmod(region, areas)
                blk, offset = divmod(addr - bases[region], lens[region])
                first = owner * heads[logical] // tp + area * hpa[logical]
                half = hpa[logical] * block * token_bytes
                cells = set()
                for byte in range(offset, offset + length):
                    k, rest = divmod(byte, half)
                    head, within = divmod(rest, block * token_bytes)
                    cells.add((logical, blk, k, first + head, within // token_bytes))
                out.append(cells)
            return out

        ours = meaning(
            local,
            bases,
            w.block_len_per_layer,
            local_hpa,
            tp_l,
            rank,
        )
        theirs = meaning(
            remote,
            meta.kv_caches_base_addr,
            meta.block_lens,
            peer_hpa,
            tp_r,
            remote_rank,
        )
        assert grid is not None and len({runs for _, runs, _ in grid.regions}) == 2
        assert len(ours) == len(theirs)
        assert ours == theirs
        # Past the whole pieces, an entry is one head's chunk of tokens in one
        # of K and V.
        whole = w.num_blocks * sum(map(len, pieces))
        assert {len(m) for m in ours[whole:]} == {block // grid.chunks_per_block}

    def test_the_remote_list_takes_the_same_chunk_grid(self):
        """`make_prepped_xfer` pairs the two lists by position, so a range
        published on one side only would pair chunk descriptors against whole
        blocks. The peer's list is built over the same two grids."""
        w = self._worker(
            tp_rank=0, tp_size=4, areas=4, slices=2, n_logical=1, block_len=256
        )
        meta = self._meta(areas=4, slices=4, n_logical=1, block_len=512)

        plain = w._build_head_matched_remote(meta, remote_tp_rank=0, remote_tp_size=1)
        with patch.object(
            type(w), "_shard_chunk_grid", return_value=uniform_grid(2, 2, regions=4)
        ) as grid:
            both = w._build_head_matched_remote(
                meta, remote_tp_rank=0, remote_tp_size=1
            )
        # Derived here rather than handed in, from the plan this list used.
        _, pieces = w._head_matched_plan(meta, 0, 1)
        assert grid.call_args.kwargs["rows"] == [p[0].rows for p in pieces]

        assert np.array_equal(both[: len(plain)], plain)
        assert len(both) == len(plain) * 5
        start, span, _ = plain[0]
        assert [(a - start, ln) for a, ln, _ in both[len(plain) : len(plain) + 4]] == [
            (0, span // 4),
            (span // 4, span // 4),
            (span // 2, span // 4),
            (span // 2 + span // 4, span // 4),
        ]

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

    def test_each_region_is_read_at_its_own_width(self):
        # Two logical regions of DIFFERENT widths, which is what tells the
        # builder's two per-region lookups -- the descriptor length and the
        # peer's page -- apart from a per-LAYER lookup. A draft model with the
        # same head count over a narrower head dimension is that shape: equal
        # bands, unequal bytes.
        w = self._worker(
            tp_rank=0,
            tp_size=1,
            areas=4,
            slices=4,
            n_logical=2,
            block_len=[512, 256],
        )
        meta = self._meta(
            areas=4, slices=4, n_logical=2, block_len=[256, 128], num_blocks=2
        )
        out = w._build_head_matched_remote(
            meta, remote_tp_rank=0, remote_tp_size=2, peer_areas=[0, 1]
        )
        # Two areas of each logical region, two blocks, two pieces each: our
        # area holds two heads and the peer's one. The piece length is our
        # region's width over the split, so it follows the region, not the
        # layer index.
        assert len(out) == 16
        lens = [ln for _, ln, _ in out]
        assert sorted(set(lens)) == [128, 256]
        assert lens.count(256) == lens.count(128) == 8
        # And the peer's own page width per region, which is the stride from one
        # block to the next and the divisor the head offset is read in: our
        # logical region 1 reads the peer's regions 4..7, at 128B each. Taking
        # the width by layer index instead lands those descriptors at 256B
        # strides -- outside the regions they name.
        assert self._decoded(out, meta) == Counter(
            {(region, block, 0): 1 for region in range(8) for block in range(2)}
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

    @pytest.mark.parametrize(
        ("tp_size", "local", "peer", "remote_tp_size", "tp_ratio", "expected"),
        [
            # Peer TP4 holds 2 of the 8 heads and replicates each over 2 areas,
            # so its slice is finer than ours and our region splits.
            (1, (4, 4, 256), (4, 2, 128), 4, -4, 2),
            # The same pair from the other end: we are the finer side, so the
            # peer's slice covers ours whole and an offset suffices.
            (4, (4, 2, 128), (4, 4, 256), 1, 4, 1),
        ],
    )
    def test_the_plan_reads_both_geometries(
        self, tp_size, local, peer, remote_tp_size, tp_ratio, expected
    ):
        # (areas, slices, block length) a side; a head is 128B on both.
        w = self._worker(
            tp_rank=0,
            tp_size=tp_size,
            areas=local[0],
            slices=local[1],
            n_logical=1,
            block_len=local[2],
        )
        w.use_host_buffer = False
        window_mode(w, None, has_swa=False)
        w.transfer_topo.tp_ratio.return_value = tp_ratio
        meta = self._meta(areas=peer[0], slices=peer[1], n_logical=1, block_len=peer[2])
        _, pieces = w._head_matched_plan(
            meta, 0, remote_tp_size, peer_areas=[0] if tp_ratio < 0 else None
        )
        assert {len(p) for p in pieces} == {expected}

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
        # so only half of it is ours to read; a longer local region holds wider
        # heads, which would run past its end. _validate_head_matched_handshake
        # rejects the same geometry from the advertised lengths, before any
        # descriptor is built.
        w = self._worker(
            tp_rank=1, tp_size=4, areas=4, slices=2, n_logical=1, block_len=384
        )
        meta = self._meta(areas=4, slices=2, n_logical=1, block_len=512)
        with pytest.raises(RuntimeError, match="a KV head is 384B .* and 128B"):
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
    def _worker(n_logical=1, cls=None):
        # Each layer over 4 chiplet areas, all 8 heads, TP1: our area holds 2
        # heads. Wired for the local dlist as well as the remote one, since the
        # contract here is that the two are emitted in the same order. The write
        # direction is asked for by name, because a peer copy is only fanned out
        # to when this side originates the bytes.
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
        if cls is not None:
            w.__class__ = cls
            # __init__ never ran, so the writer state shutdown() reaches through
            # __del__ is absent; silence it rather than leak an unraisable at GC.
            w.shutdown = lambda: None
        w.transfer_topo.is_kv_layout_blocks_first = False
        w.nixl_wrapper = MagicMock()
        w._shard_descs_per_block = {}
        w._shard_chunk_grids = {}
        w._borrowed_src_handles = set()
        return w

    def test_local_and_remote_descriptors_pair_up_by_index(self):
        from tests.vllm.distributed.kv_connector.utils import decode

        w = self._worker()
        # A peer at TP2 cuts its heads one per area, so each of our two-head
        # areas is read in two pieces from two of its regions.
        meta = TestHeadBandMatching._meta(
            areas=4, slices=4, n_logical=1, block_len=256, num_blocks=2
        )
        remote = w._build_head_matched_remote(
            meta, remote_tp_rank=0, remote_tp_size=2, peer_areas=[0, 1]
        )
        region_ids, pieces = w._head_matched_plan(meta, 0, 2, peer_areas=[0, 1])
        _, local = w._register_shard_local_xfer_handler(
            w.block_size, ("l0",), region_ids=region_ids, pieces=pieces
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

    def test_pieces_and_copies_nest_the_same_way_on_both_sides(self):
        # Both cases here run at one copy per piece, where the two innermost
        # loops collapse into one and their relative nesting is unasserted. A
        # write to a peer that replicates each slice has both: our region is cut
        # into pieces, and every piece goes to every copy.
        from tests.vllm.distributed.kv_connector.utils import decode

        w = self._worker(cls=RblnNixlPushConnectorWorker)
        meta = TestHeadBandMatching._meta(
            areas=self.AREAS, slices=2, n_logical=1, block_len=256, num_blocks=2
        )
        # The peer is finer, so our band spreads over several of its ranks and
        # only the area whose heads rank 0 owns goes to it (_fan_in_peer_areas).
        remote = w._build_head_matched_remote(
            meta, remote_tp_rank=0, remote_tp_size=4, peer_areas=[0]
        )
        region_ids, pieces = w._head_matched_plan(meta, 0, 4, peer_areas=[0])
        _, local = w._register_shard_local_xfer_handler(
            w.block_size, ("l0",), region_ids=region_ids, pieces=pieces
        )
        # 1 area x 2 blocks x 2 pieces x 2 copies.
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
        # The peer's rank 0 holds our area 0's two heads one per slice, each
        # duplicated over two of its areas: head 0 is its regions 0 and 1, head
        # 1 its 2 and 3. Counting the copies is not enough -- swapping the piece
        # and copy loops on either side keeps every count while sending head 1
        # to head 0's copy.
        assert set(pairs) == {
            ((0, 0, 0), (0, 0, 0)),
            ((0, 0, 0), (1, 0, 0)),
            ((0, 0, 1), (2, 0, 0)),
            ((0, 0, 1), (3, 0, 0)),
            ((0, 1, 0), (0, 1, 0)),
            ((0, 1, 0), (1, 1, 0)),
            ((0, 1, 1), (2, 1, 0)),
            ((0, 1, 1), (3, 1, 0)),
        }

    def test_two_layers_pair_within_their_own_layer(self):
        # Two layers, so the layer axis can be reordered at all: reversing
        # either side's layer loop then sends layer 0's pieces at layer 1's
        # regions and the pairing changes.
        from tests.vllm.distributed.kv_connector.utils import decode

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
        region_ids, pieces = w._head_matched_plan(meta, 0, 2, peer_areas=[0, 1])
        _, local = w._register_shard_local_xfer_handler(
            w.block_size, ("l0", "l1"), region_ids=region_ids, pieces=pieces
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


# The two descriptor sweeps below decode every row back to what it covers, from
# the layout alone: a region's block is K then V where they share one
# (`kv_per_block` 2), each a run of heads, each head `_TOKENS` tokens of the
# layer's token width.
_HEADS = 8
_TOKENS = 16
_TOKEN_BYTES = 4


class _Unit(NamedTuple):
    """One token of one head in one half of one block of one logical region."""

    region: int
    block: int
    half: int
    head: int
    token: int


class _RegionLayout(NamedTuple):
    """What one registered region holds, in the terms a sweep varies."""

    logical: int
    heads: tuple[int, ...]
    token_bytes: int
    kv_per_block: int
    #: The block's tokens it holds: all of them on a head cut, one area's
    #: range on a context cut.
    tokens: tuple[int, int] = (0, _TOKENS)

    @property
    def num_tokens(self) -> int:
        return self.tokens[1] - self.tokens[0]

    @property
    def block_len(self) -> int:
        return self.kv_per_block * len(self.heads) * self.num_tokens * self.token_bytes


def _region_bases(layouts, num_blocks):
    """Bases spaced past the widest region, so an address names one region."""
    span = max(lay.block_len for lay in layouts) * num_blocks
    return [span * (i + 1) for i in range(len(layouts))]


def _decode_units(rows, *, bases, layouts, num_blocks):
    """(region, the units it covers) for each descriptor row."""
    out = []
    for addr, length, _dev in np.asarray(rows).tolist():
        hits = [
            r
            for r, base in enumerate(bases)
            if base <= addr < base + num_blocks * layouts[r].block_len
        ]
        assert len(hits) == 1, f"address {addr:#x} lands in {len(hits)} regions"
        region = hits[0]
        lay = layouts[region]
        head_bytes = lay.num_tokens * lay.token_bytes
        half_bytes = len(lay.heads) * head_bytes
        block, off = divmod(addr - bases[region], lay.block_len)
        half, off = divmod(off, half_bytes)
        head_ix, off = divmod(off, head_bytes)

        def unit(half, head_ix, token, lay=lay, block=block):
            return _Unit(
                lay.logical, block, half, lay.heads[head_ix], lay.tokens[0] + token
            )

        if length > half_bytes:
            # Whole halves: a whole packed block.
            assert off == 0 and head_ix == 0 and length % half_bytes == 0
            assert half + length // half_bytes <= lay.kv_per_block
            units = [
                unit(half + k, h, t)
                for k in range(length // half_bytes)
                for h in range(len(lay.heads))
                for t in range(lay.num_tokens)
            ]
        elif length >= head_bytes:
            # A band of whole heads, every token of each.
            assert off == 0 and length % head_bytes == 0, (addr, length)
            assert head_ix + length // head_bytes <= len(lay.heads)
            units = [
                unit(half, head_ix + h, t)
                for h in range(length // head_bytes)
                for t in range(lay.num_tokens)
            ]
        else:
            # Some tokens of one head.
            assert off % lay.token_bytes == 0 and length % lay.token_bytes == 0
            first, n = off // lay.token_bytes, length // lay.token_bytes
            assert first + n <= lay.num_tokens
            units = [unit(half, head_ix, t) for t in range(first, first + n)]
        out.append((region, tuple(units)))
    return out


def _covers(rows, tag, *, regions, heads, halves, blocks, tokens, times):
    """`rows` name exactly these units, each `times` over, and nothing else.

    `heads` is one set for every region, or a dict of each region's own.
    """
    got = Counter(u for _, units in rows for u in units)
    heads_of = heads if isinstance(heads, dict) else dict.fromkeys(regions, heads)
    want = {
        _Unit(region, block, half, head, token)
        for region in regions
        for block in blocks
        for half in halves
        for head in heads_of[region]
        for token in tokens
    }
    assert set(got) == want, f"({tag}) the rows name other units than asked for"
    assert set(got.values()) == {times}, (
        f"({tag}) a unit is named {sorted(set(got.values()))} times, not {times}"
    )


def _chunk_tokens(case, chunk_span):
    """The tokens chunks `[lo, hi)` of a block hold."""
    lo, hi = chunk_span
    width = _TOKENS // case["chunks"]
    return range(lo * width, hi * width)


def _assert_shard_contract(w, case, *, peer_key, mine, peers, grid, fanout, **units):
    """A, B and C for one peer's two lists and the shard index arithmetic.

    - (A) row i of ours and row i of the peer's name the same units
    - (B) the rows the arithmetic picks for a block, or for chunks of one,
      name that block's units, or those chunks', on both lists
    - (C) each range names our units once per peer copy, and a peer
      region's once

    A sees the two lists only against each other, so a mistake both make
    alike -- a chunk range walked chunk-major, say -- passes it and fails B.
    `grid` is `(runs, chunks)`, the runs one count or one per region of ours,
    each region holding as many whole-block rows.
    """
    assert len(mine) == len(peers), "(A) the two lists differ in length"
    for i, ((_, ours), (_, theirs)) in enumerate(zip(mine, peers)):
        assert ours == theirs, f"(A) row {i} pairs different units"

    num_blocks = case["num_blocks"]

    def covers(rows, tag, **where):
        _covers(rows, tag, times=fanout, **units, **where)

    if grid is None:
        whole = len(mine)
    else:
        runs, chunks = grid
        runs = [runs] if isinstance(runs, int) else list(runs)
        whole = len(mine) * len(runs) // (len(runs) + chunks * sum(runs))
    for lo, hi in [(0, whole)] + ([(whole, len(mine))] if grid else []):
        covers(mine[lo:hi], "C", blocks=range(num_blocks), tokens=range(_TOKENS))
        named = Counter((r, u) for r, us in peers[lo:hi] for u in us)
        assert set(named.values()) == {1}, "(C) a peer region's unit is named twice"

    engine, rank = peer_key
    for block in range(num_blocks):
        ids = w._get_block_descs_ids_for_shard(engine, rank, num_blocks, [[block]])
        for rows in (mine, peers):
            covers([rows[i] for i in ids], "B", blocks=[block], tokens=range(_TOKENS))
        if grid is None:
            continue
        chunks = case["chunks"]
        for span in [(c, c + 1) for c in range(chunks)] + [(0, chunks)]:
            ids = w._chunk_descs_ids_for_shard(engine, rank, num_blocks, block, span)
            for rows in (mine, peers):
                covers(
                    [rows[i] for i in ids],
                    "B",
                    blocks=[block],
                    tokens=_chunk_tokens(case, span),
                )


def _shard_worker(w, *, layouts, num_blocks, chunked):
    """A hand-built worker wired for `_register_shard_xfer_state` over `layouts`."""
    set_shape(w, chunk_mode=chunked)
    w.engine_id = "eng"
    w.block_size = _TOKENS
    w.num_blocks = num_blocks
    w.device_id = 0
    w._has_mamba = False
    w.nixl_memory_type = "DRAM"
    w.num_regions = len(layouts)
    w.block_len_per_layer = [lay.block_len for lay in layouts]
    w.kv_caches_base_addr = {"eng": {w.tp_rank: _region_bases(layouts, num_blocks)}}
    w.kv_cache_config = MagicMock(transfer_groups=[MagicMock()])
    w.nixl_wrapper = MagicMock()
    w.src_xfer_handles_by_remote = {}
    w._shard_region_group_ids = {}
    w._shard_descs_per_block = {}
    w._shard_chunk_grids = {}
    w._shard_region_spans = {}
    w._borrowed_src_handles = set()
    w.use_host_buffer = False
    if not hasattr(w, "_logical_region_geometry"):
        # One head-cut region per layout, as a single-area region is.
        fill_region_geometry(w, areas=1, axes=[ChipletSplitAxis.HEAD] * len(layouts))
    if w._writes_into_peer:
        # __init__ never ran, so the writer state shutdown() reaches through
        # __del__ is absent.
        w.shutdown = lambda: None
    return w


def _as_grid(grid, regions):
    """The sweep's one `(runs, chunks)` for every region of a shard, as the
    `ChunkGrid` the shard lists read."""
    return None if grid is None else uniform_grid(*grid, regions=regions)


class _HeadSide(NamedTuple):
    """One side of a head-matched pairing, in the terms the sweep varies."""

    tp_rank: int
    tp_size: int
    areas: int
    slices: int
    # Each logical region's model KV heads; a draft beside its target has its own.
    kv_heads: tuple[int, ...] = (_HEADS, _HEADS)

    @property
    def heads_per_slice(self) -> int:
        return self.per_slice(0)

    def per_slice(self, logical: int) -> int:
        return self.kv_heads[logical] // (self.tp_size * self.slices)

    def heads_of(self, area: int, logical: int = 0) -> tuple[int, ...]:
        """The model heads chiplet `area` of this rank holds in `logical`.

        Ranks cut the heads into contiguous bands, a rank's slices cut its band
        again, and the `areas // slices` areas of one slice hold it alike.
        """
        slice_ = area // (self.areas // self.slices)
        width = self.per_slice(logical)
        first = (self.tp_rank * self.slices + slice_) * width
        return tuple(range(first, first + width))

    def layouts(self, kv_per_block):
        return [
            _RegionLayout(
                logical, self.heads_of(a, logical), _TOKEN_BYTES, kv_per_block
            )
            for logical in range(len(self.kv_heads))
            for a in range(self.areas)
        ]


class TestHeadMatchedDescriptorSweep:
    """The head-matched lists against what their rows mean, over many shapes.

    The contract is three things that have to agree: what each of our rows
    names, what each of the peer's rows names, and which rows the index
    arithmetic picks for a block or a chunk (`_assert_shard_contract`).
    Nothing here reads a band, a split or a grid off the code under test.
    """

    @staticmethod
    def _shapes():
        """Shapes the handshake accepts, each with the peer ranks we pair with.

        Unequal TP in both directions, at commensurate cuts and with none of
        our areas straddling two peer ranks; the refusals have their own tests.
        Each also with a draft beside the target at six times its KV heads, so
        a chunk of the draft's region is more runs than one of the target's.
        """
        locals_ = [
            _HeadSide(0, 1, 2, 2, kv_heads)
            for kv_heads in ((_HEADS, _HEADS), (_HEADS, 6 * _HEADS))
        ] + [
            _HeadSide(*side, kv_heads)
            for side in ((0, 1, 4, 4), (0, 2, 2, 2), (1, 2, 2, 2))
            for kv_heads in ((_HEADS, _HEADS), (_HEADS, 6 * _HEADS))
        ]
        peers = [  # (tp, slices, replicas)
            (1, 1, 1),
            (1, 2, 1),
            (1, 2, 2),
            (2, 1, 1),
            (2, 2, 1),
            (2, 2, 2),
            (4, 1, 1),
            (4, 1, 2),
            (4, 2, 1),
        ]
        out = []
        for local in locals_:
            ours = {h for a in range(local.areas) for h in local.heads_of(a)}
            for tp_r, slices_r, replicas_r in peers:
                cuts_l, cuts_r = local.tp_size * local.slices, tp_r * slices_r
                if (
                    tp_r == local.tp_size
                    or max(cuts_l, cuts_r) % min(cuts_l, cuts_r)
                    or local.heads_per_slice > _HEADS // tp_r
                ):
                    continue
                ranks = [
                    _HeadSide(t, tp_r, slices_r * replicas_r, slices_r, local.kv_heads)
                    for t in range(tp_r)
                ]
                paired = [
                    r
                    for r in ranks
                    if ours & {h for a in range(r.areas) for h in r.heads_of(a)}
                ]
                for direction, kv_per_block, chunks in itertools.product(
                    ("pull", "push"), (1, 2), (None, 2, 4)
                ):
                    out.append(
                        dict(
                            direction=direction,
                            kv_per_block=kv_per_block,
                            local=local,
                            peer=paired,
                            num_blocks=3,
                            chunks=chunks,
                        )
                    )
        return out

    @staticmethod
    def _expected(case, peer):
        """(peer copies, grid, our areas it serves) from the shape."""
        local = case["local"]
        theirs = {h for a in range(peer.areas) for h in peer.heads_of(a)}
        areas = [a for a in range(local.areas) if set(local.heads_of(a)) <= theirs]
        # A piece is the part of our area one peer slice holds.
        split = max(1, local.heads_per_slice // peer.heads_per_slice)
        fanout = peer.areas // peer.slices if case["direction"] == "push" else 1
        # Equal cuts move whole blocks; unequal ones read a band out of each
        # half, so the halves come apart.
        same_cuts = local.tp_size * local.slices == peer.tp_size * peer.slices
        halves = 1 if same_cuts else case["kv_per_block"]
        grid = None
        if case["chunks"] is not None:
            # A token range is one run per head of the piece, in each half the
            # piece still holds -- per region, since a draft's piece holds its
            # own heads. Our regions run logical-major over the areas served.
            runs = [
                case["kv_per_block"] // halves * (local.per_slice(logical) // split)
                for logical in range(len(local.kv_heads))
                for _ in areas
            ]
            grid = (runs, case["chunks"])
        peer_areas = None if len(areas) == local.areas else areas
        return fanout, grid, peer_areas

    def _check(self, case):
        """Assert the contract for every peer rank of `case`."""
        local_side = case["local"]
        layouts = local_side.layouts(case["kv_per_block"])
        w = TestHeadBandMatching._worker(
            tp_rank=local_side.tp_rank,
            tp_size=local_side.tp_size,
            areas=local_side.areas,
            slices=local_side.slices,
            n_logical=2,
            block_len=[layouts[lg * local_side.areas].block_len for lg in range(2)],
            kv_heads=list(local_side.kv_heads),
        )
        w.__class__ = (
            RblnNixlPullConnectorWorker
            if case["direction"] == "pull"
            else RblnNixlPushConnectorWorker
        )
        w._kv_per_block = case["kv_per_block"]
        w.local_seen_layer_names = ["l0", "l1"]
        _shard_worker(
            w,
            layouts=layouts,
            num_blocks=case["num_blocks"],
            chunked=case["chunks"] is not None,
        )
        for peer in case["peer"]:
            fanout, grid, peer_areas = self._expected(case, peer)
            peer_layouts = peer.layouts(case["kv_per_block"])
            meta = _agent_meta(
                kv_areas=peer.areas,
                kv_slices=peer.slices,
                num_blocks=case["num_blocks"],
                block_size=_TOKENS,
                kv_caches_base_addr=_region_bases(peer_layouts, case["num_blocks"]),
                block_lens=[lay.block_len for lay in peer_layouts],
            )
            # The grid is the shape's: deriving it is not what these lists are
            # held to here.
            chunk_grid = (
                None
                if grid is None
                else ChunkGrid(grid[1], tuple((0, r, grid[1]) for r in grid[0]))
            )
            with patch.object(type(w), "_shard_chunk_grid", return_value=chunk_grid):
                remote = w._build_head_matched_remote(
                    meta,
                    remote_tp_rank=peer.tp_rank,
                    remote_tp_size=peer.tp_size,
                    peer_areas=peer_areas,
                )
                w._register_shard_xfer_state(
                    "peer",
                    peer.tp_rank,
                    _TOKENS,
                    ("l0", "l1"),
                    peer_areas=peer_areas,
                    plan=w._head_matched_plan(
                        meta, peer.tp_rank, peer.tp_size, peer_areas=peer_areas
                    ),
                    remote_tp_size=peer.tp_size,
                )
            local = w.nixl_wrapper.get_xfer_descs.call_args.args[0]
            _assert_shard_contract(
                w,
                case,
                peer_key=("peer", peer.tp_rank),
                mine=_decode_units(
                    local,
                    bases=w.kv_caches_base_addr["eng"][local_side.tp_rank],
                    layouts=layouts,
                    num_blocks=case["num_blocks"],
                ),
                peers=_decode_units(
                    remote,
                    bases=meta.kv_caches_base_addr,
                    layouts=peer_layouts,
                    num_blocks=case["num_blocks"],
                ),
                grid=grid,
                fanout=fanout,
                regions=range(2),
                heads={
                    logical: sorted(
                        h
                        for a in (peer_areas or range(local_side.areas))
                        for h in local_side.heads_of(a, logical)
                    )
                    for logical in range(2)
                },
                halves=range(case["kv_per_block"]),
            )

    def test_every_shape_pairs_and_picks_what_it_names(self):
        for case in self._shapes():
            try:
                assert case["peer"], "no peer rank holds our heads"
                self._check(case)
            except AssertionError as e:
                raise AssertionError(f"{case}: {e}") from e


class TestAHeadCutMeetsAContextCutAcrossTp:
    """A head cut on one side, a context cut on the other, at unequal TP.

    One logical region over four areas. The coarser side holds a rank's whole
    band, cut by head; the finer side's rank holds fewer heads than it has
    areas, so its areas cut the block's tokens. What has to hold is the
    sweep's contract read cell by cell: row i of ours and row i of the peer's
    name the same (K/V, head, token) cells, every cell of ours is named once,
    no cell of the peer's twice, and the rows a block's ids pick are that
    block's -- or a chunk's, on the side whose areas the pieces do not split.
    """

    AREAS = 4
    NUM_BLOCKS = 3

    @classmethod
    def _layouts(cls, axis, *, tp_rank, tp_size, total, kv):
        per_rank = total // tp_size
        first = tp_rank * per_rank
        span = _TOKENS // cls.AREAS
        if axis is ChipletSplitAxis.HEAD:
            per = per_rank // cls.AREAS
            return [
                _RegionLayout(
                    0,
                    tuple(range(first + a * per, first + (a + 1) * per)),
                    _TOKEN_BYTES,
                    kv,
                )
                for a in range(cls.AREAS)
            ]
        return [
            _RegionLayout(
                0,
                tuple(range(first, first + per_rank)),
                _TOKEN_BYTES,
                kv,
                (a * span, (a + 1) * span),
            )
            for a in range(cls.AREAS)
        ]

    @classmethod
    def _latent(cls):
        """An indexer's key over the four areas, cut on tokens on either side
        and the same on every rank."""
        span = _TOKENS // cls.AREAS
        return [
            _RegionLayout(1, (0,), _TOKEN_BYTES, 1, (a * span, (a + 1) * span))
            for a in range(cls.AREAS)
        ]

    @classmethod
    def _run(
        cls,
        *,
        direction,
        total,
        kv,
        local_rank=0,
        coarse_tp=1,
        chunks=None,
        stream=False,
        latent=False,
    ):
        """Our lists against every peer rank's, as `(mine, peers)` per rank.

        `latent` adds a second logical region, an indexer beside the layer.
        """
        # Pull is TP4 rank `local_rank`, context-cut, reading the head cut at
        # `coarse_tp`, and pull_coarse a head-cut rank reading the TP4 ranks it
        # meets; push is a head-cut rank writing into them, and push_ctx a TP4
        # rank writing into the head-cut rank it meets.
        group = 4 // coarse_tp
        if direction in ("push", "pull_coarse"):
            ours = (ChipletSplitAxis.HEAD, local_rank, coarse_tp)
            theirs = (ChipletSplitAxis.TOKEN, 4)
            peer_ranks = range(local_rank * group, (local_rank + 1) * group)
            ratio = -group
        else:
            ours = (ChipletSplitAxis.TOKEN, local_rank, 4)
            theirs = (ChipletSplitAxis.HEAD, coarse_tp)
            peer_ranks, ratio = (
                range(local_rank // group, local_rank // group + 1),
                group,
            )
        axis_l, rank_l, tp_l = ours
        axis_r, tp_r = theirs
        attention = cls._layouts(
            axis_l, tp_rank=rank_l, tp_size=tp_l, total=total, kv=kv
        )
        layouts = attention + (cls._latent() if latent else [])
        names = ("l0", "l1") if latent else ("l0",)
        w = TestHeadBandMatching._worker(
            tp_rank=rank_l,
            tp_size=tp_l,
            areas=cls.AREAS,
            slices=cls.AREAS,
            n_logical=len(names),
            block_len=[layouts[i * cls.AREAS].block_len for i in range(len(names))],
            kv_heads=total,
        )
        w.__class__ = (
            RblnNixlPullConnectorWorker
            if direction.startswith("pull")
            else RblnNixlPushConnectorWorker
        )
        w._kv_per_block = kv
        w.use_mla = False
        w.local_seen_layer_names = list(names)
        fill_region_geometry(
            w,
            areas=cls.AREAS,
            axes=[axis_l] + [ChipletSplitAxis.TOKEN] * latent,
            slices=cls.AREAS,
            latent=[False] + [True] * latent,
        )
        if latent:
            w._logical_region_kv_heads[1] = None
            w._logical_region_local_heads[1] = 1
        w.transfer_topo.tp_ratio.side_effect = lambda r: {tp_r: ratio}[r]
        _shard_worker(
            w, layouts=layouts, num_blocks=cls.NUM_BLOCKS, chunked=chunks is not None
        )
        if stream:
            set_shape(w, streams_prefix=True)
        w.vllm_config = mock_vllm_config()
        w.vllm_config.scheduler_config.max_num_batched_tokens = (
            _TOKENS // cls.AREAS // (chunks or 1)
        )
        out = {}
        for rank in peer_ranks:
            peer_layouts = cls._layouts(
                axis_r, tp_rank=rank, tp_size=tp_r, total=total, kv=kv
            ) + (cls._latent() if latent else [])
            meta = _agent_meta(
                kv_areas=cls.AREAS,
                kv_split_axes=[axis_r] + [ChipletSplitAxis.TOKEN] * latent,
                region_names=[name for name in names for _ in range(cls.AREAS)],
                num_blocks=cls.NUM_BLOCKS,
                block_size=_TOKENS,
                kv_caches_base_addr=_region_bases(peer_layouts, cls.NUM_BLOCKS),
                block_lens=[lay.block_len for lay in peer_layouts],
            )
            w._check_split_axis_constraints(meta, tp_r)
            areas = w._fan_in_peer_areas(rank, tp_r)
            remote = w._build_head_matched_remote(
                meta, remote_tp_rank=rank, remote_tp_size=tp_r, peer_areas=areas
            )
            w._register_shard_xfer_state(
                "peer",
                rank,
                _TOKENS,
                names,
                peer_areas=areas,
                plan=w._head_matched_plan(meta, rank, tp_r, peer_areas=areas),
                remote_tp_size=tp_r,
            )
            local = w.nixl_wrapper.get_xfer_descs.call_args.args[0]
            mine = _decode_units(
                local,
                bases=w.kv_caches_base_addr["eng"][rank_l],
                layouts=layouts,
                num_blocks=cls.NUM_BLOCKS,
            )
            peers = _decode_units(
                remote,
                bases=meta.kv_caches_base_addr,
                layouts=peer_layouts,
                num_blocks=cls.NUM_BLOCKS,
            )
            # A rank of the finer side holds a band of the layer's heads, and
            # every rank the whole latent, which the side sending it sends from
            # the first of its ranks meeting one rank of the other side.
            sender = {"push_ctx": rank_l, "pull_coarse": rank}.get(direction, 0)
            served = [attention[a] for a in (areas or range(cls.AREAS))]
            if latent and sender % group == 0:
                served += cls._latent()
            out[rank] = (w, mine, peers, served)
        return out

    @staticmethod
    def _units(layouts, blocks, tokens=None):
        return {
            _Unit(lay.logical, b, half, head, t)
            for lay in layouts
            for b in blocks
            for half in range(lay.kv_per_block)
            for head in lay.heads
            for t in range(*lay.tokens)
            if tokens is None or t in tokens
        }

    @pytest.mark.parametrize(
        "direction, rank_l, coarse_tp, total, kv, latent",
        [
            (direction, rank, coarse_tp, total, kv, latent)
            # At TP2 the coarser side has to hold four areas' worth of heads while a
            # finer rank still holds fewer than its areas: eight heads, two a rank.
            for coarse_tp, totals in ((1, (4, 8)), (2, (8,)))
            for total in totals
            for direction, ranks in (
                ("pull", range(4)),
                ("pull_coarse", range(coarse_tp)),
                ("push", range(coarse_tp)),
                ("push_ctx", range(4)),
            )
            for rank in ranks
            # Registration seats a latent beside packed K and V only.
            for kv, latent in ((1, False), (2, False), (2, True))
            # A TP2 coarser side and pull_coarse are there for the latent's sender.
            if latent or (coarse_tp == 1 and direction != "pull_coarse")
        ],
    )
    def test_every_cell_pairs_with_the_same_cell(
        self, direction, rank_l, coarse_tp, total, kv, latent
    ):
        for rank, (w, mine, peers, served) in self._run(
            direction=direction,
            total=total,
            kv=kv,
            local_rank=rank_l,
            coarse_tp=coarse_tp,
            latent=latent,
        ).items():
            assert len(mine) == len(peers)
            for i, ((_, a), (_, b)) in enumerate(zip(mine, peers)):
                assert a == b, f"rank {rank} row {i} pairs different cells"
            named = Counter(u for _, us in mine for u in us)
            assert set(named) == self._units(served, range(self.NUM_BLOCKS))
            assert set(named.values()) == {1}
            assert set(Counter((r, u) for r, us in peers for u in us).values()) == {1}
            for block in range(self.NUM_BLOCKS):
                ids = w._get_block_descs_ids_for_shard(
                    "peer", rank, self.NUM_BLOCKS, [[block]]
                )
                for rows in (mine, peers):
                    assert {u for i in ids for u in rows[i][1]} == self._units(
                        served, [block]
                    )

    @pytest.mark.parametrize("rank_l", range(4))
    @pytest.mark.parametrize("direction", ["pull", "push_ctx"])
    def test_a_region_in_one_piece_pairs_with_the_whole_engine_list(
        self, direction, rank_l
    ):
        # One head a rank, split K/V: each region pairs in one piece, so a PP1
        # peer takes upstream's whole-engine list for ours instead of the
        # per-shard one, and row i of that list has to name row i's cells.
        ((w, _, peers, served),) = self._run(
            direction=direction, total=4, kv=1, local_rank=rank_l
        ).values()
        peer_layouts = self._layouts(
            ChipletSplitAxis.HEAD, tp_rank=0, tp_size=1, total=4, kv=1
        )
        meta = _agent_meta(
            kv_areas=self.AREAS,
            kv_split_axes=[ChipletSplitAxis.HEAD],
            num_blocks=self.NUM_BLOCKS,
            block_size=_TOKENS,
            kv_caches_base_addr=_region_bases(peer_layouts, self.NUM_BLOCKS),
            block_lens=[lay.block_len for lay in peer_layouts],
        )
        regions, pieces = w._head_matched_plan(meta, 0, 1, peer_areas=None)
        assert regions == list(range(w.num_regions))
        assert all(len(p) == 1 for p in pieces)
        w._transfer_layer_region_indices = None
        w.region_num_blocks = [self.NUM_BLOCKS] * w.num_regions
        w.block_stride_per_layer = w.block_len_per_layer
        bases = w.kv_caches_base_addr["eng"][rank_l]
        whole = _decode_units(
            w._build_fa_local(bases, 1).tolist(),
            bases=bases,
            layouts=served,
            num_blocks=self.NUM_BLOCKS,
        )

        assert [us for _, us in whole] == [us for _, us in peers]

    @pytest.mark.parametrize(
        "kv, latent", [(1, False), (2, True)], ids=["layer", "with_indexer"]
    )
    @pytest.mark.parametrize("total", [4, 8], ids=["one_head_a_rank", "two"])
    def test_a_chunk_of_the_context_cut_side_is_named_where_it_sits(
        self, total, kv, latent
    ):
        # Pulling into a context cut, a region's pieces all lie in its area's
        # token range, so one grid a region still names a chunk's cells. An
        # indexer's key is one run a block beside packed K and V.
        chunks = 2
        for rank, (w, mine, peers, served) in self._run(
            direction="pull", total=total, kv=kv, chunks=chunks, latent=latent
        ).items():
            width = _TOKENS // self.AREAS // chunks
            per_block = _TOKENS // width
            for block in range(self.NUM_BLOCKS):
                for c in range(per_block):
                    ids = w._chunk_descs_ids_for_shard(
                        "peer", rank, self.NUM_BLOCKS, block, (c, c + 1)
                    )
                    want = self._units(
                        served, [block], range(c * width, (c + 1) * width)
                    )
                    for rows in (mine, peers):
                        assert {u for i in ids for u in rows[i][1]} == want

    @pytest.mark.parametrize("total", [4, 8], ids=["one_head_a_rank", "two"])
    def test_a_streamed_write_from_the_head_cut_pairs_cell_for_cell(self, total):
        # A streamed write on its own sends whole blocks, so the pieces' token
        # ranges are named whole and the contract is the plain push's.
        for rank, (w, mine, peers, served) in self._run(
            direction="push", total=total, kv=1, stream=True
        ).items():
            assert [a for _, a in mine] == [b for _, b in peers]
            named = Counter(u for _, us in mine for u in us)
            assert set(named) == self._units(served, range(self.NUM_BLOCKS))
            assert set(named.values()) == {1}


class TestStageDescriptorSweep:
    """The equal-TP stage lists against what their rows mean.

    Upstream builds the peer's whole-block range and we add its chunk range
    (`_build_fa_remote`), cut at each row's own length. A draft layer's head is
    wider than the target's, so the rows of one list are not all one length and
    a length taken from the wrong row shows up as the wrong units.
    """

    LAYERS = ("l0", "l1", "l2", "l3")
    # The last layer is a draft whose heads are twice as wide.
    TOKEN_BYTES = (_TOKEN_BYTES, _TOKEN_BYTES, _TOKEN_BYTES, 2 * _TOKEN_BYTES)

    @classmethod
    def _shapes(cls):
        out = []
        for direction, kv_per_block, heads, pp, chunks in itertools.product(
            ("pull", "push"), (1, 2), (1, 2), (2, 4), (None, 2, 4)
        ):
            per_stage = len(cls.LAYERS) // pp
            out.append(
                dict(
                    direction=direction,
                    kv_per_block=kv_per_block,
                    heads=heads,
                    stages=[
                        cls.LAYERS[s * per_stage : (s + 1) * per_stage]
                        for s in range(pp)
                    ],
                    num_blocks=3,
                    chunks=chunks,
                )
            )
        return out

    @classmethod
    def _layouts(cls, case, layers):
        # Split K/V is a region each, packed is one region of both halves.
        rpl = 2 // case["kv_per_block"]
        return [
            _RegionLayout(
                cls.LAYERS.index(name) * rpl + k,
                tuple(range(case["heads"])),
                cls.TOKEN_BYTES[cls.LAYERS.index(name)],
                case["kv_per_block"],
            )
            for name in layers
            for k in range(rpl)
        ]

    def _check(self, case):
        layouts = self._layouts(case, self.LAYERS)
        w = TestPeerRegionView._consumer(uniform_lens=True)
        w.__class__ = (
            RblnNixlPullConnectorWorker
            if case["direction"] == "pull"
            else RblnNixlPushConnectorWorker
        )
        w._kv_per_block = case["kv_per_block"]
        w.tp_rank = 0
        w.local_seen_layer_names = list(self.LAYERS)
        w._region_is_mla = [False] * len(layouts)
        _shard_worker(
            w,
            layouts=layouts,
            num_blocks=case["num_blocks"],
            chunked=case["chunks"] is not None,
        )
        # A token range is one run per head, in each half a block holds.
        grid = (
            None
            if case["chunks"] is None
            else (case["kv_per_block"] * case["heads"], case["chunks"])
        )
        for rank, layers in enumerate(case["stages"]):
            peer_layouts = self._layouts(case, layers)
            meta = _agent_meta(
                engine_id="p",
                num_blocks=case["num_blocks"],
                block_size=_TOKENS,
                kv_caches_base_addr=_region_bases(peer_layouts, case["num_blocks"]),
                block_lens=[lay.block_len for lay in peer_layouts],
                region_names=[
                    name for name in layers for _ in range(2 // case["kv_per_block"])
                ],
                pp_rank=rank,
                pp_size=len(case["stages"]),
            )
            with patch.object(
                type(w),
                "_shard_chunk_grid",
                return_value=_as_grid(grid, len(peer_layouts)),
            ):
                remote = w._build_fa_remote(
                    TestPeerRegionView._plan(), meta, block_size_ratio=1
                )
                w._register_shard_xfer_state("p", rank, _TOKENS, tuple(layers))
            local = w.nixl_wrapper.get_xfer_descs.call_args.args[0]
            _assert_shard_contract(
                w,
                case,
                peer_key=("p", rank),
                mine=_decode_units(
                    local,
                    bases=w.kv_caches_base_addr["eng"][0],
                    layouts=layouts,
                    num_blocks=case["num_blocks"],
                ),
                peers=_decode_units(
                    remote,
                    bases=meta.kv_caches_base_addr,
                    layouts=peer_layouts,
                    num_blocks=case["num_blocks"],
                ),
                grid=grid,
                fanout=1,
                regions=sorted(lay.logical for lay in peer_layouts),
                heads=range(case["heads"]),
                halves=range(case["kv_per_block"]),
            )

    def test_every_shape_pairs_and_picks_what_it_names(self):
        for case in self._shapes():
            try:
                self._check(case)
            except AssertionError as e:
                raise AssertionError(f"{case}: {e}") from e


class TestChunkRangeSweep:
    """The whole-engine chunk range against the arithmetic that indexes it.

    Both whole-engine lists come out of `_chunk_range_descs`, so they cannot be
    held against each other; a mistake there is the same on both sides. What
    pins it is the layout and `_chunk_desc_ids`: every grid tiles each piece's
    block once, and the ids picked for chunks of a block are those chunks'
    bytes, run by run. The pieces differ in base, length, stride and device, so
    a length or a device taken from another row shows.
    """

    PIECES = [
        (0x1000, 256, 512, 0),
        (0x9000, 512, 512, 1),
        (0x20000, 1024, 2048, 0),
    ]

    @staticmethod
    def _shapes():
        return [
            dict(num_blocks=num_blocks, grid=grid, regions=regions)
            for num_blocks in (1, 3)
            for grid in ((1, 2), (2, 1), (2, 4), (4, 2))
            for regions in (1, 3)
        ]

    def _check(self, case):
        pieces = self.PIECES[: case["regions"]]
        num_blocks, (runs, chunks) = case["num_blocks"], case["grid"]
        rows = np.asarray(
            RblnNixlWorkerState._chunk_range_descs(
                pieces, num_blocks=num_blocks, grids=[case["grid"]] * len(pieces)
            )
        ).tolist()

        # (C) Each piece's block, tiled once, on that piece's device.
        tiles = defaultdict(list)
        for addr, length, dev in rows:
            hits = [
                (i, b)
                for i, (base, whole, stride, _) in enumerate(pieces)
                for b in range(num_blocks)
                if base + b * stride <= addr < base + b * stride + whole
            ]
            assert len(hits) == 1, f"(C) {addr:#x} is in {len(hits)} blocks"
            assert dev == pieces[hits[0][0]][3], "(C) a row on another device"
            tiles[hits[0]].append((addr, addr + length))
        for i, (base, whole, stride, _) in enumerate(pieces):
            for b in range(num_blocks):
                spans = sorted(tiles[(i, b)])
                start = base + b * stride
                assert spans[0][0] == start and spans[-1][1] == start + whole, "(C)"
                assert all(a[1] == b_[0] for a, b_ in zip(spans, spans[1:])), (
                    "(C) the rows of a block leave a gap or overlap"
                )

        # (B) The ids for chunks [lo, hi) of a block are those chunks' bytes in
        # every run of every piece. A run is `whole / runs` bytes and holds the
        # block's tokens once, so chunk c of it is its c-th `chunks`-th.
        for block in range(num_blocks):
            for span in [(c, c + 1) for c in range(chunks)] + [(0, chunks)]:
                ids = _chunk_desc_ids(
                    start=0,
                    cuts=[(0, runs, chunks)] * len(pieces),
                    num_blocks=num_blocks,
                    block_id=block,
                    per_block=1,
                    chunk_span=span,
                )
                picked = {(rows[i][0], rows[i][1]) for i in ids}
                want = {
                    (
                        base
                        + block * stride
                        + r * (whole // runs)
                        + c * (whole // runs // chunks),
                        whole // runs // chunks,
                    )
                    for base, whole, stride, _ in pieces
                    for r in range(runs)
                    for c in range(*span)
                }
                assert picked == want, "(B) the ids name other bytes than the chunks"
                assert len(ids) == len(want), "(B) a chunk picked twice"

    def test_every_shape_tiles_and_picks_its_chunks(self):
        for case in self._shapes():
            try:
                self._check(case)
            except AssertionError as e:
                raise AssertionError(f"{case}: {e}") from e

    @staticmethod
    def _mixed_shapes():
        # Per region `(first, runs, chunks)` of the block's chunks, as a context
        # cut beside a head cut lays them out: a region holding one chunk or
        # less has no runs, and its whole-block descriptor names it.
        return [
            dict(num_blocks=num_blocks, cuts=cuts)
            for num_blocks in (1, 3)
            for cuts in (
                ((0, 0, 1), (0, 2, 2), (1, 0, 1)),  # two areas a chunk
                ((0, 2, 1), (1, 2, 1), (0, 4, 2)),  # one area a chunk
                ((0, 1, 2), (2, 1, 2), (0, 2, 4)),  # an area two chunks
            )
        ]

    def _check_mixed(self, case):
        pieces, num_blocks, cuts = self.PIECES, case["num_blocks"], case["cuts"]
        chunks_per_block = max(first + chunks for first, _, chunks in cuts)
        whole = RblnNixlWorkerState._chunk_range_descs(
            pieces, num_blocks=num_blocks, grids=[(1, 1)] * len(pieces)
        )
        chunked = RblnNixlWorkerState._chunk_range_descs(
            pieces, num_blocks=num_blocks, grids=[(r, c) for _, r, c in cuts]
        )
        assert len(chunked) == num_blocks * sum(r * c for _, r, c in cuts), (
            "a region of no runs adds chunk rows"
        )
        rows = np.concatenate([whole, chunked]).tolist()
        for block, (lo, hi), partial_only in itertools.product(
            range(num_blocks),
            itertools.combinations(range(chunks_per_block + 1), 2),
            (False, True),
        ):
            ids = _chunk_desc_ids(
                start=len(whole),
                cuts=cuts,
                num_blocks=num_blocks,
                block_id=block,
                per_block=1,
                chunk_span=(lo, hi),
                partial_only=partial_only,
            )
            want = []
            for (base, length, stride, _), (first, runs, chunks) in zip(pieces, cuts):
                begin, end = max(lo, first) - first, min(hi, first + chunks) - first
                if begin >= end or (partial_only and begin == 0 and end == chunks):
                    continue
                start = base + block * stride
                if not runs:
                    want.append((start, length))
                    continue
                run = length // runs
                want += [
                    (start + r * run + c * (run // chunks), run // chunks)
                    for r in range(runs)
                    for c in range(begin, end)
                ]
            picked = sorted((rows[i][0], rows[i][1]) for i in ids)
            assert picked == sorted(want), (block, (lo, hi), partial_only)
            assert len(ids) == len(want), "a chunk picked twice"

    def test_a_region_of_no_runs_is_picked_by_its_whole_block(self):
        for case in self._mixed_shapes():
            try:
                self._check_mixed(case)
            except AssertionError as e:
                raise AssertionError(f"{case}: {e}") from e


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
        w._kv_per_block = 1
        w.tp_rank = tp_rank
        w._logical_region_kv_heads = [8]
        fill_region_geometry(
            w, areas=areas, axes=[ChipletSplitAxis.HEAD], slices=slices
        )
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

    def test_interleaved_replicas_follow_the_slice_their_layout_names(self):
        # Nothing promises the replicas of a slice sit side by side; the slice
        # each area holds is what decides its peer, not its position.
        w = self._worker(tp_rank=0, tp_size=4, areas=4, slices=2)
        (geometry,) = w._logical_region_geometry
        w._logical_region_geometry = [
            geometry._replace(
                areas=tuple(
                    place._replace(slice_id=a % 2)
                    for a, place in enumerate(geometry.areas)
                )
            )
        ]
        assert w._fan_in_peer_areas(0, remote_tp_size=8) == [0, 2]
        assert w._fan_in_peer_areas(1, remote_tp_size=8) == [1, 3]

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
        w._kv_per_block = 1
        # One K and one V region per layer, each expanded over `areas`.
        w.num_regions = n_layers * 2 * areas
        fill_region_geometry(
            w, areas=areas, axes=[ChipletSplitAxis.HEAD] * (n_layers * 2)
        )
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


class TestADecodeContextParallelPeerIsRefused:
    """The refusal has to see the peer's number, not the parameter default.

    `_validate_remote_agent_handshake` takes `remote_dcp_size` with a default
    of 1, so a caller that forgets it turns the guard off without changing a
    line of it -- which reads as protection while refusing nothing.
    """

    @staticmethod
    def _worker():
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
        w._shape = shape()
        w._remote_agents = {}
        w.dst_num_blocks = {}
        w.dst_region_num_blocks = {}
        w.dst_region_group_ids = {}
        w.dst_region_mem_types = {}
        w.dst_uses_region_group_mapping = {}
        w.region_group_ids = []
        w.nixl_memory_type = "VRAM"
        w.kv_caches_base_addr = defaultdict(dict)
        w.dst_xfer_side_handles = defaultdict(dict)
        w.nixl_wrapper = MagicMock()
        w.transfer_topo = MagicMock(tp_size=1)
        w._logical_region_kv_heads = [8]
        fill_region_geometry(w, areas=2, axes=[ChipletSplitAxis.HEAD], slices=2)
        w._is_head_matched_peer = MagicMock(return_value=True)
        w._register_remote_engine_prelude = MagicMock()
        w._check_d2d_region_pairing = MagicMock()
        w._check_split_axis_constraints = MagicMock()
        w._check_mla_constraints = MagicMock()
        w._build_head_matched_remote = MagicMock(return_value=[(0, 0, 0)])
        w._fan_in_peer_areas = MagicMock(return_value=None)
        return w

    def test_the_peers_size_reaches_the_refusal(self):
        w = self._worker()
        meta = _agent_meta(engine_id="peer", dcp_size=2)

        with pytest.raises(RuntimeError, match="decode-context-parallel"):
            w.add_remote_agent(meta, 0, 2, meta.dcp_size)

    def test_a_plain_peer_still_pairs(self):
        w = self._worker()
        w._validate_remote_agent_handshake = MagicMock()
        meta = _agent_meta(engine_id="peer")

        w.add_remote_agent(meta, 0, 2, meta.dcp_size)

        assert w._validate_remote_agent_handshake.call_args.args[2] == 1


class TestARefusedPeerLeavesNothingBehind:
    """The DCP refusal fires after the engine is registered with the topology.

    Making the refusal reachable made its ordering matter: the prelude writes
    an `EngineTransferInfo`, a TPMapping and the region ledger before the
    check, and a raise there never reaches `_cleanup_remote_engine` -- upstream
    swallows a handshake exception and only fails the request.
    """

    @staticmethod
    def _worker():
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
        w._shape = shape()
        w._remote_agents = {}
        w.dst_num_blocks = {}
        w.dst_region_num_blocks = {}
        w.dst_region_group_ids = {}
        w.dst_region_mem_types = {}
        w.dst_uses_region_group_mapping = {}
        w.region_group_ids = []
        w.nixl_memory_type = "VRAM"
        w.kv_caches_base_addr = defaultdict(dict)
        w.dst_xfer_side_handles = defaultdict(dict)
        w.nixl_wrapper = MagicMock()
        w.tp_mappings = {}
        w._group_spec_types = []
        w.transfer_topo = MagicMock(tp_size=1)
        w._logical_region_kv_heads = [8]
        fill_region_geometry(w, areas=1, axes=[ChipletSplitAxis.HEAD])
        w._is_head_matched_peer = MagicMock(return_value=True)
        w._check_d2d_region_pairing = MagicMock()
        w._check_split_axis_constraints = MagicMock()
        w._check_mla_constraints = MagicMock()
        w._build_head_matched_remote = MagicMock(return_value=[(0, 0, 0)])
        w._fan_in_peer_areas = MagicMock(return_value=None)
        return w

    def test_the_topology_never_sees_a_peer_it_refuses(self):
        w = self._worker()
        meta = _agent_meta(engine_id="peer", dcp_size=2)

        with pytest.raises(RuntimeError, match="decode-context-parallel"):
            w.add_remote_agent(meta, 0, 2, meta.dcp_size)

        w.transfer_topo.register_remote_engine.assert_not_called()
        assert w.tp_mappings == {}
        assert w.dst_num_blocks == {}
        assert w.kv_caches_base_addr == {}
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    @pytest.mark.parametrize("peer_slices, peer_cuts", [(2, 4), (1, 2)])
    def test_heads_cut_at_incommensurate_widths_are_refused_unregistered(
        self, peer_slices, peer_cuts
    ):
        # Three cuts here against a peer's finer four or coarser two: an area of
        # either side holds part of one of the other's, which no plan of like
        # pieces names.
        w = self._worker()
        # More areas than slices on both sides: the cut is the slices'.
        w._local_band = MagicMock(return_value=(6, 3, [0, 1, 2] * 2))
        w._peer_band = MagicMock(return_value=(4, peer_slices))
        meta = _agent_meta(engine_id="peer")

        with pytest.raises(
            RuntimeError, match=f"cuts KV heads 3 ways and the peer {peer_cuts},"
        ):
            w.add_remote_agent(meta, 0, 2, meta.dcp_size)

        w.transfer_topo.register_remote_engine.assert_not_called()
        assert w.dst_num_blocks == {}
        assert w.kv_caches_base_addr == {}
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    @pytest.mark.parametrize(
        "tp, areas, slices, peer_tp",
        [(1, 1, 1, 2), (1, 4, 2, 4), (2, 6, 3, 4)],
    )
    def test_an_area_straddling_finer_peers_is_refused_unregistered(
        self, tp, areas, slices, peer_tp
    ):
        # Commensurate cuts, but a peer rank holds less than one of our areas,
        # or (TP 2, 3 slices against TP 4) not a whole number of them.
        w = self._straddle_worker(tp, areas, slices, peer_tp)
        meta = _agent_meta(engine_id="peer")

        with pytest.raises(RuntimeError, match="straddle several peers"):
            w.add_remote_agent(meta, 0, peer_tp, meta.dcp_size)

        w.transfer_topo.register_remote_engine.assert_not_called()
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    @pytest.mark.parametrize(
        "tp, areas, slices, peer_tp",
        [(1, 4, 4, 2), (2, 8, 4, 4)],
    )
    def test_a_finer_peer_holding_several_whole_areas_registers(
        self, tp, areas, slices, peer_tp
    ):
        # The other side of the same condition: a peer rank holding two of our
        # areas is not a straddle.
        w = self._straddle_worker(tp, areas, slices, peer_tp)
        meta = _agent_meta(engine_id="peer")

        w.add_remote_agent(meta, 0, peer_tp, meta.dcp_size)

        w.transfer_topo.register_remote_engine.assert_called_once()

    def _straddle_worker(self, tp, areas, slices, peer_tp):
        # The real _fan_in_peer_areas, which refused a straddle after the peer
        # was registered.
        w = self._worker()
        del w._fan_in_peer_areas
        w._validate_remote_agent_handshake = MagicMock()
        w.use_host_buffer = False
        w.tp_rank = 0
        w.transfer_topo.tp_size = tp
        w.transfer_topo.total_num_kv_heads = 12 * tp
        w.transfer_topo.tp_ratio = MagicMock(return_value=-(peer_tp // tp))
        ids = [a * slices // areas for a in range(areas)]
        w._local_band = MagicMock(return_value=(areas, slices, ids))
        w._peer_band = MagicMock(return_value=(slices, slices))
        return w

    @pytest.mark.parametrize(
        "check",
        [
            "_check_d2d_region_pairing",
            "_check_split_axis_constraints",
            "_check_mla_constraints",
        ],
    )
    def test_the_layout_is_checked_before_the_peer_is_registered(self, check):
        # The head-cut check reads the peer's axes by our layout, so a peer laid
        # out otherwise has to be refused first, and before the prelude.
        w = self._worker()
        setattr(w, check, MagicMock(side_effect=RuntimeError("laid out otherwise")))
        meta = _agent_meta(engine_id="peer")

        with pytest.raises(RuntimeError, match="laid out otherwise"):
            w.add_remote_agent(meta, 0, 2, meta.dcp_size)

        w.transfer_topo.register_remote_engine.assert_not_called()
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    @pytest.mark.parametrize("peer_tp", [4, 3])
    def test_an_mla_peer_of_another_tp_is_refused_as_mla_unregistered(self, peer_tp):
        # Through the real check: refused by name ahead of the head-cut check,
        # which would refuse TP 2 against TP 3 under its own message.
        w = self._worker()
        del w._check_mla_constraints
        del w._is_head_matched_peer
        w.use_mla = True
        w._logical_region_kv_heads = [None]
        w._logical_region_latent = [True]
        w.use_host_buffer = False
        w.transfer_topo.tp_size = 2
        w.transfer_topo.tp_ratio = MagicMock(side_effect=lambda r: 1 if r == 2 else -r)
        meta = _agent_meta(engine_id="peer")

        with pytest.raises(RuntimeError, match="MLA is not supported"):
            w.add_remote_agent(meta, 0, peer_tp, meta.dcp_size)

        w.transfer_topo.register_remote_engine.assert_not_called()
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    def _mixed_worker(self):
        """A non-MLA engine at TP 2 of a head-cut attention region and a
        token-cut latent, with the real `_check_mla_constraints`."""
        w = self._worker()
        del w._check_mla_constraints
        w.use_mla = False
        w.use_host_buffer = False
        w.transfer_topo.tp_size = 2
        w.transfer_topo.tp_ratio = MagicMock(
            side_effect=lambda r: 2 // r if r <= 2 else -(r // 2)
        )
        fill_region_geometry(
            w,
            areas=4,
            axes=[ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN],
            slices=[2, 4],
        )
        w._logical_region_kv_heads = [8, None]
        w._logical_region_latent = [False, True]
        return w

    @pytest.mark.parametrize(
        ("ours_tp", "peer_tp"),
        [(4, 1), (1, 4)],
        ids=["finer_registers", "coarser_registers"],
    )
    @pytest.mark.parametrize(
        ("finer", "coarser"),
        [
            # A latent cut finer than the attention beside it.
            (
                ((ChipletSplitAxis.HEAD, 2), (ChipletSplitAxis.TOKEN, 4)),
                ((ChipletSplitAxis.HEAD, 4), (ChipletSplitAxis.TOKEN, 4)),
            ),
            # An uncut latent, recorded on HEAD with one slice.
            (
                ((ChipletSplitAxis.TOKEN, 4), (ChipletSplitAxis.HEAD, 1)),
                ((ChipletSplitAxis.HEAD, 4), (ChipletSplitAxis.HEAD, 1)),
            ),
        ],
        ids=["token_cut", "uncut"],
    )
    def test_a_latent_cut_unlike_its_attention_reaches_registration_both_ways(
        self, ours_tp, peer_tp, finer, coarser
    ):
        # Neither band counts a latent, which pairs by its own slices, so the
        # same two engines register each other in either direction.
        ours, theirs = (finer, coarser) if ours_tp > peer_tp else (coarser, finer)
        w = self._worker()
        del w._check_mla_constraints
        del w._check_split_axis_constraints
        del w._is_head_matched_peer
        w.use_mla = False
        w.use_host_buffer = False
        w.transfer_topo.tp_size = ours_tp
        w.transfer_topo.tp_ratio = MagicMock(
            return_value=ours_tp // peer_tp or -(peer_tp // ours_tp)
        )
        fill_region_geometry(
            w,
            areas=4,
            axes=[axis for axis, _ in ours],
            slices=[slices for _, slices in ours],
            latent=[False, True],
        )
        w.block_size = 16
        w._logical_region_kv_heads = [8, None]
        w.local_seen_layer_names = ["l0", "l1"]
        meta = _agent_meta(
            engine_id="peer",
            kv_areas=4,
            kv_split_axes=[axis for axis, _ in theirs],
            kv_area_slices=[area_slices(4, slices, 1)[0] for _, slices in theirs],
            region_names=["l0"] * 4 + ["l1"] * 4,
            block_lens=[8192] * 8,
            kv_caches_base_addr=[0x1000 * (i + 1) for i in range(8)],
        )

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=AssertionError("reached the prelude"),
            ),
            pytest.raises(AssertionError, match="reached the prelude"),
        ):
            w._add_remote_agent_head_matched(meta, 0, peer_tp, meta.dcp_size)

    def test_two_context_cuts_are_refused_by_their_axis_unregistered(self):
        # Through the real check, ahead of the head-cut check, which would
        # refuse TP 2 against TP 3 under its own message.
        w = self._worker()
        del w._check_split_axis_constraints
        del w._is_head_matched_peer
        fill_region_geometry(w, areas=1, axes=[ChipletSplitAxis.TOKEN])
        w.use_mla = False
        w.use_host_buffer = False
        w.transfer_topo.tp_size = 2
        w.transfer_topo.tp_ratio = MagicMock(side_effect=lambda r: 1 if r == 2 else -r)
        meta = _agent_meta(
            engine_id="peer", kv_split_axes=[ChipletSplitAxis.TOKEN], region_names=[]
        )

        with pytest.raises(RuntimeError, match="context-cut on both sides"):
            w.add_remote_agent(meta, 0, 3, meta.dcp_size)

        w.transfer_topo.register_remote_engine.assert_not_called()
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    def test_a_peer_of_another_kv_layout_is_refused_by_it_unregistered(self):
        # Through the real check, ahead of the head-cut check, which would
        # refuse TP 2 against TP 3 under its own message.
        w = self._worker()
        del w._check_d2d_region_pairing
        w.use_host_buffer = False
        w.transfer_topo.tp_size = 2
        w.transfer_topo.tp_ratio = MagicMock(side_effect=lambda r: 1 if r == 2 else -r)
        meta = _agent_meta(engine_id="peer", kv_per_block=2)

        with pytest.raises(RuntimeError, match="use_custom_kernel"):
            w.add_remote_agent(meta, 0, 3, meta.dcp_size)

        w.transfer_topo.register_remote_engine.assert_not_called()
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    def test_a_sliding_window_engine_refuses_another_tp_unregistered(self):
        w = self._worker()
        del w._check_d2d_region_pairing
        w._shape = shape(has_swa=True)
        w._window_grid_cut = None
        w.block_len_per_layer = [8192]
        w.local_seen_layer_names = ["layer.0"]
        w.num_regions = 1
        w.use_host_buffer = False
        w.transfer_topo.tp_size = 2
        w.transfer_topo.tp_ratio = MagicMock(side_effect=lambda r: 1 if r == 2 else -r)
        meta = _agent_meta(engine_id="peer", region_names=[])

        with pytest.raises(RuntimeError, match="sliding-window"):
            w.add_remote_agent(meta, 0, 4, meta.dcp_size)

        w.transfer_topo.register_remote_engine.assert_not_called()
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    def test_a_context_cut_pairs_a_head_cut_of_any_count(self):
        # Six KV heads: our TP3 context cut holds two a rank, and the TP1 peer
        # cuts them into two slices of three. Neither count divides the other,
        # yet every piece is one head over our token range. Bands from geometry.
        w = self._worker()
        del w._check_split_axis_constraints
        w.use_host_buffer = False
        w.transfer_topo.tp_size = 3
        w._logical_region_kv_heads = [6]
        fill_region_geometry(w, areas=4, axes=[ChipletSplitAxis.TOKEN], slices=4)
        meta = _agent_meta(
            engine_id="peer",
            kv_areas=4,
            kv_slices=2,
            kv_split_axes=[ChipletSplitAxis.HEAD],
            region_names=[],
        )

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=RuntimeError("past the cut check"),
            ),
            pytest.raises(RuntimeError, match="past the cut check"),
        ):
            w.add_remote_agent(meta, 0, 1, meta.dcp_size)

    def test_a_head_cut_against_a_finer_context_cut_is_a_straddle(self):
        # The same pair the other way round: one of our TP1 head slices spans
        # two TP3 peers, which the straddle check names.
        w = self._worker()
        del w._check_split_axis_constraints
        w.use_host_buffer = False
        w._logical_region_kv_heads = [6]
        fill_region_geometry(w, areas=4, axes=[ChipletSplitAxis.HEAD], slices=2)
        meta = _agent_meta(
            engine_id="peer",
            kv_areas=4,
            kv_split_axes=[ChipletSplitAxis.TOKEN],
            region_names=[],
        )

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=AssertionError("registered"),
            ),
            pytest.raises(RuntimeError, match="straddle several peers"),
        ):
            w.add_remote_agent(meta, 0, 3, meta.dcp_size)

    def test_a_context_cut_against_a_head_cut_passes_with_chunks(self):
        # The other way round, every piece lies in our area's one token range,
        # which a chunk range names: chunk_mode is not refused here.
        w = self._worker()
        w.transfer_topo.tp_size = 2
        w.local_seen_layer_names = ["l0"]
        fill_region_geometry(w, areas=4, axes=[ChipletSplitAxis.TOKEN], slices=4)
        set_shape(w, chunk_mode=True)
        w._local_band = MagicMock(return_value=(4, 4, [0, 1, 2, 3]))
        w._peer_band = MagicMock(return_value=(4, 4))
        meta = _agent_meta(engine_id="peer", kv_split_axes=[ChipletSplitAxis.HEAD])

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=RuntimeError("past the cut check"),
            ),
            pytest.raises(RuntimeError, match="past the cut check"),
        ):
            w.add_remote_agent(meta, 0, 1, meta.dcp_size)

    @pytest.mark.parametrize(
        "cls, knob, raised",
        [
            (
                RblnNixlPullConnectorWorker,
                {"chunk_mode": True},
                "which chunk_mode cannot name",
            ),
            (
                RblnNixlPushConnectorWorker,
                {"chunk_mode": True},
                "which chunk_mode cannot name",
            ),
            (
                RblnNixlPushConnectorWorker,
                {"streams_prefix": True},
                "past the cut check",
            ),
        ],
        ids=["pull_chunked", "push_chunked", "push_streamed"],
    )
    def test_a_head_cut_against_a_context_cut_refuses_chunks_unregistered(
        self, cls, knob, raised
    ):
        # Our head cut meets the peer's context cut, reading or writing: a
        # region pairs in pieces of the peer's token spans, which a chunk range
        # cannot place. A streamed write names whole blocks and pairs.
        w = self._worker()
        w.__class__ = cls
        w.local_seen_layer_names = ["l0"]
        fill_region_geometry(w, areas=4, axes=[ChipletSplitAxis.HEAD], slices=4)
        set_shape(w, **knob)
        w._local_band = MagicMock(return_value=(4, 4, [0, 1, 2, 3]))
        w._peer_band = MagicMock(return_value=(4, 4))
        meta = _agent_meta(engine_id="peer", kv_split_axes=[ChipletSplitAxis.TOKEN])

        with (
            patch.object(
                cls,
                "_register_remote_engine_prelude",
                side_effect=RuntimeError("past the cut check"),
            ),
            pytest.raises(RuntimeError, match=raised),
        ):
            w.add_remote_agent(meta, 0, 4, meta.dcp_size)

    def test_each_logical_region_is_counted_on_its_own_axis(self):
        # Our TP2 cuts region 0's heads 2 x 4 ways and region 1's tokens, which
        # cuts its heads only twice: region 1's areas alone straddle TP4 peers.
        w = self._worker()
        w.transfer_topo.tp_size = 2
        w._logical_region_kv_heads = [8, 8]
        w.local_seen_layer_names = ["l0"]
        fill_region_geometry(
            w, areas=4, axes=[ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN], slices=4
        )
        w._local_band = MagicMock(return_value=(4, 4, [0, 1, 2, 3]))
        w._peer_band = MagicMock(return_value=(2, 2))
        meta = _agent_meta(engine_id="peer", kv_split_axes=[ChipletSplitAxis.HEAD] * 2)

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=AssertionError("registered"),
            ),
            pytest.raises(RuntimeError, match="heads of logical region 1"),
        ):
            w.add_remote_agent(meta, 0, 4, meta.dcp_size)

    def test_each_peer_region_is_counted_on_its_own_axis(self):
        # The peer's TP2 cuts region 0's tokens (two head cuts) and region 1's
        # heads 2 x 3 ways; only region 1 is incommensurate with our four.
        w = self._worker()
        w._logical_region_kv_heads = [8, 8]
        w.local_seen_layer_names = ["l0"]
        fill_region_geometry(w, areas=4, axes=[ChipletSplitAxis.HEAD] * 2, slices=4)
        w._local_band = MagicMock(return_value=(4, 4, [0, 1, 2, 3]))
        w._peer_band = MagicMock(return_value=(3, 3))
        meta = _agent_meta(
            engine_id="peer",
            kv_split_axes=[ChipletSplitAxis.TOKEN, ChipletSplitAxis.HEAD],
        )

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=AssertionError("registered"),
            ),
            pytest.raises(RuntimeError, match="for logical region 1 .* the peer 6"),
        ):
            w.add_remote_agent(meta, 0, 2, meta.dcp_size)

    def test_a_peer_stage_s_regions_are_read_at_its_own_positions(self):
        # The peer's stage holds our layer l1 as its region 0: its axis is read
        # there, ours at region 1, and the head cut meets the context cut. At
        # TP2 the peer would straddle region 0's single cut.
        w = self._worker()
        w._logical_region_kv_heads = [8, 8]
        w.local_seen_layer_names = ["l0", "l1"]
        fill_region_geometry(
            w, areas=4, axes=[ChipletSplitAxis.TOKEN, ChipletSplitAxis.HEAD], slices=4
        )
        set_shape(w, chunk_mode=True)
        w._local_band = MagicMock(return_value=(4, 4, [0, 1, 2, 3]))
        w._peer_band = MagicMock(return_value=(4, 4))
        meta = _agent_meta(engine_id="peer", kv_split_axes=[ChipletSplitAxis.TOKEN])

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=AssertionError("registered"),
            ),
            pytest.raises(RuntimeError, match="logical region 1 is head-cut here"),
        ):
            w._add_remote_agent_head_matched(
                meta, 0, 2, meta.dcp_size, registered_layer_names=("l1",)
            )

    @pytest.mark.parametrize(
        ("cls", "tp", "peer_tp", "ours", "theirs"),
        [
            # A TP1 side writing into TP4: our latent's one cut against TP4.
            (
                RblnNixlPushConnectorWorker,
                1,
                4,
                [ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN],
                [ChipletSplitAxis.TOKEN] * 2,
            ),
            # A TP1 side reading from TP4: our latent's one cut against four
            # peers, each holding all of it.
            (
                RblnNixlPullConnectorWorker,
                1,
                4,
                [ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN],
                [ChipletSplitAxis.TOKEN] * 2,
            ),
            # The side with more ranks writes: a TP4 side writing into TP1.
            (
                RblnNixlPushConnectorWorker,
                4,
                1,
                [ChipletSplitAxis.TOKEN] * 2,
                [ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN],
            ),
        ],
        ids=["coarser_writes", "coarser_reads", "finer_writes"],
    )
    def test_a_latent_straddles_no_peer_and_reaches_registration(
        self, cls, tp, peer_tp, ours, theirs
    ):
        # Every peer rank holds the whole latent, so it straddles none; only the
        # attention region is counted.
        w = self._worker()
        w.__class__ = cls
        del w._check_mla_constraints
        del w._check_split_axis_constraints
        del w._is_head_matched_peer
        w.use_mla = False
        w.use_host_buffer = False
        w.transfer_topo.tp_size = tp
        w.transfer_topo.tp_ratio = MagicMock(return_value=tp // peer_tp or -peer_tp)
        fill_region_geometry(w, areas=4, axes=ours, slices=[4, 4], latent=[False, True])
        w.block_size = 16
        w._logical_region_kv_heads = [8, None]
        w.local_seen_layer_names = ["l0", "l1"]
        meta = _agent_meta(
            engine_id="peer",
            kv_areas=4,
            kv_split_axes=theirs,
            kv_area_slices=[[0, 1, 2, 3]] * 2,
            region_names=["l0"] * 4 + ["l1"] * 4,
            block_lens=[8192] * 8,
            kv_caches_base_addr=[0x1000 * (i + 1) for i in range(8)],
        )

        with (
            patch.object(
                cls,
                "_register_remote_engine_prelude",
                side_effect=AssertionError("reached the prelude"),
            ),
            pytest.raises(AssertionError, match="reached the prelude"),
        ):
            w._add_remote_agent_head_matched(meta, 0, peer_tp, meta.dcp_size)

    def test_an_attention_region_beside_a_latent_still_straddles(self):
        # The exemption is the latent region's: our attention is cut two ways,
        # so each TP4 peer rank holds part of one area.
        w = self._worker()
        w.__class__ = RblnNixlPushConnectorWorker
        del w._check_mla_constraints
        del w._check_split_axis_constraints
        del w._is_head_matched_peer
        w.use_mla = False
        w.use_host_buffer = False
        w.transfer_topo.tp_size = 1
        w.transfer_topo.tp_ratio = MagicMock(return_value=-4)
        fill_region_geometry(
            w,
            areas=4,
            axes=[ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN],
            slices=[2, 4],
            latent=[False, True],
        )
        w._logical_region_kv_heads = [8, None]
        w.local_seen_layer_names = ["l0", "l1"]
        meta = _agent_meta(
            engine_id="peer",
            kv_areas=4,
            kv_split_axes=[ChipletSplitAxis.TOKEN] * 2,
            kv_area_slices=[[0, 1, 2, 3]] * 2,
            region_names=["l0"] * 4 + ["l1"] * 4,
            block_lens=[8192] * 8,
            kv_caches_base_addr=[0x1000 * (i + 1) for i in range(8)],
        )

        with (
            patch.object(
                RblnNixlPushConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=AssertionError("reached the prelude"),
            ),
            pytest.raises(RuntimeError, match="logical region 0, so one chiplet area"),
        ):
            w._add_remote_agent_head_matched(meta, 0, 4, meta.dcp_size)

    @pytest.mark.parametrize(
        (
            "attention_slices",
            "latent_slices",
            "peer_latent",
            "chunked",
            "stage",
            "raised",
        ),
        [
            (
                4,
                2,
                [0, 1, 2, 3],
                True,
                False,
                "logical region 1, a latent, holds a block in 2 token range",
            ),
            (4, 2, [0, 1, 2, 3], False, False, "reached the prelude"),
            (4, 4, [0, 1, 2, 3], True, False, "reached the prelude"),
            # Ours finer than the peer's: each of our ranges lies in one of its.
            (4, 4, [0, 0, 1, 1], True, False, "reached the prelude"),
            # The refusal is the latent's; attention cut coarser pairs as it does
            # without a latent.
            (2, 4, [0, 1, 2, 3], True, False, "reached the prelude"),
            # A peer stage of l1 alone: its region 0 is our region 1.
            (
                4,
                2,
                [0, 1, 2, 3],
                True,
                True,
                "logical region 1, a latent, holds a block in 2 token range",
            ),
            # An uncut latent is recorded on HEAD, and named as a latent.
            (
                4,
                1,
                [0, 1, 2, 3],
                True,
                False,
                "logical region 1, a latent, holds a block in 1 token range",
            ),
        ],
        ids=[
            "coarser_chunked",
            "coarser_whole",
            "alike_chunked",
            "finer_chunked",
            "attention_coarser_chunked",
            "coarser_chunked_stage",
            "uncut_chunked",
        ],
    )
    def test_a_latent_in_coarser_token_ranges_refuses_chunks_unregistered(
        self, attention_slices, latent_slices, peer_latent, chunked, stage, raised
    ):
        # Our latent's two token ranges each span two of the peer's four, so a
        # chunk of ours is pieces of the peer's; whole blocks still pair.
        w = self._worker()
        del w._check_mla_constraints
        del w._check_split_axis_constraints
        del w._is_head_matched_peer
        w.use_mla = False
        w.use_host_buffer = False
        w.transfer_topo.tp_size = 4
        w.transfer_topo.tp_ratio = MagicMock(return_value=4)
        fill_region_geometry(
            w,
            areas=4,
            axes=[
                ChipletSplitAxis.TOKEN,
                ChipletSplitAxis.HEAD if latent_slices == 1 else ChipletSplitAxis.TOKEN,
            ],
            slices=[attention_slices, latent_slices],
            latent=[False, True],
        )
        set_shape(w, chunk_mode=chunked)
        w.block_size = 16
        w._logical_region_kv_heads = [8, None]
        w.local_seen_layer_names = ["l0", "l1"]
        layers = ["l1"] if stage else ["l0", "l1"]
        meta = _agent_meta(
            engine_id="peer",
            kv_areas=4,
            kv_split_axes=[ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN][
                -len(layers) :
            ],
            kv_area_slices=[peer_latent, peer_latent][-len(layers) :],
            region_names=[name for name in layers for _ in range(4)],
            block_lens=[8192] * 4 * len(layers),
            kv_caches_base_addr=[0x1000 * (i + 1) for i in range(4 * len(layers))],
        )

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=AssertionError("reached the prelude"),
            ),
            pytest.raises((RuntimeError, AssertionError), match=raised),
        ):
            w._add_remote_agent_head_matched(
                meta, 0, 1, meta.dcp_size, registered_layer_names=tuple(layers)
            )

    def test_a_context_cut_straddles_a_finer_peer_unregistered(self):
        # Every area holds all of this rank's heads over a quarter of the
        # tokens, so each TP4 peer rank's head quarter is part of every area.
        w = self._worker()
        w.local_seen_layer_names = ["l0"]
        fill_region_geometry(w, areas=4, axes=[ChipletSplitAxis.TOKEN], slices=4)
        w._local_band = MagicMock(return_value=(4, 4, [0, 1, 2, 3]))
        w._peer_band = MagicMock(return_value=(4, 4))
        meta = _agent_meta(engine_id="peer", kv_split_axes=[ChipletSplitAxis.HEAD])

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=AssertionError("registered"),
            ),
            pytest.raises(RuntimeError, match="straddle several peers"),
        ):
            w.add_remote_agent(meta, 0, 4, meta.dcp_size)

    @pytest.mark.parametrize(
        "axis, slices",
        [(ChipletSplitAxis.TOKEN, 4), (ChipletSplitAxis.HEAD, 1)],
        ids=["context_cut", "replicated"],
    )
    def test_a_rank_short_of_one_kv_head_is_refused_unregistered(self, axis, slices):
        # Four KV heads at TP8: upstream gives each rank one head replicated
        # across two ranks, which registration records as no head band. Our
        # bands and the peer's are read from the geometry, not mocked.
        w = self._worker()
        del w._check_split_axis_constraints
        w.use_host_buffer = False
        w.transfer_topo.tp_size = 8
        w._logical_region_kv_heads = [None]
        fill_region_geometry(w, areas=4, axes=[axis], slices=slices)
        meta = _agent_meta(
            engine_id="peer",
            kv_areas=4,
            kv_split_axes=[ChipletSplitAxis.HEAD],
            region_names=[],
        )

        with (
            patch.object(
                RblnNixlPullConnectorWorker,
                "_register_remote_engine_prelude",
                side_effect=AssertionError("registered"),
            ),
            pytest.raises(RuntimeError, match="fewer KV heads than"),
        ):
            w.add_remote_agent(meta, 0, 1, meta.dcp_size)

    def test_the_prelude_carries_the_peers_dcp_size(self):
        # `EngineTransferInfo.remote_dcp_size` defaults to 1, so a prelude that
        # does not pass it describes every peer as DCP-1 -- and upstream's own
        # validation asserts the topology agrees with the argument.
        w = self._worker()
        # Two areas, so the peer's TP 2 holds whole ones.
        fill_region_geometry(w, areas=2, axes=[ChipletSplitAxis.HEAD], slices=2)
        w._validate_remote_agent_handshake = MagicMock()
        meta = _agent_meta(engine_id="peer", dcp_size=1)

        w.add_remote_agent(meta, 0, 2, meta.dcp_size)

        info = w.transfer_topo.register_remote_engine.call_args.args[1]
        assert info.remote_dcp_size == 1
        assert (
            w.transfer_topo.compute_tp_mapping.call_args is None
            or "remote_dcp_size" in w.transfer_topo.compute_tp_mapping.call_args.kwargs
        )


class TestOnePermutationPerView:
    """`_regions_viewed_as` reorders `block_len_per_layer` for the duration.

    Its own docstring says the lengths are reordered "not just routed through
    the accessor that used to own this translation" -- so inside the scope the
    index IS the peer's position and the accessor must not translate again.
    """

    @staticmethod
    def _worker():
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.block_len_per_layer = [10, 20, 30]
        w._viewed_region_ids = None
        w._mamba_ssm_size = (0, 0)
        w.transfer_topo = MagicMock(virtually_split_kv_in_blocks=False)
        return w

    def test_a_peer_position_reads_the_region_it_names(self):
        w = self._worker()
        # The peer's region 0 is our region 2, its 1 is our 0, its 2 is our 1.
        with w._regions_viewed_as([2, 0, 1]):
            got = [w.get_backend_aware_kv_block_len(i) for i in range(3)]
        assert got == [30, 10, 20]

    def test_outside_a_view_the_index_is_our_own(self):
        w = self._worker()
        assert [w.get_backend_aware_kv_block_len(i) for i in range(3)] == [10, 20, 30]


class TestTheRemoteRegionLedger:
    """vllm 0.30 put four per-engine region lists beside ``dst_num_blocks``.

    Upstream writes all five the first time it pairs a peer, inside its own
    ``add_remote_agent``. Two RBLN routes register a peer themselves and never
    reach that body, while the transfer paths index the four with no guard
    (``pull_worker`` for the mapping flag, ``push_worker`` for all three).
    """

    @staticmethod
    def _worker(*, region_group_ids=(0, 0)):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._remote_agents = defaultdict(dict)
        w.dst_num_blocks = {}
        w.dst_region_num_blocks = {}
        w.dst_region_group_ids = {}
        w.dst_uses_region_group_mapping = {}
        w.dst_region_mem_types = {}
        w.kv_caches_base_addr = defaultdict(dict)
        w.dst_xfer_side_handles = defaultdict(dict)
        w.region_group_ids = list(region_group_ids)
        w.nixl_memory_type = "VRAM"
        w.nixl_wrapper = MagicMock()
        w._logical_region_kv_heads = [8]
        fill_region_geometry(w, areas=2, axes=[ChipletSplitAxis.HEAD], slices=2)
        w._kv_per_block = 1
        w.transfer_topo = MagicMock(tp_size=1)
        w._register_remote_engine_prelude = MagicMock()
        w._validate_remote_agent_handshake = MagicMock()
        w._check_d2d_region_pairing = MagicMock()
        w._check_split_axis_constraints = MagicMock()
        w._check_mla_constraints = MagicMock()
        w._build_head_matched_remote = MagicMock(return_value=[(0, 0, 0)])
        w._fan_in_peer_areas = MagicMock(return_value=None)
        return w

    @staticmethod
    def _meta():
        return _agent_meta(
            engine_id="peer", kv_caches_base_addr=[0x1000, 0x2000], num_blocks=4
        )

    def test_head_matched_registration_records_the_ledger(self):
        w = self._worker()

        w._add_remote_agent_head_matched(self._meta(), 0, 2)

        # A peer that advertises none of them is described by its own block
        # count over its own region count, which is what upstream falls back to.
        assert w.dst_region_num_blocks["peer"] == [4, 4]
        assert w.dst_region_group_ids["peer"] == [0, 0]
        assert w.dst_uses_region_group_mapping["peer"] is False
        assert w.dst_region_mem_types["peer"] == ["VRAM", "VRAM"]

    def test_the_peers_own_lists_win(self):
        # A peer that advertises them is not described by our geometry: it is
        # the side whose regions these descriptors address.
        w = self._worker()
        meta = _agent_meta(
            engine_id="peer",
            kv_caches_base_addr=[0x1000, 0x2000],
            num_blocks=4,
            region_num_blocks=[7, 9],
            region_group_ids=[0, 1],
            region_mem_types=["DRAM", "DRAM"],
        )

        w._add_remote_agent_head_matched(meta, 0, 2)

        assert w.dst_region_num_blocks["peer"] == [7, 9]
        assert w.dst_region_group_ids["peer"] == [0, 1]
        # Two distinct groups is exactly what turns the mapping on.
        assert w.dst_uses_region_group_mapping["peer"] is True
        assert w.dst_region_mem_types["peer"] == ["DRAM", "DRAM"]


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
        w._kv_per_block = 1
        w.use_host_buffer = host_buffer
        w.block_len_per_layer = [64] * n_local
        # The check compares regions PER LAYER, so it needs both figures.
        w.num_regions = n_local
        w.local_seen_layer_names = [f"layer.{i}" for i in range(n_layers)]
        window_mode(
            w, sw_ratio, has_swa=(sw_ratio is not None) if has_swa is None else has_swa
        )
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
            region_names=[f"layer.{i}" for i in range(n_layers)],
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
        # straddles two peers needs the chiplet cuts, so
        # _add_remote_agent_head_matched checks it before registering the peer.
        w = self._worker(host_buffer=False, tp_ratio=-2, n_local=56, tp_size=2)
        w._check_d2d_region_pairing(self._meta(56), remote_tp_size=4)  # no raise

    @pytest.mark.parametrize("sw_ratio", [4, None])
    def test_heterogeneous_tp_with_swa_raises(self, sw_ratio):
        # A sliding window is refused with model parallelism whether or not the
        # window mode is on: `sw_ratio=None` is the model with it off, which
        # keys on `has_swa` alone.
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
        w._kv_per_block = 1
        w._engine_ttl = 1.0
        w._engine_last_active = {"eng": time.perf_counter() - 10.0}
        w.nixl_wrapper = MagicMock()
        w.src_xfer_handles_by_remote = {("eng", 0, 16): 100}
        w._shard_region_group_ids = {("eng", 0): (0,)}
        w._shard_descs_per_block = {("eng", 0): (1,)}
        w._shard_chunk_grids = {("eng", 0): None}
        w._shard_region_spans = {("eng", 0): (0,)}
        w._borrowed_src_handles = set()
        w._remote_shard_layer_names = defaultdict(dict, {"eng": {0: ("l0",)}})
        w._overlapping_ranks = defaultdict(list, {"eng": [0]})
        w._remote_pp_size = {"eng": 1}
        # The 0.30.0 sweep drops a stale engine's in-flight reads too.
        w._recving_transfers = defaultdict(list)

        with patch.object(NixlBaseConnectorWorker, "_cleanup_remote_engine"):
            NixlBaseConnectorWorker._evict_stale_engines(w)

        # The real override ran: our per-stage state is gone, not just upstream's.
        assert "eng" not in w._overlapping_ranks
        assert [k for k in w.src_xfer_handles_by_remote if k[0] == "eng"] == []
        w.nixl_wrapper.release_dlist_handle.assert_called_once_with(100)

    def test_cleanup_purges_per_stage_state_and_releases_handles(self):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
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
        w._shard_descs_per_block = {
            ("eng", 0): (1,),
            ("eng", 1): (2,),
            ("other", 0): (1,),
        }
        w._shard_chunk_grids = {
            ("eng", 0): (1, 2),
            ("eng", 1): None,
            ("other", 0): None,
        }
        w._shard_region_spans = {key: (0,) for key in w._shard_chunk_grids}
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
        # A grid left behind would index a range the re-handshake did not build.
        assert [k for k in w._shard_chunk_grids if k[0] == "eng"] == []
        assert [k for k in w._shard_region_spans if k[0] == "eng"] == []
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
        w._kv_per_block = 1
        w.nixl_wrapper = MagicMock()
        w.src_xfer_handles_by_remote = {("eng", 0, 16): 100, ("eng", 1, 16): 101}
        w._borrowed_src_handles = {("eng", 1, 16)}
        w._shard_region_group_ids = {}
        w._shard_descs_per_block = {}
        w._shard_chunk_grids = {}
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
    def _worker(*, local_len=128, block_size_ratio=1, layout="LBNHC", kv_heads=8):
        # `local_len` and `kv_heads` take a list for one entry per logical region;
        # the transfer table repeats each across the 4 chiplet areas.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
        w.tp_rank = 1
        w.kv_cache_layout = layout
        lens = list(local_len) if isinstance(local_len, list) else [local_len]
        # 8 heads over TP4 is 2 per rank, cut into 2 slices -> 1 head per area.
        w.block_len_per_layer = [ln for ln in lens for _ in range(4)]
        w._logical_region_kv_heads = (
            list(kv_heads) if isinstance(kv_heads, list) else [kv_heads] * len(lens)
        )
        fill_region_geometry(
            w, areas=4, axes=[ChipletSplitAxis.HEAD] * len(lens), slices=2
        )
        topo = MagicMock()
        topo.tp_size = 4
        topo.total_num_kv_heads = w._logical_region_kv_heads[0]
        topo.block_size_ratio.return_value = block_size_ratio
        w.transfer_topo = topo
        w.vllm_config = mock_vllm_config()
        w.vllm_config.parallel_config.tensor_parallel_size = 4
        w.vllm_config.speculative_config = None
        return w

    @staticmethod
    def _meta(*, remote_len, layout="LBNHC"):
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

    def test_a_draft_at_another_tp_is_refused_by_name(self):
        # A separate draft may run at TP 1 under a TP4 target, and then every
        # rank holds all of its heads. Unnamed, that surfaced as an assertion
        # on a missing head band, far from the setting that caused it.
        w = self._worker(local_len=[128, 128], kv_heads=[8, None])
        w.vllm_config.speculative_config = SimpleNamespace(
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=1)
        )
        with pytest.raises(RuntimeError, match="draft at TP 1 beside a target at TP 4"):
            w._validate_head_matched_handshake(
                self._meta(remote_len=[256, 256]), remote_tp_size=1
            )

    def test_a_draft_at_the_target_tp_takes_the_head_check(self):
        # An eagle draft inherits the target's TP, which is the shape that
        # pairs; it goes on to the byte check like any other region.
        w = self._worker(local_len=128)
        w.vllm_config.speculative_config = SimpleNamespace(
            draft_parallel_config=SimpleNamespace(tensor_parallel_size=4)
        )
        w._validate_head_matched_handshake(
            self._meta(remote_len=256), remote_tp_size=1
        )  # no raise

    @pytest.mark.parametrize("remote_len, pairs", [(256, True), (128, False)])
    def test_a_context_cut_is_checked_per_head_and_token(self, remote_len, pairs):
        # Our TP4 rank holds one head, its areas each a quarter of the block's
        # tokens; the TP1 peer's areas hold one head over the whole block. A
        # cell is one token of one head, so a peer area costs four of ours.
        w = self._worker(local_len=64, kv_heads=4)
        fill_region_geometry(w, areas=4, axes=[ChipletSplitAxis.TOKEN], slices=4)
        meta = _agent_meta(
            block_size=16,
            kv_cache_layout="LBNHC",
            kv_areas=4,
            kv_split_axes=[ChipletSplitAxis.HEAD],
            block_lens=[remote_len] * 4,
        )
        if pairs:
            w._validate_head_matched_handshake(meta, remote_tp_size=1)  # no raise
        else:
            with pytest.raises(RuntimeError, match="a KV head occupies"):
                w._validate_head_matched_handshake(meta, remote_tp_size=1)

    def test_each_region_is_drawn_on_its_own_axis(self):
        # Region 0 cuts our rank's four heads, region 1 the block's tokens.
        w = self._worker(local_len=[64, 64], kv_heads=[16, 16])
        fill_region_geometry(
            w, areas=4, axes=[ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN], slices=4
        )
        # The peer's are the other way round.
        meta = _agent_meta(
            block_size=16,
            kv_areas=4,
            kv_split_axes=[ChipletSplitAxis.TOKEN, ChipletSplitAxis.HEAD],
            block_lens=[256] * 8,
        )
        ours, peer = w._area_rects(meta, 1, 1, 0, 1)
        assert ours == [AreaRect((4, 8), (4 * a, 4 * a + 4)) for a in range(4)]
        assert [rect for _, rect, _ in peer] == [
            AreaRect((4 * a, 4 * a + 4), (0, 16)) for a in range(4)
        ]
        # A peer stage's position need not be ours: its region 0 is our 1.
        _, peer = w._area_rects(meta, 1, 0, 0, 1)
        assert [rect for _, rect, _ in peer] == [
            AreaRect((0, 16), (4 * a, 4 * a + 4)) for a in range(4)
        ]

    def test_a_context_cut_s_replicas_hold_their_slice_s_tokens(self):
        # Four areas over two slices: areas 0, 1 hold slice 0's half of the
        # block, areas 2, 3 slice 1's.
        w = self._worker(local_len=[64, 64], kv_heads=[16, 16])
        fill_region_geometry(
            w, areas=4, axes=[ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN], slices=2
        )
        meta = _agent_meta(
            block_size=16,
            kv_areas=4,
            kv_split_axes=[ChipletSplitAxis.HEAD] * 2,
            block_lens=[256] * 8,
        )
        ours, _ = w._area_rects(meta, 1, 1, 0, 1)
        assert ours == [
            AreaRect((4, 8), (8 * (a // 2), 8 * (a // 2) + 8)) for a in range(4)
        ]

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
            kv_cache_layout="LBNHC",
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
        w = self._worker(layout="LBNHC")
        with pytest.raises(RuntimeError, match="peer KV layout"):
            w._validate_head_matched_handshake(
                self._meta(remote_len=256, layout="LBHNC"), remote_tp_size=1
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
        w._kv_per_block = 1
        w._remote_agents = {}
        w.dst_num_blocks = {}
        # Registering a peer now records its region ledger beside the block
        # count, which the transfer paths read unguarded.
        w.dst_region_num_blocks = {}
        w.dst_region_group_ids = {}
        w.dst_region_mem_types = {}
        w.dst_uses_region_group_mapping = {}
        w.region_group_ids = []
        w.kv_caches_base_addr = {"eng": {}}
        w.dst_xfer_side_handles = {"eng": {}}
        w.nixl_memory_type = "VRAM"
        w._logical_region_kv_heads = [8, 8]
        fill_region_geometry(w, areas=4, axes=[ChipletSplitAxis.HEAD] * 2, slices=2)
        w.transfer_topo = MagicMock(tp_size=4)
        w._check_d2d_region_pairing = MagicMock()
        w._check_split_axis_constraints = MagicMock()
        w._check_mla_constraints = MagicMock()
        w.local_seen_layer_names = ["l1", "l2"]
        w.nixl_wrapper = MagicMock()
        w.nixl_wrapper.add_remote_agent.return_value = "agent"
        meta = MagicMock()
        meta.engine_id = "eng"
        meta.kv_area_slices = area_slices(4, 2, 2)
        meta.kv_split_axes = [ChipletSplitAxis.HEAD] * 2

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
        assert prelude.call_args.args == (meta, 2, 1)
        assert validate.call_args.args == (meta, 2, 1)

    def test_a_shard_already_registered_is_not_registered_again(self):
        # Mirrors upstream's own cache check, which it keeps behind a TODO for
        # a refresh path neither side has yet: within one handshake the engine's
        # agent map is still empty, so today nothing reaches this.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
        # Flat rank 2 of a TP2 peer is its (pp 1, tp 0) -- not a palindrome, so
        # the pair's order has to be right and not merely its members.
        w._remote_agents = {"eng": {(1, 0): "already"}}
        w.nixl_wrapper = MagicMock()
        meta = MagicMock()
        meta.engine_id = "eng"

        assert w._add_remote_agent_head_matched(meta, 2, 2) == "already"
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    @staticmethod
    def _slice_worker(*, slices, kv_heads):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
        w._remote_agents = {}
        w.nixl_wrapper = MagicMock()
        w._logical_region_kv_heads = list(kv_heads)
        w.use_mla = False
        fill_region_geometry(
            w,
            areas=4,
            axes=[ChipletSplitAxis.HEAD] * len(list(slices)),
            slices=list(slices),
        )
        return w

    def test_uneven_region_slice_counts_are_refused_before_an_agent_is_added(self):
        # Silent without the refusal: the smaller count divides the larger side's
        # heads per rank, so a wrong band transfers with nothing objecting.
        w = self._slice_worker(slices=[4, 4, 2, 2], kv_heads=[8, 8, 32, 32])
        meta = MagicMock()
        meta.engine_id = "eng"

        with pytest.raises(RuntimeError, match="no one geometry bands every region"):
            w._add_remote_agent_head_matched(meta, 1, 2)

        # Refused at the entry, so no peer state was taken on.
        w.nixl_wrapper.add_remote_agent.assert_not_called()

    def test_the_loud_direction_is_refused_by_the_same_check(self):
        # The mirror case already fails, but inside `_slice_head_bounds` as
        # "owns 2 KV heads cut into 4 logical slice(s)" -- which names the
        # arithmetic rather than the cause. One condition covers both directions.
        w = self._slice_worker(slices=[2, 2, 4, 4], kv_heads=[8, 8, 32, 32])

        with pytest.raises(RuntimeError, match="no one geometry bands every region"):
            w._local_band(2)

    def test_equal_counts_placed_differently_are_refused(self):
        # One band picks our areas for every region, so the banded regions have
        # to place their slices alike; equal counts are not enough. Here the
        # draft's replicas alternate where the target's sit side by side.
        w = self._slice_worker(slices=[2, 2, 2, 2], kv_heads=[8, 8, 32, 32])
        alternating = w._logical_region_geometry[2]._replace(
            areas=tuple(AreaPlace(0, 0, sid) for sid in (0, 1, 0, 1))
        )
        w._logical_region_geometry[2:] = [alternating, alternating]

        with pytest.raises(RuntimeError, match="no one geometry bands every region"):
            w._local_band(2)

    def test_a_peer_whose_regions_differ_in_geometry_is_refused(self):
        # The same holds for the peer: one split serves all of its regions.
        w = self._slice_worker(slices=[4, 4], kv_heads=[8, 8])
        meta = SimpleNamespace(kv_area_slices=[[0, 1, 2, 3], [0, 0, 1, 1]])

        with pytest.raises(RuntimeError, match="the peer's KV cache entries"):
            w._peer_band(meta, 2, None)

    def test_uniform_slice_counts_are_not_refused(self):
        # A differing KV geometry is supported; only a differing chiplet CUT is
        # not. Refusing this would close the case the head bands exist for.
        w = self._slice_worker(slices=[4, 4, 4, 4], kv_heads=[8, 8, 32, 32])

        assert w._local_band(2) == (4, 4, (0, 1, 2, 3))

    def test_a_region_without_a_head_axis_is_not_counted(self):
        # An SSM state is not cut by KV heads, so its slice count says nothing
        # about the head bands -- and the refusal that does name Mamba sits
        # further in, past this one.
        w = self._slice_worker(slices=[4, 4, 1, 1], kv_heads=[8, 8, None, None])

        assert w._local_band(2) == (4, 4, (0, 1, 2, 3))

    def test_a_peer_region_paired_with_one_of_ours_without_a_band_is_not_counted(
        self,
    ):
        # A peer region is the layer of ours it pairs with, so ours says which
        # of the peer's band by head. The peer stage starts at our l1, so its
        # region 0 is the one without a band.
        w = self._slice_worker(slices=[2, 4, 2], kv_heads=[8, None, 8])
        w.local_seen_layer_names = ["l0", "l1", "l2"]
        meta = SimpleNamespace(kv_area_slices=[[0, 1, 2, 3], [0, 0, 1, 1]])

        assert w._peer_band(meta, 2, ("l1", "l2")) == (4, 2)

    def test_a_peer_with_a_different_tp_degree_is_routed_to_head_matching(self):
        # The fork every head-matched path hangs off: unequal TP means position
        # is the wrong key, so upstream's add_remote_agent must not be reached.
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
        window_mode(w, None, has_swa=False)
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

        assert head_matched.call_args.args == (meta, 1, 2, 1)
        base.assert_not_called()


class TestSplitAxisConstraints:
    # The guard that reads the advertised axis. Every other check here compares
    # counts; two peers can agree on every count and still mean different axes
    # by them, which is the one thing a count cannot say.

    @staticmethod
    def _worker(
        *,
        axis,
        tp_ratio=1,
        host_buffer=False,
        trim=False,
        block_size=16,
        sw_ratio=None,
        cls=RblnNixlPullConnectorWorker,
        stream=False,
        areas=1,
    ):
        w = object.__new__(cls)
        w._kv_per_block = 1
        w.use_host_buffer = host_buffer
        # `_agent_meta`'s four layers, one logical region each.
        w.local_seen_layer_names = [f"layer.{i}" for i in range(4)]
        fill_region_geometry(w, areas=areas, axes=[axis] * 4, slices=areas)
        w.block_size = block_size
        window_mode(
            w,
            sw_ratio,
            chunk_mode=trim,
            streams_prefix=stream,
            wants_stream=stream,
            writes_into_peer=stream,
        )
        topo = MagicMock()
        topo.tp_size = 2
        topo.tp_ratio.return_value = tp_ratio
        w.transfer_topo = topo
        return w

    def test_a_peer_that_cut_another_axis_is_rejected(self):
        w = self._worker(axis=ChipletSplitAxis.HEAD)
        meta = _agent_meta(kv_split_axes=[ChipletSplitAxis.TOKEN] * 4)
        with pytest.raises(RuntimeError, match=r"layer layer.0 of its KV cache"):
            w._check_split_axis_constraints(meta, 2)

    def test_a_context_cut_on_both_sides_is_rejected_with_unequal_tp(self):
        # A pairing this code has not opened: both sides cut tokens.
        w = self._worker(axis=ChipletSplitAxis.TOKEN, tp_ratio=2)
        meta = _agent_meta(kv_split_axes=[ChipletSplitAxis.TOKEN] * 4)
        with pytest.raises(RuntimeError, match="context-cut on both sides"):
            w._check_split_axis_constraints(meta, 1)

    @pytest.mark.parametrize(
        ("ours", "theirs"),
        [
            (ChipletSplitAxis.TOKEN, ChipletSplitAxis.HEAD),
            (ChipletSplitAxis.HEAD, ChipletSplitAxis.TOKEN),
        ],
        ids=["we_cut_tokens", "the_peer_cuts_tokens"],
    )
    def test_a_head_cut_meets_a_context_cut_across_tp(self, ours, theirs):
        # The axes differ, so no area of one side is an area of the other;
        # `pair_area` pairs what each holds, and nothing here is refused.
        w = self._worker(axis=ours, tp_ratio=2)
        meta = _agent_meta(kv_split_axes=[theirs] * 4)
        w._check_split_axis_constraints(meta, 1)  # no raise

    def test_a_peer_cutting_another_number_of_regions_is_refused_across_tp(self):
        # Across TP the axes are compared region by region, which pairs nothing
        # when the lists differ in length; that is a mismatch, not a pass. A
        # peer naming its layers is caught on the count first.
        w = self._worker(axis=ChipletSplitAxis.HEAD, tp_ratio=2)
        meta = _agent_meta(kv_split_axes=[ChipletSplitAxis.HEAD] * 3)
        meta.region_names = []
        with pytest.raises(RuntimeError, match=r"\(3 region\(s\) against 4\)"):
            w._check_split_axis_constraints(meta, 1)

    def test_a_latent_at_equal_tp_is_compared_by_its_axis(self):
        # Area k pairs with area k, so a latent keeps its axis in the
        # comparison; only across TP is it left out.
        w = self._worker(axis=ChipletSplitAxis.HEAD)
        w._logical_region_latent[-1] = True
        w._logical_region_geometry[-1] = w._logical_region_geometry[-1]._replace(
            axis=ChipletSplitAxis.TOKEN
        )
        meta = _agent_meta(
            kv_split_axes=[ChipletSplitAxis.HEAD] * 3 + [ChipletSplitAxis.TOKEN]
        )

        w._check_split_axis_constraints(meta, 2)  # no raise

    def test_a_context_cut_at_equal_tp_passes(self):
        # Area k pairs with area k and no band is consulted, so the bytes are
        # right; rejecting this would refuse the only shape that does work.
        w = self._worker(axis=ChipletSplitAxis.TOKEN)
        meta = _agent_meta(kv_split_axes=[ChipletSplitAxis.TOKEN] * 4)
        w._check_split_axis_constraints(meta, 2)  # no raise

    def test_a_trimming_peer_must_size_its_block_the_same(self):
        # Same axis, equal TP, and still wrong: an area is a fraction of a
        # block, so a peer whose block holds a different number of tokens gives
        # the same area a different token range while the byte counts fit.
        w = self._worker(axis=ChipletSplitAxis.TOKEN, trim=True, block_size=16)
        meta = _agent_meta(
            kv_split_axes=[ChipletSplitAxis.TOKEN] * 4,
            block_size=32,
        )
        with pytest.raises(RuntimeError, match="count a block the same way"):
            w._check_split_axis_constraints(meta, 2)

    def test_a_windowed_peer_must_size_its_block_the_same(self):
        # The window range divides the peer's block by OUR `window_ratio`, so an
        # unequal block size names a length that is not the peer's window --
        # and the descriptor counts match either way, so nothing reports it.
        w = self._worker(axis=ChipletSplitAxis.HEAD, sw_ratio=4, block_size=16)
        meta = _agent_meta(block_size=32, kv_split_axes=[ChipletSplitAxis.HEAD] * 4)
        with pytest.raises(RuntimeError, match="count a block the same way"):
            w._check_split_axis_constraints(meta, 1)

    def test_a_streaming_peer_must_size_its_block_the_same(self):
        # Streaming alone cuts no block, but it places the blocks a prefill
        # closes by counting them on both sides -- so it needs the peer to hold
        # the same count, whether or not chunk mode is on.
        w = self._worker(
            axis=ChipletSplitAxis.TOKEN,
            stream=True,
            block_size=16,
            cls=RblnNixlPushConnectorWorker,
        )
        meta = _agent_meta(
            kv_split_axes=[ChipletSplitAxis.TOKEN] * 4,
            block_size=32,
        )
        with pytest.raises(RuntimeError, match="count a block the same way"):
            w._check_split_axis_constraints(meta, 2)

    def test_a_matching_block_size_passes_while_trimming(self):
        w = self._worker(axis=ChipletSplitAxis.TOKEN, trim=True, block_size=16)
        meta = _agent_meta(
            kv_split_axes=[ChipletSplitAxis.TOKEN] * 4,
            block_size=16,
        )
        w._check_split_axis_constraints(meta, 2)  # no raise

    def test_a_differing_block_size_is_upstreams_when_not_trimming(self):
        # The refusal is scoped to the trim: without it this pairing is
        # upstream's and unchanged.
        w = self._worker(axis=ChipletSplitAxis.TOKEN, block_size=16)
        meta = _agent_meta(
            kv_split_axes=[ChipletSplitAxis.TOKEN] * 4,
            block_size=32,
        )
        w._check_split_axis_constraints(meta, 2)  # no raise

    def test_a_peer_laying_a_regions_slices_over_other_areas_is_refused(self):
        # Equal TP pairs area k with the peer's area k, so agreeing on the axis
        # is not enough: each area has to hold the same slice on both sides.
        w = self._worker(axis=ChipletSplitAxis.HEAD, areas=4)
        meta = _agent_meta(
            kv_area_slices=[[0, 1, 2, 3]] * 2 + [[1, 0, 3, 2]] + [[0, 1, 2, 3]],
        )
        with pytest.raises(RuntimeError, match="logical region 0 of layer layer.2"):
            w._check_split_axis_constraints(meta, 1)

    @pytest.mark.parametrize("token_region", [0, -1], ids=["target", "draft"])
    def test_a_context_cut_on_both_sides_beside_a_head_cut_is_rejected(
        self, token_region
    ):
        # One region cut on its tokens on both sides is enough, however the
        # regions beside it are cut.
        w = self._worker(axis=ChipletSplitAxis.HEAD, tp_ratio=2)
        geometry = w._logical_region_geometry
        geometry[token_region] = geometry[token_region]._replace(
            axis=ChipletSplitAxis.TOKEN
        )
        meta = _agent_meta(kv_split_axes=[g.axis for g in geometry])

        with pytest.raises(RuntimeError, match="context-cut on both sides"):
            w._check_split_axis_constraints(meta, 2)

    def test_a_stage_advertising_another_count_of_regions_a_layer_is_refused(self):
        # Its axes are sliced a layer at a time, so a count off by one shifts
        # every layer's slice onto the next one's.
        w = self._worker(axis=ChipletSplitAxis.HEAD)
        meta = _agent_meta(
            region_names=["layer.2", "layer.3"],
            kv_area_slices=[[0]] * 3,
        )

        with pytest.raises(RuntimeError, match="cut axes over 2 layer"):
            w._check_split_axis_constraints(meta, 1)

    def test_a_head_matched_peer_is_compared_on_its_axes_alone(self):
        # Across TP the head bands read the peer's slice ids area by area, so
        # how it lays them out is its own.
        w = self._worker(axis=ChipletSplitAxis.HEAD, areas=4, tp_ratio=2)
        meta = _agent_meta(kv_area_slices=[[1, 0, 3, 2]] * 4)

        w._check_split_axis_constraints(meta, 2)  # no raise

    def test_a_target_and_draft_cut_alike_on_both_sides_pass(self):
        w = self._worker(axis=ChipletSplitAxis.TOKEN)
        geometry = w._logical_region_geometry
        geometry[-1] = geometry[-1]._replace(axis=ChipletSplitAxis.HEAD)
        meta = _agent_meta(kv_split_axes=[g.axis for g in geometry])

        w._check_split_axis_constraints(meta, 1)  # no raise

    def test_a_draft_cut_another_way_is_refused_by_name(self):
        # The engines agree on the target and differ on the draft, which no
        # count says: the areas and slices of a draft are the same numbers
        # under either axis.
        w = self._worker(axis=ChipletSplitAxis.TOKEN)
        geometry = w._logical_region_geometry
        geometry[-1] = geometry[-1]._replace(axis=ChipletSplitAxis.HEAD)
        meta = _agent_meta(kv_split_axes=[ChipletSplitAxis.TOKEN] * 4)

        with pytest.raises(RuntimeError, match="layer layer.3 of its KV cache"):
            w._check_split_axis_constraints(meta, 1)

    @pytest.mark.parametrize(
        ("differs_at", "refused"), [("layer.0", False), ("layer.2", True)]
    )
    def test_a_pipeline_stage_is_compared_on_the_layers_it_holds(
        self, differs_at, refused
    ):
        # A stage advertises its own layers' axes, so position 0 of its list
        # is our layer 2 -- read by position, the first case refuses and the
        # second passes.
        w = self._worker(axis=ChipletSplitAxis.HEAD)
        names = [f"layer.{i}" for i in range(4)]
        geometry = w._logical_region_geometry
        at = names.index(differs_at)
        geometry[at] = geometry[at]._replace(axis=ChipletSplitAxis.TOKEN)
        meta = _agent_meta(
            region_names=["layer.2", "layer.3"],
            kv_split_axes=[ChipletSplitAxis.HEAD] * 2,
        )

        if refused:
            with pytest.raises(RuntimeError, match="layer layer.2 of its KV cache"):
                w._check_split_axis_constraints(meta, 1)
        else:
            w._check_split_axis_constraints(meta, 1)  # no raise

    def test_host_bounce_is_exempt(self):
        # Host staging registers one logical buffer per layer and never expands
        # per area, so its HEAD default describes a peer of any axis.
        w = self._worker(axis=ChipletSplitAxis.HEAD, tp_ratio=2, host_buffer=True)
        meta = _agent_meta(kv_split_axes=[ChipletSplitAxis.TOKEN] * 4)
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
        window_mode(worker, 2)
        worker._has_mamba = False
        worker.use_mla = False
        worker.tp_rank = 0
        worker._group_spec_types = ()
        worker.nixl_memory_type = "DRAM"

        topo = MagicMock(virtually_split_kv_in_blocks=False)
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
        assert validate.call_args.args == (meta, 1, 1)

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
            transfer_topology=topo,
            remote_tp_size=1,
            group_spec_types=(),
            remote_dcp_size=1,
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

    @staticmethod
    def _swa_remote_worker(monkeypatch):
        worker = build_worker(monkeypatch, num_blocks=8, block_size=64)
        window_mode(worker, 2)
        worker._has_mamba = False
        worker.use_mla = False
        worker.tp_rank = 0
        worker.device_id = 0
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
        worker.kv_caches_base_addr[worker.engine_id] = {0: [0x1000, 0x2000]}
        worker.block_len_per_layer = [256, 256]
        worker.dst_xfer_side_handles = collections.defaultdict(dict)
        worker.src_xfer_handles_by_block_size = {}
        worker.src_blocks_data = []
        worker.nixl_wrapper = MagicMock()
        worker.nixl_wrapper.add_remote_agent.return_value = "remote-agent-name"
        return worker

    def test_both_lists_cut_a_block_the_same_way(self, monkeypatch):
        # The one thing the two builders must agree on and nothing else checks:
        # a chunk's offset inside its block, and its length. Different bases,
        # different strides, same cut.
        worker = self._swa_remote_worker(monkeypatch)
        meta = _remote_agent_meta()

        with (
            patched_in_package("compute_tp_mapping", MagicMock()),
            patch.object(worker, "_validate_remote_agent_handshake"),
            patch.object(worker, "get_backend_aware_kv_block_len", return_value=256),
            patch.object(
                type(worker),
                "_shard_chunk_grid",
                return_value=uniform_grid(2, 2, regions=2),
            ),
        ):
            worker.add_remote_agent(meta, 0, 1)
            remote = worker.nixl_wrapper.get_xfer_descs.call_args[0][0]
            worker.register_local_xfer_handler(64)
            local = worker.nixl_wrapper.get_xfer_descs.call_args[0][0]

        def cut(descs, whole, base):
            # The first block of the first region, as (offset in block, length).
            return [(addr - base, ln) for addr, ln, _ in descs[whole : whole + 4]]

        # 2 regions x 8 blocks whole, then sw_ratio granules of each.
        window_end = 2 * 8 * (1 + worker._shape.window_ratio)
        # Non-empty, or the comparison is two empty lists agreeing.
        assert (
            cut(remote, window_end, 0x5000)
            == cut(local, window_end, 0x1000)
            == [
                (0, 64),
                (64, 64),
                (128, 64),
                (192, 64),
            ]
        )

    def test_the_peer_list_takes_the_third_range_too(self, monkeypatch):
        # A prepared transfer pairs the two lists by position, so a range one
        # side carries and the other does not pairs chunk descriptors against
        # whole blocks -- and the length check ahead of a transfer compares how
        # many indices each side named, not how far they reach.
        worker = self._swa_remote_worker(monkeypatch)
        meta = _remote_agent_meta()

        with (
            patched_in_package("compute_tp_mapping", MagicMock()),
            patch.object(worker, "_validate_remote_agent_handshake"),
            patch.object(worker, "get_backend_aware_kv_block_len", return_value=256),
            patch.object(
                type(worker),
                "_shard_chunk_grid",
                return_value=uniform_grid(2, 2, regions=2),
            ),
        ):
            worker.add_remote_agent(meta, 0, 1)

        blocks_data = worker.nixl_wrapper.get_xfer_descs.call_args[0][0]
        # 2 regions x 8 blocks, whole then sw_ratio granules of each, and last
        # 2 regions x 8 blocks x 2 runs x 2 chunks.
        assert len(blocks_data) == 16 * (1 + worker._shape.window_ratio) + 64
        assert np.array_equal(
            blocks_data[48:52],
            [
                (0x5000, 64, 1),
                (0x5040, 64, 1),
                (0x5080, 64, 1),
                (0x50C0, 64, 1),
            ],
        )

    def test_a_smaller_remote_block_shortens_descs_and_adds_a_local_handle(
        self, monkeypatch
    ):
        # A peer with a smaller block holds less per block than we do, so a desc
        # can only span the peer's block length -- on both sides, which is why a
        # second local handler keyed by the peer's block size is registered.
        worker = build_worker(monkeypatch, num_blocks=4, block_size=64)
        window_mode(worker, 2)
        worker._has_mamba = False
        worker.use_mla = False
        worker.tp_rank = 0
        worker._group_spec_types = ()
        worker.nixl_memory_type = "DRAM"

        topo = MagicMock(virtually_split_kv_in_blocks=False)
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


# What each layout puts in one block.
PACKED = 2  # rbln_custom_ops: (num_blocks, 2, H, 1, S, D)
SPLIT = 1  # rbln_triton_ops: (2, num_blocks, H, 1, S, D)


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
    BLOCK = 16

    def _pieces(self, kv_per_block, peer_page=PEER_PAGE):
        full = (0, self.BLOCK)
        return pair_area(
            AreaRect((self.WANTED_HEAD, self.WANTED_HEAD + 2), full),
            self.LOCAL_PAGE,
            [
                (0, AreaRect((0, self.PEER_HEADS), full), peer_page),
                (1, AreaRect((self.PEER_HEADS, 2 * self.PEER_HEADS), full), peer_page),
            ],
            kv_per_block=kv_per_block,
            every_holder=False,
            region_id=0,
        )

    def _descs(self, kv_per_block):
        """(address, length) per block and piece, as the remote list has them."""
        return [
            (self.PEER_BASE + block * self.PEER_PAGE + p.remote_offset, p.length)
            for block in range(self.NUM_BLOCKS)
            for p in self._pieces(kv_per_block)
        ]

    def _decode(self, addr, kv_per_block):
        """(block, K-or-V, head) for an address inside the peer's region."""
        off = addr - self.PEER_BASE
        block, within_block = divmod(off, self.PEER_PAGE)
        kv_stride = self.PEER_PAGE // kv_per_block
        kv, within_kv = divmod(within_block, kv_stride)
        return block, kv, within_kv // (kv_stride // self.PEER_HEADS)

    def test_a_packed_block_is_read_at_the_head_the_pairing_asked_for(self):
        descs = self._descs(kv_per_block=2)
        # Two blocks, each named once in K and once in V, all at our head.
        assert [self._decode(a, 2) for a, _ in descs] == [
            (0, 0, self.WANTED_HEAD),
            (0, 1, self.WANTED_HEAD),
            (1, 0, self.WANTED_HEAD),
            (1, 1, self.WANTED_HEAD),
        ]

    @pytest.mark.parametrize("kv_per_block", [1, 2])
    def test_no_descriptor_runs_off_the_end_of_its_half(self, kv_per_block):
        kv_stride = self.PEER_PAGE // kv_per_block
        for addr, length in self._descs(kv_per_block):
            within_kv = (addr - self.PEER_BASE) % self.PEER_PAGE % kv_stride
            assert within_kv + length <= kv_stride

    def test_a_peer_block_that_does_not_halve_is_refused(self):
        # The length comes off the wire, so it is the peer's claim rather than
        # ours; halving it silently would move a byte short of every V.
        with pytest.raises(RuntimeError, match="range"):
            self._pieces(kv_per_block=2, peer_page=self.PEER_PAGE + 1)

    def test_the_two_halves_are_a_stride_apart(self):
        addrs = [a for a, _ in self._descs(kv_per_block=2)]
        kv_stride = self.PEER_PAGE // 2
        # K and V of the same block sit `kv_stride` apart, K first.
        assert addrs[1] - addrs[0] == kv_stride
        assert addrs[3] - addrs[2] == kv_stride


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
        window_mode(w, None, has_swa=False)
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
        # Explicit, because a mock answers this truthy and the pairing reads it
        # as a peer that holds a sliding-window group.
        meta.swa_kernel_block = 0
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


class TestTwoKernelGeometriesNeverPair:
    """Which of `block_size` and `sliding_window` the kernel addresses the
    cache in is the runner's choice, so two engines off one build can differ.
    A window range is cut by that number, so the lists would name different
    pieces of the same block while every byte count still matched."""

    @staticmethod
    def _worker(swa_kernel_block, *, windowed=True):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.use_host_buffer = False
        window_mode(w, 8, has_swa=True)
        w._swa_kernel_blocks = {swa_kernel_block} if swa_kernel_block else set()
        w._window_grid_cut = (1, 8) if windowed else None
        w._kv_per_block = 1
        w.block_len_per_layer = [64]
        w.num_regions = 1
        w.local_seen_layer_names = ["layer.0"]
        topo = MagicMock()
        topo.tp_ratio.return_value = 1
        w.transfer_topo = topo
        return w

    @staticmethod
    def _meta(swa_kernel_block):
        meta = MagicMock()
        meta.kv_per_block = 1
        meta.swa_kernel_block = swa_kernel_block
        meta.kv_caches_base_addr = [1000]
        meta.registered_layer_names = ["layer.0"]
        return meta

    def test_a_peer_on_the_other_geometry_is_refused(self):
        with pytest.raises(RuntimeError, match="sub-block prefix caching"):
            self._worker(128)._check_d2d_region_pairing(
                self._meta(1024), remote_tp_size=1
            )

    def test_a_peer_on_our_geometry_pairs(self):
        self._worker(128)._check_d2d_region_pairing(self._meta(128), remote_tp_size=1)

    def test_a_peer_holding_no_window_pairs(self):
        # Zero is what a shard cutting no window range advertises, which a PP
        # stage without such a group is; refusing it would refuse the stage.
        self._worker(128)._check_d2d_region_pairing(self._meta(0), remote_tp_size=1)

    def test_without_a_window_range_the_geometry_does_not_pair(self):
        # The regression this guards: two hybrids whose runners chose
        # differently, neither cutting a window range. Their lists are whole
        # blocks of the full-attention view, which is the same shape either
        # way, so the pair is describable and has to stay allowed.
        self._worker(128, windowed=False)._check_d2d_region_pairing(
            self._meta(1024), remote_tp_size=1
        )


class TestAPackedBlockNeedsAnEqualPeerBlockSize:
    """The equal-TP peer hands its descriptors to upstream unchanged, and
    upstream cuts a block into `block_size_ratio` equal pieces to pair each with
    one peer block. Under a packed block that cut falls on the K/V boundary, so
    the combination is refused rather than transferred wrong.
    """

    @staticmethod
    def _worker(kv_per_block, block_size_ratio):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w._kv_per_block = kv_per_block
        w.transfer_topo = MagicMock()
        w.transfer_topo.block_size_ratio.return_value = block_size_ratio
        for check in (
            "_check_split_axis_constraints",
            "_check_mla_constraints",
            "_check_d2d_region_pairing",
        ):
            setattr(w, check, lambda *a, **k: None)
        # Unequal TP pairs by head band and refuses this on its own; the path
        # this guards is the one that reaches upstream.
        w._is_head_matched_peer = lambda _: False
        return w

    @staticmethod
    def _meta():
        meta = MagicMock(spec=RblnNixlAgentMetadata)
        meta.pp_size = 1
        meta.block_size = 64
        return meta

    def test_a_peer_with_a_different_block_size_is_refused(self):
        worker = self._worker(PACKED, block_size_ratio=2)

        with pytest.raises(RuntimeError, match="equal P/D block sizes"):
            worker._validate_remote_agent_handshake(self._meta(), remote_tp_size=1)

    @pytest.mark.parametrize(
        "kv_per_block, block_size_ratio",
        [
            pytest.param(PACKED, 1, id="packed_equal_block_size"),
            pytest.param(SPLIT, 2, id="separate_regions_unequal_block_size"),
        ],
    )
    def test_upstream_still_gets_the_cases_it_handles(
        self, kv_per_block, block_size_ratio
    ):
        # Separate regions are a whole block of K or of V, so upstream's cut
        # stays between token ranges however the two sides size their blocks.
        worker = self._worker(kv_per_block, block_size_ratio)
        meta = self._meta()

        with patch.object(
            NixlBaseConnectorWorker, "_validate_remote_agent_handshake"
        ) as upstream:
            worker._validate_remote_agent_handshake(meta, remote_tp_size=1)

        # The DCP size travels with it: upstream 0.30 takes it as a third
        # argument, and dropping it would validate against the default.
        upstream.assert_called_once_with(meta, 1, 1)


class TestPairArea:
    """`pair_area`: the pieces one of our areas is exchanged in.

    A piece is the intersection of two areas' rectangles. What has to hold is
    that it is one contiguous range on BOTH sides, and that only the ranges
    neither side cuts are merged.
    """

    FULL = (0, 16)

    def _pair(self, local, peers, *, kv_per_block=1, page=256, write=False):
        return pair_area(
            AreaRect(local, self.FULL),
            page,
            [
                (region, AreaRect(heads, self.FULL), page_r)
                for region, heads, page_r in peers
            ],
            kv_per_block=kv_per_block,
            every_holder=write,
            region_id=0,
        )

    @pytest.mark.parametrize(
        "local, peers",
        [
            pytest.param((0, 2), [(0, (0, 1), 128), (1, (1, 2), 128)], id="peer_finer"),
            pytest.param((2, 4), [(0, (0, 4), 512)], id="we_are_finer"),
        ],
    )
    def test_a_packed_block_breaks_in_two_when_the_cuts_differ(self, local, peers):
        # One side reads a head band out of the other's block, and under a packed
        # block that band sits once in K and once in V.
        pieces = self._pair(local, peers, kv_per_block=2)
        assert [p.local_offset for p in pieces][:2] == [0, 128]
        assert len(pieces) == 2 * len({p.remote_region for p in pieces})

    @pytest.mark.parametrize("page_r", [128, 512])
    def test_a_replica_holding_another_block_length_is_refused(self, page_r):
        # A copy of the same heads in a shorter block would be written past its
        # end; in a longer one, it lays them out in a way we do not know.
        with pytest.raises(RuntimeError, match=f"{page_r}B a block, against 256B here"):
            self._pair((0, 2), [(0, (0, 2), 256), (1, (0, 2), page_r)], write=True)

    def test_a_peer_head_of_another_width_is_refused(self):
        # The handshake compares area 0 of a region only; a later area whose
        # heads are wider would be read short, from the start of its head.
        with pytest.raises(RuntimeError, match="128B a block .* and 256B"):
            self._pair((0, 1), [(0, (0, 2), 512)], page=128)

    def test_a_token_of_another_width_is_refused(self):
        # The sides hold different token ranges, so a head's width is not
        # comparable; a token costs 16B here and 8B on the peer.
        with pytest.raises(RuntimeError, match="a KV head is 64B a block over 4"):
            pair_area(
                AreaRect((0, 1), (0, 4)),
                64,
                [(0, AreaRect((0, 1), (0, 16)), 128)],
                kv_per_block=1,
                every_holder=False,
                region_id=0,
            )

    @pytest.mark.parametrize("write, holders", [(False, [0]), (True, [0, 1])])
    def test_a_token_range_s_replicas_are_read_once_and_written_each(
        self, write, holders
    ):
        # Two peer areas hold the same head over the same tokens.
        pieces = pair_area(
            AreaRect((0, 1), (0, 4)),
            64,
            [(r, AreaRect((0, 1), (0, 16)), 256) for r in (0, 1)],
            kv_per_block=1,
            every_holder=write,
            region_id=0,
        )
        assert [p.remote_region for p in pieces] == holders
        assert {(p.remote_offset, p.length) for p in pieces} == {(0, 64)}

    def test_a_packed_block_stays_one_range_at_equal_cuts(self):
        # Both sides name whole blocks, and K and V are adjacent inside one.
        assert self._pair((0, 2), [(0, (0, 2), 256)], kv_per_block=2) == [
            Piece(0, 0, 0, 256, 4)
        ]

    @pytest.mark.parametrize(
        "local, peers",
        [
            pytest.param((0, 2), [(0, (0, 1), 128), (1, (1, 2), 128)], id="peer_finer"),
            pytest.param((2, 4), [(0, (0, 4), 512)], id="we_are_finer"),
        ],
    )
    def test_separate_regions_never_break(self, local, peers):
        # A region is K or V alone, so a head band is contiguous however the two
        # sides cut heads: one piece per peer slice.
        pieces = self._pair(local, peers)
        assert len(pieces) == len(peers)

    @pytest.mark.parametrize(
        "peers, expected",
        [
            # Peer coarser: an offset into its slice, one piece.
            ([(0, (0, 4), 512)], 1),
            # Peer finer: one piece per slice of it our area spans.
            ([(0, (0, 1), 128), (1, (1, 2), 128)], 2),
            ([(r, (r, r + 1), 128) for r in range(3)], 2),
        ],
    )
    def test_an_area_splits_only_where_the_peer_is_finer(self, peers, expected):
        assert len(self._pair((0, 2), peers)) == expected

    def test_a_replica_is_read_once_and_written_every_time(self):
        # Reading any copy answers; writing one leaves the peer's other chiplets
        # on stale KV. The local range repeats while the destination moves.
        peers = [(0, (0, 2), 256), (1, (0, 2), 256)]
        read = self._pair((0, 2), peers)
        written = self._pair((0, 2), peers, write=True)
        assert [p.remote_region for p in read] == [0]
        assert [p.remote_region for p in written] == [0, 1]
        assert {p.local_offset for p in written} == {0}

    def test_heads_the_peer_does_not_hold_are_refused(self):
        # The peer holds half of what this area needs, so the other half would
        # never arrive; a read that reports success there is the failure.
        with pytest.raises(
            RuntimeError,
            match=r"outside the peer's range: the areas offered \(0\.\.0 x 0\.\.15\)",
        ):
            self._pair((0, 2), [(0, (0, 1), 128)])


class TestADeadPeerIsReported:
    # The heartbeat is the one call made against a producer every step, so a
    # connection that died after the handshake surfaces here first. Upstream
    # logs it and moves on, which leaves the reads against that peer unreported
    # and its agents in place.

    @staticmethod
    def _worker():
        w = RblnNixlPullConnectorWorker.__new__(RblnNixlPullConnectorWorker)
        w.engine_id = "local"
        w.nixl_wrapper = MagicMock()
        w._recving_metadata = {}
        w._recving_transfers = collections.defaultdict(list)
        w._recv_failures = set()
        w._is_hma_required = False
        w.xfer_stats = MagicMock()
        w._engines_to_rehandshake = set()
        w._remote_agents = {"eng": {(0, 0): "agent0", (0, 1): "agent1"}}
        w._hb_handshake_notif_only = False
        # Handshake already done for every engine a heartbeat names.
        w._ensure_handshake = MagicMock(return_value=None)
        w._cleanup_remote_engine = MagicMock(
            side_effect=lambda eid, **_: w._remote_agents.pop(eid)
        )
        return w

    @staticmethod
    def _meta(engine_id="eng", local_ids=((1, 2),)):
        meta = MagicMock()
        meta.remote = MagicMock(engine_id=engine_id)
        meta.local_block_ids = [list(g) for g in local_ids]
        return meta

    @staticmethod
    def _heartbeat(req_ids=frozenset(), *, dcp_size=1, pp_size=1):
        metadata = MagicMock()
        metadata.heartbeat_by_engine = {
            "eng": MagicMock(
                req_ids=set(req_ids),
                host="h",
                port=1,
                tp_size=1,
                dcp_size=dcp_size,
                pp_size=pp_size,
            )
        }
        return metadata

    def test_the_handshake_gets_the_arguments_upstream_names(self):
        # This override mirrors upstream's body, and the call is positional, so
        # a parameter added between the ones it passes lands every later
        # argument one place left without raising. Bind against the real
        # signature rather than a literal tuple: the point is the contract.
        w = self._worker()

        w._send_heartbeats(self._heartbeat(dcp_size=3, pp_size=2))

        call = w._ensure_handshake.call_args
        bound = inspect.signature(NixlBaseConnectorWorker._ensure_handshake).bind(
            w, *call.args, **call.kwargs
        )
        bound.apply_defaults()
        assert bound.arguments["dcp_size"] == 3
        assert bound.arguments["pp_size"] == 2
        assert bound.arguments["notif_agents_only"] is False

    def test_a_push_consumer_s_heartbeat_handshake_is_notif_only_at_any_pp(self):
        # A producer without pipeline stages included: a full handshake would
        # run the writing side's checks, and a refusal there fails the
        # PUSH_REG waiting on the same handshake.
        w = self._worker()
        w._hb_handshake_notif_only = True

        w._send_heartbeats(self._heartbeat(pp_size=1))

        call = w._ensure_handshake.call_args
        bound = inspect.signature(NixlBaseConnectorWorker._ensure_handshake).bind(
            w, *call.args, **call.kwargs
        )
        assert bound.arguments["notif_agents_only"] is True

    def test_a_failing_heartbeat_fails_that_peers_reads(self):
        w = self._worker()
        w._recving_metadata = {"r0": self._meta(), "r1": self._meta("other")}
        w._recving_transfers["r0"].append(7)
        w.nixl_wrapper.send_notif.side_effect = RuntimeError("conn is gone")

        w._send_heartbeats(self._heartbeat({"r0"}))

        assert w._recv_failures == {"r0"}
        # The in-flight handle goes with it: left in place, _pop_done_transfers
        # would report r0 a second time with its metadata already gone.
        w.nixl_wrapper.release_xfer_handle.assert_called_once_with(7)
        assert "r0" not in w._recving_transfers
        assert w._engines_to_rehandshake == set()

    def test_a_failing_heartbeat_drops_the_handshake_state(self):
        # _ensure_handshake skips the reconnect while the entry is there, so
        # the pair never recovers until the process dies.
        w = self._worker()
        w.nixl_wrapper.send_notif.side_effect = RuntimeError("conn is gone")

        w._send_heartbeats(self._heartbeat())

        w._cleanup_remote_engine.assert_called_once_with("eng", log_eviction=False)
        assert "eng" not in w._remote_agents
        # One attempt, not one per agent of an engine already declared gone.
        assert w.nixl_wrapper.send_notif.call_count == 1

    def test_a_teardown_that_itself_fails_still_frees_the_reconnect(self):
        # The teardown talks to the agent it is dropping, so the dead
        # connection can fail it halfway; the entry still has to go.
        w = self._worker()
        w.nixl_wrapper.send_notif.side_effect = RuntimeError("conn is gone")
        w._cleanup_remote_engine = MagicMock(side_effect=RuntimeError("agent is gone"))

        w._send_heartbeats(self._heartbeat())

        assert "eng" not in w._remote_agents


class TestAReadThatMovedNoBlock:
    # An empty local list means no async load was scheduled, so the request is
    # not in WAITING_FOR_REMOTE_KVS -- naming it to the scheduler trips
    # `assert req_id in self.requests`. Both producers of such a read reach the
    # failure path: a full prefix hit and an aborted request's release notify.

    def test_it_is_not_reported(self):
        w = TestADeadPeerIsReported._worker()
        w._recving_metadata = {"r0": TestADeadPeerIsReported._meta(local_ids=())}

        w._handle_failed_transfer("r0", None, w._recv_failures)

        assert not w._recv_failures

    def test_a_second_report_says_the_same(self):
        # The handshake done-callback runs on the executor thread, so it and
        # the heartbeat can reach one request. Consuming the entry would leave
        # the second one with no block list to judge by.
        w = TestADeadPeerIsReported._worker()
        w._recving_metadata = {"r0": TestADeadPeerIsReported._meta(local_ids=())}

        w._handle_failed_transfer("r0", None, w._recv_failures)
        w._handle_failed_transfer("r0", None, w._recv_failures)

        assert not w._recv_failures


class TestMetadataTakenMidSweep:
    # `_handle_dead_engine` sweeps `_recving_metadata` while that same
    # executor thread may be dropping entries from it.

    class _PopsWhenRead:
        """A read whose peer cannot be named without losing another entry."""

        def __init__(self, store, victim):
            self._store = store
            self._victim = victim
            self.local_block_ids = [[1, 2]]

        @property
        def remote(self):
            self._store.pop(self._victim, None)
            return MagicMock(engine_id="eng")

    def test_the_sweep_survives_it(self):
        w = TestADeadPeerIsReported._worker()
        store: dict = {}
        store["r0"] = self._PopsWhenRead(store, "r1")
        store["r1"] = TestADeadPeerIsReported._meta()
        store["r2"] = TestADeadPeerIsReported._meta()
        w._recving_metadata = store

        w._handle_dead_engine("eng")

        # Reading the live view would raise here; the sweep still covers every
        # read it set out to, including the one taken from under it.
        assert w._recv_failures == {"r0", "r1", "r2"}
