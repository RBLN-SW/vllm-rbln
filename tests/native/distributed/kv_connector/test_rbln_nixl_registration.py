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


# Unit coverage: turning this engine's KV tensors into a region table and
# registering that memory with NIXL -- the page and region arithmetic, the head
# band each region carries, and the layouts registration refuses outright.

import sys
import types
from typing import Any
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlBaseConnectorWorker,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    UniformTypeKVCacheSpecs,
)

from tests.native.distributed.kv_connector.utils import (
    KvGeometry,
    build_worker,
    patch_in_package,
    patched_in_package,
)
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


def _impl_layer_spec(page_size_bytes=4096, num_kv_heads=8):
    # Full-attention spec stand-in: .page_size_bytes and .num_kv_heads are read,
    # and it must fail the isinstance(MambaSpec/UniformTypeKVCacheSpecs) checks.
    # num_kv_heads defaults to the 8 the tensors _impl_kv_caches builds.
    spec = MagicMock(spec=FullAttentionSpec)
    spec.page_size_bytes = page_size_bytes
    spec.num_kv_heads = num_kv_heads
    return spec


def _impl_kv_caches(num_blocks=128, base_ptr=0x10000, names=("l0", "l1")):
    # Full-layer tensors: shape (K/V, num_blocks, heads, ..., dim). Only the
    # entry's address is read here; the per-region bytes come from _split_kv.
    kv = {}
    for i, name in enumerate(names):
        t = MagicMock()
        t.shape = (2, num_blocks, 8, 1, 64, 64)
        t.data_ptr.return_value = base_ptr + i * 0x10000
        t.get_device.return_value = 0
        t.zero_ = MagicMock()
        kv[name] = t
    return kv


def _mla_kv_caches(num_blocks=128, base_ptr=0x10000, page_size_bytes=4096):
    # Two MLA layers: a 3D latent cache, block axis first, no K/V split. The
    # whole entry is the region, so its bytes must be num_blocks x the page.
    kv = {}
    for i, name in enumerate(("l0", "l1")):
        t = MagicMock()
        t.shape = (num_blocks, page_size_bytes)
        t.numel.return_value = num_blocks * page_size_bytes
        t.element_size.return_value = 1
        t.data_ptr.return_value = base_ptr + i * 0x10000
        t.get_device.return_value = 0
        t.zero_ = MagicMock()
        kv[name] = t
    return kv


def _split_kv(num_blocks):
    # Fake TransferTopology.get_transfer_cache_regions: K and V as two region
    # tensors, each with shape[0] == num_blocks. Each half is sized from THIS
    # layer's spec, so its byte footprint agrees with the spec -- what the
    # per-region span check asserts.
    def _split(cache, spec):
        half = spec.page_size_bytes // 2
        regions = []
        for off in (0, 0x100):
            region = MagicMock()
            region.shape = (num_blocks, half)
            region.numel.return_value = num_blocks * half
            region.element_size.return_value = 1
            region.data_ptr.return_value = cache.data_ptr() + off
            regions.append(region)
        return regions

    return _split


def _patch_worker_nixl_symbols(
    topo, *, mamba_spec=None, uniform_spec=None, topology_cls=None
):
    # The isinstance() targets get a dummy class, which the Full-attention specs
    # cannot match; a test that wants one of those branches passes the real
    # class instead. Patched across the package because these are
    # `from x import Y` bindings, so the binding that matters is the one in
    # whichever module holds the function under test.
    msgspec_mock = MagicMock()
    msgspec_mock.msgpack.Encoder.return_value.encode.return_value = b"meta"
    return patch_in_package(
        TransferTopology=topology_cls or MagicMock(return_value=topo),
        compute_nixl_compatibility_hash=MagicMock(return_value="hash"),
        MambaSpec=mamba_spec or type("MambaSpec", (), {}),
        UniformTypeKVCacheSpecs=uniform_spec or type("UniformTypeKVCacheSpecs", (), {}),
        NixlAgentMetadata=MagicMock(),
        NixlHandshakePayload=MagicMock(),
        msgspec=msgspec_mock,
    )


def _impl_xfer_result(
    base_addrs=(0x20000, 0x20100, 0x30000, 0x30100),
    block_lens=(256, 256, 256, 256),
    slice_ids=None,
    n_shards=1,
    slices=1,
):
    xfer = MagicMock()
    xfer.base_addrs = list(base_addrs)
    xfer.block_lens = list(block_lens)
    xfer.reg_handle = "reg-handle"
    xfer.n_shards = n_shards
    xfer.slices = slices
    # Parallel to base_addrs: which logical slice each expanded region carries.
    # One area per region by default, so every region is its entry whole.
    xfer.slice_ids = list(slice_ids) if slice_ids is not None else [0] * len(base_addrs)
    return xfer


def _fake_nixl_rbln(xfer_result):
    module: Any = types.ModuleType("nixl_rbln")
    module.register_kv_regions = MagicMock(return_value=xfer_result)
    module.ensure_rbln_backend = MagicMock()
    return module


def _prep_impl_worker(monkeypatch, *, num_blocks=128, block_size=64):
    # A D2D worker back-filled with the attributes upstream __init__ would set.
    worker = build_worker(
        monkeypatch,
        kv_buffer_device="rbln",
        num_blocks=num_blocks,
        block_size=block_size,
        nixl_available=True,
    )
    worker.tp_rank = 0
    worker.world_size = 1
    worker.use_mla = False
    worker._has_mamba = False
    worker.attn_backends = []
    worker.backend_name = "rbln"
    worker.kv_cache_layout = "HND"
    worker._mamba_ssm_size = 0
    worker.model_config = MagicMock()
    worker.model_config.get_total_num_kv_heads.return_value = 8
    worker.host_xfer_buffers = {}
    worker.nixl_wrapper = MagicMock()
    worker.nixl_wrapper.get_agent_metadata.return_value = b"agent-meta"
    worker.kv_caches_base_addr = {worker.engine_id: {0: None}}
    worker._registered_descs = []
    worker.dst_num_blocks = {}
    worker.src_xfer_handles_by_block_size = {}
    return worker


class TestRegisterKvCaches:
    def test_d2d_stashes_and_defers(self, monkeypatch):
        # D2D can't register until warm-up materializes memory: stash and return.
        worker = build_worker(monkeypatch, kv_buffer_device="rbln")
        worker.register_kv_caches({"layer0": "tensor"})
        assert worker._pending_kv_caches == {"layer0": "tensor"}

    def test_host_bounce_creates_backend_and_delegates(self, monkeypatch):
        # Host-bounce with the adapter creates the RBLN backend on the agent,
        # then delegates registration to upstream.
        worker = build_worker(monkeypatch, kv_buffer_device="cpu", nixl_available=True)
        worker.nixl_wrapper = "wrapper"
        worker._layer_specs = {"layer0": _impl_layer_spec()}
        worker.block_len_per_layer = [2048, 2048]
        ensured = []
        monkeypatch.setattr(
            sys.modules["nixl_rbln"],
            "ensure_rbln_backend",
            lambda wrapper, device_id=0: ensured.append((wrapper, device_id)),
            raising=False,
        )
        delegated = []
        monkeypatch.setattr(
            NixlBaseConnectorWorker,
            "register_kv_caches",
            lambda self, kv: delegated.append(kv),
        )
        worker.register_kv_caches({"layer0": "tensor"})
        assert ensured == [("wrapper", 0)]
        assert delegated == [{"layer0": "tensor"}]
        assert worker._pending_kv_caches is None
        # Host staging needs the per-region counts too: a pipelined peer reaches
        # the per-head handshake check through it.
        assert worker._logical_region_kv_heads == [8, 8]

    def test_host_bounce_rejects_differing_per_layer_sizes(self, monkeypatch):
        # Pins the refusal at this path's own point rather than upstream's assert.
        worker = build_worker(monkeypatch, kv_buffer_device="cpu", nixl_available=True)
        worker.nixl_wrapper = "wrapper"
        worker._layer_specs = {
            "l0": _impl_layer_spec(page_size_bytes=4096),
            "l1": _impl_layer_spec(page_size_bytes=16384, num_kv_heads=32),
        }
        monkeypatch.setattr(
            sys.modules["nixl_rbln"],
            "ensure_rbln_backend",
            lambda wrapper, device_id=0: None,
            raising=False,
        )
        delegated = []
        monkeypatch.setattr(
            NixlBaseConnectorWorker,
            "register_kv_caches",
            lambda self, kv: delegated.append(kv),
        )
        with pytest.raises(RuntimeError, match="host staging cannot register"):
            worker.register_kv_caches({"l0": "tensor", "l1": "tensor"})
        assert delegated == []


class TestInitializeHostXferBuffer:
    def test_allocates_one_buffer_per_layer_preserving_order(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_buffer_device="cpu")
        worker.kv_cache_layout = "HND"
        kv_caches = {
            "l0": torch.zeros(2, 4, dtype=torch.float16),
            "l1": torch.zeros(3, 5, dtype=torch.float16),
        }
        worker.initialize_host_xfer_buffer(kv_caches)
        assert list(worker.host_xfer_buffers.keys()) == ["l0", "l1"]
        assert worker.host_xfer_buffers["l0"].shape == (2, 4)
        assert worker.host_xfer_buffers["l1"].shape == (3, 5)

    def test_rejects_non_hnd_layout(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_buffer_device="cpu")
        worker.kv_cache_layout = "NHD"
        with pytest.raises(AssertionError, match="HND"):
            worker.initialize_host_xfer_buffer(
                {"l0": torch.zeros(2, 4, dtype=torch.float16)}
            )

    def test_mla_accepts_any_layout(self, monkeypatch):
        # MLA has no head axis to order, so upstream advertises no required
        # layout and the resolved value is whatever the fallback picked. The
        # 3D latent shape must survive unchanged.
        worker = build_worker(monkeypatch, kv_buffer_device="cpu", use_mla=True)
        worker.kv_cache_layout = "NHD"
        worker.initialize_host_xfer_buffer(
            {"l0": torch.zeros(4, 64, 576, dtype=torch.float16)}
        )
        assert worker.host_xfer_buffers["l0"].shape == (4, 64, 576)

    def test_rejects_odd_byte_footprint(self, monkeypatch):
        # The page-aligned host buffer is backed by an fp16 (2-byte) allocation,
        # so a cache whose byte footprint is odd cannot be tiled.
        worker = build_worker(monkeypatch, kv_buffer_device="cpu")
        worker.kv_cache_layout = "HND"
        with pytest.raises(AssertionError, match="multiple of 2"):
            worker.initialize_host_xfer_buffer(
                {"l0": torch.zeros(1, dtype=torch.uint8)}  # 1 byte
            )


class TestSetHostXferBufferOps:
    def test_noop_when_kv_buffer_is_not_cpu(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_buffer_device="rbln")
        worker.set_host_xfer_buffer_ops("copy_op")
        assert not hasattr(worker, "copy_blocks")

    def test_assigns_copy_on_host_bounce(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_buffer_device="cpu")
        worker.use_host_buffer = True
        worker.set_host_xfer_buffer_ops("copy_op")
        assert worker.copy_blocks == "copy_op"


class TestRegisterKvCachesImpl:
    # The deferred D2D body: hands the logical K/V regions to
    # nixl_rbln.register_kv_regions and absorbs the returned transfer tables.
    def test_registers_with_vram_segment_and_captures_xfer_tables(self, monkeypatch):
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        xfer_result = MagicMock()
        # One transfer region per (logical region x chiplet area): two layers
        # split into K/V, one area.
        xfer_result.base_addrs = [0x20000, 0x20100, 0x30000, 0x30100]
        xfer_result.block_lens = [256, 256, 256, 256]
        xfer_result.reg_handle = "reg-handle"
        xfer_result.n_shards = 1
        xfer_result.slices = 1
        xfer_result.slice_ids = [0] * len(xfer_result.base_addrs)
        fake = _fake_nixl_rbln(xfer_result)

        topo = MagicMock(
            is_kv_layout_blocks_first=False,
            cross_layers_blocks=False,
        )
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel") as mock_rebel,
            patch.object(
                worker,
                "register_local_xfer_handler",
                return_value=("local-handle", [(0x0, 0, 0)]),
            ),
        ):
            mock_rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
            worker._register_kv_caches_impl(kv_caches)

        # rbln_ctx_ptr comes from rebel.context_of(kv_tensor), not a runtime handle.
        mock_rebel.context_of.assert_called_once_with(next(iter(kv_caches.values())))

        # nixl-rbln invoked once, with the D2D VRAM segment + resolved ctx ptr.
        fake.register_kv_regions.assert_called_once()
        called = fake.register_kv_regions.call_args.kwargs
        assert called["mem"] == "VRAM"
        assert called["rbln_ctx_ptr"] == 0x1000

        # Returned transfer tables absorbed into worker state.
        assert worker.device_id == 0
        assert worker.block_len_per_layer == [256, 256, 256, 256]
        assert worker.kv_caches_base_addr[worker.engine_id][0] == [
            0x20000,
            0x20100,
            0x30000,
            0x30100,
        ]
        assert worker._registered_descs == ["reg-handle"]

        # 4 regions (2 layers x K/V), layout-blocks-first=False so no x2.
        assert worker.num_regions == 4
        assert worker.num_descs == 4 * worker.num_blocks
        # Full attention is head-sharded, so no region transfers REPLICATE.
        assert worker._region_is_mla == [False] * 4

        # Final hand-offs into upstream's transfer state.
        assert worker.device_kv_caches is kv_caches
        assert worker.dst_num_blocks[worker.engine_id] == worker.num_blocks
        assert (
            worker.src_xfer_handles_by_block_size[worker.block_size] == "local-handle"
        )

    def test_layout_blocks_first_doubles_region_count(self, monkeypatch):
        # is_kv_layout_blocks_first flips the region count to 2x (K and V share a
        # region tensor), which cascades into num_descs.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        xfer_result = MagicMock()
        xfer_result.base_addrs = [0x20000, 0x20100, 0x30000, 0x30100]
        xfer_result.block_lens = [256, 256, 256, 256]
        xfer_result.reg_handle = "reg-handle"
        xfer_result.n_shards = 1
        xfer_result.slices = 1
        xfer_result.slice_ids = [0] * len(xfer_result.base_addrs)
        fake = _fake_nixl_rbln(xfer_result)

        topo = MagicMock(
            is_kv_layout_blocks_first=True,
            cross_layers_blocks=False,
        )
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel") as mock_rebel,
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            mock_rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
            worker._register_kv_caches_impl(kv_caches)

        assert worker.num_regions == 8  # 4 base addrs x 2 (blocks-first)
        assert worker.num_descs == 8 * worker.num_blocks

    def test_mla_registers_one_replicated_region_per_layer_and_area(self, monkeypatch):
        # MLA is key-only, so a layer contributes one logical region rather than
        # two, and every chiplet area of it carries the same latent -> REPLICATE.
        worker = _prep_impl_worker(monkeypatch)
        worker.use_mla = True
        worker._kv_split_axis = KVSplitAxis.NON_HEAD  # see the head-axis case
        spec = MagicMock(spec=MLAAttentionSpec)
        spec.page_size_bytes = 4096
        spec.num_kv_heads = 1
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _mla_kv_caches(num_blocks=worker.num_blocks)

        areas = 4
        xfer_result = MagicMock()
        # 2 layers x 1 region x 4 areas, each area a full-length replica (g=1).
        xfer_result.base_addrs = [0x20000 + 0x1000 * i for i in range(2 * areas)]
        xfer_result.block_lens = [4096] * (2 * areas)
        xfer_result.reg_handle = "reg-handle"
        xfer_result.n_shards = areas
        xfer_result.slices = 1
        xfer_result.slice_ids = [0] * (2 * areas)
        fake = _fake_nixl_rbln(xfer_result)

        topo = MagicMock(
            is_kv_layout_blocks_first=False,
            cross_layers_blocks=False,
        )
        # split_k_and_v is False for MLA upstream, so one region per layer.
        topo.get_transfer_cache_regions.side_effect = lambda cache, _spec: [cache]

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel") as mock_rebel,
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            mock_rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
            worker._register_kv_caches_impl(kv_caches)

        assert len(fake.register_kv_regions.call_args.args[1]) == 2  # logical regions
        assert worker._region_is_mla == [True] * (2 * areas)
        assert worker.num_regions == 2 * areas
        assert worker.num_descs == 2 * areas * worker.num_blocks
        assert (worker._kv_areas, worker._kv_slices) == (areas, 1)
        # One head over one slice: nothing was cut, so the axis stays HEAD.
        assert worker._kv_split_axis is KVSplitAxis.HEAD

    def test_region_flags_must_cover_every_transfer_region(self, monkeypatch):
        # A logical region count that does not account for the returned table
        # would mislabel regions, so it fails rather than guesses.
        worker = _prep_impl_worker(monkeypatch)
        worker.use_mla = True
        spec = MagicMock(spec=MLAAttentionSpec)
        spec.page_size_bytes = 4096
        spec.num_kv_heads = 1
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _mla_kv_caches(num_blocks=worker.num_blocks)

        xfer_result = MagicMock()
        xfer_result.base_addrs = [0x20000, 0x21000, 0x22000]  # not 2 x n_shards
        xfer_result.block_lens = [4096] * 3
        xfer_result.reg_handle = "reg-handle"
        xfer_result.n_shards = 4
        xfer_result.slices = 1
        xfer_result.slice_ids = [0] * 3
        fake = _fake_nixl_rbln(xfer_result)

        topo = MagicMock(
            is_kv_layout_blocks_first=False,
            cross_layers_blocks=False,
        )
        topo.get_transfer_cache_regions.side_effect = lambda cache, _spec: [cache]

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel") as mock_rebel,
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
            pytest.raises(AssertionError, match="transfer region"),
        ):
            mock_rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
            worker._register_kv_caches_impl(kv_caches)

    def test_constructs_transfer_topology_with_expected_kwargs(self, monkeypatch):
        # The happy-path test fully fakes TransferTopology, so a wrong ctor kwarg
        # would slip through; this pins the exact call.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        xfer_result = MagicMock()
        xfer_result.base_addrs = [0x20000, 0x20100, 0x30000, 0x30100]
        xfer_result.block_lens = [256, 256, 256, 256]
        xfer_result.reg_handle = "reg-handle"
        xfer_result.n_shards = 1
        xfer_result.slices = 1
        xfer_result.slice_ids = [0] * len(xfer_result.base_addrs)
        fake = _fake_nixl_rbln(xfer_result)

        topo = MagicMock(
            is_kv_layout_blocks_first=False,
            cross_layers_blocks=False,
        )
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)
        topology_cls = MagicMock(return_value=topo)

        with (
            _patch_worker_nixl_symbols(topo, topology_cls=topology_cls),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel") as mock_rebel,
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            mock_rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
            worker._register_kv_caches_impl(kv_caches)
            topology_cls.assert_called_once_with(
                tp_rank=0,
                tp_size=1,
                block_size=64,
                engine_id="test-engine",
                is_mla=False,
                total_num_kv_heads=8,
                attn_backends=[],
                tensor_shape=(2, 128, 8, 1, 64, 64),
                is_mamba=False,
            )

    # The three shape branches below all decide `full_block_len`, the per-block
    # stride handed to nixl-rbln. Baseline for comparison is the first test in
    # this class: a 4096B page over two K/V regions is 2048B per block.

    def test_a_group_spec_is_unwrapped_to_this_layer_s_spec(self, monkeypatch):
        # A uniform-type group publishes one spec object for every layer of the
        # group; the geometry lives on the member spec, and the group has no page
        # size of its own to read.
        worker = _prep_impl_worker(monkeypatch)
        member = _impl_layer_spec(page_size_bytes=4096)
        group = MagicMock(spec=UniformTypeKVCacheSpecs)
        group.kv_cache_specs = {"l0": member, "l1": member}
        worker._layer_specs = {"l0": group, "l1": group}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        fake = _fake_nixl_rbln(_impl_xfer_result())
        topo = MagicMock(is_kv_layout_blocks_first=False, cross_layers_blocks=False)
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo, uniform_spec=UniformTypeKVCacheSpecs),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel"),
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            worker._register_kv_caches_impl(kv_caches)

        regions = fake.register_kv_regions.call_args.args[1]
        assert [block_len for _, _, block_len in regions] == [2048] * 4

    def test_a_draft_layer_with_its_own_page_size_is_registered(self, monkeypatch):
        # Requiring one size for every non-MLA tensor would reject this outright;
        # registration has to describe each region by its own geometry.
        worker = _prep_impl_worker(monkeypatch)
        target = _impl_layer_spec(page_size_bytes=4096, num_kv_heads=8)
        draft = _impl_layer_spec(page_size_bytes=16384, num_kv_heads=32)
        worker._layer_specs = {"l0": target, "l1": target, "l2": draft}
        # A head count is accepted only if some model here has it, so the draft
        # has to be declared.
        spec_cfg = MagicMock(method="eagle3")
        spec_cfg.draft_model_config.model = "draft"
        spec_cfg.draft_model_config.revision = None
        spec_cfg.draft_model_config.code_revision = None
        spec_cfg.draft_model_config.get_total_num_kv_heads.return_value = 32
        worker.vllm_config.speculative_config = spec_cfg
        # The draft layer is named past the target's depth, so it comes last.
        kv_caches = _impl_kv_caches(
            num_blocks=worker.num_blocks, names=("l0", "l1", "l2")
        )

        fake = _fake_nixl_rbln(
            _impl_xfer_result(
                base_addrs=(0x20000, 0x20100, 0x30000, 0x30100, 0x40000, 0x40100),
                block_lens=(2048, 2048, 2048, 2048, 8192, 8192),
            )
        )
        topo = MagicMock(is_kv_layout_blocks_first=False, cross_layers_blocks=False)
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel"),
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            worker._register_kv_caches_impl(kv_caches)

        regions = fake.register_kv_regions.call_args.args[1]
        # Each half is that layer's own page, not the first layer's.
        assert [block_len for _, _, block_len in regions] == [
            2048,
            2048,
            2048,
            2048,
            8192,
            8192,
        ]
        # Without a per-region count the draft's regions would be banded at the
        # target's width.
        assert worker._logical_region_kv_heads == [8, 8, 8, 8, 32, 32]
        assert worker.num_regions == 6

    def test_replicated_heads_get_no_head_band(self, monkeypatch):
        # 4 heads at TP 8: upstream floors the share at 1, so the product reads
        # 8 and passes the divisibility guard (`_layer_kv_heads`). No count is
        # recorded, so the head-band paths refuse the region.
        worker = _prep_impl_worker(monkeypatch)
        worker.world_size = 8
        worker.model_config.get_total_num_kv_heads.return_value = 4
        spec = _impl_layer_spec(page_size_bytes=4096, num_kv_heads=1)
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        fake = _fake_nixl_rbln(_impl_xfer_result())
        topo = MagicMock(is_kv_layout_blocks_first=False, cross_layers_blocks=False)
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel"),
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            worker._register_kv_caches_impl(kv_caches)

        # Registration itself still serves.
        assert worker._logical_region_kv_heads == [None] * 4
        with pytest.raises(AssertionError, match="no head band"):
            worker._region_kv_heads(0)

    def test_two_models_sharing_a_per_rank_share_of_one_is_refused(self, monkeypatch):
        # Target 4 heads and draft 8 at TP 8: both report 1 per rank, so the
        # target's replicated layer is indistinguishable from the draft's real 8.
        # Banded at 8, rank 3 would claim head 3 while it holds head 1.
        worker = _prep_impl_worker(monkeypatch)
        worker.world_size = 8
        worker.model_config.get_total_num_kv_heads.return_value = 4
        spec_cfg = MagicMock(method="eagle3")
        spec_cfg.draft_model_config.model = "draft"
        spec_cfg.draft_model_config.revision = None
        spec_cfg.draft_model_config.code_revision = None
        spec_cfg.draft_model_config.get_total_num_kv_heads.return_value = 8
        worker.vllm_config.speculative_config = spec_cfg
        spec = _impl_layer_spec(page_size_bytes=4096, num_kv_heads=1)
        worker._layer_specs = {"l0": spec, "l1": spec}

        assert worker._layer_kv_heads("l0") is None

    def test_a_head_count_a_model_in_the_engine_has_is_kept(self, monkeypatch):
        # The counterpart: 8 heads over TP 8 replicates nothing, so the band is
        # real and recorded.
        worker = _prep_impl_worker(monkeypatch)
        worker.world_size = 8
        worker.model_config.get_total_num_kv_heads.return_value = 8
        spec = _impl_layer_spec(page_size_bytes=4096, num_kv_heads=1)
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        fake = _fake_nixl_rbln(_impl_xfer_result())
        topo = MagicMock(is_kv_layout_blocks_first=False, cross_layers_blocks=False)
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel"),
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            worker._register_kv_caches_impl(kv_caches)

        assert worker._logical_region_kv_heads == [8] * 4

    def test_each_region_records_its_own_slice_count(self, monkeypatch):
        # Each region's own count comes from `slice_ids`, not from the `slices`
        # scalar, which describes the LAST region only -- a draft's.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        fake = _fake_nixl_rbln(
            _impl_xfer_result(
                base_addrs=tuple(0x20000 + 0x100 * i for i in range(16)),
                block_lens=(256,) * 16,
                # Two entries of four areas each: the first tiled over all four,
                # the second cut in two and replicated across pairs.
                slice_ids=(0, 1, 2, 3, 0, 1, 2, 3, 0, 0, 1, 1, 0, 0, 1, 1),
                n_shards=4,
                slices=2,
            )
        )
        topo = MagicMock(is_kv_layout_blocks_first=False, cross_layers_blocks=False)
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel"),
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            worker._register_kv_caches_impl(kv_caches)

        assert worker._logical_region_slices == [4, 4, 2, 2]
        # The scalar the library reported is only the last region's, which is
        # exactly why the list above cannot be derived from it.
        assert worker._kv_slices == 2

    def test_a_region_whose_span_disagrees_with_its_spec_is_rejected(self, monkeypatch):
        # Without the span check a spec/allocation disagreement shifts every
        # descriptor past block 0.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec(page_size_bytes=4096)
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        def _short_regions(cache, layer_spec):
            regions = _split_kv(worker.num_blocks)(cache, layer_spec)
            # One block's worth missing, with shape[0] still right so the block
            # count assert cannot catch it.
            regions[0].numel.return_value -= 2048
            return regions

        fake = _fake_nixl_rbln(_impl_xfer_result())
        topo = MagicMock(is_kv_layout_blocks_first=False, cross_layers_blocks=False)
        topo.get_transfer_cache_regions.side_effect = _short_regions

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel"),
            pytest.raises(AssertionError, match="is not 128 blocks of 2048B"),
        ):
            worker._register_kv_caches_impl(kv_caches)

    def test_mixed_mla_and_non_mla_layers_are_rejected(self, monkeypatch):
        # An MLA draft under a non-MLA target: the latent would be iterated as a
        # K/V pair and the region list would count blocks as regions.
        worker = _prep_impl_worker(monkeypatch)
        target = _impl_layer_spec(page_size_bytes=4096, num_kv_heads=8)
        latent = MagicMock(spec=MLAAttentionSpec)
        latent.page_size_bytes = 4096
        latent.num_kv_heads = 1
        worker._layer_specs = {"l0": target, "l1": latent}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        fake = _fake_nixl_rbln(_impl_xfer_result())
        topo = MagicMock(is_kv_layout_blocks_first=False, cross_layers_blocks=False)
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel"),
            pytest.raises(RuntimeError, match="mix MLA and non-MLA"),
        ):
            worker._register_kv_caches_impl(kv_caches)

        # Refused before anything was handed to nixl-rbln.
        fake.register_kv_regions.assert_not_called()

    def test_cross_layer_blocks_with_differing_page_sizes_are_rejected(
        self, monkeypatch
    ):
        # Without this the page keeps the KV-cache tensor count as a factor, so
        # the stride describes no layer.
        worker = _prep_impl_worker(monkeypatch)
        target = _impl_layer_spec(page_size_bytes=4096, num_kv_heads=8)
        draft = _impl_layer_spec(page_size_bytes=16384, num_kv_heads=32)
        worker._layer_specs = {"l0": target, "l1": draft}
        worker.kv_cache_config = MagicMock(kv_cache_tensors=[object(), object()])
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        fake = _fake_nixl_rbln(_impl_xfer_result())
        topo = MagicMock(is_kv_layout_blocks_first=False, cross_layers_blocks=True)
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel"),
            pytest.raises(RuntimeError, match="cross-layer blocks require one page"),
        ):
            worker._register_kv_caches_impl(kv_caches)

    def test_cross_layer_blocks_scale_the_page_by_the_tensor_count(self, monkeypatch):
        # One tensor holds every layer's blocks, so a layer's page covers the
        # whole set and the stride from one block to the next is that much wider.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec(page_size_bytes=4096)
        worker._layer_specs = {"l0": spec, "l1": spec}
        worker.kv_cache_config = MagicMock(kv_cache_tensors=[object(), object()])
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        fake = _fake_nixl_rbln(_impl_xfer_result())
        topo = MagicMock(is_kv_layout_blocks_first=False, cross_layers_blocks=True)
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel"),
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            worker._register_kv_caches_impl(kv_caches)

        regions = fake.register_kv_regions.call_args.args[1]
        assert [block_len for _, _, block_len in regions] == [2048 * 2] * 4

    def test_a_mamba_state_is_strided_by_the_logical_block(self, monkeypatch):
        # An SSM state has no K/V split and is counted in logical blocks, so its
        # page already covers every physical block one logical block expands to
        # -- unlike attention, where the page is one physical block.
        worker = _prep_impl_worker(monkeypatch)
        worker._has_mamba = True
        worker._logical_num_blocks = 8
        worker._physical_blocks_per_logical_kv_block = 2
        spec = MagicMock(spec=MambaSpec)
        spec.page_size_bytes = 4096
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        def _one_region(cache, _spec):
            region = MagicMock()
            region.shape = (worker._logical_num_blocks, 4)
            region.data_ptr.return_value = cache.data_ptr()
            return [region]

        fake = _fake_nixl_rbln(
            _impl_xfer_result(base_addrs=[0x20000, 0x30000], block_lens=[4096, 4096])
        )
        topo = MagicMock(is_kv_layout_blocks_first=False, cross_layers_blocks=False)
        topo.get_transfer_cache_regions.side_effect = _one_region

        with (
            _patch_worker_nixl_symbols(topo, mamba_spec=MambaSpec),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel"),
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            worker._register_kv_caches_impl(kv_caches)

        regions = fake.register_kv_regions.call_args.args[1]
        assert [block_len for _, _, block_len in regions] == [2048, 2048]

    def test_one_head_over_several_slices_is_a_non_head_cut(self, monkeypatch):
        # A single KV head cannot be cut along the head axis -- the compiler
        # replicates it instead -- so distinct slices can only have come from
        # another axis. This is the geometry a sparse-MLA model registers.
        worker = _prep_impl_worker(monkeypatch)
        worker.use_mla = True
        spec = MagicMock(spec=MLAAttentionSpec)
        spec.page_size_bytes = 4096
        spec.num_kv_heads = 1
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _mla_kv_caches(num_blocks=worker.num_blocks)

        areas = 4
        xfer_result = MagicMock()
        xfer_result.base_addrs = [0x20000 + 0x1000 * i for i in range(2 * areas)]
        xfer_result.block_lens = [1024] * (2 * areas)
        xfer_result.reg_handle = "reg-handle"
        xfer_result.n_shards = areas
        xfer_result.slices = areas
        xfer_result.slice_ids = [0, 1, 2, 3] * 2
        fake = _fake_nixl_rbln(xfer_result)

        topo = MagicMock(
            is_kv_layout_blocks_first=False,
            _cross_layers_blocks=False,
            cross_layers_blocks=False,
        )
        topo.get_transfer_cache_regions.side_effect = lambda cache, _spec: [cache]

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel") as mock_rebel,
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            mock_rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
            worker._register_kv_caches_impl(kv_caches)

        assert worker._kv_split_axis is KVSplitAxis.NON_HEAD

    def test_the_same_slice_count_over_several_heads_stays_head(self, monkeypatch):
        # Identical areas and slices to the case above, and the opposite answer:
        # 8 heads divide into 4 slices, so head tiling explains it and the
        # derivation must not claim more than it can prove.
        worker = _prep_impl_worker(monkeypatch)
        # HEAD is also the field's initial value, so start from the other one:
        # otherwise the assertion below passes just as well when the derivation
        # never runs at all.
        worker._kv_split_axis = KVSplitAxis.NON_HEAD
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        areas = 4
        xfer_result = MagicMock()
        xfer_result.base_addrs = [0x20000 + 0x1000 * i for i in range(4 * areas)]
        xfer_result.block_lens = [64] * (4 * areas)
        xfer_result.reg_handle = "reg-handle"
        xfer_result.n_shards = areas
        xfer_result.slices = areas
        xfer_result.slice_ids = [0, 1, 2, 3] * 4
        fake = _fake_nixl_rbln(xfer_result)

        topo = MagicMock(
            is_kv_layout_blocks_first=False,
            _cross_layers_blocks=False,
            cross_layers_blocks=False,
        )
        topo.get_transfer_cache_regions.side_effect = _split_kv(worker.num_blocks)

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel") as mock_rebel,
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            mock_rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
            worker._register_kv_caches_impl(kv_caches)

        assert worker._kv_split_axis is KVSplitAxis.HEAD

    def test_regions_cut_on_different_axes_are_rejected(self, monkeypatch):
        # One axis is advertised per engine, so regions that disagree cannot be
        # described -- and either choice mislabels the other's areas while every
        # byte count still adds up. Both layers are latents so the two axes
        # differ by geometry alone (a mixed MLA engine is refused earlier).
        worker = _prep_impl_worker(monkeypatch)
        worker.use_mla = True
        latent = MagicMock(spec=MLAAttentionSpec)
        latent.page_size_bytes = 4096
        latent.num_kv_heads = 1
        worker._layer_specs = {"l0": latent, "l1": latent}
        kv_caches = _mla_kv_caches(num_blocks=worker.num_blocks)

        areas = 4
        xfer_result = MagicMock()
        xfer_result.base_addrs = [0x20000 + 0x1000 * i for i in range(2 * areas)]
        xfer_result.block_lens = [1024] * (2 * areas)
        xfer_result.reg_handle = "reg-handle"
        xfer_result.n_shards = areas
        xfer_result.slices = areas
        xfer_result.slice_ids = [0, 1, 2, 3] + [0] * areas
        fake = _fake_nixl_rbln(xfer_result)

        topo = MagicMock(
            is_kv_layout_blocks_first=False,
            _cross_layers_blocks=False,
            cross_layers_blocks=False,
        )
        topo.get_transfer_cache_regions.side_effect = lambda cache, _spec: [cache]

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patched_in_package("rebel") as mock_rebel,
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
            pytest.raises(RuntimeError, match="not all cut on the same axis"),
        ):
            mock_rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
            worker._register_kv_caches_impl(kv_caches)


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
        w.vllm_config.speculative_config = None
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

    def test_an_ssm_group_is_refused_before_any_registration(self):
        # Not a PP constraint: the region table this connector publishes is one
        # entry per chiplet area, and an SSM group's cache is one blocks-first
        # region per layer. Both paths pass through register_kv_caches, so the
        # refusal belongs there.
        w = self._worker(pp_size=1, has_mamba=True)
        w.kv_buffer_device = "rbln"
        with pytest.raises(RuntimeError, match="Mamba/SSM"):
            w.register_kv_caches({"l0": MagicMock()})

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
        has_mamba=False,
        cross_layers=False,
    ):
        w = object.__new__(cls or RblnNixlPullConnectorWorker)
        # __init__ never ran, so the writer state shutdown() reaches through
        # __del__ is absent; silence it rather than leak an unraisable at GC.
        w.shutdown = lambda: None
        w.compat_hash = "BASE"
        # _check_pp_constraints reads these; a plain PP producer passes.
        w.vllm_config = MagicMock()
        w.vllm_config.parallel_config.pipeline_parallel_size = pp_size
        w.vllm_config.speculative_config = None
        w.transfer_topo = MagicMock()
        w.transfer_topo.cross_layers_blocks = cross_layers
        w._has_mamba = has_mamba
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
        with patched_in_package("get_pp_group", MagicMock(return_value=pp_group)):
            w._publish_handshake_metadata(self._base_meta(), layer_names)
        return w

    def test_publishing_is_where_the_pp_guard_fires(self):
        # The guard is only reached from here, so a test that calls it directly
        # cannot tell whether anything still does.
        with pytest.raises(RuntimeError, match="cross-layer-blocks"):
            self._publish(pp_rank=0, pp_size=2, layer_names=["l0"], cross_layers=True)

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
        w._has_mamba = False
        w._use_rbln_nixl_backend = False
        w.xfer_handshake_metadata = MagicMock(
            agent_metadata_bytes=msgspec.msgpack.encode(self._base_meta())
        )
        w._publish_handshake_metadata = MagicMock()
        kv_caches = {"l0": MagicMock(), "l1": MagicMock()}
        # Registration reads the layer specs and the transfer table upstream
        # fills, to record each region's head count.
        w._layer_specs = {
            name: MagicMock(page_size_bytes=4096, num_kv_heads=8) for name in kv_caches
        }
        w.block_len_per_layer = [2048, 2048, 2048, 2048]
        w.world_size = 1
        # A layer's head count is accepted only if a model in this engine has it:
        # the target's, or a speculative draft's where there is one.
        w.model_config = MagicMock()
        w.model_config.get_total_num_kv_heads.return_value = 8
        w.vllm_config = MagicMock()
        w.vllm_config.speculative_config = None

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


class TestFinalize:
    def test_no_pending_is_noop(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_buffer_device="rbln")
        worker._pending_kv_caches = None
        worker.finalize_kv_cache_registration()  # must not raise

    def test_pending_dispatches_to_impl_and_clears(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_buffer_device="rbln")
        worker._pending_kv_caches = {"layer0": 1}
        registered: list = []
        worker._register_kv_caches_impl = registered.append
        worker.finalize_kv_cache_registration()
        assert registered == [{"layer0": 1}]
        assert worker._pending_kv_caches is None

    def test_double_call_is_idempotent(self, monkeypatch):
        worker = build_worker(monkeypatch, kv_buffer_device="rbln")
        worker._pending_kv_caches = {"layer0": 1}
        calls: list = []
        worker._register_kv_caches_impl = calls.append
        worker.finalize_kv_cache_registration()
        worker.finalize_kv_cache_registration()  # pending already cleared
        assert len(calls) == 1


class TestD2dRegistrationReachesTheAdapterAndThePeer:
    # Two things the D2D path does whose absence is silent: neither shows up as
    # a failure anywhere else, so each has one test here and nowhere.

    def test_each_half_of_a_layer_gets_its_own_byte_offset(self, make_worker):
        # K and V of one layer are told apart by this offset alone: zero it and
        # both register at the same address. The adapter derives every base
        # from it, so the connector's answer is what has to be right.
        w = make_worker(kv_cache=KvGeometry(layers=("l0", "l1")))
        regions = sys.modules["nixl_rbln"].regions_seen
        assert len(regions) == 1, "registration hands the adapter one batch"
        halves = [regions[0][i : i + 2] for i in range(0, len(regions[0]), 2)]
        assert len(halves) == 2, "two layers, K and V each"
        for k, v in halves:
            assert k[1] == 0
            # The V half starts exactly one K half in: the block length the third
            # element carries, over the worker's own block count.
            assert v[1] == w.num_blocks * k[2]

    def test_registration_republishes_what_a_peer_pairs_on(self, make_worker):
        # Upstream's publish carries no layer names and no chiplet geometry, so
        # without this a D2D peer has nothing to match on. The geometry has to
        # differ from `RblnNixlAgentMetadata`'s defaults (one area, one slice)
        # or a publish that dropped it would decode to the same answer.
        w = make_worker(kv_cache=KvGeometry(layers=("l0", "l1"), areas=4, slices=4))
        assert w.xfer_handshake_metadata is not None
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert list(decoded.registered_layer_names) == ["l0", "l1"]
        assert (decoded.kv_areas, decoded.kv_slices) == (4, 4)


class TestTopologyAccess:
    def test_reading_the_topology_before_registration_is_refused(self, make_worker):
        # Without the assert the property returns None, and the failure moves to
        # whichever reader dereferences it first.
        w = make_worker(register=False)
        with pytest.raises(AssertionError, match="before the KV caches are registered"):
            _ = w.topo


class TestWhatRegistrationSettles:
    def test_registration_derives_the_region_table_from_the_tensors(self, make_worker):
        geo = KvGeometry(layers=("l0", "l1"), heads=8, num_blocks=4)
        w = make_worker(kv_cache=geo)
        assert w.transfer_topo is not None
        # Two layers, K and V apart -> four regions of one block length each.
        assert len(w.block_len_per_layer) == 4
        assert w._logical_region_kv_heads == [8, 8, 8, 8]
        assert w._kv_split_axis is KVSplitAxis.HEAD
        bases = w.kv_caches_base_addr[w.engine_id][w.tp_rank]
        assert len(bases) == 4
        # Real addresses, and each region a distinct span of its tensor.
        assert len(set(bases)) == 4 and all(base > 0 for base in bases)

    @pytest.mark.parametrize("draft_kv_heads", [8, 4])
    def test_a_draft_layer_takes_its_own_models_head_count(
        self, make_worker, draft_kv_heads
    ):
        # Equal head counts leave one page size for the group; unequal ones make
        # the engine wrap the group spec, and the draft's regions are then half
        # the size. Both are real configurations of the same connector.
        geo = KvGeometry(layers=("l0", "l1"), draft_layers=("l1",))
        w = make_worker(kv_cache=geo, draft_kv_heads=draft_kv_heads)
        assert w._logical_region_kv_heads == [8, 8, draft_kv_heads, draft_kv_heads]
        # The connector's own answer, not the fixture's: `full_block_len` is what
        # it handed the adapter, taken from the unwrapped layer spec's page size.
        lens = [full for _, _, full in sys.modules["nixl_rbln"].regions_seen[0]]
        assert lens[2] * 8 == lens[0] * draft_kv_heads

    def test_a_draft_moves_the_compatibility_hash(self, make_worker):
        # The hash is the only thing that stops a producer running a draft from
        # pairing with a consumer that is not, and their region tables differ.
        # `rbln_compat_hash` folds the speculative config, but only the publish
        # passes it, so the factor is unpinned unless a drafted worker is
        # compared with one built the same way without.
        plain = make_worker(kv_cache=KvGeometry(layers=("l0", "l1")))
        drafted = make_worker(
            kv_cache=KvGeometry(layers=("l0", "l1"), draft_layers=("l1",)),
            draft_kv_heads=4,
        )
        assert plain.compat_hash and drafted.compat_hash
        assert plain.compat_hash != drafted.compat_hash

    def test_an_undeclared_head_count_has_no_band(self, make_worker):
        # Neither the target's count nor any declared draft's: replicated heads
        # report a product no model in this engine has, and _layer_kv_heads
        # refuses to name a band rather than compute one from the wrong model.
        geo = KvGeometry(layers=("l0", "l1"), per_layer_heads={"l1": 2})
        w = make_worker(kv_cache=geo)
        assert w._logical_region_kv_heads == [8, 8, None, None]
