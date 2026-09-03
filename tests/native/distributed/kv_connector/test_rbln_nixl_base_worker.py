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

# RblnNixlPullConnectorWorker built through its real __init__ with the upstream base
# and the external NIXL calls faked, so assertions land on the RBLN bookkeeping
# around them rather than on the faked returns.

import collections
import sys
import types
from typing import Any, cast
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.config import CacheConfig
from vllm.distributed.kv_transfer.kv_connector.utils import EngineTransferInfo
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import NixlBaseConnectorWorker
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

import vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.base_worker as wm
import vllm_rbln.envs as envs
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)


def _sliding_window_spec(*, block_size, sliding_window):
    spec = MagicMock(spec=SlidingWindowSpec)
    spec.block_size = block_size
    spec.sliding_window = sliding_window
    return spec


def _build_worker(
    monkeypatch,
    *,
    kv_buffer_device="cpu",
    num_blocks=128,
    block_size=64,
    specs=None,
    nixl_available=True,
    swa_view_opt=False,
    use_mla=False,
):
    """The worker via its real __init__, with upstream's stubbed to set only what
    the RBLN overrides read and `nixl_rbln` faked present or absent."""
    module = types.ModuleType("nixl_rbln") if nixl_available else None
    monkeypatch.setitem(sys.modules, "nixl_rbln", module)
    monkeypatch.setattr(envs, "VLLM_RBLN_NIXL_SWA_VIEW_OPT", swa_view_opt)

    def fake_super_init(self, vllm_config, engine_id, kv_cache_config):
        self.vllm_config = vllm_config
        self.engine_id = engine_id
        self.kv_cache_config = kv_cache_config
        self.kv_buffer_device = kv_buffer_device
        self.use_mla = use_mla
        self._block_size = {}
        # Read by `_layer_kv_heads` to recover a model-wide count from a spec's
        # per-rank share.
        self.world_size = 1
        self.model_config = MagicMock()
        self.model_config.get_total_num_kv_heads.return_value = 8
        # Upstream's own __init__ sets this to None;
        # register_kv_caches reads it after super().register_kv_caches().
        self.xfer_handshake_metadata = None
        # add_remote_agent asks for tp_ratio before deciding whether upstream's
        # positional pairing applies; 1 keeps these cases homogeneous.
        self.transfer_topo = MagicMock()
        self.transfer_topo.tp_ratio.return_value = 1

    monkeypatch.setattr(NixlBaseConnectorWorker, "__init__", fake_super_init)

    vllm_config = MagicMock()
    vllm_config.cache_config = CacheConfig(block_size=block_size)
    # No speculative decoding: the compat hash then folds what it always did.
    vllm_config.speculative_config = None
    # _check_pp_constraints compares pipeline_parallel_size <= 1; give it a real
    # int (a MagicMock would raise TypeError). 1 == the non-PP default here.
    vllm_config.parallel_config.pipeline_parallel_size = 1
    kv_cache_config = MagicMock()
    kv_cache_config.num_blocks = num_blocks
    kv_cache_config.kv_cache_groups = [
        MagicMock(kv_cache_spec=spec) for spec in (specs or [])
    ]
    return RblnNixlPullConnectorWorker(vllm_config, "test-engine", kv_cache_config)


class TestBackendSelection:
    def test_host_bounce_with_adapter_uses_rbln_backend(self, monkeypatch):
        worker = _build_worker(monkeypatch, kv_buffer_device="cpu", nixl_available=True)
        assert worker._use_rbln_nixl_backend is True
        assert worker.use_host_buffer is True
        assert not hasattr(worker, "nixl_memory_type")  # VRAM only set for D2D

    def test_d2d_with_adapter_registers_vram(self, monkeypatch):
        worker = _build_worker(
            monkeypatch, kv_buffer_device="rbln", nixl_available=True
        )
        assert worker._use_rbln_nixl_backend is True
        assert worker.nixl_memory_type == "VRAM"
        assert worker.use_host_buffer is False

    def test_host_bounce_without_adapter_falls_back_to_upstream(self, monkeypatch):
        worker = _build_worker(
            monkeypatch, kv_buffer_device="cpu", nixl_available=False
        )
        assert worker._use_rbln_nixl_backend is False
        assert worker.use_host_buffer is True

    def test_d2d_without_adapter_is_rejected(self, monkeypatch):
        # D2D needs the RBLN backend; without nixl_rbln there is no way to
        # register device memory, so construction must fail loudly.
        with pytest.raises(RuntimeError, match="nixl-rbln"):
            _build_worker(monkeypatch, kv_buffer_device="rbln", nixl_available=False)


class TestLogicalBlockPinning:
    def test_pins_logical_block_counts(self, monkeypatch):
        # num_blocks / block_size are pinned to the logical values, and the
        # physical-per-logical ratio stays 1 (no kernel-ratio multiplication).
        worker = _build_worker(monkeypatch, num_blocks=128, block_size=64)
        assert worker.num_blocks == 128
        assert worker.block_size == 64
        assert worker._physical_blocks_per_logical_kv_block == 1
        assert worker._logical_num_blocks == 128
        assert worker._pending_kv_caches is None


class TestSwaViewRatio:
    def test_opt_off_keeps_ratio_none(self, monkeypatch):
        worker = _build_worker(
            monkeypatch,
            swa_view_opt=False,
            specs=[_sliding_window_spec(block_size=64, sliding_window=16)],
        )
        assert worker._sw_ratio is None
        # The window is still detected -- it gates the model parallelism guards
        # whether or not the view-opt is on.
        assert worker._has_swa

    def test_pure_full_attention_keeps_ratio_none(self, monkeypatch):
        # A non-sliding-window group contributes no ratio.
        worker = _build_worker(monkeypatch, swa_view_opt=True, specs=[MagicMock()])
        assert worker._sw_ratio is None

    def test_sliding_window_derives_block_over_window_ratio(self, monkeypatch):
        worker = _build_worker(
            monkeypatch,
            swa_view_opt=True,
            specs=[_sliding_window_spec(block_size=64, sliding_window=16)],
        )
        assert worker._sw_ratio == 4

    def test_window_equal_to_block_collapses_to_none(self, monkeypatch):
        # ratio 1 means the SWA view equals the full block -> no trimming.
        worker = _build_worker(
            monkeypatch,
            swa_view_opt=True,
            specs=[_sliding_window_spec(block_size=64, sliding_window=64)],
        )
        assert worker._sw_ratio is None

    def test_full_attention_groups_are_skipped(self, monkeypatch):
        # The hybrid shape: a model interleaves full-attention and sliding-window
        # layers, so the ratio has to come from the windowed groups alone.
        worker = _build_worker(
            monkeypatch,
            swa_view_opt=True,
            specs=[MagicMock(), _sliding_window_spec(block_size=64, sliding_window=16)],
        )
        assert worker._sw_ratio == 4

    def test_consistent_ratio_across_groups(self, monkeypatch):
        worker = _build_worker(
            monkeypatch,
            swa_view_opt=True,
            specs=[
                _sliding_window_spec(block_size=64, sliding_window=16),
                _sliding_window_spec(block_size=64, sliding_window=16),
            ],
        )
        assert worker._sw_ratio == 4

    def test_mismatched_ratios_are_rejected(self, monkeypatch):
        with pytest.raises(AssertionError, match="single SWA ratio"):
            _build_worker(
                monkeypatch,
                swa_view_opt=True,
                specs=[
                    _sliding_window_spec(block_size=64, sliding_window=16),
                    _sliding_window_spec(block_size=64, sliding_window=32),
                ],
            )

    def test_window_not_dividing_block_is_rejected(self, monkeypatch):
        with pytest.raises(AssertionError):
            _build_worker(
                monkeypatch,
                swa_view_opt=True,
                specs=[_sliding_window_spec(block_size=64, sliding_window=15)],
            )

    def test_mla_with_view_opt_is_rejected_at_startup(self, monkeypatch):
        # The dual desc range and a key-only latent have not been combined,
        # so fail at construction rather than at the first handshake.
        with pytest.raises(RuntimeError, match="SWA_VIEW_OPT"):
            _build_worker(
                monkeypatch,
                swa_view_opt=True,
                use_mla=True,
                specs=[_sliding_window_spec(block_size=64, sliding_window=16)],
            )


class TestRegisterKvCaches:
    def test_d2d_stashes_and_defers(self, monkeypatch):
        # D2D can't register until warm-up materializes memory: stash and return.
        worker = _build_worker(monkeypatch, kv_buffer_device="rbln")
        worker.register_kv_caches({"layer0": "tensor"})
        assert worker._pending_kv_caches == {"layer0": "tensor"}

    def test_host_bounce_creates_backend_and_delegates(self, monkeypatch):
        # Host-bounce with the adapter creates the RBLN backend on the agent,
        # then delegates registration to upstream.
        worker = _build_worker(monkeypatch, kv_buffer_device="cpu", nixl_available=True)
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
        worker = _build_worker(monkeypatch, kv_buffer_device="cpu", nixl_available=True)
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


class TestFinalize:
    def test_no_pending_is_noop(self, monkeypatch):
        worker = _build_worker(monkeypatch, kv_buffer_device="rbln")
        worker._pending_kv_caches = None
        worker.finalize_kv_cache_registration()  # must not raise

    def test_pending_dispatches_to_impl_and_clears(self, monkeypatch):
        worker = _build_worker(monkeypatch, kv_buffer_device="rbln")
        worker._pending_kv_caches = {"layer0": 1}
        registered: list = []
        worker._register_kv_caches_impl = registered.append
        worker.finalize_kv_cache_registration()
        assert registered == [{"layer0": 1}]
        assert worker._pending_kv_caches is None

    def test_double_call_is_idempotent(self, monkeypatch):
        worker = _build_worker(monkeypatch, kv_buffer_device="rbln")
        worker._pending_kv_caches = {"layer0": 1}
        calls: list = []
        worker._register_kv_caches_impl = calls.append
        worker.finalize_kv_cache_registration()
        worker.finalize_kv_cache_registration()  # pending already cleared
        assert len(calls) == 1


class TestInitializeHostXferBuffer:
    def test_allocates_one_buffer_per_layer_preserving_order(self, monkeypatch):
        worker = _build_worker(monkeypatch, kv_buffer_device="cpu")
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
        worker = _build_worker(monkeypatch, kv_buffer_device="cpu")
        worker.kv_cache_layout = "NHD"
        with pytest.raises(AssertionError, match="HND"):
            worker.initialize_host_xfer_buffer(
                {"l0": torch.zeros(2, 4, dtype=torch.float16)}
            )

    def test_mla_accepts_any_layout(self, monkeypatch):
        # MLA has no head axis to order, so upstream advertises no required
        # layout and the resolved value is whatever the fallback picked. The
        # 3D latent shape must survive unchanged.
        worker = _build_worker(monkeypatch, kv_buffer_device="cpu", use_mla=True)
        worker.kv_cache_layout = "NHD"
        worker.initialize_host_xfer_buffer(
            {"l0": torch.zeros(4, 64, 576, dtype=torch.float16)}
        )
        assert worker.host_xfer_buffers["l0"].shape == (4, 64, 576)

    def test_rejects_odd_byte_footprint(self, monkeypatch):
        # The page-aligned host buffer is backed by an fp16 (2-byte) allocation,
        # so a cache whose byte footprint is odd cannot be tiled.
        worker = _build_worker(monkeypatch, kv_buffer_device="cpu")
        worker.kv_cache_layout = "HND"
        with pytest.raises(AssertionError, match="multiple of 2"):
            worker.initialize_host_xfer_buffer(
                {"l0": torch.zeros(1, dtype=torch.uint8)}  # 1 byte
            )


class TestSwaViewDelegation:
    # Both collapse to the upstream Full-only implementation when _sw_ratio is
    # None; the SWA dual-range paths are exercised in the Swa classes below.
    def test_register_local_xfer_handler_delegates_when_no_swa(self, monkeypatch):
        worker = _build_worker(monkeypatch)  # _sw_ratio is None
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
        worker = _build_worker(monkeypatch)  # _sw_ratio is None

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
        worker = _build_worker(monkeypatch)  # _sw_ratio is None
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
        worker = _build_worker(
            monkeypatch,
            swa_view_opt=True,
            specs=[_sliding_window_spec(block_size=64, sliding_window=16)],
        )
        worker._remote_agents = {"peer": {0: "cached-name"}}
        super_calls = []
        monkeypatch.setattr(
            NixlBaseConnectorWorker,
            "add_remote_agent",
            lambda self, *a, **k: super_calls.append(1),
        )
        result = worker.add_remote_agent(MagicMock(engine_id="peer"), 0, 1)
        assert result == "cached-name"
        assert super_calls == []


class TestSetHostXferBufferOps:
    def test_noop_when_kv_buffer_is_not_cpu(self, monkeypatch):
        worker = _build_worker(monkeypatch, kv_buffer_device="rbln")
        worker.set_host_xfer_buffer_ops("copy_op")
        assert not hasattr(worker, "copy_blocks")

    def test_assigns_copy_on_host_bounce(self, monkeypatch):
        worker = _build_worker(monkeypatch, kv_buffer_device="cpu")
        worker.use_host_buffer = True
        worker.set_host_xfer_buffer_ops("copy_op")
        assert worker.copy_blocks == "copy_op"


# Heavy D2D paths: the deferred registration body and the SWA dual-range descs.
# The real method bodies run; only the external NIXL pieces are faked.


def _prep_impl_worker(monkeypatch, *, num_blocks=128, block_size=64):
    # A D2D worker back-filled with the attributes upstream __init__ would set.
    worker = _build_worker(
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


def _patch_worker_nixl_symbols(topo, *, mamba_spec=None, uniform_spec=None):
    # Patch the directly-imported NIXL/spec symbols so the impl runs without the
    # real nixl package. The isinstance() targets get a dummy class, which the
    # Full-attention specs cannot match; a test that wants one of those branches
    # passes the real class instead and builds a spec against it.
    msgspec_mock = MagicMock()
    msgspec_mock.msgpack.Encoder.return_value.encode.return_value = b"meta"
    return patch.multiple(
        wm,
        TransferTopology=MagicMock(return_value=topo),
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
            patch.object(wm, "rebel") as mock_rebel,
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
            patch.object(wm, "rebel") as mock_rebel,
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
            patch.object(wm, "rebel") as mock_rebel,
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
            patch.object(wm, "rebel") as mock_rebel,
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

        with (
            _patch_worker_nixl_symbols(topo),
            patch.dict(sys.modules, {"nixl_rbln": fake}),
            patch.object(wm, "rebel") as mock_rebel,
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            mock_rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
            worker._register_kv_caches_impl(kv_caches)
            cast(MagicMock, wm.TransferTopology).assert_called_once_with(
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
            patch.object(wm, "rebel"),
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
            patch.object(wm, "rebel"),
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
            patch.object(wm, "rebel"),
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
        ):
            worker._register_kv_caches_impl(kv_caches)

        # Registration itself still serves.
        assert worker._logical_region_kv_heads == [None] * 4
        with pytest.raises(AssertionError, match="no head band"):
            worker._region_kv_heads(0)

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
            patch.object(wm, "rebel"),
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
            patch.object(wm, "rebel"),
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
            patch.object(wm, "rebel"),
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
            patch.object(wm, "rebel"),
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
            patch.object(wm, "rebel"),
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
            patch.object(wm, "rebel"),
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
            patch.object(wm, "rebel"),
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
            patch.object(wm, "rebel") as mock_rebel,
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
            patch.object(wm, "rebel") as mock_rebel,
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
            patch.object(wm, "rebel") as mock_rebel,
            patch.object(worker, "register_local_xfer_handler", return_value=("h", [])),
            pytest.raises(RuntimeError, match="not all cut on the same axis"),
        ):
            mock_rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
            worker._register_kv_caches_impl(kv_caches)


class TestRegisterLocalXferHandlerSwa:
    # With _sw_ratio set, register_local_xfer_handler emits a dual desc range
    # (Full then SWA at the same base addresses over length_divisors [1, ratio]).
    def test_swa_builds_dual_desc_ranges(self, monkeypatch):
        worker = _build_worker(monkeypatch, num_blocks=4, block_size=64)
        worker._sw_ratio = 2
        worker._has_mamba = False
        worker.tp_rank = 0
        worker.device_id = 0
        worker.transfer_topo = MagicMock(is_kv_layout_blocks_first=False)
        worker.kv_caches_base_addr = {worker.engine_id: {0: [0x1000, 0x2000]}}
        worker.block_len_per_layer = [256, 256]
        worker.nixl_memory_type = "DRAM"
        worker.nixl_wrapper = MagicMock()

        with patch.object(worker, "get_backend_aware_kv_block_len", return_value=256):
            worker.register_local_xfer_handler(64)  # block_size_ratio == 1

        worker.nixl_wrapper.get_xfer_descs.assert_called_once()
        blocks_data = worker.nixl_wrapper.get_xfer_descs.call_args[0][0]
        # 2 passes (Full + SWA) x 2 base addrs x num_blocks(4) = 16; a Full-only
        # pass would be 8 -> the dual range is confirmed.
        assert len(blocks_data) == 16
        # SWA descs sit at the SAME addresses as the Full descs (a length-trimmed
        # view), so the second pass repeats the first pass's addresses.
        addrs = [d[0] for d in blocks_data]
        assert addrs[:8] == addrs[8:]
        # Full desc length 256; SWA length trimmed by _sw_ratio -> 128.
        assert blocks_data[0][1] == 256
        assert blocks_data[8][1] == 128


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
        worker = _build_worker(monkeypatch, num_blocks=4, block_size=64)
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
            patch.object(
                wm, "compute_tp_mapping", return_value=mapping_sentinel
            ) as ctm,
            patch.object(worker, "_validate_remote_agent_handshake"),
            patch.object(worker, "get_backend_aware_kv_block_len", return_value=256),
        ):
            out = worker.add_remote_agent(meta, 0, 1)

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
        worker = _build_worker(monkeypatch, num_blocks=4, block_size=64)
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
            patch.object(wm, "compute_tp_mapping", return_value=MagicMock()),
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


class TestComputeDescIds:
    # Routes block ids into the Full range (offset 0) or the SWA range (offset
    # num_full_descs) by group spec, expanded across regions.
    def test_none_ratio_delegates_to_super(self, monkeypatch):
        worker = _build_worker(monkeypatch)  # _sw_ratio is None
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
        worker = _build_worker(monkeypatch)
        worker._sw_ratio = 2
        worker.num_regions = 2
        full_spec = MagicMock()  # not a SlidingWindowSpec
        worker._group_specs = [
            full_spec,
            _sliding_window_spec(block_size=64, sliding_window=32),
        ]

        # dst_num_blocks=4 -> num_full_descs = num_regions(2) * 4 = 8.
        out = worker._compute_desc_ids([[0, 1], [2]], 4, None, 1)

        # Full ids [0,1] -> r*4 + id: 0,1 then 4,5. SWA id [2] -> r*4 + 2 + 8.
        assert list(out) == [0, 1, 4, 5, 10, 14]

    def test_block_size_ratio_scales_block_span(self, monkeypatch):
        # A block_size_ratio widens the per-region block span (num_blocks *= ratio),
        # shifting both the region stride and the SWA offset.
        worker = _build_worker(monkeypatch)
        worker._sw_ratio = 2
        worker.num_regions = 1
        worker._group_specs = [_sliding_window_spec(block_size=64, sliding_window=32)]

        # dst_num_blocks=2, ratio=2 -> num_blocks=4, num_full_descs = 1*4 = 4.
        out = worker._compute_desc_ids([[1]], 2, 2.0, 1)
        # single region: 0*4 + 1 + offset(4) = 5.
        assert list(out) == [5]

    def test_rejects_multi_physical_blocks_per_logical(self, monkeypatch):
        # The SWA desc formula indexes physical blocks directly; the connector
        # pins one physical block per logical, so >1 is rejected.
        worker = _build_worker(monkeypatch)
        worker._sw_ratio = 2
        worker.num_regions = 1
        worker._group_specs = [_sliding_window_spec(block_size=64, sliding_window=32)]
        with pytest.raises(AssertionError, match="physical_blocks_per_logical"):
            worker._compute_desc_ids([[0]], 4, None, 2)

    def test_empty_groups_yield_empty(self, monkeypatch):
        worker = _build_worker(monkeypatch)
        worker._sw_ratio = 2
        worker.num_regions = 1
        worker._group_specs = [MagicMock()]
        out = worker._compute_desc_ids([[]], 4, None, 1)
        assert out.size == 0
