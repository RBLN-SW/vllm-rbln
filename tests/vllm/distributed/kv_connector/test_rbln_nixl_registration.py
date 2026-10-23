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

import inspect
import sys
import types
from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, patch

import msgspec
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlAgentMetadata,
    NixlBaseConnectorWorker,
)
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
    compute_nixl_compatibility_hash,
)
from vllm.v1.kv_cache_interface import (
    ChunkedLocalAttentionSpec,
    FullAttentionSpec,
    KVCacheTensor,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from tests.vllm.distributed.kv_connector.utils import (
    KvGeometry,
    area_slices,
    build_worker,
    fill_region_geometry,
    mock_vllm_config,
    patch_in_package,
    patched_in_package,
    set_shape,
    sliding_window_spec,
    window_mode,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl import (
    registration,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    ChipletSplitAxis,
    RblnNixlAgentMetadata,
    rbln_compat_hash,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (
    RblnNixlPullConnectorWorker,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.push_worker import (
    RblnNixlPushConnectorWorker,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
    AreaPlace,
    ChunkGrid,
    RegionGeometry,
)


def _impl_layer_spec(page_size_bytes=4096, num_kv_heads=8):
    # Full-attention spec stand-in: .page_size_bytes and .num_kv_heads are read,
    # and it must fail the isinstance(MambaSpec/UniformTypeKVCacheSpecs) checks.
    # num_kv_heads defaults to the 8 the tensors _impl_kv_caches builds.
    spec = MagicMock(spec=FullAttentionSpec)
    spec.page_size_bytes = page_size_bytes
    spec.num_kv_heads = num_kv_heads
    return spec


#: Every entry the impl tests build, by address. The device is asked about an
#: allocation by address rather than handed the tensor, so this is what answers.
_IMPL_ENTRIES: dict[int, Any] = {}


def _impl_kv_caches(
    num_blocks=128, base_ptr=0x10000, names=("l0", "l1"), heads=8, block_tokens=64
):
    # Full-layer tensors: shape (K/V, num_blocks, heads, ..., dim). `heads` and
    # `block_tokens` are the two counts the shard rule reads, so they are what
    # makes a cut fall on one axis or the other; the byte counts stay as the
    # two _split_kv halves of the default 4096B page add up to, and do not move
    # with them.
    kv = {}
    for i, name in enumerate(names):
        t = MagicMock()
        t.shape = (2, num_blocks, heads, 1, block_tokens, 64)
        t.numel.return_value = num_blocks * 4096
        t.element_size.return_value = 1
        t.data_ptr.return_value = base_ptr + i * 0x10000
        t.storage_offset.return_value = 0
        t.get_device.return_value = 0
        _IMPL_ENTRIES[t.data_ptr()] = t
        kv[name] = t
    return kv


def _mla_kv_caches(
    num_blocks=128, base_ptr=0x10000, page_size_bytes=4096, block_tokens=64
):
    # Two MLA layers: a latent cache (blocks, tokens, latent), block axis first,
    # no head axis and no K/V split. The whole entry is the region, so its bytes
    # must be num_blocks x the page.
    kv = {}
    for i, name in enumerate(("l0", "l1")):
        t = MagicMock()
        t.shape = (num_blocks, block_tokens, page_size_bytes // 64)
        t.numel.return_value = num_blocks * page_size_bytes
        t.element_size.return_value = 1
        t.data_ptr.return_value = base_ptr + i * 0x10000
        t.storage_offset.return_value = 0
        t.get_device.return_value = 0
        _IMPL_ENTRIES[t.data_ptr()] = t
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
    topo, *, mamba_spec=None, uniform_spec=None, topology_cls=None, hash_fn=None
):
    # The isinstance() targets get a dummy class, which the Full-attention specs
    # cannot match; a test that wants one of those branches passes the real
    # class instead. Patched across the package because these are
    # `from x import Y` bindings, so the binding that matters is the one in
    # whichever module holds the function under test.
    msgspec_mock = MagicMock()
    msgspec_mock.msgpack.Encoder.return_value.encode.return_value = b"meta"
    return patch_in_package(
        RblnTransferTopology=topology_cls or MagicMock(return_value=topo),
        compute_nixl_compatibility_hash=hash_fn or MagicMock(return_value="hash"),
        MambaSpec=mamba_spec or type("MambaSpec", (), {}),
        UniformTypeKVCacheSpecs=uniform_spec or type("UniformTypeKVCacheSpecs", (), {}),
        NixlAgentMetadata=MagicMock(),
        NixlHandshakePayload=MagicMock(),
        msgspec=msgspec_mock,
    )


# The split K/V layout `_impl_kv_caches` builds: (K/V, blocks, heads, 1, tokens,
# dim). The topology finds the two dims by probing the backend with sentinels.
HEAD_AXIS = 2
TOKEN_AXIS = 4
# A latent cache, (blocks, tokens, latent), as `_mla_kv_caches` builds it.
MLA_TOKEN_AXIS = 1


# Pipeline stages of a model that mixes attention backends, each holding a
# different set of them. A peer pairs with every stage, so every stage has to
# publish the same hash.
_STAGES = (
    ("BACKEND_A", ("BACKEND_A", "BACKEND_B")),
    ("BACKEND_B", ("BACKEND_B",)),
    ("BACKEND_A", ("BACKEND_A",)),
)


def _hash_by_backend():
    return MagicMock(side_effect=lambda cfg, name, **kw: f"hash:{name}")


def _backends(names):
    return [MagicMock(**{"get_name.return_value": name}) for name in names]


def _hashed_backend_name():
    """The `attn_backend_name` the last upstream hash call was given."""
    call = registration.compute_nixl_compatibility_hash.call_args
    assert call is not None, "nothing reached the upstream hash"
    bound = inspect.signature(compute_nixl_compatibility_hash).bind(
        *call.args, **call.kwargs
    )
    return bound.arguments["attn_backend_name"]


def _probe_kv_shape(topo):
    topo.kv_head_dim, topo.kv_token_dim = HEAD_AXIS, TOKEN_AXIS


def _probe_mla_shape(topo):
    topo.kv_head_dim, topo.kv_token_dim = None, MLA_TOKEN_AXIS


def _area_base(entry, area):
    # Synthetic device addresses, an entry's areas far enough apart that no two
    # regions overlap.
    return (entry.data_ptr() << 12) + area * 0x100000


def _impl_buffers(*, areas=1, slices=1, size=None, per_entry=None):
    """`nixl_rbln.entry_areas` over the impl tests' entries: `areas`
    chiplet buffers an entry, holding one of `slices` equal pieces of its bytes
    each. `per_entry` gives an entry, by its address, an (areas, slices) of its
    own, and `size` an area a byte count that is not its share."""

    def entry_areas(vaddr):
        count, pieces = (per_entry or {}).get(vaddr, (areas, slices))
        entry = _IMPL_ENTRIES[vaddr]
        share = entry.numel() * entry.element_size() // pieces
        return tuple(
            SimpleNamespace(
                node_id=0,
                chiplet_id=a,
                device_addr=_area_base(entry, a),
                size=size or share,
            )
            for a in range(count)
        )

    return entry_areas


def _fake_nixl_rbln(entry_areas=None):
    module: Any = types.ModuleType("nixl_rbln")
    module.ensure_rbln_backend = MagicMock()
    module.entry_areas = MagicMock(side_effect=entry_areas or _impl_buffers())
    return module


def _impl_topo(regions, **attrs):
    """The topology registration builds, splitting an entry into `regions`;
    `attrs` overrides the layout flags, all off by default."""
    attrs = {
        "virtually_split_kv_in_blocks": False,
        "_cross_layers_blocks": False,
        "cross_layers_blocks": False,
        **attrs,
    }
    topo = MagicMock(**attrs)
    topo.get_transfer_cache_regions.side_effect = regions
    return topo


@contextmanager
def _registering(worker, topo, *, buffers=None, **symbols):
    """What `_register_kv_caches_impl` reaches past the worker, faked: the
    package's symbols (`symbols` as `_patch_worker_nixl_symbols` takes them),
    nixl_rbln, and rebel answering for each entry's storage through `buffers`.
    Yields the two fakes as `nixl_rbln` and `rebel`."""
    nixl_rbln = _fake_nixl_rbln(buffers)
    with (
        _patch_worker_nixl_symbols(topo, **symbols),
        patch.dict(sys.modules, {"nixl_rbln": nixl_rbln}),
        patched_in_package("rebel") as rebel,
        patch.object(
            worker,
            "register_local_xfer_handler",
            return_value=("local-handle", [(0x0, 0, 0)]),
        ),
    ):
        rebel.context_of.return_value.rbln_ctx_ptr = 0x1000
        yield SimpleNamespace(nixl_rbln=nixl_rbln, rebel=rebel)


def _prep_impl_worker(
    monkeypatch,
    *,
    num_blocks=128,
    block_size=64,
    names=("l0", "l1"),
    specs=None,
    chunk_mode=False,
    chunk_tokens=0,
    push_stream=False,
    cls=None,
):
    # A D2D worker back-filled with the attributes upstream __init__ would set.
    worker = build_worker(
        monkeypatch,
        kv_buffer_device="rbln",
        num_blocks=num_blocks,
        block_size=block_size,
        nixl_available=True,
        specs=specs,
        chunk_mode=chunk_mode,
        chunk_tokens=chunk_tokens,
        push_stream=push_stream,
        cls=cls,
    )
    worker.tp_rank = 0
    worker.world_size = 1
    worker.use_mla = False
    worker._has_mamba = False
    worker.attn_backends = []
    worker.backend_name = "rbln"
    worker.kv_cache_layout = "LBHNC"
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
    # `transfer_groups` is a SUBSET of `kv_cache_groups` -- the same objects,
    # filtered by `enable_kv_transfer` -- so a config where one is not a slice
    # of the other cannot exist. The two views still disagree here, the way a
    # config with a non-transferring group does: a group that opted out sits
    # ahead of the one that registers, so counting cache groups numbers the
    # registered layers one too high.
    opted_out = SimpleNamespace(layer_names=["opted-out"], kv_cache_spec=None)
    transferring = SimpleNamespace(layer_names=list(names), kv_cache_spec=None)
    worker.kv_cache_config.kv_cache_groups = [opted_out, transferring]
    worker.kv_cache_config.transfer_groups = [transferring]
    worker.kv_cache_config.transfer_group_index_by_layer = dict.fromkeys(names, 0)
    return worker


class TestRegisterKvCaches:
    def test_d2d_stashes_and_defers(self, monkeypatch):
        # D2D can't register until warm-up materializes memory: stash and return.
        worker = build_worker(monkeypatch, kv_buffer_device="rbln")
        worker.register_kv_caches({"layer0": "tensor"})
        assert worker._pending_kv_caches == {"layer0": "tensor"}

    @pytest.mark.parametrize("kv_buffer_device", ["rbln", "cpu"])
    @pytest.mark.parametrize(
        "names, expected",
        [
            (("swa0", "full0", "full1"), ["full0", "full1"]),
            (("full1", "swa0", "full0"), ["full1", "full0"]),
        ],
    )
    def test_a_buffer_is_registered_once_through_its_full_attention_layer(
        self, monkeypatch, kv_buffer_device, names, expected
    ):
        # The runner hands over every layer. A sliding-window layer whose view
        # aliases a full-attention layer's buffer counts blocks differently, so
        # the buffer is registered once, through the full-attention layer.
        worker = build_worker(
            monkeypatch,
            kv_buffer_device=kv_buffer_device,
            nixl_available=kv_buffer_device == "rbln",
            layer_names=names,
        )
        # Two tensors from byte 0: the layers at one position share a buffer.
        worker.kv_cache_config.kv_cache_tensors = [
            KVCacheTensor(size=0, layers=["swa0"], layer_stride=1, block_stride=1),
            KVCacheTensor(
                size=0, layers=["full0", "full1"], layer_stride=1, block_stride=1
            ),
        ]
        worker._layer_specs = {
            "swa0": MagicMock(
                spec=SlidingWindowSpec, page_size_bytes=4096, num_kv_heads=8
            ),
            "full0": _impl_layer_spec(),
            "full1": _impl_layer_spec(),
        }

        worker.tp_rank = 0

        def register_upstream(kv_caches):
            worker.block_len_per_layer = [4096] * len(kv_caches)
            worker.kv_caches_base_addr = {
                worker.engine_id: {worker.tp_rank: [0x1000] * len(kv_caches)}
            }

        upstream_register = MagicMock(side_effect=register_upstream)
        monkeypatch.setattr(
            NixlBaseConnectorWorker, "register_kv_caches", upstream_register
        )

        worker.register_kv_caches(dict.fromkeys(names, "tensor"))

        if kv_buffer_device == "cpu":
            upstream_register.assert_called_once()
            registered = upstream_register.call_args.args[0]
            assert worker._pending_kv_caches is None
        else:
            upstream_register.assert_not_called()
            registered = worker._pending_kv_caches
        assert list(registered) == expected
        assert worker.local_seen_layer_names == expected

    def test_a_kv_sharing_layer_is_not_registered(self, monkeypatch):
        # A layer reading another's cache (kv_sharing_target_layer_name) owns no
        # buffer and has no spec in this connector's config, yet the runner
        # hands it over with the rest.
        worker = build_worker(monkeypatch, kv_buffer_device="rbln", layer_names=("l0",))

        worker.register_kv_caches({"l0": "tensor", "l1_shared": "tensor"})

        assert list(worker._pending_kv_caches) == ["l0"]

    def test_the_block_count_comes_from_the_allocation_not_the_estimate(
        self, monkeypatch
    ):
        # A dynamic-KV resize settles the count after warm-up and registers
        # only then, so what `__init__` copied is the pre-compile estimate.
        worker = build_worker(monkeypatch, kv_buffer_device="rbln", num_blocks=4)
        worker.vllm_config.cache_config.num_gpu_blocks = 128

        worker.register_kv_caches({"layer0": "tensor"})

        assert worker.num_blocks == 128
        assert worker._logical_num_blocks == 128

    @pytest.mark.parametrize(
        "stripe_width, passed",
        [(None, {}), (0, {"stripe_width": 0}), (4096, {"stripe_width": 4096})],
    )
    def test_host_bounce_creates_backend_and_delegates(
        self, monkeypatch, stripe_width, passed
    ):
        # Host-bounce with the adapter creates the RBLN backend on the agent,
        # then delegates registration to upstream. The width rides on that
        # call, which is the only place this path can pass one.
        worker = build_worker(
            monkeypatch,
            kv_buffer_device="cpu",
            nixl_available=True,
            stripe_width=stripe_width,
        )
        worker.nixl_wrapper = "wrapper"
        worker._layer_specs = {"layer0": _impl_layer_spec()}
        worker.block_len_per_layer = [2048, 2048]
        # Upstream fills the base addresses beside the lengths.
        worker.tp_rank = 0
        worker.kv_caches_base_addr = {
            worker.engine_id: {
                worker.tp_rank: [
                    0x1000 * (i + 1) for i in range(len(worker.block_len_per_layer))
                ]
            }
        }
        ensured = []
        monkeypatch.setattr(
            sys.modules["nixl_rbln"],
            "ensure_rbln_backend",
            lambda wrapper, device_id=0, **kw: ensured.append((wrapper, device_id, kw)),
            raising=False,
        )
        delegated = []
        monkeypatch.setattr(
            NixlBaseConnectorWorker,
            "register_kv_caches",
            lambda self, kv: delegated.append(kv),
        )
        worker.register_kv_caches({"layer0": "tensor"})
        assert ensured == [("wrapper", 0, passed)]
        assert delegated == [{"layer0": "tensor"}]
        assert worker._pending_kv_caches is None
        # Host staging needs the per-region counts too: a pipelined peer reaches
        # the per-head handshake check through it.
        assert worker._logical_region_kv_heads == [8, 8]
        # And the geometry it advertises, which a peer now checks on arrival:
        # a staging buffer is one area holding a region whole.
        assert worker._logical_region_geometry == [
            RegionGeometry(ChipletSplitAxis.HEAD, 1, (AreaPlace(0x1000, 2048, 0),)),
            RegionGeometry(ChipletSplitAxis.HEAD, 1, (AreaPlace(0x2000, 2048, 0),)),
        ]

    def test_host_staging_publishes_one_hash_across_pipeline_stages(self, monkeypatch):
        # This path keeps upstream's registration, whose hash takes the rank's
        # first backend; what it publishes must not. Upstream's registration is
        # stubbed out, so the hash and payload it would leave are seeded here.
        names, published = [], []
        for backend_name, backends in _STAGES:
            worker = build_worker(
                monkeypatch, kv_buffer_device="cpu", nixl_available=True
            )
            worker.backend_name = backend_name
            worker.attn_backends = _backends(backends)
            worker.nixl_wrapper = "wrapper"
            worker.topo.cross_layers_blocks = False
            worker._layer_specs = {"layer0": _impl_layer_spec()}
            worker.block_len_per_layer = [2048, 2048]
            worker.tp_rank = 0
            worker.kv_caches_base_addr = {
                worker.engine_id: {worker.tp_rank: [0x1000, 0x2000]}
            }
            worker.compat_hash = f"upstream-{backend_name}"
            worker.xfer_handshake_metadata = MagicMock()
            monkeypatch.setattr(
                sys.modules["nixl_rbln"],
                "ensure_rbln_backend",
                lambda wrapper, device_id=0, **kw: None,
                raising=False,
            )
            monkeypatch.setattr(
                NixlBaseConnectorWorker, "register_kv_caches", lambda self, kv: None
            )
            with _patch_worker_nixl_symbols(MagicMock(), hash_fn=_hash_by_backend()):
                worker.register_kv_caches({"layer0": "tensor"})
                names.append(_hashed_backend_name())
            published.append(worker.compat_hash)

        assert len(set(names)) == len(set(published)) == 1

    @pytest.mark.parametrize(
        "stripe_width, passed",
        [(None, {}), (0, {"stripe_width": 0}), (4096, {"stripe_width": 4096})],
    )
    def test_a_stripe_width_reaches_the_adapter_only_when_named(
        self, monkeypatch, stripe_width, passed
    ):
        # 0 is a width the adapter takes, so it cannot stand for the knob
        # nobody set; absence is what keeps the plugin's own default.
        worker = build_worker(
            monkeypatch,
            kv_buffer_device="cpu",
            nixl_available=True,
            stripe_width=stripe_width,
        )
        worker.nixl_wrapper = "wrapper"
        worker._layer_specs = {"layer0": _impl_layer_spec()}
        worker.block_len_per_layer = [2048, 2048]
        # Upstream fills the base addresses beside the lengths.
        worker.tp_rank = 0
        worker.kv_caches_base_addr = {
            worker.engine_id: {
                worker.tp_rank: [
                    0x1000 * (i + 1) for i in range(len(worker.block_len_per_layer))
                ]
            }
        }
        seen = []
        monkeypatch.setattr(
            sys.modules["nixl_rbln"],
            "ensure_rbln_backend",
            lambda wrapper, device_id=0, **kw: seen.append(kw),
            raising=False,
        )
        monkeypatch.setattr(
            NixlBaseConnectorWorker, "register_kv_caches", lambda self, kv: None
        )
        worker.register_kv_caches({"layer0": "tensor"})

        assert seen == [passed]

    def test_a_stripe_width_given_a_bool_is_refused(self, monkeypatch):
        # It goes through the same typed knob path as the rest: read straight
        # off the extra config, `true` would arrive as a one-byte stripe.
        with pytest.raises(RuntimeError, match="stripe_width"):
            build_worker(monkeypatch, stripe_width=True)

    def test_host_bounce_rejects_differing_per_layer_sizes(self, monkeypatch):
        # Pins the refusal at this path's own point rather than upstream's assert.
        worker = build_worker(
            monkeypatch,
            kv_buffer_device="cpu",
            nixl_available=True,
            layer_names=("l0", "l1"),
        )
        worker.nixl_wrapper = "wrapper"
        worker._layer_specs = {
            "l0": _impl_layer_spec(page_size_bytes=4096),
            "l1": _impl_layer_spec(page_size_bytes=16384, num_kv_heads=32),
        }
        monkeypatch.setattr(
            sys.modules["nixl_rbln"],
            "ensure_rbln_backend",
            lambda wrapper, device_id=0, **kw: None,
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
        worker.kv_cache_layout = "LBHNC"
        kv_caches = {
            "l0": torch.zeros(2, 4, dtype=torch.float16),
            "l1": torch.zeros(3, 5, dtype=torch.float16),
        }
        worker.initialize_host_xfer_buffer(kv_caches)
        assert list(worker.host_xfer_buffers.keys()) == ["l0", "l1"]
        assert worker.host_xfer_buffers["l0"].shape == (2, 4)
        assert worker.host_xfer_buffers["l1"].shape == (3, 5)

    def test_rejects_a_layout_that_splits_the_block(self, monkeypatch):
        # vllm 0.30 renamed the layouts, and the resolved name for this backend
        # is one of three. Only the identity permutation keeps [H, N, C]
        # together, which is what the descriptor math addresses.
        worker = build_worker(monkeypatch, kv_buffer_device="cpu")
        worker.kv_cache_layout = "LBNHC"
        with pytest.raises(AssertionError, match=r"\[H, N, C\] contiguous"):
            worker.initialize_host_xfer_buffer(
                {"l0": torch.zeros(2, 4, dtype=torch.float16)}
            )

    def test_rejects_a_k_v_first_cache(self, monkeypatch):
        # The rbln_triton_ops cache puts K and V on the leading axis, and
        # upstream's registration -- which this path delegates to -- cuts
        # regions by `cache.stride(0)`, which then spans a whole half. It
        # raises with the shapes alone, so the remedy has to be named here.
        worker = build_worker(monkeypatch, kv_buffer_device="cpu", num_blocks=4)
        worker.kv_cache_layout = "LBHNC"
        with pytest.raises(RuntimeError, match="K/V-first"):
            worker.initialize_host_xfer_buffer(
                {"l0": torch.zeros(2, 4, 8, dtype=torch.float16)}
            )

    def test_accepts_a_blocks_first_cache(self, monkeypatch):
        # The same two axes the other way round is what rbln_custom_ops
        # allocates, and it is the shape this path supports.
        worker = build_worker(monkeypatch, kv_buffer_device="cpu", num_blocks=4)
        worker.kv_cache_layout = "LBHNC"
        worker.initialize_host_xfer_buffer(
            {"l0": torch.zeros(4, 2, 8, dtype=torch.float16)}
        )
        assert worker.host_xfer_buffers["l0"].shape == (4, 2, 8)

    def test_mla_accepts_any_layout(self, monkeypatch):
        # MLA has no head axis to order, so upstream advertises no required
        # layout and the resolved value is whatever the fallback picked. The
        # 3D latent shape must survive unchanged.
        worker = build_worker(monkeypatch, kv_buffer_device="cpu", use_mla=True)
        worker.kv_cache_layout = "LBNHC"
        worker.initialize_host_xfer_buffer(
            {"l0": torch.zeros(4, 64, 576, dtype=torch.float16)}
        )
        assert worker.host_xfer_buffers["l0"].shape == (4, 64, 576)

    def test_rejects_odd_byte_footprint(self, monkeypatch):
        # The page-aligned host buffer is backed by an fp16 (2-byte) allocation,
        # so a cache whose byte footprint is odd cannot be tiled.
        worker = build_worker(monkeypatch, kv_buffer_device="cpu")
        worker.kv_cache_layout = "LBHNC"
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
    # The deferred D2D body: lays the logical K/V regions over each entry's
    # chiplet areas, registers the areas and builds the transfer tables.
    def test_the_compat_hash_gets_the_transfer_mode(self, monkeypatch):
        # vllm 0.30 turned the third parameter into `transfer_mode`. It has a
        # default, so anything passed there is accepted: a bool lands in it
        # without raising and the hash stops separating push from pull, which
        # is the one thing upstream added it for. Bind against the real
        # signature -- the point is which parameter the value reaches.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        topo = _impl_topo(_split_kv(worker.num_blocks))

        with _registering(worker, topo):
            worker._register_kv_caches_impl(kv_caches)
            call = registration.compute_nixl_compatibility_hash.call_args

        bound = inspect.signature(compute_nixl_compatibility_hash).bind(
            *call.args, **call.kwargs
        )
        bound.apply_defaults()
        assert bound.arguments["transfer_mode"] == worker._TRANSFER_MODE

    def test_the_compat_hash_is_equal_across_pipeline_stages(self, monkeypatch):
        names, published = [], []
        for backend_name, backends in _STAGES:
            worker = _prep_impl_worker(monkeypatch)
            worker.backend_name = backend_name
            worker.attn_backends = _backends(backends)
            spec = _impl_layer_spec()
            worker._layer_specs = {"l0": spec, "l1": spec}
            kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)
            with _registering(
                worker,
                _impl_topo(_split_kv(worker.num_blocks)),
                hash_fn=_hash_by_backend(),
            ):
                worker._register_kv_caches_impl(kv_caches)
                names.append(_hashed_backend_name())
            published.append(worker.compat_hash)

        assert len(set(names)) == len(set(published)) == 1

    def test_region_group_ids_index_the_transfer_groups(self, monkeypatch):
        # Upstream's descriptor machinery indexes `transfer_groups`, so a
        # region's group id is a position in THAT list. The fixture puts a
        # group that opted out of transfer ahead of the one that registers, so
        # counting cache groups would number every region 1 instead of 0.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        topo = _impl_topo(_split_kv(worker.num_blocks))

        with _registering(worker, topo):
            worker._register_kv_caches_impl(kv_caches)

        assert worker.region_group_ids == [0, 0, 0, 0]
        assert worker._uses_region_group_mapping is False

    def test_registers_with_vram_segment_and_captures_xfer_tables(self, monkeypatch):
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        topo = _impl_topo(_split_kv(worker.num_blocks))

        with _registering(worker, topo) as env:
            worker._register_kv_caches_impl(kv_caches)

        # rbln_ctx_ptr comes from rebel.context_of(kv_tensor), not a runtime handle.
        env.rebel.context_of.assert_called_once_with(next(iter(kv_caches.values())))

        # The backend gets the resolved ctx ptr, and the areas the VRAM segment.
        backend = env.nixl_rbln.ensure_rbln_backend.call_args
        assert backend.kwargs["rbln_ctx_ptr"] == 0x1000
        register = worker.nixl_wrapper.register_memory.call_args
        assert register.kwargs == {"mem_type": "VRAM", "backends": ["RBLN"]}

        # One area an entry, so a region sits at its offset into that area.
        assert worker.device_id == 0
        assert worker.block_len_per_layer == [2048, 2048, 2048, 2048]
        assert worker.kv_caches_base_addr[worker.engine_id][0] == [
            _area_base(entry, 0) + offset
            for entry in kv_caches.values()
            for offset in (0, 0x100)
        ]
        assert worker._registered_descs == [
            worker.nixl_wrapper.register_memory.return_value
        ]

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

    def test_a_block_split_in_two_doubles_region_count(self, monkeypatch):
        # virtually_split_kv_in_blocks indexes each block's two halves apart, so
        # the region count doubles and cascades into num_descs.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        topo = _impl_topo(
            _split_kv(worker.num_blocks), virtually_split_kv_in_blocks=True
        )

        with _registering(worker, topo):
            worker._register_kv_caches_impl(kv_caches)

        assert worker.num_regions == 8  # 4 base addrs x 2 halves
        assert worker.num_descs == 8 * worker.num_blocks

    def test_mla_registers_one_replicated_region_per_layer_and_area(self, monkeypatch):
        # MLA is key-only, so a layer contributes one logical region rather than
        # two, and every chiplet area of it carries the same latent -> REPLICATE.
        worker = _prep_impl_worker(monkeypatch)
        worker.use_mla = True
        spec = MagicMock(spec=MLAAttentionSpec)
        spec.page_size_bytes = 4096
        spec.num_kv_heads = 1
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _mla_kv_caches(num_blocks=worker.num_blocks)

        areas = 4
        buffers = _impl_buffers(areas=areas)

        # split_k_and_v is False for MLA upstream, so one region per layer.
        topo = _impl_topo(lambda cache, _spec: [cache])
        _probe_mla_shape(topo)

        with _registering(worker, topo, buffers=buffers) as env:
            worker._register_kv_caches_impl(kv_caches)

        # one query per entry
        assert env.nixl_rbln.entry_areas.call_count == 2
        assert worker._region_is_mla == [True] * (2 * areas)
        assert worker.num_regions == 2 * areas
        assert worker.num_descs == 2 * areas * worker.num_blocks
        assert worker._kv_areas == areas
        assert [g.slices for g in worker._logical_region_geometry] == [1, 1]
        # Nothing was cut, so every area holds the latent whole.
        assert [g.axis for g in worker._logical_region_geometry] == [
            ChipletSplitAxis.HEAD
        ] * 2

    def test_constructs_transfer_topology_with_expected_kwargs(self, monkeypatch):
        # The happy-path test fully fakes TransferTopology, so a wrong ctor kwarg
        # would slip through; this pins the exact call.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        topo = _impl_topo(_split_kv(worker.num_blocks))
        topology_cls = MagicMock(return_value=topo)

        with _registering(worker, topo, topology_cls=topology_cls):
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

        topo = _impl_topo(_split_kv(worker.num_blocks))

        with _registering(worker, topo, uniform_spec=UniformTypeKVCacheSpecs):
            worker._register_kv_caches_impl(kv_caches)

        assert worker.block_len_per_layer == [2048] * 4

    def test_a_draft_layer_with_its_own_page_size_is_registered(self, monkeypatch):
        # Requiring one size for every non-MLA tensor would reject this outright;
        # registration has to describe each region by its own geometry.
        worker = _prep_impl_worker(monkeypatch, names=("l0", "l1", "l2"))
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

        topo = _impl_topo(_split_kv(worker.num_blocks))

        with _registering(worker, topo):
            worker._register_kv_caches_impl(kv_caches)

        # Each half is that layer's own page, not the first layer's.
        assert worker.block_len_per_layer == [
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

        topo = _impl_topo(_split_kv(worker.num_blocks))

        with _registering(worker, topo):
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

        topo = _impl_topo(_split_kv(worker.num_blocks))

        with _registering(worker, topo):
            worker._register_kv_caches_impl(kv_caches)

        assert worker._logical_region_kv_heads == [8] * 4

    def test_each_region_records_its_own_slice_count(self, monkeypatch):
        # Each region's own count comes from its entry's layout; the engine
        # scalar is the LAST region's only -- a draft's.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec()
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        # Two entries of four areas each: the first tiled over all four, the
        # second cut in two and replicated across pairs.
        buffers = _impl_buffers(per_entry={0x10000: (4, 4), 0x20000: (4, 2)})
        topo = _impl_topo(_split_kv(worker.num_blocks))
        _probe_kv_shape(topo)

        with _registering(worker, topo, buffers=buffers):
            worker._register_kv_caches_impl(kv_caches)

        assert [g.slices for g in worker._logical_region_geometry] == [4, 4, 2, 2]

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

        topo = _impl_topo(_short_regions)

        with (
            pytest.raises(AssertionError, match="is not 128 blocks of 2048B"),
            _registering(worker, topo),
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

        topo = _impl_topo(_split_kv(worker.num_blocks))

        with (
            pytest.raises(RuntimeError, match="mix MLA and non-MLA"),
            _registering(worker, topo),
        ):
            worker._register_kv_caches_impl(kv_caches)

        # Refused before anything was registered.
        worker.nixl_wrapper.register_memory.assert_not_called()

    def test_cross_layer_blocks_with_differing_page_sizes_are_rejected(
        self, monkeypatch
    ):
        # Without this the page keeps the KV-cache tensor count as a factor, so
        # the stride describes no layer.
        worker = _prep_impl_worker(monkeypatch)
        target = _impl_layer_spec(page_size_bytes=4096, num_kv_heads=8)
        draft = _impl_layer_spec(page_size_bytes=16384, num_kv_heads=32)
        worker._layer_specs = {"l0": target, "l1": draft}
        worker.kv_cache_config = MagicMock(
            kv_cache_tensors=[object(), object()],
            kv_cache_groups=[
                SimpleNamespace(layer_names=["l0", "l1"], kv_cache_spec=None)
            ],
        )
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        topo = _impl_topo(_split_kv(worker.num_blocks), cross_layers_blocks=True)

        with (
            pytest.raises(RuntimeError, match="cross-layer blocks require one page"),
            _registering(worker, topo),
        ):
            worker._register_kv_caches_impl(kv_caches)

    def test_cross_layer_blocks_scale_the_page_by_the_tensor_count(self, monkeypatch):
        # One tensor holds every layer's blocks, so a layer's page covers the
        # whole set and the stride from one block to the next is that much wider.
        worker = _prep_impl_worker(monkeypatch)
        spec = _impl_layer_spec(page_size_bytes=4096)
        worker._layer_specs = {"l0": spec, "l1": spec}
        worker.kv_cache_config = MagicMock(
            kv_cache_tensors=[object(), object()],
            kv_cache_groups=[
                SimpleNamespace(layer_names=["l0", "l1"], kv_cache_spec=None)
            ],
        )
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)

        topo = _impl_topo(_split_kv(worker.num_blocks), cross_layers_blocks=True)

        with _registering(worker, topo):
            worker._register_kv_caches_impl(kv_caches)

        assert worker.block_len_per_layer == [2048 * 2] * 4

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

        topo = _impl_topo(_one_region)

        with _registering(worker, topo, mamba_spec=MambaSpec):
            worker._register_kv_caches_impl(kv_caches)

        assert worker.block_len_per_layer == [2048, 2048]

    def test_regions_cut_on_different_axes_keep_their_own(self, monkeypatch):
        # A target and its draft can be cut differently, and either axis
        # applied to both mislabels the other's areas while every byte count
        # still adds up. Both layers are latents so the two axes differ by
        # geometry alone (a mixed MLA engine is refused earlier).
        worker = _prep_impl_worker(monkeypatch)
        worker.use_mla = True
        latent = MagicMock(spec=MLAAttentionSpec)
        latent.page_size_bytes = 4096
        latent.num_kv_heads = 1
        worker._layer_specs = {"l0": latent, "l1": latent}
        kv_caches = _mla_kv_caches(num_blocks=worker.num_blocks)

        areas = 4
        # The first layer cut on its tokens, the second whole on every area.
        buffers = _impl_buffers(
            per_entry={0x10000: (areas, areas), 0x20000: (areas, 1)},
        )

        topo = _impl_topo(lambda cache, _spec: [cache])
        _probe_mla_shape(topo)

        with _registering(worker, topo, buffers=buffers):
            worker._register_kv_caches_impl(kv_caches)

        assert [g.axis for g in worker._logical_region_geometry] == [
            ChipletSplitAxis.TOKEN,
            ChipletSplitAxis.HEAD,
        ]


class TestTheAxisFollowsTheShardRule:
    """The counts cannot tell a head cut from a token cut by themselves once a
    rank holds more than one head but fewer than it has chiplets -- the shape a
    tensor-parallel GQA model reaches. What decides is the rule the attention
    op picks its shard mode by, read off the cache's own heads and block
    tokens; the device says only how many areas there are and how big."""

    def _run(self, monkeypatch, *, heads=8, block_tokens=64, slices=1, buffers=None):
        worker = _prep_impl_worker(monkeypatch)
        spec = MagicMock(spec=FullAttentionSpec)
        spec.page_size_bytes = 4096
        spec.num_kv_heads = heads
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(
            num_blocks=worker.num_blocks, heads=heads, block_tokens=block_tokens
        )

        topo = _impl_topo(_split_kv(worker.num_blocks))
        _probe_kv_shape(topo)

        # Four areas an entry: two layers, K and V each, so four logical
        # regions over four areas.
        with _registering(
            worker, topo, buffers=buffers or _impl_buffers(areas=4, slices=slices)
        ) as env:
            worker._register_kv_caches_impl(kv_caches)
        worker.fake_nixl_rbln = env.nixl_rbln
        return worker

    def test_a_rank_too_short_of_heads_takes_the_token_cut(self, monkeypatch):
        # Tensor-parallel GQA: two heads are fewer than the four chiplets, and
        # a block of 256 tokens carries a 64-channel shard each, so the op the
        # converter picks is the one that cuts the tokens.
        worker = self._run(monkeypatch, heads=2, block_tokens=256, slices=4)
        assert {g.axis for g in worker._logical_region_geometry} == {
            ChipletSplitAxis.TOKEN
        }

    def test_a_rank_with_heads_to_spare_takes_the_head_cut(self, monkeypatch):
        # The same wide block, but the heads divide: head-TP comes first.
        worker = self._run(monkeypatch, heads=8, block_tokens=256, slices=4)
        assert {g.axis for g in worker._logical_region_geometry} == {
            ChipletSplitAxis.HEAD
        }

    def test_a_block_too_narrow_to_shard_keeps_the_head_cut(self, monkeypatch):
        # Too few heads to divide AND a block no shard can take 64 channels
        # of: the token cut is unavailable, so the band is replicated instead.
        worker = self._run(monkeypatch, heads=2, block_tokens=64, slices=2)
        assert {g.axis for g in worker._logical_region_geometry} == {
            ChipletSplitAxis.HEAD
        }
        assert [g.slices for g in worker._logical_region_geometry] == [2] * 4

    def test_a_replicated_slice_sits_on_adjacent_chiplets(self, monkeypatch):
        # Two bands over four areas: a shard's nodes are a contiguous block of
        # ids, so the copies of a band are neighbours rather than interleaved.
        worker = self._run(monkeypatch, heads=2, block_tokens=64, slices=2)
        assert {
            tuple(place.slice_id for place in geometry.areas)
            for geometry in worker._logical_region_geometry
        } == {(0, 0, 1, 1)}

    def test_an_uncut_entry_is_replicated_not_guessed(self, monkeypatch):
        worker = self._run(monkeypatch, heads=2, slices=1)
        assert {g.axis for g in worker._logical_region_geometry} == {
            ChipletSplitAxis.HEAD
        }
        assert [g.slices for g in worker._logical_region_geometry] == [1] * 4

    def test_each_area_sits_at_its_own_range_of_the_region(self, monkeypatch):
        # A token cut in four: area a holds a quarter of every block, at the
        # quarter of the region's offset into its own allocation.
        worker = self._run(monkeypatch, heads=2, block_tokens=256, slices=4)
        entries = _impl_kv_caches(num_blocks=worker.num_blocks)
        expected = [
            _area_base(entry, a) + offset // 4
            for entry in entries.values()
            for offset in (0, 0x100)
            for a in range(4)
        ]
        assert worker.kv_caches_base_addr[worker.engine_id][0] == expected
        assert worker.block_len_per_layer == [2048 // 4] * 16

    def test_every_area_is_registered_once_on_the_rbln_backend(self, monkeypatch):
        worker = self._run(monkeypatch, heads=2, block_tokens=256, slices=4)
        entries = _impl_kv_caches(num_blocks=worker.num_blocks)
        size = worker.num_blocks * 4096 // 4
        worker.nixl_wrapper.register_memory.assert_called_once_with(
            [
                (_area_base(entry, a), size, 0, "")
                for entry in entries.values()
                for a in range(4)
            ],
            mem_type="VRAM",
            backends=["RBLN"],
        )
        assert worker._registered_descs == [
            worker.nixl_wrapper.register_memory.return_value
        ]
        backend = worker.fake_nixl_rbln.ensure_rbln_backend.call_args
        assert backend.kwargs["rbln_ctx_ptr"] == 0x1000

    def test_entries_over_one_storage_register_its_areas_once(self, monkeypatch):
        # Layers sharing one KV tensor reach registration as entries of their
        # own over one storage, whose areas are the same allocations; an MR
        # each time would register an allocation twice.
        shared = _impl_kv_caches()["l0"]
        areas_of = _impl_buffers(areas=4, slices=4)
        worker = self._run(
            monkeypatch,
            heads=2,
            block_tokens=256,
            buffers=lambda _vaddr: areas_of(shared.data_ptr()),
        )

        (descs,) = worker.nixl_wrapper.register_memory.call_args.args
        assert descs == [
            (_area_base(shared, a), 128 * 4096 // 4, 0, "") for a in range(4)
        ]

    def test_an_entry_inside_its_storage_is_refused(self, monkeypatch):
        # The areas describe the storage from its first byte, so their
        # addresses would land an entry that starts further in on other bytes.
        worker = _prep_impl_worker(monkeypatch)
        spec = MagicMock(spec=FullAttentionSpec)
        spec.page_size_bytes = 4096
        spec.num_kv_heads = 8
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(num_blocks=worker.num_blocks)
        kv_caches["l1"].storage_offset.return_value = 64
        topo = _impl_topo(_split_kv(worker.num_blocks))
        _probe_kv_shape(topo)

        buffers = _impl_buffers(areas=4, slices=1)
        with (
            pytest.raises(RuntimeError, match="64 element"),
            _registering(worker, topo, buffers=buffers),
        ):
            worker._register_kv_caches_impl(kv_caches)

    def test_areas_of_differing_sizes_are_refused(self, monkeypatch):
        # One cut gives every area the same share, so sizes that differ are
        # not the cut this derives.
        even = _impl_buffers(areas=4, slices=4)

        def uneven(vaddr):
            areas = list(even(vaddr))
            areas[-1].size //= 2
            return tuple(areas)

        with pytest.raises(RuntimeError, match="gives every area the same"):
            self._run(monkeypatch, heads=2, block_tokens=256, buffers=uneven)

    def test_an_area_count_the_compiler_does_not_cut_for_is_refused(self, monkeypatch):
        # The shard mode was picked for four chiplets; over any other count
        # the predicates say nothing about which piece an area holds.
        with pytest.raises(RuntimeError, match="picks its shard mode"):
            self._run(
                monkeypatch,
                heads=2,
                block_tokens=256,
                buffers=_impl_buffers(areas=3, slices=3),
            )

    def test_an_area_not_holding_its_share_of_the_entry_is_refused(self, monkeypatch):
        with pytest.raises(RuntimeError, match="do not cut it evenly"):
            self._run(
                monkeypatch,
                heads=2,
                block_tokens=256,
                buffers=_impl_buffers(areas=4, slices=4, size=4096),
            )

    def test_a_cut_that_does_not_tile_the_dim_is_refused(self, monkeypatch):
        # Six heads over four areas: the rule puts the cut on the head dim,
        # and four equal bands are not what six heads make.
        with pytest.raises(RuntimeError, match="does not tile"):
            self._run(monkeypatch, heads=6, slices=4)

    def test_entries_over_different_area_counts_are_refused(self, monkeypatch):
        buffers = _impl_buffers(per_entry={0x10000: (4, 4), 0x20000: (1, 1)})

        with pytest.raises(RuntimeError, match="different chiplet area counts"):
            self._run(monkeypatch, heads=2, block_tokens=256, buffers=buffers)


class TestTheFramesOverOneStorage:
    """A storage several ops read is cut once, so the frames they read it in
    have to agree on the cut -- which the producer is handed them to check."""

    @staticmethod
    def _entry(shape):
        entry = MagicMock()
        entry.shape = shape
        entry.numel.return_value = 4096
        entry.element_size.return_value = 1
        entry.data_ptr.return_value = 0x40000
        entry.storage_offset.return_value = 0
        _IMPL_ENTRIES[0x40000] = entry
        return entry

    @staticmethod
    def _geometry(entry, spec, windows=()):
        """The geometry of one region of `entry`, whose storage `windows` are
        read in besides."""
        worker = object.__new__(RblnNixlPullConnectorWorker)
        worker.transfer_topo = SimpleNamespace(
            kv_head_dim=HEAD_AXIS, kv_token_dim=TOKEN_AXIS
        )
        worker._swa_group_views = [(entry.data_ptr(), view) for view in windows]
        with patch.dict(
            sys.modules,
            {"nixl_rbln": _fake_nixl_rbln(_impl_buffers(areas=4, slices=4))},
        ):
            geometries, _ = worker._region_geometries([(entry, spec, 0, 4096)])
        return geometries[0]

    def test_a_window_frame_that_agrees_leaves_the_entry_s_cut(self):
        # Same heads, blocks of its own: the head cut names the same bytes in
        # both, so the window frame changes nothing about the entry's.
        entry = self._entry((4, 2, 8, 1, 64, 16))
        geometry = self._geometry(
            entry, _impl_layer_spec(), windows=[(32, 2, 8, 1, 8, 16)]
        )

        assert geometry.axis is ChipletSplitAxis.HEAD
        assert [place.slice_id for place in geometry.areas] == [0, 1, 2, 3]

    def test_a_window_kernel_reading_whole_blocks_leaves_the_entry_s_cut(self):
        # A hybrid model's window group aliases the full-attention storage, and
        # where its kernel addresses whole blocks the frame it reads is the
        # entry's own. The op that owns the storage names the cut: put that one
        # frame through the window rule as well and the two answers split, on a
        # shard of 80 tokens that tiles no 64-lane group.
        entry = self._entry((4, 2, 2, 1, 320, 16))
        geometry = self._geometry(
            entry, _impl_layer_spec(), windows=[(4, 2, 2, 1, 320, 16)]
        )

        assert geometry.axis is ChipletSplitAxis.TOKEN
        assert [place.slice_id for place in geometry.areas] == [0, 1, 2, 3]

    def test_frames_whose_rules_disagree_are_refused(self):
        # The window predicate takes the block it reads, which is not the one
        # the full-attention layer reads: a shard of 96 tokens tiles no 64-lane
        # group, so that op stays on head-TP while the other cuts the tokens.
        entry = self._entry((4, 2, 2, 1, 256, 16))
        with pytest.raises(RuntimeError, match="cut it on"):
            self._geometry(entry, _impl_layer_spec(), windows=[(11, 2, 2, 1, 96, 16)])

    def test_frames_of_differing_length_on_the_cut_dim_are_refused(self):
        # Both read it as a token cut, but a quarter of 256 tokens and a
        # quarter of 64 are not the same bytes.
        entry = self._entry((4, 2, 2, 1, 256, 16))
        with pytest.raises(RuntimeError, match="on the cut dim"):
            self._geometry(entry, _impl_layer_spec(), windows=[(16, 2, 2, 1, 64, 16)])

    def test_a_window_layer_that_owns_its_storage_is_read_by_the_window_rule(self):
        # A pure sliding-window pool hands its own layer over, so the entry's
        # frame is a window kernel's. Read by the full-attention rule a
        # 128-token block over 2 heads stays on head-TP, which 4 chiplets do
        # not tile; read by the window rule its quarter tiles a lane group and
        # the tokens are cut.
        entry = self._entry((4, 2, 2, 1, 128, 16))
        spec = MagicMock(spec=SlidingWindowSpec)
        spec.page_size_bytes = 4096
        spec.num_kv_heads = 2
        geometry = self._geometry(entry, spec)

        assert geometry.axis is ChipletSplitAxis.TOKEN
        assert [place.slice_id for place in geometry.areas] == [0, 1, 2, 3]

    def test_a_cut_storage_whose_op_is_unknown_is_refused(self):
        # A chunked-local layer is an attention spec this path holds no shard
        # contract for; a cut storage of one has no cut to name.
        entry = self._entry((4, 2, 2, 1, 256, 16))
        with pytest.raises(RuntimeError, match="shard contract this path"):
            self._geometry(entry, MagicMock(spec=ChunkedLocalAttentionSpec))


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
        w._kv_per_block = 1
        w.vllm_config = mock_vllm_config()
        w.vllm_config.parallel_config.pipeline_parallel_size = pp_size
        w.vllm_config.speculative_config = None
        w.transfer_topo = MagicMock()
        w.transfer_topo.cross_layers_blocks = cross_layers
        w._has_mamba = has_mamba
        window_mode(
            w, sw_ratio, has_swa=(sw_ratio is not None) if has_swa is None else has_swa
        )
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
        # `sw_ratio=None` is the model with window mode off: a sliding window
        # bars pipelining on its own, which is what `has_swa` exists for.
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
    def _base_meta(layer_names=("l0", "l1")):
        # One region per layer here, which is what the layer names derive from.
        names = list(layer_names)
        return NixlAgentMetadata(
            engine_id="eng",
            agent_metadata=b"agent",
            kv_caches_base_addr=[0x1000, 0x2000][: len(names)],
            device_id=0,
            num_blocks=4,
            block_lens=[8192] * len(names),
            block_strides=[8192] * len(names),
            kv_cache_layout="LBHNC",
            block_size=16,
            ssm_sizes=(0, 0),
            attn_backend_name="RBLN",
            physical_blocks_per_logical_kv_block=1,
            region_names=names,
        )

    def _publish(
        self,
        *,
        pp_rank,
        pp_size,
        layer_names,
        areas=1,
        slices=1,
        axes=(ChipletSplitAxis.HEAD,),
        cls=None,
        has_mamba=False,
        cross_layers=False,
        base_meta=None,
        swa_kernel_blocks=frozenset(),
        slice_ids=None,
    ):
        w = object.__new__(cls or RblnNixlPullConnectorWorker)
        w._kv_per_block = 1
        # __init__ never ran, so the writer state shutdown() reaches through
        # __del__ is absent; silence it rather than leak an unraisable at GC.
        w.shutdown = lambda: None
        # _check_pp_constraints reads these; a plain PP producer passes.
        w.vllm_config = mock_vllm_config()
        w.vllm_config.parallel_config.pipeline_parallel_size = pp_size
        w.vllm_config.speculative_config = None
        w.transfer_topo = MagicMock()
        w.transfer_topo.cross_layers_blocks = cross_layers
        w._has_mamba = has_mamba
        window_mode(w, None, has_swa=False)
        w._swa_kernel_blocks = set(swa_kernel_blocks)
        w.use_mla = False
        # Chiplet geometry travels with the metadata so a consumer with a
        # different TP degree can match head bands. One logical region over one
        # area unless a test names more.
        fill_region_geometry(w, areas=areas, axes=list(axes), slices=slices)
        if slice_ids is not None:
            w._logical_region_geometry = [
                g._replace(
                    areas=tuple(
                        p._replace(slice_id=i) for p, i in zip(g.areas, slice_ids)
                    )
                )
                for g in w._logical_region_geometry
            ]
        pp_group = MagicMock()
        pp_group.rank_in_group = pp_rank
        pp_group.world_size = pp_size
        with (
            patched_in_package("get_pp_group", MagicMock(return_value=pp_group)),
            patched_in_package(
                "compute_nixl_compatibility_hash", MagicMock(return_value="BASE")
            ),
        ):
            w._publish_handshake_metadata(
                self._base_meta(layer_names) if base_meta is None else base_meta
            )
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
        assert w.compat_hash == rbln_compat_hash(
            "BASE", writes_into_peer=True, cross_layers_blocks=False
        )

    def test_the_region_ledger_survives_the_re_wrap(self):
        # The published payload is our subclass rebuilt field by field from
        # upstream's. A field left out is not an error: it takes the dataclass
        # default, so the peer reads `None` and describes our regions by its
        # own geometry.
        base = self._base_meta()
        base.region_num_blocks = [3, 3]
        base.region_group_ids = [0, 1]
        base.region_names = ["l0", "l0"]
        base.region_mem_types = ["VRAM", "VRAM"]
        base.dcp_size = 1
        base.pcp_size = 1

        w = self._publish(pp_rank=0, pp_size=1, layer_names=["l0"], base_meta=base)

        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert decoded.region_num_blocks == [3, 3]
        assert decoded.region_group_ids == [0, 1]
        assert decoded.region_names == ["l0", "l0"]
        assert decoded.region_mem_types == ["VRAM", "VRAM"]

    def test_cross_layer_blocks_reach_the_hash(self):
        # vllm 0.30 dropped the flag from upstream's own factors, so a peer that
        # scales its page by the tensor count and one that does not would match
        # unless this path folds it in.
        plain = self._publish(pp_rank=0, pp_size=1, layer_names=["l0"])
        crossed = self._publish(
            pp_rank=0, pp_size=1, layer_names=["l0"], cross_layers=True
        )
        assert plain.compat_hash != crossed.compat_hash

    def test_advertises_the_split_axis_of_every_region(self):
        # A consumer cannot derive them: the areas and slices it also receives
        # are the same numbers under either axis (see TestSplitAxisConstraints),
        # and a draft can be cut on the other one.
        axes = [ChipletSplitAxis.TOKEN, ChipletSplitAxis.HEAD]
        w = self._publish(
            pp_rank=0,
            pp_size=1,
            layer_names=["l0"],
            areas=4,
            slices=4,
            axes=axes,
        )
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert decoded.kv_split_axes == axes

    def test_advertises_chiplet_geometry(self):
        """Head-band matching on the consumer needs the producer's areas/slices;
        they cannot be derived from the address list without assuming exactly
        two regions per layer."""
        w = self._publish(pp_rank=0, pp_size=1, layer_names=["l0"], areas=4, slices=2)
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert decoded.kv_area_slices == [[0, 0, 1, 1]]

    def test_advertises_the_slices_in_area_order(self):
        # A peer picks the area it reads a slice from by position in this
        # list, so interleaved replicas have to go out as laid out.
        w = self._publish(
            pp_rank=0,
            pp_size=1,
            layer_names=["l0"],
            areas=4,
            slices=2,
            slice_ids=[0, 1, 0, 1],
        )
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert decoded.kv_area_slices == [[0, 1, 0, 1]]

    def test_advertises_the_kernel_block_the_window_is_cut_by(self):
        """A peer cuts its window range by its own runner's answer, and the two
        sides have to be cut by one number. Nothing else in the blob says which
        -- `block_size` and `sliding_window` are both legal values of it."""
        w = self._publish(
            pp_rank=0, pp_size=1, layer_names=["l0"], swa_kernel_blocks={128}
        )
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert decoded.swa_kernel_block == 128

    def test_a_shard_holding_no_window_advertises_zero(self):
        # Not None: the field is an int over the wire, and zero is what the
        # pairing reads as "nothing to disagree with".
        w = self._publish(pp_rank=0, pp_size=1, layer_names=["l0"])
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert decoded.swa_kernel_block == 0

    def test_a_shard_whose_groups_disagree_advertises_zero(self):
        # Two answers cannot be advertised as one, and the peer cuts our
        # addresses by whatever this number says. Zero withdraws the claim.
        w = self._publish(
            pp_rank=0, pp_size=1, layer_names=["l0"], swa_kernel_blocks={16, 64}
        )
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert decoded.swa_kernel_block == 0

    def test_wraps_upstream_and_folds_compat(self):
        w = self._publish(pp_rank=1, pp_size=2, layer_names=["l7", "l8"])
        # compat hash folded with our version and direction, mirrored into the
        # payload. A read-path worker must publish the read-path hash.
        assert w.compat_hash == rbln_compat_hash(
            "BASE", writes_into_peer=False, cross_layers_blocks=False
        )
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
        w.engine_id = "eng"
        w._kv_per_block = 1
        w.kv_buffer_device = "cpu"
        w._has_mamba = False
        w._use_rbln_nixl_backend = False
        w.xfer_handshake_metadata = MagicMock(
            agent_metadata_bytes=msgspec.msgpack.encode(self._base_meta())
        )
        w._publish_handshake_metadata = MagicMock()
        kv_caches = {"l0": MagicMock(), "l1": MagicMock()}
        w.kv_cache_config = SimpleNamespace(
            kv_cache_tensors=[
                KVCacheTensor(
                    size=0, layers=list(kv_caches), layer_stride=1, block_stride=1
                )
            ],
            # No sliding-window group, so nothing names a window frame.
            transfer_groups=[],
        )
        # Registration reads the layer specs and the transfer table upstream
        # fills, to record each region's head count.
        w._layer_specs = {
            name: MagicMock(page_size_bytes=4096, num_kv_heads=8) for name in kv_caches
        }
        w.block_len_per_layer = [2048, 2048, 2048, 2048]
        # Upstream fills the base addresses beside the lengths.
        w.tp_rank = 0
        w.kv_caches_base_addr = {
            w.engine_id: {
                w.tp_rank: [0x1000 * (i + 1) for i in range(len(w.block_len_per_layer))]
            }
        }
        w.world_size = 1
        # A layer's head count is accepted only if a model in this engine has it:
        # the target's, or a speculative draft's where there is one.
        w.model_config = MagicMock()
        w.model_config.get_total_num_kv_heads.return_value = 8
        w.vllm_config = mock_vllm_config()
        w.vllm_config.speculative_config = None

        with patch.object(NixlBaseConnectorWorker, "register_kv_caches"):
            w.register_kv_caches(kv_caches)

        assert w.local_seen_layer_names == ["l0", "l1"]
        w._publish_handshake_metadata.assert_called_once()
        (published_meta,) = w._publish_handshake_metadata.call_args[0]
        # Upstream's metadata is handed over decoded, not as bytes. The layers
        # ride on it now, in `region_names`, rather than beside it.
        assert published_meta.engine_id == "eng"

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
    # Three things the D2D path does whose absence is silent: none shows up as
    # a failure anywhere else, so each has one test here and nowhere.

    def test_a_layers_region_spans_its_whole_entry(self, make_worker):
        # Every base comes from the region's offset and every block from its
        # length, so those are what have to be right. A packed block holds K
        # and V, so a layer is one region at offset 0 whose blocks cover the
        # entry -- the descriptor path cuts the halves out of it later
        # (`_kv_per_block`), registration does not.
        geometry = KvGeometry(layers=("l0", "l1"))
        w = make_worker(kv_cache=geometry)
        entries = geometry.kv_caches()
        assert len(w.block_len_per_layer) == 2, "one region per layer"
        entry_bytes = {e.numel() * e.element_size() for e in entries.values()}
        assert {w.num_blocks * length for length in w.block_len_per_layer} == (
            entry_bytes
        )

    @pytest.mark.parametrize(
        "stripe_width, passed",
        [(None, {}), (0, {"stripe_width": 0}), (4096, {"stripe_width": 4096})],
    )
    def test_a_stripe_width_reaches_the_adapter_only_when_named(
        self, make_worker, stripe_width, passed
    ):
        # A backend-lifecycle override, and this path creates the backend just
        # before it registers -- so this is where it has to ride. The 0 case is
        # what separates "nobody named it" from a width of zero.
        make_worker(stripe_width=stripe_width)
        assert sys.modules["nixl_rbln"].backend_kwargs_seen == [passed]

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
        assert decoded.kv_area_slices == [[0, 1, 2, 3]] * 2


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
        # Two layers, K and V inside one block -> one region per layer. The
        # split layout gives two of them, which `_kv_per_block` tells apart.
        assert len(w.block_len_per_layer) == 2
        assert w._logical_region_kv_heads == [8, 8]
        assert {g.axis for g in w._logical_region_geometry} == {ChipletSplitAxis.HEAD}
        bases = w.kv_caches_base_addr[w.engine_id][w.tp_rank]
        assert len(bases) == 2
        # Real addresses, and each region a distinct span of its tensor.
        assert len(set(bases)) == 2 and all(base > 0 for base in bases)

    @pytest.mark.parametrize("draft_kv_heads", [8, 4])
    def test_a_draft_layer_takes_its_own_models_head_count(
        self, make_worker, draft_kv_heads
    ):
        # Equal head counts leave one page size for the group; unequal ones make
        # the engine wrap the group spec, and the draft's regions are then half
        # the size. Both are real configurations of the same connector.
        geo = KvGeometry(layers=("l0", "l1"), draft_layers=("l1",))
        w = make_worker(kv_cache=geo, draft_kv_heads=draft_kv_heads)
        assert w._logical_region_kv_heads == [8, draft_kv_heads]
        # The connector's own answer, not the fixture's: one area, so a region's
        # block is the unwrapped layer spec's page size whole.
        lens = list(w.block_len_per_layer)
        assert lens[1] * 8 == lens[0] * draft_kv_heads

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

    def test_every_layer_of_a_window_group_names_its_own_storage(self, make_worker):
        # A hybrid model's window group reads one storage per layer, and the
        # frame it reads each in lays that storage out beside the entry's own.
        # Here the second layer's two frames pick opposite axes: a 480-token
        # block over 2 heads is a token cut, while the 48-token view its kernel
        # reads tiles no lane group and stays on head-TP.
        geo = KvGeometry(
            layers=("l0", "l1"),
            spec="swa",
            sliding_window=240,
            block_size=480,
            swa_kernel_block=48,
            per_layer_heads={"l1": 2},
            areas=4,
            slices=4,
        )
        with pytest.raises(RuntimeError, match="cut it on"):
            make_worker(kv_cache=geo)

    def test_a_context_cut_over_a_packed_block_registers(self, make_worker):
        # The run count carries the layout -- a token range of one head sits
        # once in K and once in V -- so a packed block over a token cut needs
        # nothing of its own.
        geo = KvGeometry(layers=("l0",), heads=1, areas=4, slices=4, cut="token")
        w = make_worker(kv_cache=geo)
        assert {g.axis for g in w._logical_region_geometry} == {ChipletSplitAxis.TOKEN}

    def test_a_draft_cut_on_another_axis_registers_and_says_so(self, make_worker):
        # The target rank holds one head, too few to spread over its four
        # chiplets, so a 1k block sends it to the token cut; its draft holds
        # four and takes the head cut. One engine, two axes.
        geo = KvGeometry(
            layers=("l0", "l1"),
            heads=1,
            areas=4,
            slices=4,
            draft_layers=("l1",),
        )
        w = make_worker(kv_cache=geo, draft_kv_heads=4)

        per_layer = geo.regions_per_layer
        token, head = ChipletSplitAxis.TOKEN, ChipletSplitAxis.HEAD
        axes = [token] * per_layer + [head] * per_layer
        assert [g.axis for g in w._logical_region_geometry] == axes
        decoded = msgspec.msgpack.Decoder(RblnNixlAgentMetadata).decode(
            w.xfer_handshake_metadata.agent_metadata_bytes
        )
        assert decoded.kv_split_axes == axes

    @pytest.mark.parametrize(
        "step, token_cut, head_cut",
        [
            # A 1k block's target area holds 256 tokens: a 128-token step cuts
            # it in two, and the draft's whole block in eight. The block packs
            # K and V, so a chunk is a run in each.
            (128, lambda area: (area * 2, 2, 2), (0, 2, 8)),
            # A 512-token step is two target areas, each whole, so the target
            # needs no chunk descriptor; the draft's block is cut in two.
            (512, lambda area: (area // 2, 0, 1), (0, 2, 2)),
        ],
    )
    def test_a_draft_cut_on_another_axis_registers_a_chunk_range(
        self, make_worker, step, token_cut, head_cut
    ):
        geo = KvGeometry(
            layers=("l0", "l1"),
            heads=1,
            areas=4,
            slices=4,
            draft_layers=("l1",),
        )
        w = make_worker(
            kv_cache=geo, draft_kv_heads=4, chunk_mode=True, prefill_step=step
        )

        per_layer = geo.regions_per_layer * geo.areas
        target = tuple(
            token_cut(area) for _ in range(geo.regions_per_layer) for area in range(4)
        )
        assert w._chunk_grid == ChunkGrid(
            geo.block_size // step, target + (head_cut,) * per_layer
        )

    def test_an_undeclared_head_count_has_no_band(self, make_worker):
        # Neither the target's count nor any declared draft's: replicated heads
        # report a product no model in this engine has, and _layer_kv_heads
        # refuses to name a band rather than compute one from the wrong model.
        geo = KvGeometry(layers=("l0", "l1"), per_layer_heads={"l1": 2})
        w = make_worker(kv_cache=geo)
        assert w._logical_region_kv_heads == [8, None]


# What each layout puts in one block.
PACKED = 2  # rbln_custom_ops: (num_blocks, 2, H, 1, S, D)
SPLIT = 1  # rbln_triton_ops: (2, num_blocks, H, 1, S, D)


class TestTheLayoutReachesTheDescriptors:
    """`_kv_per_block` is worker state; these pin what reads it."""

    @staticmethod
    def _worker(kv_per_block):
        w = object.__new__(RblnNixlPullConnectorWorker)
        w.use_host_buffer = False
        window_mode(w, None)
        w._kv_per_block = kv_per_block
        # The chunk grid asks for the head bands before it asks anything else.
        set_shape(w, chunk_mode=False)
        w._logical_region_kv_heads = [8]
        fill_region_geometry(w, areas=1, axes=[ChipletSplitAxis.HEAD])
        topo = MagicMock()
        topo.tp_size = 1
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
        meta.kv_area_slices = area_slices(2, 2, 1)
        assert w._peer_kv_runs(meta, remote_tp_size=1) == expected

    def test_a_peer_of_our_own_width_never_splits(self):
        # Both sides name whole blocks, so the layout does not matter.
        w = self._worker(PACKED)
        meta = MagicMock()
        meta.kv_area_slices = area_slices(1, 1, 1)
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
        meta.kv_area_slices = area_slices(2, 2, 1)
        assert w._peer_kv_runs(meta, remote_tp_size=1) == 1


class TestTailBlockTrim:
    # Whether a context-cut engine may leave a last block's empty areas out,
    # and the geometries whose areas do not name a token range at all.

    @staticmethod
    def _register(monkeypatch, *, areas, slices, block_tokens=64, **kw):
        worker = _prep_impl_worker(
            monkeypatch,
            specs=[MagicMock(spec=FullAttentionSpec)],
            **kw,
        )
        worker.use_mla = True
        spec = MagicMock(spec=MLAAttentionSpec)
        spec.page_size_bytes = 4096
        spec.num_kv_heads = 1
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _mla_kv_caches(
            num_blocks=worker.num_blocks, block_tokens=block_tokens
        )

        # A latent has one head, so the compiler cuts it on its tokens.
        buffers = _impl_buffers(areas=areas, slices=slices)

        # the chunk grid asks the topology for a real head band
        topo = _impl_topo(lambda cache, _spec: [cache], tp_size=1)
        _probe_mla_shape(topo)

        with _registering(worker, topo, buffers=buffers):
            worker._register_kv_caches_impl(kv_caches)
        return worker

    def test_a_context_cut_into_whole_areas_enables_the_trim(self, monkeypatch):
        worker = self._register(monkeypatch, areas=4, slices=4, chunk_mode=True)
        assert {g.axis for g in worker._logical_region_geometry} == {
            ChipletSplitAxis.TOKEN
        }
        assert worker._shape.chunk_mode is True

    def test_registration_leaves_the_grid_a_transfer_reads_back(self, monkeypatch):
        # A transfer picks its range by this, and nothing else sets it: left
        # unset, the third range is registered and never selected -- every
        # request goes whole while the longer dlist is still paid for. The span
        # has to be wider than the chunk, or the size collapses the grid to
        # None and the assertion holds for the wrong reason.
        worker = self._register(
            monkeypatch,
            areas=4,
            slices=4,
            chunk_mode=True,
            block_size=1024,
            chunk_tokens=128,
        )

        assert worker._chunk_grid == worker._shard_chunk_grid(
            range(worker.num_regions), block_size=worker.block_size, split=1
        )
        assert worker._chunk_grid is not None

    @pytest.mark.parametrize(
        "sizing, line",
        [
            (
                {"block_size": 1024, "chunk_tokens": 128},
                "chunk_mode on: 128-token chunks, 8 a block.",
            ),
            ({"block_size": 128}, "chunk_mode on: 128-token chunks, 1 a block."),
        ],
        ids=["cut", "collapsed"],
    )
    def test_the_knob_on_says_so(self, monkeypatch, caplog, sizing, line):
        # A chunk is never narrower than one prefill step, so a step as wide as
        # the block (the widest the platform allows) leaves every block whole.
        # The knob is still on either way, and the line says what it cuts a
        # block into.
        with caplog.at_level("INFO"):
            self._register(monkeypatch, areas=4, slices=4, chunk_mode=True, **sizing)

        assert f"RBLN NIXL (D2D): {line}" in [r.getMessage() for r in caplog.records]

    def test_the_flag_off_leaves_the_same_geometry_alone(self, monkeypatch, caplog):
        # Same cut, opposite answer: nothing about the geometry turns this on.
        with caplog.at_level("INFO"):
            worker = self._register(monkeypatch, areas=4, slices=4, chunk_mode=False)
        assert {g.axis for g in worker._logical_region_geometry} == {
            ChipletSplitAxis.TOKEN
        }
        assert worker._shape.chunk_mode is False
        assert "RBLN NIXL (D2D): chunk_mode off." in [
            r.getMessage() for r in caplog.records
        ]

    def test_a_replicated_context_cut_names_each_areas_token_range(self, monkeypatch):
        # Two areas a slice: each copy holds its slice's range, which is what
        # a chunk of the block is read off, not the area's position.
        worker = self._register(
            monkeypatch,
            areas=4,
            slices=2,
            chunk_mode=True,
            block_size=1024,
            chunk_tokens=128,
        )

        assert [worker._region_span_index(r) for r in range(4)] == [0, 0, 1, 1]
        assert [first for first, _, _ in worker._chunk_grid.regions[:4]] == [0, 0, 4, 4]

    def test_areas_that_do_not_divide_the_block_are_refused(self, monkeypatch):
        # 6 tokens over 4 areas: no area is a whole number of them.
        with pytest.raises(RuntimeError, match="does not tile"):
            self._register(
                monkeypatch, areas=4, slices=4, block_tokens=6, chunk_mode=True
            )

    def test_host_staging_is_refused_before_anything_registers(self, monkeypatch):
        # Host staging keeps one full-shape buffer per layer, so the flag would
        # otherwise be silently inert: `_register_kv_caches_impl` is D2D-only.
        with pytest.raises(RuntimeError, match="host staging"):
            build_worker(monkeypatch, kv_buffer_device="cpu", chunk_mode=True)

    def test_host_staging_refuses_the_window_knob_too(self, monkeypatch):
        # The window range sits in the same lists a chunk range would, so host
        # staging has no more room for one than for the other.
        with pytest.raises(RuntimeError, match="host staging"):
            build_worker(monkeypatch, kv_buffer_device="cpu", swa_window_mode=True)

    def test_host_staging_refuses_streaming_on_the_side_that_would_do_it(
        self, monkeypatch
    ):
        # A prefix offer is named in per-shard descriptors, which host staging
        # has none of. It used to be dropped in silence, leaving an operator
        # who asked for it with an engine that never streamed.
        with pytest.raises(RuntimeError, match="host staging"):
            build_worker(
                monkeypatch,
                kv_buffer_device="cpu",
                push_stream=True,
                cls=RblnNixlPushConnectorWorker,
            )

    def test_host_staging_leaves_the_reading_side_alone(self, monkeypatch):
        # One `--kv-transfer-config` reaches both ends, and the knob names
        # something only the writer does. Refusing it here would refuse the
        # consumer of a pair whose producer legitimately asked.
        worker = build_worker(monkeypatch, kv_buffer_device="cpu", push_stream=True)

        assert worker._shape.streams_prefix is False


class TestChunkModeWithASlidingWindow:
    """A hybrid engine is let into chunk mode because it owns the whole-engine
    descriptor lists, which is where a chunk range can sit. The per-shard lists
    it would otherwise be sent to cannot name two KV groups: their
    region-to-group map holds one group per region, and under HMA both groups
    share every region. A window range is a separate knob, not the price of
    admission."""

    @staticmethod
    def _register(
        monkeypatch, *, specs, chunk_mode=True, axis=None, push_stream=False, cls=None
    ):
        worker = _prep_impl_worker(
            monkeypatch,
            specs=specs,
            chunk_mode=chunk_mode,
            push_stream=push_stream,
            cls=cls,
        )
        # A context cut is a rank of one head over a block wide enough for a
        # shard of it to carry 64 channels; a head cut is the default 8 heads
        # over a narrow block. The axis is derived from that pair, so asking
        # for it here means building the counts that make it.
        context_cut = axis is ChipletSplitAxis.TOKEN
        spec = _impl_layer_spec(
            page_size_bytes=512, num_kv_heads=1 if context_cut else 8
        )
        worker._layer_specs = {"l0": spec, "l1": spec}
        kv_caches = _impl_kv_caches(
            num_blocks=worker.num_blocks,
            heads=1 if context_cut else 8,
            block_tokens=256 if context_cut else 64,
        )

        # Four logical regions (two layers, K and V), each expanded into the
        # areas a context cut gives it.
        areas = 4 if context_cut else 1
        buffers = _impl_buffers(areas=areas, slices=areas)

        # 8 heads of a 256B region: the block has to be small enough that a
        # token of one head is at least a byte, or the geometry is a fiction.
        # The step is at most the block, as the platform requires.
        worker.block_size = 8
        worker.vllm_config.scheduler_config.max_num_batched_tokens = 8
        # the chunk grid asks the topology for a real head band
        topo = _impl_topo(_split_kv(worker.num_blocks), tp_size=1)
        _probe_kv_shape(topo)

        with _registering(worker, topo, buffers=buffers):
            worker._register_kv_caches_impl(kv_caches)
        return worker

    @staticmethod
    def _hybrid_specs():
        # What a gpt-oss-shaped model hands the worker: one full-attention
        # group and one sliding-window group, both over the same regions.
        return [
            MagicMock(spec=FullAttentionSpec),
            sliding_window_spec(block_size=64, sliding_window=16),
        ]

    def test_a_hybrid_enters_chunk_mode_without_a_window_range(self, monkeypatch):
        # The second range is not the price of admission: a hybrid owns the
        # whole-engine lists on its own.
        worker = self._register(monkeypatch, specs=self._hybrid_specs())

        assert worker._shape.window_ratio is None
        assert worker._shape.chunk_mode is True
        assert worker._own_engine_layout

    def test_two_groups_without_a_window_are_still_refused(self, monkeypatch):
        # Nothing gave this engine a second range, so a third has nowhere to go.
        with pytest.raises(RuntimeError, match="sliding window whose view"):
            self._register(
                monkeypatch,
                specs=[MagicMock(spec=FullAttentionSpec)] * 2,
            )

    def test_a_context_cut_with_a_window_is_refused(self, monkeypatch):
        # The whole-engine lists a window keeps name a block's chunks without
        # naming which span holds the last token, so a block cut into several
        # spans would send chunks addressed past the request's own blocks.
        with pytest.raises(RuntimeError, match="needs a head cut"):
            self._register(
                monkeypatch,
                specs=self._hybrid_specs(),
                axis=ChipletSplitAxis.TOKEN,
            )

    def test_a_streamed_hybrid_on_a_context_cut_is_refused(self, monkeypatch):
        # A hybrid stays on the whole-engine lists whichever asked for the
        # chunks, and those lists name every region's chunks with no way to say
        # which span holds the last token. Streaming reaches them without the
        # knob, so the refusal cannot be the knob's alone.
        with pytest.raises(RuntimeError, match="needs a head cut"):
            self._register(
                monkeypatch,
                specs=self._hybrid_specs(),
                chunk_mode=False,
                push_stream=True,
                axis=ChipletSplitAxis.TOKEN,
                cls=RblnNixlPushConnectorWorker,
            )

    def test_a_window_with_no_full_group_to_trim_is_refused(self, monkeypatch):
        # The view alone is not the point: a chunk cuts full-attention blocks,
        # and an engine with none of them would register a range it never uses.
        with pytest.raises(RuntimeError, match="one full-attention"):
            self._register(
                monkeypatch,
                specs=[sliding_window_spec(block_size=64, sliding_window=16)],
            )

    def test_a_streamed_engine_with_no_full_group_is_refused(self, monkeypatch):
        # Streaming names part of a block without the knob, so the refusal
        # cannot be the knob's: this engine registers, and the write path's
        # asserts then name a group it has none of, mid-transfer.
        with pytest.raises(RuntimeError, match="one full-attention"):
            self._register(
                monkeypatch,
                specs=[sliding_window_spec(block_size=64, sliding_window=16)],
                chunk_mode=False,
                push_stream=True,
                cls=RblnNixlPushConnectorWorker,
            )
