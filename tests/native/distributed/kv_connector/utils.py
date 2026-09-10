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

"""Builders for the RBLN NIXL connector tests.

``conftest.py::make_worker`` runs the worker's real ``__init__`` on a real
VllmConfig and KVCacheConfig over real KV tensors, faking what it cannot reach
here: the NIXL agent and its config, the ``nixl_rbln`` adapter, ``rebel``'s
context lookup, ``get_device``, the TP-rank accessors and the view-opt env
flag. ``build_worker`` below goes further and replaces upstream's ``__init__``.

``KvGeometry`` is why that is affordable. One object produces all three views of
a geometry -- the KVCacheConfig the worker is built with, the tensors it
registers, and the sharding the adapter reports -- so a test varies one value
and the three cannot drift apart. An MLA geometry has only the last two: the
engine config here is non-MLA, so no worker can be built over an MLA cache.
Import lazily from a fixture: this module pulls in vllm.config.
"""

from __future__ import annotations

import contextlib
import functools
from dataclasses import dataclass, field
from typing import Any

import torch

# A hub id would revalidate config.json over the network on every build.
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# RBLN needs an explicit block_size, and VllmConfig validates it against the
# platform's prefix block size -- tests/native/v1/worker/utils.py pins the same
# value for the same reason.
BLOCK_SIZE = 1024
# No geometry case varies this: a head's size does not enter the region
# arithmetic, only the byte total it contributes to.
HEAD_SIZE = 64


@functools.cache
def engine_config(
    *,
    kv_buffer_device: str = "rbln",
    speculative_model: str | None = None,
    block_size: int = BLOCK_SIZE,
    kv_role: str = "kv_both",
) -> Any:
    """A real VllmConfig with a kv_transfer_config, through EngineArgs.

    Cached on the keyword tuple: the first build pays for
    ``create_engine_config`` and backend resolution, every later one with the
    same shape is free. The connector never mutates it.
    """
    from vllm.config import KVTransferConfig

    from tests.native.vllm_config import make_vllm_config

    extra: dict[str, Any] = {}
    if speculative_model is not None:
        extra["speculative_config"] = {
            "model": speculative_model,
            "num_speculative_tokens": 2,
        }
    return make_vllm_config(
        model=MODEL,
        block_size=block_size,
        kv_transfer_config=KVTransferConfig(
            kv_connector="RblnNixlConnector",
            kv_role=kv_role,
            kv_buffer_device=kv_buffer_device,
        ),
        **extra,
    )


def mock_vllm_config(**options) -> Any:
    """A mocked config that answers this connector's knob lookups.

    A bare ``MagicMock`` answers every attribute with a truthy mock, so a knob
    read from one would come back ON -- the opposite of every default here, and
    silently. This asks for the knobs the test means and leaves the rest at
    their defaults, so a test that forgets one gets the default rather than a
    mock.
    """
    from unittest.mock import MagicMock

    config = MagicMock()
    # The one attribute outside the knobs that a mock answers wrongly rather
    # than harmlessly: a chunk size is floored by it, and arithmetic on a mock
    # raises instead of producing a chunk size that is merely wrong. The value
    # is the one `engine_config` builds, so a mocked config and a real one size
    # a chunk the same way.
    config.scheduler_config.max_num_batched_tokens = 128
    set_mock_connector_options(config, **options)
    return config


def set_mock_connector_options(vllm_config, **options) -> None:
    """The same, on a mocked config a test already has.

    Replacing it would take the rest of the mock's shape with it -- the
    scheduler config a chunk size is floored by, for one.
    """
    vllm_config.kv_transfer_config.get_from_extra_config.side_effect = (
        lambda key, default: options.get(key, default)
    )


def draft_model_dir(dest: Any, kv_heads: int) -> str:
    """A draft model config with ``kv_heads`` KV heads, and nothing else new.

    vLLM refuses a draft whose vocabulary differs from the target's, so a draft
    with a head count of its own is built by copying the target's config and
    changing that one field. Tokenizer, vocabulary and architecture stay the
    target's, which leaves the head count as the only axis that varies.

    One directory per head count, written under the caller's temporary path.
    Nothing loads weights from it: the connector reads the config's shapes.
    """
    import json
    import pathlib

    from tests.native.vllm_config import local_model_path

    out = pathlib.Path(dest) / f"draft-kv{kv_heads}"
    if not out.exists():
        config = json.loads(
            (pathlib.Path(local_model_path(MODEL)) / "config.json").read_text()
        )
        config["num_key_value_heads"] = kv_heads
        # One decoder layer, as a real eagle draft has.
        config["num_hidden_layers"] = 1
        out.mkdir(parents=True)
        (out / "config.json").write_text(json.dumps(config))
    return str(out)


@dataclass(frozen=True)
class KvGeometry:
    """What the device looks like, in the terms a geometry test varies.

    ``per_layer_heads`` gives one layer a head count of its own, which is the
    speculative-draft case: the layers of one group no longer share a page
    size. ``areas`` is the chiplet count one logical region expands into and
    ``slices`` how many distinct head bands those areas hold; ``areas >
    slices`` would mean a replicated band, which this does not model.
    """

    layers: tuple[str, ...] = ("l0", "l1")
    heads: int = 8
    block_size: int = BLOCK_SIZE
    num_blocks: int = 4
    spec: str = "full"
    sliding_window: int | None = None
    areas: int = 1
    slices: int = 1
    per_layer_heads: dict[str, int] = field(default_factory=dict)
    draft_layers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        # The byte arithmetic here partitions one allocation, but a replicated
        # band is a separate copy per chiplet: the adapter divides a block by
        # the slice count, not the area count, and groups adjacent areas as
        # replicas of one slice. The table would be one no device produces.
        assert self.areas == self.slices, (
            f"KvGeometry does not model a replicated head band "
            f"({self.areas} areas over {self.slices} slices)"
        )

    def with_draft_heads(self, heads: int) -> KvGeometry:
        """Give every draft layer the head count its model declares.

        The fixture calls this once the config is built, so a draft layer's
        geometry cannot drift from the speculative config the same worker was
        constructed with -- which is the pairing `_layer_kv_heads` checks.
        """
        from dataclasses import replace

        return replace(
            self,
            per_layer_heads={
                **self.per_layer_heads,
                **{name: heads for name in self.draft_layers},
            },
        )

    def heads_of(self, layer: str) -> int:
        return self.per_layer_heads.get(layer, self.heads)

    # --- view 1: the config the worker is constructed with -------------------
    def kv_cache_config(self, *, dtype: torch.dtype = torch.bfloat16) -> Any:
        from vllm.v1.kv_cache_interface import (
            FullAttentionSpec,
            KVCacheConfig,
            KVCacheGroupSpec,
            KVCacheTensor,
            SlidingWindowSpec,
            UniformTypeKVCacheSpecs,
        )

        specs = {}
        for name in self.layers:
            common = dict(
                block_size=self.block_size,
                num_kv_heads=self.heads_of(name),
                head_size=HEAD_SIZE,
                dtype=dtype,
            )
            # The engine config this view is built against is a non-MLA model,
            # so `use_mla` is False and registration would split K/V against an
            # MLA tensor. The other two views model MLA; this one cannot.
            assert self.spec != "mla", "no MLA worker can be built over this config"
            if self.spec == "swa":
                assert self.sliding_window is not None, "swa needs a window"
                specs[name] = SlidingWindowSpec(
                    **common, sliding_window=self.sliding_window
                )
            else:
                specs[name] = FullAttentionSpec(**common)

        # One group over every layer, which is what the engine hands the worker.
        # Layers with differing page sizes reach it wrapped, exactly as
        # _get_kv_cache_groups_uniform_type produces -- the worker unwraps per
        # layer, and that asymmetry is a thing the tests must be able to build.
        members = list(specs.values())
        if len({s.page_size_bytes for s in members}) > 1:
            group_spec: Any = UniformTypeKVCacheSpecs(
                block_size=self.block_size, kv_cache_specs=specs
            )
        else:
            group_spec = members[0]

        return KVCacheConfig(
            num_blocks=self.num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=specs[name].page_size_bytes * self.num_blocks,
                    shared_by=[name],
                )
                for name in self.layers
            ],
            kv_cache_groups=[
                KVCacheGroupSpec(
                    layer_names=list(self.layers), kv_cache_spec=group_spec
                )
            ],
        )

    # --- view 2: the tensors it registers ------------------------------------
    def kv_caches(self, *, dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
        """Real tensors, with only ``get_device`` overridden.

        A real RBLN device tensor would force the ``use_device`` mark, which
        spawns the test and blinds coverage; the D2D path asserts
        ``cache.get_device() >= 0`` and nothing else about the device. So a
        subclass answers that one probe and leaves ``data_ptr``, ``numel``,
        ``element_size`` and ``shape`` to torch -- the region arithmetic under
        test then runs on real addresses and real byte counts.
        """

        class _OnDevice(torch.Tensor):
            @staticmethod
            def get_device() -> int:  # type: ignore[override]
                return 0

        out = {}
        for name in self.layers:
            heads = self.heads_of(name)
            shape: tuple[int, ...]
            if self.spec == "mla":
                # No K/V split: the whole entry is one region, block axis first.
                shape = (self.num_blocks, 1, self.block_size, HEAD_SIZE * heads)
            else:
                shape = (2, self.num_blocks, heads, 1, self.block_size, HEAD_SIZE)
            out[name] = torch.zeros(shape, dtype=dtype).as_subclass(_OnDevice)
        return out

    # --- view 3: what the adapter reports back -------------------------------
    def xfer_tables(self, kv_caches: dict[str, Any]) -> Any:
        """``nixl_rbln.register_kv_regions``' result, from the real addresses.

        Bases come from the tensors' own ``data_ptr`` so a descriptor decoded
        back to (region, block, head offset) lands on the region it names.
        """
        regions_per_layer = 1 if self.spec == "mla" else 2
        base_addrs: list[int] = []
        block_lens: list[int] = []
        slice_ids: list[int] = []
        for name in self.layers:
            t = kv_caches[name]
            entry_bytes = t.numel() * t.element_size()
            half = entry_bytes // regions_per_layer
            for r in range(regions_per_layer):
                # One area per chiplet, each holding 1/areas of the region.
                area_bytes = half // self.areas
                for a in range(self.areas):
                    base_addrs.append(t.data_ptr() + r * half + a * area_bytes)
                    block_lens.append(area_bytes // self.num_blocks)
                    slice_ids.append(a)
        return _XferTables(
            base_addrs=base_addrs,
            block_lens=block_lens,
            slice_ids=slice_ids,
            n_shards=self.areas,
            slices=self.slices,
            reg_handle=object(),
        )


@dataclass
class _XferTables:
    base_addrs: list[int]
    block_lens: list[int]
    slice_ids: list[int]
    n_shards: int
    slices: int
    reg_handle: Any


class FakeNixlAgent:
    """The out-of-process NIXL agent.

    Deliberately wider than the suite reaches: upstream calls the agent too, so
    a name dropped for being unreached comes back as an ``AttributeError`` the
    first time a test drives that path. Handles are distinct values so a test
    can tell one dlist from another.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._next = 0

    def _handle(self, *args: Any, **kwargs: Any) -> int:
        self._next += 1
        return self._next

    def get_reg_descs(self, *a: Any, **k: Any) -> Any:
        return self._handle(*a, **k)

    def register_memory(self, *a: Any, **k: Any) -> Any:
        return self._handle(*a, **k)

    def deregister_memory(self, *a: Any, **k: Any) -> Any:
        # Upstream's shutdown reaches this from __del__; without it the workers
        # that registered memory raise an ignored AttributeError there.
        return self._handle(*a, **k)

    def get_xfer_descs(self, *a: Any, **k: Any) -> Any:
        return self._handle(*a, **k)

    def prep_xfer_dlist(self, *a: Any, **k: Any) -> Any:
        return self._handle(*a, **k)

    def release_dlist_handle(self, *a: Any, **k: Any) -> Any:
        return self._handle(*a, **k)

    def release_xfer_handle(self, *a: Any, **k: Any) -> Any:
        return self._handle(*a, **k)

    def make_prepped_xfer(self, *a: Any, **k: Any) -> Any:
        return self._handle(*a, **k)

    def transfer(self, *a: Any, **k: Any) -> Any:
        return self._handle(*a, **k)

    def send_notif(self, *a: Any, **k: Any) -> Any:
        return self._handle(*a, **k)

    def add_remote_agent(self, *a: Any, **k: Any) -> Any:
        self._handle(*a, **k)
        return "remote-agent"

    def remove_remote_agent(self, *a: Any, **k: Any) -> Any:
        return self._handle(*a, **k)

    def get_agent_metadata(self) -> bytes:
        return b"agent-meta"

    def get_new_notifs(self) -> dict[str, list[bytes]]:
        return {}

    def check_xfer_state(self, *a: Any, **k: Any) -> str:
        return "DONE"


def fake_nixl_rbln(geometry: KvGeometry, kv_caches: dict[str, Any]) -> Any:
    """A stand-in ``nixl_rbln`` module over ``geometry``'s real addresses."""
    import types

    module: Any = types.ModuleType("nixl_rbln")
    module.ensure_rbln_backend = lambda *a, **k: None
    # The connector's own answer, kept so a test can assert it. The tables come
    # from the geometry, so nothing else here would ever read `regions` -- and
    # the byte offset in it is the only thing telling a layer's K from its V.
    module.regions_seen = []

    def _register_kv_regions(wrapper: Any, regions: Any, *a: Any, **k: Any) -> Any:
        module.regions_seen.append(list(regions))
        return geometry.xfer_tables(kv_caches)

    module.register_kv_regions = _register_kv_regions
    return module


# --------------------------------------------------------------------------- #
# Descriptor decoding
# --------------------------------------------------------------------------- #
# A descriptor list is asserted by decoding each (addr, len) back to what it
# means and checking the local-to-peer pairing, not by matching the order the
# loops happen to emit. The decoder takes the bases and block lengths it is
# decoding against -- for the peer side, the ones the peer advertised -- so it
# shares no constant with the code under test.


def decode(
    descs: list[tuple[int, int, int]],
    *,
    bases: list[int],
    block_lens: list[int],
    num_blocks: int,
) -> list[tuple[int, int, int]]:
    """(region, block, piece) for each descriptor.

    ``piece`` is the descriptor's index inside the region's block, in units of
    its own length: a 512-byte region read in 256-byte descriptors has pieces 0
    and 1, which is the head band the descriptor names. Nothing else is needed
    to say what an address means, so this shares no constant with the code
    under test -- it takes the bases and block lengths it decodes against, and
    for a peer those are the ones the peer advertised.

    Requires the regions not to overlap. They cannot: every base is a distinct
    span of a real tensor, or a synthetic address a builder spaced apart.
    """
    out = []
    for addr, length, _dev in descs:
        hits = [
            r
            for r, base in enumerate(bases)
            if base <= addr < base + num_blocks * block_lens[r]
        ]
        assert len(hits) == 1, f"address {addr:#x} lands in {len(hits)} regions"
        region = hits[0]
        block, byte_off = divmod(addr - bases[region], block_lens[region])
        assert byte_off % length == 0, (
            f"offset {byte_off} in region {region} is not a whole number of "
            f"{length}-byte pieces"
        )
        out.append((region, block, byte_off // length))
    return out


def peer_meta(
    geometry: KvGeometry,
    *,
    engine_id: str = "peer-engine",
    compat_hash: str = "compat",
    base: int = 0x7000_0000,
    tp_size: int = 1,
    pp_rank: int = 0,
    pp_size: int = 1,
    layers: tuple[str, ...] | None = None,
) -> Any:
    """A peer's handshake payload, as the side channel would deliver it.

    ``base`` is a synthetic address space: the peer's tensors are in another
    process, so its bases are numbers to us. They only have to be far enough
    apart that decode() can attribute an address to one region, which the
    stride below guarantees.
    """
    import msgspec
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl.metadata import (
        NixlHandshakePayload,
    )

    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
        RblnNixlAgentMetadata,
    )

    names = list(layers if layers is not None else geometry.layers)
    regions_per_layer = 1 if geometry.spec == "mla" else 2
    n_regions = len(names) * regions_per_layer * geometry.areas
    # One region's bytes per layer -- a draft layer owns fewer heads, so the
    # lengths differ across regions -- then a stride wider than the widest
    # region so no two regions can claim the same address.
    block_lens = [
        (geometry.block_size * HEAD_SIZE * geometry.heads_of(name) * 2)
        // geometry.areas
        for name in names
        for _ in range(regions_per_layer * geometry.areas)
    ]
    stride = max(block_lens) * geometry.num_blocks * 2
    base_addrs = [base + i * stride for i in range(n_regions)]

    agent = RblnNixlAgentMetadata(
        engine_id=engine_id,
        agent_metadata=b"peer-agent",
        kv_caches_base_addr=base_addrs,
        device_id=0,
        num_blocks=geometry.num_blocks,
        block_lens=block_lens,
        attn_backend_name="RBLN_FLASH_ATTN",
        kv_cache_layout="HND",
        block_size=geometry.block_size,
        ssm_sizes=(0, 0),
        physical_blocks_per_logical_kv_block=1,
        pp_rank=pp_rank,
        pp_size=pp_size,
        registered_layer_names=names,
        kv_areas=geometry.areas,
        kv_slices=geometry.slices,
    )
    return NixlHandshakePayload(
        compatibility_hash=compat_hash,
        agent_metadata_bytes=msgspec.msgpack.Encoder().encode(agent),
    )


def _package_modules() -> list[Any]:
    """Every loaded module of the rbln_nixl package."""
    import sys

    package = "vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl"
    return [m for name, m in sys.modules.items() if name.startswith(package)]


def _bindings_of(name: str) -> list[Any]:
    """The modules of the package that bind ``name``.

    ``from x import Y`` binds Y in the importing module, so patching it on one
    module stops reaching the code the moment that code moves to a sibling.
    Every module that binds the name is patched instead, which is what lets a
    module split leave the tests alone. A name no module binds is a patch that
    would silently do nothing, so it raises.
    """
    modules = [m for m in _package_modules() if hasattr(m, name)]
    assert modules, (
        f"no module of the rbln_nixl package binds {name!r}; a patch on a name "
        "nothing binds silently does nothing"
    )
    return modules


def patch_in_package(**names: Any) -> Any:
    """``patch.object(module, name, value)`` across the package.

    Returns a context manager; enter it around the call under test.
    """
    import contextlib
    from unittest.mock import patch

    # Resolve every name before entering any patch: `_bindings_of` raises for a
    # name nothing binds, and a caller passes several at once, so entering as we
    # go would leave the earlier ones patched for the rest of the session.
    targets = [
        (module, name, value)
        for name, value in names.items()
        for module in _bindings_of(name)
    ]
    stack = contextlib.ExitStack()
    for module, name, value in targets:
        stack.enter_context(patch.object(module, name, value))
    return stack


def setattr_in_package(monkeypatch: Any, **names: Any) -> None:
    """``monkeypatch.setattr(module, name, value)`` across the package.

    The same substitution as patch_in_package for a fixture that already has
    monkeypatch and wants it undone at the end of the test rather than at the
    end of a `with`.
    """
    for name, value in names.items():
        for module in _bindings_of(name):
            monkeypatch.setattr(module, name, value)


@contextlib.contextmanager
def patched_in_package(name: str, value: Any = None) -> Any:
    """patch_in_package for a single name, yielding the substitute.

    Reads back the way ``patch.object(...) as m`` does. ``value`` defaults to a
    fresh MagicMock.
    """
    from unittest.mock import MagicMock

    substitute = MagicMock() if value is None else value
    with patch_in_package(**{name: substitute}):
        yield substitute


# ---------------------------------------------------------------------------- #
# A worker without a KV geometry
# ---------------------------------------------------------------------------- #
# `conftest.py::make_worker` builds the real thing over real tensors. These
# two stub upstream's __init__ down to what the RBLN overrides read, for cases
# that pin a spec or a missing `nixl_rbln` and have no geometry to go with it.


def sliding_window_spec(*, block_size, sliding_window):
    from unittest.mock import MagicMock

    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    spec = MagicMock(spec=SlidingWindowSpec)
    spec.block_size = block_size
    spec.sliding_window = sliding_window
    return spec


def build_worker(
    monkeypatch,
    *,
    kv_buffer_device="cpu",
    num_blocks=128,
    block_size=64,
    specs=None,
    nixl_available=True,
    swa_view_opt=False,
    use_mla=False,
    chunk_mode=False,
    chunk_bytes=0,
):
    """The worker via its real __init__, with upstream's stubbed to set only what
    the RBLN overrides read and `nixl_rbln` faked present or absent."""
    import sys
    import types
    from unittest.mock import MagicMock

    from vllm.config import CacheConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
        NixlBaseConnectorWorker,
    )

    import vllm_rbln.envs as envs
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (  # noqa: E501
        RblnNixlPullConnectorWorker,
    )

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
        # Upstream's __init__ derives this from the group specs; the connector
        # refuses a Mamba/SSM group at registration, so the stub has to carry it.
        self._has_mamba = False
        # add_remote_agent asks for tp_ratio before deciding whether upstream's
        # positional pairing applies; 1 keeps these cases homogeneous.
        self.transfer_topo = MagicMock()
        self.transfer_topo.tp_ratio.return_value = 1

    monkeypatch.setattr(NixlBaseConnectorWorker, "__init__", fake_super_init)

    vllm_config = mock_vllm_config(chunk_mode=chunk_mode, chunk_bytes=chunk_bytes)
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
