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
context lookup, ``get_device``, the TP-rank accessors and the window-mode env
flag. ``build_worker`` below goes further and replaces upstream's ``__init__``.

``KvGeometry`` is why that is affordable. One object produces all three views of
a geometry -- the KVCacheConfig the worker is built with, the tensors it
registers, and the layout rebel reports for them -- so a test varies one value
and the three cannot drift apart. An MLA geometry has only the last two: the
engine config here is non-MLA, so no worker can be built over an MLA cache.
Import lazily from a fixture: this module pulls in vllm.config.
"""

from __future__ import annotations

import contextlib
import functools
import weakref
from collections.abc import Sequence
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (  # noqa: E501
        TransferShape,
    )

# A hub id would revalidate config.json over the network on every build.
MODEL = "meta-llama/Llama-3.2-1B-Instruct"

# RBLN needs an explicit block_size, and VllmConfig validates it against the
# platform's prefix block size -- tests/vllm/v1/worker/utils.py pins the same
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
    stripe_width: int | None = None,
) -> Any:
    """A real VllmConfig with a kv_transfer_config, through EngineArgs.

    Cached on the keyword tuple: the first build pays for
    ``create_engine_config`` and backend resolution, every later one with the
    same shape is free. The connector never mutates it.
    """
    from vllm.config import KVTransferConfig
    from vllm.v1.attention.backends.utils import resolve_kv_cache_layout

    from tests.vllm.vllm_config import make_vllm_config
    from vllm_rbln.v1.attention.backends.flash_attention import (
        RBLNFlashAttentionBackend,
    )

    extra: dict[str, Any] = {}
    if speculative_model is not None:
        extra["speculative_config"] = {
            "model": speculative_model,
            "num_speculative_tokens": 2,
        }
    config = make_vllm_config(
        model=MODEL,
        block_size=block_size,
        kv_transfer_config=KVTransferConfig(
            kv_connector="RblnNixlConnector",
            kv_role=kv_role,
            kv_buffer_device=kv_buffer_device,
            kv_connector_extra_config=(
                {} if stripe_width is None else {"stripe_width": stripe_width}
            ),
        ),
        **extra,
    )
    # A NIXL worker reads the resolved layout in __init__, and the engine core
    # is what resolves it -- which these tests skip by building the worker
    # directly. Run the real resolver over the backend's own preference list
    # rather than naming a layout here, so the tests see what production sees.
    resolve_kv_cache_layout(
        config,
        [
            [
                layout.name
                for layout in RBLNFlashAttentionBackend.supported_kv_cache_layouts()
            ]
        ],
    )
    return config


def set_connector_options(monkeypatch, vllm_config, **options) -> None:
    """Put this connector's knobs on a config, for the test's duration.

    They live in ``kv_connector_extra_config`` rather than the environment, so
    a test sets them where production reads them. ``setitem`` rather than a
    fresh config because `vllm_config_for` caches on its keyword tuple and the
    object is shared -- monkeypatch takes the entries back out.
    """
    for key, value in options.items():
        monkeypatch.setitem(
            vllm_config.kv_transfer_config.kv_connector_extra_config, key, value
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
    # Attributes outside the knobs that a mock answers wrongly: comparing a mock
    # with a number raises. The values are what `engine_config` builds by
    # default, so a mocked config and a real one size a chunk and a prefill the
    # same way.
    config.scheduler_config.max_num_batched_tokens = 128
    config.num_prefill_lookahead_tokens = 0
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

    from tests.vllm.vllm_config import local_model_path

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


#: What a layer's sliding-window companion is called; `heads_of` reads through it.
_SWA_SUFFIX = ".swa"


#: The allocations `_on_device` made, by address. A device-side query takes an
#: address rather than a tensor, so this is what answers one.
_DEVICE_ALLOCS: weakref.WeakValueDictionary[int, Any] = weakref.WeakValueDictionary()


def _on_device(shape: tuple[int, ...], dtype: torch.dtype) -> Any:
    """A real tensor with only ``get_device`` overridden.

    A real RBLN device tensor would force the ``use_device`` mark, which spawns
    the test and blinds coverage; the D2D path asserts ``cache.get_device() >= 0``
    and nothing else about the device. So a subclass answers that one probe and
    leaves ``data_ptr``, ``numel``, ``element_size`` and ``shape`` to torch --
    the region arithmetic under test then runs on real addresses and real byte
    counts.
    """

    class _OnDevice(torch.Tensor):
        @staticmethod
        def get_device() -> int:  # type: ignore[override]
            return 0

    tensor = torch.zeros(shape, dtype=dtype).as_subclass(_OnDevice)
    _DEVICE_ALLOCS[tensor.data_ptr()] = tensor
    return tensor


@dataclass(frozen=True)
class KvGeometry:
    """What the device looks like, in the terms a geometry test varies.

    ``per_layer_heads`` gives one layer a head count of its own, which is the
    speculative-draft case: the layers of one group no longer share a page
    size. ``areas`` is the chiplet count one logical region expands into and
    ``slices`` how many distinct pieces those areas hold; ``areas > slices``
    replicates, which this models only for a whole entry (``slices == 1``),
    since that is the one case no convention decides. ``cut`` names the axis
    for the peer alone -- the local side derives its own from the shapes.
    """

    layers: tuple[str, ...] = ("l0", "l1")
    heads: int = 8
    block_size: int = BLOCK_SIZE
    num_blocks: int = 4
    spec: str = "full"
    sliding_window: int | None = None
    areas: int = 1
    slices: int = 1
    #: Which axis the PEER's regions are cut on. The local side derives its own
    #: from the shapes in `kv_caches`, so this only shapes `peer_meta`.
    cut: str = "head"
    per_layer_heads: dict[str, int] = field(default_factory=dict)
    draft_layers: tuple[str, ...] = ()
    #: Tokens a block holds in the view the sliding-window kernel reads, which
    #: the runner picks independently of `block_size`. None means it reads whole
    #: blocks. Registered tensors are unaffected -- a pool hands over its
    #: full-attention layer -- so this shapes the window layer beside it, and
    #: the frame the device serves that layer.
    swa_kernel_block: int | None = None

    def __post_init__(self) -> None:
        assert self.cut in ("head", "token")
        assert self.areas == self.slices or self.slices == 1, (
            f"KvGeometry models a replica only of a whole entry "
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
        return self.per_layer_heads.get(layer.removesuffix(_SWA_SUFFIX), self.heads)

    @property
    def swa_layers(self) -> tuple[str, ...]:
        """The sliding-window layer each layer's storage also serves.

        A hybrid model (gpt-oss) gives its two attention types a cache group
        each over one buffer, so a pool hands over both: the full-attention
        layer, which is what gets registered, and the window layer, a view of
        the same storage in the shape its kernel reads (`swa_kernel_block`).
        """
        return (
            tuple(f"{name}{_SWA_SUFFIX}" for name in self.layers)
            if (self.spec == "swa")
            else ()
        )

    def swa_view_shape(self, layer: str) -> tuple[int, ...]:
        """The shape a window layer reads its storage in."""
        from vllm_rbln.v1.attention.backends.flash_attention import (
            RBLNFlashAttentionBackend,
        )

        kernel_block = self.swa_kernel_block or self.block_size
        return RBLNFlashAttentionBackend.get_kv_cache_shape(
            num_blocks=self.num_blocks * (self.block_size // kernel_block),
            block_size=kernel_block,
            num_kv_heads=self.heads_of(layer),
            head_size=HEAD_SIZE,
        )

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
        for name in (*self.layers, *self.swa_layers):
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
            if name in self.swa_layers:
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
        members = [specs[name] for name in self.layers]
        if len({s.page_size_bytes for s in members}) > 1:
            group_spec: Any = UniformTypeKVCacheSpecs(
                block_size=self.block_size, kv_cache_specs=specs
            )
        else:
            group_spec = members[0]

        # One group: its layers sit a page apart inside each block, and the pool
        # spans the whole group. A window layer takes the extent of the layer it
        # follows rather than one of its own -- the two read one buffer, which is
        # what `canonical_kv_layers` collapses to the full-attention one.
        pool = (
            sum(specs[name].page_size_bytes for name in self.layers) * self.num_blocks
        )
        extents = {
            name: (
                sum(
                    specs[earlier].page_size_bytes * self.num_blocks
                    for earlier in list(self.layers)[:index]
                ),
                specs[name].page_size_bytes * self.num_blocks,
            )
            for index, name in enumerate(self.layers)
        }
        groups = [
            KVCacheGroupSpec(layer_names=list(self.layers), kv_cache_spec=group_spec)
        ]
        if self.swa_layers:
            groups.append(
                KVCacheGroupSpec(
                    layer_names=list(self.swa_layers),
                    kv_cache_spec=specs[self.swa_layers[0]],
                )
            )
        return KVCacheConfig(
            num_blocks=self.num_blocks,
            kv_cache_tensors=[
                KVCacheTensor(
                    size=pool,
                    layers=[name],
                    layer_stride=extents[name.removesuffix(_SWA_SUFFIX)][1],
                    block_stride=specs[name].page_size_bytes,
                    offset=extents[name.removesuffix(_SWA_SUFFIX)][0],
                )
                for name in (*self.layers, *self.swa_layers)
            ],
            kv_cache_groups=groups,
        )

    # --- view 2: the tensors it registers ------------------------------------
    @property
    def regions_per_layer(self) -> int:
        """How many regions one layer registers, from the layout the backend
        declares. MLA is key-only and a packed block holds K and V, so both are
        one region; the split layout keeps them apart and gives two.
        """
        from vllm_rbln.v1.attention.backends.flash_attention import (
            RBLNFlashAttentionBackend,
        )

        if self.spec == "mla":
            return 1
        # A count no other axis can equal, so `index` names the block axis.
        probe = 1234567
        shape = RBLNFlashAttentionBackend.get_kv_cache_shape(
            num_blocks=probe,
            block_size=self.block_size,
            num_kv_heads=self.heads,
            head_size=HEAD_SIZE,
        )
        return 1 if shape.index(probe) == 0 else 2

    def kv_caches(self, *, dtype: torch.dtype = torch.bfloat16) -> dict[str, Any]:
        """What a pool hands over for each layer, in the view it allocates."""

        from vllm_rbln.v1.attention.backends.flash_attention import (
            RBLNFlashAttentionBackend,
        )

        out = {}
        for name in self.layers:
            heads = self.heads_of(name)
            shape: tuple[int, ...]
            if self.spec == "mla":
                # No K/V split: the whole entry is one region, block axis first,
                # and no head axis -- the RBLN MLA backend's shape.
                shape = (self.num_blocks, self.block_size, HEAD_SIZE * heads)
            else:
                # Ask the backend rather than spelling the axes out: which of
                # them carries the blocks is what the topology reads to decide
                # whether K and V share one, so a literal here could describe a
                # layout the product never builds.
                shape = RBLNFlashAttentionBackend.get_kv_cache_shape(
                    num_blocks=self.num_blocks,
                    block_size=self.block_size,
                    num_kv_heads=heads,
                    head_size=HEAD_SIZE,
                )
            out[name] = _on_device(shape, dtype)
        # The window layers read the same storage in the shape their kernel
        # addresses it in, so a view of it is what the pool hands over.
        for name in self.swa_layers:
            base = out[name.removesuffix(_SWA_SUFFIX)]
            out[name] = base.view(self.swa_view_shape(name))
        return out

    # --- view 3: what the device answers about an entry's storage ------------
    def device_buffers(self, vaddr: int) -> tuple[Any, ...]:
        """``nixl_rbln.entry_areas``: one handle per chiplet area.

        One handle per chiplet area, with nothing about how they were cut --
        which is all a released rebel answers, and so all the derivation that
        lays the regions over them has to go on.
        """
        entry = _DEVICE_ALLOCS[vaddr]
        area_bytes = entry.numel() * entry.element_size() // self.slices
        return tuple(
            SimpleNamespace(
                node_id=0,
                chiplet_id=area,
                device_addr=vaddr + area * area_bytes,
                size=area_bytes,
            )
            for area in range(self.areas)
        )


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


def area_slices(areas: int, slices: int, regions: int) -> list[list[int]]:
    """`kv_area_slices` for `regions` logical regions of `areas` areas over
    `slices` slices, the replicas of a slice adjacent: what a producer of one
    such geometry publishes."""
    replicas = areas // slices
    return [[a // replicas for a in range(areas)] for _ in range(regions)]


def fill_region_geometry(
    w: Any, *, areas: int, axes: Sequence[Any], slices: int | Sequence[int] = 1
) -> Any:
    """The chiplet geometry registration lays out, for a worker a test built by
    hand: one logical region per entry of `axes`, each over `areas` areas and
    cut into `slices` (one count for every region, or one per region), the
    replicas of a slice adjacent as in `area_slices`. Nothing that reads this
    reads an address, so the places carry none.
    """
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
        AreaPlace,
        RegionGeometry,
    )

    counts = [slices] * len(axes) if isinstance(slices, int) else list(slices)
    assert len(counts) == len(axes), (counts, axes)
    geometries = []
    for axis, count in zip(axes, counts):
        assert areas % count == 0, f"{areas} areas over {count} slices"
        (ids,) = area_slices(areas, count, 1)
        places = tuple(AreaPlace(0, 0, slice_id) for slice_id in ids)
        geometries.append(RegionGeometry(axis, count, places))
    w._kv_areas = areas
    w._logical_region_geometry = geometries
    w._logical_region_latent = [False] * len(axes)
    return w


def fake_nixl_rbln(entry_areas: Any = None) -> Any:
    """A stand-in ``nixl_rbln`` module: the backend lifecycle and the chiplet
    areas of an entry, which is all the connector asks of the adapter."""
    import types

    module: Any = types.ModuleType("nixl_rbln")
    # Backend-lifecycle overrides ride this call on the D2D path, and the
    # adapter treats a missing one as "keep the plugin default", so what is
    # absent is as much the contract as what is present.
    module.backend_kwargs_seen = []

    def _ensure_rbln_backend(wrapper: Any, device_id: int, **k: Any) -> None:
        module.backend_kwargs_seen.append(
            {n: v for n, v in k.items() if n != "rbln_ctx_ptr"}
        )

    module.ensure_rbln_backend = _ensure_rbln_backend
    module.entry_areas = entry_areas
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
        ChipletSplitAxis,
        RblnNixlAgentMetadata,
    )

    names = list(layers if layers is not None else geometry.layers)
    regions_per_layer = geometry.regions_per_layer
    n_regions = len(names) * regions_per_layer * geometry.areas
    # How many of K and V sit in one region: a packed block holds both, so its
    # region is twice a split one. MLA is key-only and has no second half.
    kv_in_a_region = 1 if geometry.spec == "mla" else 2 // regions_per_layer
    # One region's bytes per layer -- a draft layer owns fewer heads, so the
    # lengths differ across regions -- then a stride wider than the widest
    # region so no two regions can claim the same address.
    block_lens = [
        (geometry.block_size * HEAD_SIZE * geometry.heads_of(name) * 2 * kv_in_a_region)
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
        block_strides=block_lens,
        attn_backend_name="RBLN_FLASH_ATTN",
        kv_cache_layout="LBHNC",
        block_size=geometry.block_size,
        ssm_sizes=(0, 0),
        physical_blocks_per_logical_kv_block=1,
        pp_rank=pp_rank,
        pp_size=pp_size,
        # Region-major, area-minor and one entry per region, which is the order
        # and the length a producer publishes; `registered_layer_names` folds it
        # back to layers. Per-layer would read the same only at one area.
        region_names=[
            name for name in names for _ in range(regions_per_layer * geometry.areas)
        ],
        region_num_blocks=[geometry.num_blocks] * n_regions,
        region_mem_types=["VRAM"] * n_regions,
        region_group_ids=[0] * n_regions,
        kv_area_slices=area_slices(
            geometry.areas, geometry.slices, n_regions // geometry.areas
        ),
        kv_split_axes=[
            ChipletSplitAxis.TOKEN
            if geometry.slices > 1 and geometry.cut == "token"
            else ChipletSplitAxis.HEAD
        ]
        * (n_regions // geometry.areas),
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


def shape(**over) -> TransferShape:
    """A `TransferShape` with the fields a test means, and `_couple`'s answer
    for the ones they carry.

    Hand-built workers skip `__init__`, so they have to be handed the shape the
    reduction would have produced. Naming only the fields under test keeps a
    case from asserting a value it never meant to pin; everything the named
    ones do not reach is off.
    """
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
        TransferShape,
    )

    fields = dict(
        chunk_mode=False,
        wants_window=False,
        wants_stream=False,
        streams_prefix=False,
        window_ratio=None,
        counted_group=0,
        has_swa=False,
        groups=1,
        use_host_buffer=False,
        writes_into_peer=False,
    )
    _couple(fields, over)
    return TransferShape(**fields)


def _couple(fields: dict, over: dict) -> None:
    """Carry a field's companions with it, so a case cannot ask for a shape
    the reduction has no way to return. A ratio without a window is one, and so
    is a sliding window that is the engine's only group -- the one place a
    chunk range is sized in would then be the window's own. A prefix does not
    stream unless the knob asked for it, this side writes into the peer and
    the buffer is the device's: the reduction reads all three. A counted group
    is an index into the groups, so it bounds how many there are, and the side
    that writes into the peer is the worker's class, not a case's choice.
    """
    if over.get("window_ratio") is not None:
        fields["has_swa"] = True
        fields["groups"] = 2
        fields["wants_window"] = True
    if over.get("has_swa") and "groups" not in over:
        fields["groups"] = 2
    counted = over.get("counted_group")
    if counted:
        fields["groups"] = max(fields["groups"], over.get("groups", counted + 1))
    if over.get("streams_prefix"):
        fields["wants_stream"] = True
        fields["writes_into_peer"] = True
        fields["use_host_buffer"] = False
    fields.update(over)


def set_shape(worker, **over) -> None:
    """Change named fields of a worker's shape, leaving the rest as they were.

    A hand-built worker states its shape once; a case that turns one knob on
    says only that, rather than restating the whole shape and pinning fields it
    never meant to. A worker with no shape yet starts from `shape`'s, where
    everything is off.
    """
    import dataclasses

    current = getattr(worker, "_shape", None) or shape(
        writes_into_peer=type(worker)._writes_into_peer
    )
    fields = dataclasses.asdict(current)
    over.setdefault("writes_into_peer", type(worker)._writes_into_peer)
    # Through the same couplings `shape` applies: a change made here reaches
    # the same object, so a route round them is a route to the same impossible
    # shape.
    _couple(fields, over)
    worker._shape = dataclasses.replace(current, **fields)


def uniform_grid(runs: int, chunks: int, *, regions: int, spans: int = 1) -> Any:
    """A `ChunkGrid` whose regions all cut a block alike.

    `spans` areas of `chunks` chunks each, a chunk `runs` byte runs -- a
    context cut where `spans` is the area count, a head cut at one. Region
    positions run area-minor, as every list does.
    """
    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
        ChunkGrid,
    )

    return ChunkGrid(
        spans * chunks,
        tuple((p % spans * chunks, runs, chunks) for p in range(regions)),
    )


def sliding_window_spec(*, block_size, sliding_window):
    from unittest.mock import MagicMock

    from vllm.v1.kv_cache_interface import SlidingWindowSpec

    spec = MagicMock(spec=SlidingWindowSpec)
    spec.block_size = block_size
    spec.sliding_window = sliding_window
    return spec


def window_mode(worker: Any, ratio: int | None, *, runs: int = 1, **over) -> None:
    """Put a worker in window mode the way registration would.

    The shape and the geometry, which registration settles at different times:
    `shape` carries the ratio the knobs asked for, and the two fields after it
    carry what the runner's binding turned out to be. The default is the
    geometry where a granule IS a kernel block and so one byte run.

    Extra keywords go to `shape`, for a case that means more than a window.
    """
    over.setdefault("writes_into_peer", type(worker)._writes_into_peer)
    worker._shape = shape(window_ratio=ratio, **over)
    if ratio is None:
        worker._swa_kernel_blocks = set()
        worker._window_grid_cut = None
        return
    block_size = getattr(worker, "block_size", None)
    worker._swa_kernel_blocks = (
        set()
        if block_size is None
        else {block_size // ratio if runs == 1 else block_size}
    )
    # Derived where the rule can answer, not restated: a helper spelling the
    # answer out keeps agreeing with a rule that has moved. The other geometry
    # is the one `_window_grid` reaches through `_block_runs`, which a
    # hand-built worker has no bands for.
    worker._window_grid_cut = (
        worker._window_grid() if runs == 1 and block_size is not None else (runs, ratio)
    )


def build_worker(
    monkeypatch,
    *,
    kv_buffer_device="cpu",
    num_blocks=128,
    block_size=64,
    specs=None,
    non_transfer_specs=None,
    nixl_available=True,
    swa_window_mode=False,
    use_mla=False,
    pp_size=1,
    dcp_size=1,
    pcp_size=1,
    stripe_width=None,
    layer_names=("layer0",),
    chunk_mode=False,
    chunk_tokens=0,
    push_stream=False,
    cls=None,
    kv_role="kv_both",
    link_down_exit_s=0,
):
    """The worker via its real __init__, with upstream's stubbed to set only what
    the RBLN overrides read and `nixl_rbln` faked present or absent.

    `cls` picks the direction. It is the read side by default, and the write
    side is asked for by name -- the shape puts the direction in, so a knob
    only the writer acts on is inert on the other.
    """
    import sys
    import types
    from collections import defaultdict
    from unittest.mock import MagicMock

    from vllm.config import CacheConfig, SchedulerConfig
    from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
        NixlBaseConnectorWorker,
    )
    from vllm.v1.kv_cache_interface import KVCacheTensor

    from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.pull_worker import (  # noqa: E501
        RblnNixlPullConnectorWorker,
    )

    module = types.ModuleType("nixl_rbln") if nixl_available else None
    monkeypatch.setitem(sys.modules, "nixl_rbln", module)

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
        # Upstream's __init__ fills this from the group specs; tests that need
        # a real spec set their own.
        self._layer_specs = dict.fromkeys(layer_names)
        # add_remote_agent asks for tp_ratio before deciding whether upstream's
        # positional pairing applies; 1 keeps these cases homogeneous.
        self.transfer_topo = MagicMock()
        self.transfer_topo.tp_ratio.return_value = 1
        # State vllm 0.30.0 added to the real __init__ and the inherited entry
        # points now read. The parallel sizes are the single-shard values these
        # tests build against, so no rank ever offsets off them.
        self.dcp_size = dcp_size
        self.pcp_size = pcp_size
        self.dcp_rank = 0
        self.pcp_rank = 0
        self.dst_region_num_blocks = {}
        self.dst_region_group_ids = {}
        self.dst_region_mem_types = {}
        self.dst_uses_region_group_mapping = {}
        self._recving_transfers = defaultdict(list)
        self._engine_ttl = 0.0

    monkeypatch.setattr(NixlBaseConnectorWorker, "__init__", fake_super_init)

    vllm_config = mock_vllm_config(
        chunk_mode=chunk_mode,
        chunk_tokens=chunk_tokens,
        swa_window_mode=swa_window_mode,
        stripe_width=stripe_width,
        push_stream=push_stream,
        link_down_exit_s=link_down_exit_s,
    )
    vllm_config.cache_config = CacheConfig(block_size=block_size)
    # What the worker sets before it builds the connector; `register_kv_caches`
    # takes the count from here.
    vllm_config.cache_config.num_gpu_blocks = num_blocks
    # The shape reads the requested buffer device off the config, which is
    # where an engine states it; the worker attribute is derived from it.
    vllm_config.kv_transfer_config.kv_buffer_device = kv_buffer_device
    # No speculative decoding: the compat hash then folds what it always did.
    vllm_config.speculative_config = None
    # _check_pp_constraints compares pipeline_parallel_size <= 1; a MagicMock
    # would raise TypeError there, so give it a real int.
    vllm_config.parallel_config.pipeline_parallel_size = pp_size
    vllm_config.scheduler_config = SchedulerConfig(
        is_encoder_decoder=False,
        max_model_len=128,
        # A chunk size is floored by this, and the real default is wide enough
        # to collapse every grid a test builds; keep the one `mock_vllm_config`
        # names so replacing the mocked config here does not resize a chunk.
        max_num_batched_tokens=128,
    )
    vllm_config.kv_transfer_config.kv_role = kv_role
    kv_cache_config = MagicMock()
    kv_cache_config.num_blocks = num_blocks
    # The pool lists exactly the layers the groups carry: `canonical_kv_layers`
    # reads it to pick one layer per buffer, so a tensor naming a different set
    # sends registration after layers nobody built. `layer_names` is for the
    # cases that build no spec and still name their layers. A MagicMock answers
    # `len()` with 0, so the two views must differ for a case to tell them apart.
    all_specs = list(specs or []) + list(non_transfer_specs or [])
    kv_cache_config.kv_cache_tensors = [
        KVCacheTensor(
            size=0,
            layers=[f"g{i}.l0" for i in range(len(all_specs))] or list(layer_names),
            layer_stride=1,
            block_stride=1,
        )
    ]
    all_groups = [
        MagicMock(kv_cache_spec=spec, layer_names=[f"g{i}.l0"])
        for i, spec in enumerate(all_specs)
    ]
    kv_cache_config.kv_cache_groups = all_groups
    kv_cache_config.transfer_groups = all_groups[: len(specs or [])]
    kv_cache_config.transfer_group_ids = tuple(range(len(specs or [])))
    worker_cls = cls or RblnNixlPullConnectorWorker
    return worker_cls(vllm_config, "test-engine", kv_cache_config)


def fake_sysfs_net(tmp_path, **operstate: str):
    """A `/sys/class/net` with a veth and the given physical links."""
    (tmp_path / "eth0").mkdir()
    (tmp_path / "eth0" / "operstate").write_text("up\n")
    for name, state in operstate.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "device").touch()
        (tmp_path / name / "operstate").write_text(f"{state}\n")
    return tmp_path
