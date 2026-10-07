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

from collections import defaultdict
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.nixl import (
    NixlBaseConnectorWorker,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.handshake import (
    RblnNixlHandshakeMixin,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
    connector_option,
    transfer_shape,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.registration import (
    RblnNixlRegistrationMixin,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
    RequestTail,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.transfer import (
    RblnNixlTransferMixin,
)
from vllm_rbln.logger import init_logger

if TYPE_CHECKING:
    from vllm.v1.kv_cache_interface import KVCacheConfig

logger = init_logger(__name__)


class RblnNixlWorkerBase(
    RblnNixlHandshakeMixin,
    RblnNixlRegistrationMixin,
    RblnNixlTransferMixin,
    NixlBaseConnectorWorker,
):
    """Everything the transfer direction does not decide: memory registration,
    the handshake, region pairing, descriptor construction, topology guards.
    Mixed with whichever direction class moves the bytes.
    """

    def __init__(
        self, vllm_config: VllmConfig, engine_id: str, kv_cache_config: "KVCacheConfig"
    ) -> None:
        # Settled first because nothing in it is an attribute upstream sets --
        # the knobs and the groups are both arguments. That is what lets the
        # scheduler run the same reduction and reach the same answer.
        self._shape = transfer_shape(
            vllm_config,
            kv_cache_config.transfer_groups,
            writes_into_peer=self._writes_into_peer,
        )

        super().__init__(vllm_config, engine_id, kv_cache_config)

        # The descriptor arithmetic addresses a rank's whole region, which is
        # why a context-parallel peer is refused at the handshake; ours would be
        # a slice for the same reason, and upstream swaps the TP rank and size
        # for the PCP pair once both are sharded. Refusing here rather than
        # advertising a size the topology built below does not carry.
        if self.dcp_size != 1 or self.pcp_size != 1:
            raise RuntimeError(
                "RBLN NIXL does not support a context-parallel engine: this "
                f"worker reports dcp_size={self.dcp_size}, "
                f"pcp_size={self.pcp_size}."
            )

        # nixl-rbln present -> RBLN backend (host-bounce DRAM_SEG / D2D VRAM_SEG);
        # absent -> upstream UCX/DRAM defaults, and D2D (kv_buffer_device="rbln")
        # is rejected below since it needs the RBLN backend.
        try:
            import nixl_rbln  # noqa: F401

            self._use_rbln_nixl_backend = True
        except ImportError:
            self._use_rbln_nixl_backend = False

        if self._use_rbln_nixl_backend:
            self.nixl_backends = ["RBLN"]
            # D2D registers VRAM (device dmabuf); host-bounce keeps DRAM.
            if self.kv_buffer_device == "rbln":
                self.nixl_memory_type = "VRAM"
        elif self.kv_buffer_device == "rbln":
            raise RuntimeError(
                "kv_buffer_device='rbln' (D2D) requires the 'nixl-rbln' "
                "adapter package; install it or set kv_buffer_device='cpu' "
                "to fall back to the upstream NIXL (UCX) host-bounce path."
            )
        else:
            logger.info(
                "RBLN NIXL: nixl-rbln not available — "
                "using upstream NIXL (UCX) on the host-bounce path."
            )

        # `RblnPlatform.device_type = "cpu"` makes upstream skip the host
        # buffer; restore it — NIXL cannot register RBLN device memory.
        self.use_host_buffer = self._shape.use_host_buffer
        if self.use_host_buffer:
            # Each knob needs the descriptor lists of the direct path. Refused
            # here rather than left inert, since an operator who named one is
            # owed the reason it cannot be served. `push_stream` only on the
            # side that would act on it: the other gets the same config and
            # the shape has already made it inert there.
            for knob, asked in (
                ("chunk_mode", self._shape.chunk_mode),
                ("swa_window_mode", self._shape.wants_window),
                (
                    "push_stream",
                    self._shape.wants_stream and self._shape.writes_into_peer,
                ),
            ):
                if asked:
                    raise RuntimeError(
                        f"RBLN NIXL: {knob} needs the descriptor lists of the "
                        "direct path; host staging registers one full-shape "
                        "buffer per layer and gives a narrowed peer a handle "
                        "upstream built, and a second range extends neither."
                    )

        # 0 is a width the adapter takes, so it cannot stand for "nobody named
        # one" -- this knob carries its absence instead.
        self._stripe_width = connector_option(
            vllm_config, "stripe_width", None, takes=int
        )
        self._listen_ip = connector_option(vllm_config, "listen_ip", None, takes=str)

        self._pending_kv_caches: dict[str, torch.Tensor] | None = None

        # --- Chiplet geometry of one KV entry (D2D only) ---
        # Set from nixl_rbln.register_kv_regions. Host-bounce registers logical
        # full-shape buffers and never expands per area, so the defaults below
        # are its permanent (and correct) values.
        self._kv_areas: int = 1
        self._kv_slices: int = 1
        # And which axis they came from -- the two counts alone do not say.
        self._kv_split_axis: KVSplitAxis = KVSplitAxis.HEAD
        # The extent of the `kv` axis inside a region's block
        # (`get_kv_cache_shape`), which is one unless the attention cache packs
        # both. Host staging registers whole logical buffers and stays here.
        self._kv_per_block: int = 1
        # Model-wide counts, not this rank's share. None where the layer has
        # no head band (`_layer_kv_heads`).
        self._logical_region_kv_heads: list[int | None] = []

        # `_kv_slices` above is only the LAST region's, which describes every
        # region until a speculative draft gives them different geometries.
        self._logical_region_slices: list[int] = []

        # --- Pipeline-parallel (PP) P/D state (empty / inert for pp_size == 1) ---
        # Per remote producer shard, the ordered KV-cache layer names it owns,
        # keyed by engine_id -> global_rank (= pp_rank * tp_size + tp_rank,
        # == pp_rank when tensor parallelism is off).
        self._remote_shard_layer_names: defaultdict[str, dict[int, tuple[str, ...]]] = (
            defaultdict(dict)
        )
        # engine_id -> producer pp_size (discovered at handshake).
        self._remote_pp_size: dict[str, int] = {}
        # engine_id -> the producer stages (flat global ranks) whose layers this
        # rank owns; the per-shard transfer path walks exactly these.
        self._overlapping_ranks: defaultdict[str, list[int]] = defaultdict(list)
        self._engines_to_rehandshake: set[str] = set()
        self._link_down_since = None
        self._link_down_exit_s = connector_option(vllm_config, "link_down_exit_s", 0)
        if (
            self._link_down_exit_s > 0
            and vllm_config.kv_transfer_config.kv_role != "kv_producer"
        ):
            raise RuntimeError(
                "link_down_exit_s recycles a KV producer whose links died; a "
                "consumer keeps its running requests and is recycled from outside."
            )
        # Per producer shard, a local xfer dlist scoped to that shard's local
        # region subset, keyed by (engine_id, global_rank, block_size); and the
        # shard's per-region KV-group ids, keyed by (engine_id, global_rank).
        self.src_xfer_handles_by_remote: dict[tuple[str, int, int], int] = {}
        # Which of those entries point at a handle upstream owns, so cleanup
        # drops the entry without releasing what other peers still use.
        self._borrowed_src_handles: set[tuple[str, int, int]] = set()
        self._shard_region_group_ids: dict[tuple[str, int], tuple[int, ...]] = {}
        # How many descriptors each of that shard's regions is cut into
        # (_head_split).
        self._shard_descs_per_block: dict[tuple[str, int], int] = {}
        # Per peer shard, the grid its chunk range was built over, or None
        # where its lists carry no such range. See `_shard_chunk_grid`.
        self._shard_chunk_grids: dict[tuple[str, int], tuple[int, int] | None] = {}
        # This engine's own grid, set once registration knows the geometry.
        # None wherever a chunk is the whole span (see `_shard_chunk_grid`).
        self._chunk_grid: tuple[int, int] | None = None
        # The window range's grid, and the observation it is chosen from. The
        # observation is the runner's; the grid adds this rank's own cut.
        self._window_grid_cut: tuple[int, int] | None = None
        self._swa_kernel_blocks: set[int] = set()
        # Parked for the length of one upstream call (`_tail_viewed_as`); what
        # it carries is `RequestTail`.
        self._request_tail: RequestTail | None = None
        # Ordered local KV-cache layer names (one per layer), captured at
        # register_kv_caches.
        self.local_seen_layer_names: list[str] = []

        # Pin to logical values. Upstream would otherwise multiply by the
        # attention backend's kernel ratio, which doesn't reflect per-spec
        # ratios in hybrid models.
        self.num_blocks = self.kv_cache_config.num_blocks
        self.block_size = self.vllm_config.cache_config.block_size
        self._physical_blocks_per_logical_kv_block = 1
        self._logical_num_blocks = self.num_blocks

        # SWA window mode: a second range at the same NIXL base addrs as the
        # Full range, cut into the granules a window sits in. How those are
        # laid out is the kernel's geometry, which registration observes.
        # Storage and host copies stay Full; which knobs ask for the range,
        # and whether this engine can serve one, the shape has settled.
        self._group_specs: list[Any] = [
            g.kv_cache_spec for g in self.kv_cache_config.transfer_groups
        ]
        if self._shape.wants_window and not self._shape.has_window_range:
            # Doing nothing is right here -- a granule would be the block, so
            # the range would repeat what the whole one names. Saying so is
            # what was missing: the knob is set and nothing follows.
            logger.info(
                "RBLN NIXL: swa_window_mode registered no window range. This "
                "engine has no window to cut by -- no sliding-window group, or "
                "one as wide as its block."
            )
        if (
            self._shape.wants_stream
            and self._shape.writes_into_peer
            and not self._shape.streams_prefix
        ):
            raise RuntimeError(
                "RBLN NIXL: push_stream hands over a prefill's closed "
                "prefix ahead of the request, and this engine's groups are "
                "two with neither sliding -- so no list it can be given says "
                "which of them a batch filled: a per-shard list names one "
                "group, and a sliding window is what puts the whole-engine "
                "lists in its own hands."
            )
        # A backstop: `patches/attention.py` refuses a sliding-window MLA layer
        # while the engine is being built, and this connector is reached on the
        # vllm model path alone, where that patch runs. So no such group. The
        # two desc ranges `register_local_xfer_handler` builds and a key-only
        # latent have not been combined.
        assert not (self._shape.has_window_range and self.use_mla), (
            "RBLN NIXL: SWA window mode is not supported with a "
            "sliding-window MLA cache."
        )
