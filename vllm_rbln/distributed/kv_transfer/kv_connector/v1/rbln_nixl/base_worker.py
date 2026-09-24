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
from vllm.distributed.kv_transfer.kv_connector.v1.nixl.utils import (
    get_representative_spec_type,
)
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    SlidingWindowSpec,
    UniformTypeKVCacheSpecs,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.handshake import (
    RblnNixlHandshakeMixin,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
    connector_option,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.registration import (
    RblnNixlRegistrationMixin,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.transfer import (
    RblnNixlTransferMixin,
)
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

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
        # Upstream's PP>1 refusal reads "not FullAttentionSpec" as "the region
        # count varies per layer", misjudging one merged group of full
        # attention specs: uniform per layer, yet not a FullAttentionSpec.
        # Mamba and sliding window merge the same way and do vary, so suppress
        # for that shape alone. TODO: drop once upstream tests uniformity.
        groups = kv_cache_config.kv_cache_groups
        group_spec = groups[0].kv_cache_spec if len(groups) == 1 else None
        suppress = isinstance(group_spec, UniformTypeKVCacheSpecs) and issubclass(
            get_representative_spec_type(group_spec), FullAttentionSpec
        )
        scheduler_config = vllm_config.scheduler_config
        hma_disabled = scheduler_config.disable_hybrid_kv_cache_manager
        scheduler_config.disable_hybrid_kv_cache_manager = hma_disabled or suppress
        try:
            super().__init__(vllm_config, engine_id, kv_cache_config)
        finally:
            scheduler_config.disable_hybrid_kv_cache_manager = hma_disabled
        if suppress:
            # What upstream would have computed: `any()` over the one group is
            # True, since UniformTypeKVCacheSpecs is not a FullAttentionSpec.
            self._is_hma_required = not hma_disabled

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
        self.use_host_buffer = self.kv_buffer_device == "cpu"
        if self.use_host_buffer:
            # Either knob puts a second descriptor range on the lists. Refused
            # here rather than left inert, since an operator who named one is
            # owed the reason it cannot be served.
            for knob in ("chunk_mode", "swa_window_mode"):
                if connector_option(vllm_config, knob, False):
                    raise RuntimeError(
                        f"RBLN NIXL: {knob} needs the descriptor lists of the "
                        "direct path; host staging registers one full-shape "
                        "buffer per layer and gives a narrowed peer a handle "
                        "upstream built, and a second range extends neither."
                    )

        # 0 is "nobody named one": a stripe is a byte width, so no width is a
        # width the adapter is never handed.
        # 0 is a width the adapter takes, so it cannot stand for "nobody named
        # one" -- this knob carries its absence instead.
        self._stripe_width = connector_option(
            vllm_config, "stripe_width", None, takes=int
        )

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
        # Whether a transfer may carry less than a whole block. What it
        # leaves out is a token range of that block.
        self._chunk_mode: bool = False

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
        # The window range's grid, and the observation it is cut by. Both are
        # the runner's answer, so both wait for registration.
        self._window_grid_cut: tuple[int, int] | None = None
        self._swa_kernel_blocks: set[int] = set()
        # How far the request being transferred fills its last block, parked
        # for the length of one upstream call (`_tail_viewed_as`).
        self._request_tail: tuple[int | None, int | None] | None = None
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
        # Full range, cutting each block into the kernel blocks a window moves
        # in. Storage and host copies stay Full.
        self._group_specs: list[Any] = [
            g.kv_cache_spec for g in self.kv_cache_config.kv_cache_groups
        ]
        # Whether the model has a sliding window at all, which decides the model
        # parallelism guards; `_sw_ratio` is the window mode's desc layout and only
        # ever set when that flag is on.
        self._has_swa = any(
            isinstance(spec, SlidingWindowSpec) for spec in self._group_specs
        )
        self._sw_ratio: int | None = None
        swa_window_mode = connector_option(self.vllm_config, "swa_window_mode", False)
        if self._has_swa and swa_window_mode:
            ratios: set[int] = set()
            for spec in self._group_specs:
                if not isinstance(spec, SlidingWindowSpec):
                    continue
                if spec.block_size % spec.sliding_window != 0:
                    # Upstream's block table refuses this where the kernel
                    # addresses the cache in windows; where it addresses whole
                    # blocks the engine starts, and this is then the only place
                    # that sees a window no granule can tile.
                    raise RuntimeError(
                        "RBLN NIXL: a window range cuts a block into windows, "
                        f"so a {spec.sliding_window}-token window has to "
                        f"divide the {spec.block_size}-token block this "
                        "engine's manager leases. Turn swa_window_mode off."
                    )
                ratio = spec.block_size // spec.sliding_window
                ratios.add(ratio)
                if ratio == 1:
                    continue
                # Which granule the range names is read off the request's token
                # count, and that is where the window is only where it slides.
                # This spec's manager leases one block a request and the runner
                # reads its first granule, wherever the count points.
                if isinstance(spec, RBLNSlidingWindowSpec):
                    raise RuntimeError(
                        "RBLN NIXL: a window range needs a window that moves "
                        "through its block, and this engine pins every one to "
                        "the block's first kernel block. Turn swa_window_mode "
                        "off."
                    )
            if len(ratios) > 1:
                # The builder reads a group as windowed from its spec and then
                # cuts it by the one ratio this engine carries, so a group that
                # tiles its block differently would be named in another group's
                # granules -- part of its block, with the descriptor count
                # unchanged.
                raise RuntimeError(
                    "RBLN NIXL: every sliding-window group has to cut its "
                    "block into the same number of kernel blocks, and this "
                    f"engine's groups cut it into {sorted(ratios)} kernel "
                    "block(s)."
                )
            self._sw_ratio = next((r for r in ratios if r != 1), None)
            if self._sw_ratio is None:
                # Doing nothing is right here -- a granule would be the block,
                # so the range would repeat what the whole one names. Saying so
                # is what was missing: the knob is set and nothing follows.
                logger.info(
                    "RBLN NIXL: swa_window_mode registered no window range. "
                    "Every sliding-window group here holds a window as wide as "
                    "its block, so a granule is the block."
                )
            if self._sw_ratio is not None:
                # Fail at startup rather than at the first handshake: the
                # two desc ranges `register_local_xfer_handler` builds and a
                # key-only latent have not been combined.
                if self.use_mla:
                    raise RuntimeError(
                        "RBLN NIXL: SWA window mode is not supported with a "
                        "sliding-window MLA cache."
                    )
                logger.info(
                    "SWA window mode on: %d sliding_window-sized desc(s) per "
                    "block alongside the Full descs at shared base addrs.",
                    self._sw_ratio,
                )
