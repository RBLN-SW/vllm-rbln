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
from vllm.v1.kv_cache_interface import (
    SlidingWindowSpec,
)

import vllm_rbln.envs as envs
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.handshake import (
    RblnNixlHandshakeMixin,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.metadata import (
    KVSplitAxis,
)
from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.registration import (
    RblnNixlRegistrationMixin,
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
        super().__init__(vllm_config, engine_id, kv_cache_config)

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

        self._pending_kv_caches: dict[str, torch.Tensor] | None = None

        # --- Chiplet geometry of one KV entry (D2D only) ---
        # Set from nixl_rbln.register_kv_regions. Host-bounce registers logical
        # full-shape buffers and never expands per area, so the defaults below
        # are its permanent (and correct) values.
        self._kv_areas: int = 1
        self._kv_slices: int = 1
        # And which axis they came from -- the two counts alone do not say.
        self._kv_split_axis: KVSplitAxis = KVSplitAxis.HEAD

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

        # SWA view-opt: publish a second sliding_window-length desc range at the
        # same NIXL base addrs as the Full range, so SWA groups transport only the
        # populated prefix (kernel slot 0 is pinned at the block base). Storage and
        # host copies stay Full; _sw_ratio is None collapses to upstream Full-only.
        # `register_local_xfer_handler` builds that second range and documents it.
        self._group_specs: list[Any] = [
            g.kv_cache_spec for g in self.kv_cache_config.kv_cache_groups
        ]
        # Whether the model has a sliding window at all, which decides the model
        # parallelism guards; `_sw_ratio` is the view-opt's desc layout and only
        # ever set when that flag is on.
        self._has_swa = any(
            isinstance(spec, SlidingWindowSpec) for spec in self._group_specs
        )
        self._sw_ratio: int | None = None
        if self._has_swa and envs.VLLM_RBLN_NIXL_SWA_VIEW_OPT:
            for spec in self._group_specs:
                if not isinstance(spec, SlidingWindowSpec):
                    continue
                assert spec.block_size % spec.sliding_window == 0
                ratio = spec.block_size // spec.sliding_window
                if ratio == 1:
                    continue
                if self._sw_ratio is None:
                    self._sw_ratio = ratio
                else:
                    assert self._sw_ratio == ratio, (
                        "RBLN NIXL connector assumes a single SWA ratio "
                        f"across groups, got {self._sw_ratio} vs {ratio}"
                    )
            if self._sw_ratio is not None:
                # Fail at startup rather than at the first handshake: the
                # two desc ranges `register_local_xfer_handler` builds and a
                # key-only latent have not been combined.
                if self.use_mla:
                    raise RuntimeError(
                        "RBLN NIXL: VLLM_RBLN_NIXL_SWA_VIEW_OPT is not "
                        "supported with a sliding-window MLA cache."
                    )
                logger.info(
                    "VLLM_RBLN_NIXL_SWA_VIEW_OPT=1: trimming SWA-group "
                    "RDMA payload by 1/%d (sliding_window-sized descs "
                    "alongside Full descs at shared base addrs).",
                    self._sw_ratio,
                )
