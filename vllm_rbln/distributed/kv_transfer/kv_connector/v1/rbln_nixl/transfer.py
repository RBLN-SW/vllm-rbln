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

import numpy as np
from vllm.distributed.kv_transfer.kv_connector.utils import (
    BlockIds,
)
from vllm.v1.kv_cache_interface import (
    SlidingWindowSpec,
)

from vllm_rbln.distributed.kv_transfer.kv_connector.v1.rbln_nixl.state import (
    RblnNixlWorkerState,
)


class RblnNixlTransferMixin(RblnNixlWorkerState):
    """Issuing one transfer: which descriptors a request needs, and how a
    completion is named.

    The transfer lifetime, and the only one that runs per request: registration
    produces the region table and the handshake produces the pairing, both once,
    and this reads what they left.
    """

    def _compute_desc_ids(
        self,
        block_ids: BlockIds,
        dst_num_blocks: int,
        block_size_ratio: float | None,
        physical_blocks_per_logical: int,
        region_num_blocks: list[int] | None = None,
        region_group_ids: list[int] | None = None,
        uses_region_group_mapping: bool | None = None,
    ) -> np.ndarray:
        if self._sw_ratio is None:
            # No SWA view opt: upstream's Full/SSM desc layout applies, and the
            # 0.30.0 per-region arguments belong to it.
            return super()._compute_desc_ids(
                block_ids,
                dst_num_blocks,
                block_size_ratio,
                physical_blocks_per_logical,
                region_num_blocks=region_num_blocks,
                region_group_ids=region_group_ids,
                uses_region_group_mapping=uses_region_group_mapping,
            )

        # The SWA desc formula below indexes physical blocks directly; the
        # connector pins one physical block per logical block, so the
        # physical_blocks_per_logical argument does not apply here.
        assert physical_blocks_per_logical == 1, (
            "RBLN NIXL connector assumes physical_blocks_per_logical == 1"
        )

        num_blocks = dst_num_blocks
        if block_size_ratio is not None:
            num_blocks = int(num_blocks * block_size_ratio)

        # Both lists run region-major, block, then K/V, so a block's descriptors
        # are `_kv_per_block` consecutive ids (`register_local_xfer_handler`).
        kv_per_block = self._kv_per_block
        region_ids = np.arange(self.num_regions)[:, None]
        num_full_descs = self.num_regions * num_blocks * kv_per_block
        all_descs: list[np.ndarray] = []
        for g, group in enumerate(block_ids):
            if not group:
                continue
            is_sw = isinstance(self._group_specs[g], SlidingWindowSpec)
            offset = num_full_descs if is_sw else 0
            group_arr = np.asarray(group)[None, :]
            block_ids_2d = (region_ids * num_blocks + group_arr)[..., None]
            all_descs.append(
                (
                    block_ids_2d * kv_per_block + np.arange(kv_per_block) + offset
                ).flatten()
            )
        return np.concatenate(all_descs) if all_descs else np.empty(0, dtype=int)

    def _get_block_descs_ids_for_shard(
        self,
        engine_id: str,
        global_rank: int,
        num_blocks: int,
        block_ids: BlockIds,
    ) -> np.ndarray:
        region_group_ids = self._shard_region_group_ids[(engine_id, global_rank)]
        per_block = self._shard_descs_per_block[(engine_id, global_rank)]
        # Converted once, not once per region: this runs per request, and every
        # region of a layer names the same group.
        group_arrays = [np.asarray(g, dtype=np.int64) for g in block_ids]
        desc_ids: list[np.ndarray] = []
        for region_id, group_id in enumerate(region_group_ids):
            group_arr = group_arrays[group_id]
            if group_arr.size == 0:
                continue
            block_ix = region_id * num_blocks + group_arr
            if per_block == 1:
                desc_ids.append(block_ix)
                continue
            # Both dlists are laid out region-major, then block, then piece, so
            # one block becomes `per_block` consecutive descriptors on each side.
            desc_ids.append(
                (
                    block_ix[:, None] * per_block + np.arange(per_block, dtype=np.int64)
                ).ravel()
            )
        if not desc_ids:
            return np.empty(0, dtype=np.int64)
        return np.concatenate(desc_ids)

    def _xfer_notif_id(
        self,
        engine_id: str,
        remote_request_id: str,
        remote_tp_size: int,
        *,
        count_stages: bool = True,
    ) -> bytes:
        """Notification carrying what the peer on this path waits for.

        NOTE(RBLN): the two paths send different units, because vllm 0.30
        changed one of the two protocols and left the other alone. The read
        path's peer counts notifications and settles the request when the
        total arrives, deriving nothing from the number, so it is handed that
        total: our readers of one peer rank, times our stages, since a finer
        pipeline has each stage report for its own layers. The write path's
        peer still divides by its own TP, so it is handed the quantity that
        division expects, and it must not count stages -- the consumer's own
        accounting multiplies the producer's stage count back in, off the
        `pp_size` its `ReqMeta` carries, so they would be counted twice.
        """
        peers = max(1, self.world_size // remote_tp_size)
        if not count_stages:
            return f"{remote_request_id}:{peers * remote_tp_size}".encode()
        remote_pp = self._remote_pp_size.get(engine_id, 1)
        local_pp = self.vllm_config.parallel_config.pipeline_parallel_size
        return f"{remote_request_id}:{peers * max(1, local_pp // remote_pp)}".encode()
