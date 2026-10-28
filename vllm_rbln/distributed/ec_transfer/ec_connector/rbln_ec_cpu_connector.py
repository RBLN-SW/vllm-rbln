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

"""Upstream's ECCPUConnector on the vllm model path.

The RBLN runner encodes and gathers on the host, so the shared region and the
encoder cache are both host memory: a save or a load is a plain copy, done when
the call returns and reported on the same step.
"""

from typing import TYPE_CHECKING

import torch
from vllm.distributed.ec_transfer.ec_connector.base import (
    ECConnectorBase,
    ECConnectorRole,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.common import (
    ECCPUConnectorMetadata,
    ECCPUWorkerMetadata,
)
from vllm.distributed.ec_transfer.ec_connector.cpu.connector import ECCPUConnector
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import ECCPUWorker

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class RblnECCPUWorker(ECCPUWorker):
    """Copies between the encoder cache and the shared region on the host."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        super().__init__(vllm_config)
        self._completed_saves: list[str] = []
        self._completed_loads: list[int] = []

    def save_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        mm_hash: str,
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        if not self._is_save_rank:
            return
        block_ids = connector_metadata.saves.get(mm_hash)
        if block_ids is None:
            return
        src = encoder_cache[mm_hash]
        # A load reads each block back as one row of the model dtype.
        rows = src.view(torch.int8)
        assert (
            src.dtype == self._dtype
            and rows.dim() == 2
            and rows.shape[1] == self._region.block_size_bytes
            and rows.shape[0] <= len(block_ids)
        ), (
            f"EC: the encoder output of mm_hash={mm_hash} ({src.dtype}, "
            f"shape={list(src.shape)}) does not fit {len(block_ids)} rows of "
            f"{self._region.block_size_bytes} bytes in {self._dtype}."
        )
        self._region.blocks[block_ids[: rows.shape[0]]] = rows
        self._completed_saves.append(mm_hash)

    def start_load_caches(
        self,
        encoder_cache: dict[str, torch.Tensor],
        connector_metadata: ECCPUConnectorMetadata,
    ) -> None:
        for mm_hash, (transfer_id, block_ids) in connector_metadata.loads.items():
            encoder_cache[mm_hash] = self._region.blocks[block_ids].view(self._dtype)
            self._completed_loads.append(transfer_id)

    def build_connector_worker_meta(self) -> ECCPUWorkerMetadata | None:
        if not self._completed_saves and not self._completed_loads:
            return None
        meta = ECCPUWorkerMetadata(
            completed_saves=self._completed_saves,
            completed_loads=self._completed_loads,
        )
        self._completed_saves, self._completed_loads = [], []
        return meta


class RblnECCPUConnector(ECCPUConnector):
    """ECCPUConnector whose worker copies on the host."""

    def __init__(self, vllm_config: "VllmConfig", role: ECConnectorRole) -> None:
        # NOTE(RBLN): skip ECCPUConnector.__init__, which accepts only the V2
        # model runner; the RBLN runner returns the EC output on every step,
        # which is what that check asks for.
        ECConnectorBase.__init__(self, vllm_config=vllm_config, role=role)
        self.connector_worker = None
        self.connector_scheduler = None
        if role == ECConnectorRole.WORKER:
            self.connector_worker = self._make_worker(vllm_config)
        elif role == ECConnectorRole.SCHEDULER:
            self.connector_scheduler = self._make_scheduler(vllm_config)
        else:
            raise ValueError(f"Unknown ECConnectorRole: {role}")

    def _make_worker(self, vllm_config: "VllmConfig") -> RblnECCPUWorker:
        return RblnECCPUWorker(vllm_config)
