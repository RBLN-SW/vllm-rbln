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

from collections.abc import Collection
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
from vllm.distributed.ec_transfer.ec_connector.cpu.scheduler import ECCPUScheduler
from vllm.distributed.ec_transfer.ec_connector.cpu.worker import ECCPUWorker

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.multimodal.inputs import MultiModalFeatureSpec
    from vllm.v1.request import Request


class RblnECCPUScheduler(ECCPUScheduler):
    """Admits reusable runner outputs without requiring a remote transfer."""

    def __init__(self, vllm_config: "VllmConfig") -> None:
        self._local_cache_hashes: Collection[str] = ()
        super().__init__(vllm_config)

    def ensure_cache_available(
        self,
        request: "Request",
        num_computed_tokens: int,
        local_cache_hashes: Collection[str] | None = None,
    ) -> bool:
        self._local_cache_hashes = local_cache_hashes or ()
        try:
            return super().ensure_cache_available(request, num_computed_tokens)
        finally:
            self._local_cache_hashes = ()

    def _admit_item(
        self,
        request: "Request",
        feature: "MultiModalFeatureSpec",
        num_computed_tokens: int,
        now: float,
    ) -> bool:
        # NOTE(RBLN): skip transfers for inputs already in the runner cache.
        # Keep the connector's existing transfer polling and timeout handling.
        if feature.identifier in self._local_cache_hashes:
            self._deferred_since.pop((request.request_id, feature.identifier), None)
            return True
        return super()._admit_item(request, feature, num_computed_tokens, now)


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
        # One byte row per region block; a load views them back in the model dtype.
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

    def _make_scheduler(self, vllm_config: "VllmConfig") -> RblnECCPUScheduler:
        # NOTE(RBLN): a producer and a consumer reach each other only through
        # NIXL, so it is always on for them. ec_both keeps upstream's default,
        # a CPU tier of its own.
        ec_config = vllm_config.ec_transfer_config
        if ec_config.ec_role != "ec_both":
            ec_config.ec_connector_extra_config["ec_enable_nixl"] = True
        return RblnECCPUScheduler(vllm_config)

    def ensure_cache_available(
        self,
        request: "Request",
        num_computed_tokens: int,
        local_cache_hashes: Collection[str] | None = None,
    ) -> bool:
        scheduler = self.connector_scheduler
        assert isinstance(scheduler, RblnECCPUScheduler)
        return scheduler.ensure_cache_available(
            request, num_computed_tokens, local_cache_hashes
        )

    def start_save_caches(
        self, encoder_cache: dict[str, torch.Tensor] | None = None, **kwargs
    ) -> None:
        # NOTE(RBLN): ECConnectorModelRunnerMixin._get_ec_connector_output() calls
        # this before encoding to save outputs already in the runner cache.
        # The runner saves new outputs later through maybe_save_ec_to_connector().
        if encoder_cache is None:
            return
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECCPUConnectorMetadata)
        for mm_hash in metadata.saves:
            if mm_hash in encoder_cache:
                self.save_caches(encoder_cache, mm_hash)

    def update_state_after_cache_hit(self, request: "Request", index: int) -> None:
        scheduler = self.connector_scheduler
        assert scheduler is not None
        mm_hash = request.mm_features[index].identifier
        if self.is_producer and scheduler._cache.get(mm_hash) is None:
            # Producer: allocate shared-memory blocks and queue their IDs
            # in _pending_saves, delivered to the worker as metadata.saves.
            # Consumer: pin ready blocks and queue loads.
            # ec_both: both.
            scheduler.update_state_after_alloc(request, index)
