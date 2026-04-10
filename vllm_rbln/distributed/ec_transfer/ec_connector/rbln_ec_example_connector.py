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

from typing import Any

import safetensors.torch
import torch
from rebel.kv_cache import aligned_tensor
from vllm.config import VllmConfig
from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorRole
from vllm.distributed.ec_transfer.ec_connector.example_connector import (
    ECExampleConnector,
    ECExampleConnectorMetadata,
)

from vllm_rbln.logger import init_logger

logger = init_logger(__name__)


class RblnECExampleConnector(ECExampleConnector):
    """RBLN-adapted EC connector that uses memory-aligned CPU tensors.

    Subclasses the upstream file-based ECExampleConnector with RBLN-specific
    changes:
      1. Encoder cache tensors are allocated via ``aligned_tensor``
         (required for efficient DMA with RBLN NPU).
      2. Tensors reside on CPU by default. Swap ``_allocate_cache_tensor``
         to target a different device when NPU-side caching is ready.
      3. Stores multiple tensors per mm_hash (inputs_embeds, position_embed,
         rope_deltas) as a dict, enabling full prefill-param transfer.
    """

    def __init__(self, vllm_config: VllmConfig, role: ECConnectorRole):
        super().__init__(vllm_config=vllm_config, role=role)

        ec_transfer_config = vllm_config.ec_transfer_config
        assert ec_transfer_config is not None
        self._ec_buffer_device: str = ec_transfer_config.ec_buffer_device or "cpu"

    def _allocate_cache_tensor(self, source: torch.Tensor) -> torch.Tensor:
        """Allocate a memory-aligned tensor and copy *source* into it.

        Currently allocates on CPU via ``rebel.kv_cache.aligned_tensor``.
        To move the cache to another device (e.g. NPU) in the future,
        override this method or extend with a device-transfer step after
        the aligned copy.
        """
        buf = aligned_tensor(source.numel()).reshape(source.shape)
        buf.copy_(source)
        return buf

    # ------------------------------------------------------------------
    # Worker-side overrides
    # ------------------------------------------------------------------

    def start_load_caches(
        self, encoder_cache: dict[str, Any], **kwargs
    ) -> None:
        metadata = self._get_connector_metadata()
        assert isinstance(metadata, ECExampleConnectorMetadata)
        assert encoder_cache is not None

        for mm_data in metadata.mm_datas:
            if mm_data.mm_hash in encoder_cache:
                continue
            filename = self._generate_filename_debug(mm_data.mm_hash)
            raw_tensors = safetensors.torch.load_file(filename)
            aligned: dict[str, torch.Tensor] = {}
            for key, tensor in raw_tensors.items():
                aligned[key] = self._allocate_cache_tensor(tensor)
            encoder_cache[mm_data.mm_hash] = aligned
            logger.debug("Loaded encoder cache for hash %s", mm_data.mm_hash)

    def save_caches(
        self, encoder_cache: dict[str, Any], mm_hash: str, **kwargs
    ) -> None:
        if not self.is_producer:
            return

        filename = self._generate_filename_debug(mm_hash)
        data = encoder_cache[mm_hash]

        if isinstance(data, dict):
            tensors = {
                k: v.detach().cpu() if v.device.type != "cpu" else v.detach()
                for k, v in data.items()
            }
        else:
            tensor = data.detach()
            if tensor.device.type != "cpu":
                tensor = tensor.cpu()
            tensors = {"ec_cache": tensor}

        safetensors.torch.save_file(tensors, filename)
        logger.debug("Saved encoder cache for hash %s", mm_hash)
