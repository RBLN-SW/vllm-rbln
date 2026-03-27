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

from __future__ import annotations

from collections.abc import Sequence

import torch

from lmcache.v1.gpu_connector.gpu_connectors import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata


class RBLNLMCacheTensorConnector(GPUConnectorInterface):
    """LMCache connector for the RBLN KV layout in CPU compatibility mode.

    Current RBLN workers expose KV cache tensors on CPU, but the runtime path is
    still logically the RBLN connector path. This connector therefore implements
    LMCache's transfer surface for the RBLN paged KV layout while assuming the
    tensors are CPU-visible.
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        *,
        use_gpu: bool = False,
        device: torch.device | None = None,
    ) -> None:
        del use_gpu
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.device = torch.device("cpu") if device is None else device
        self.kvcaches: list[torch.Tensor] | None = None

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: torch.device | None = None,
    ) -> RBLNLMCacheTensorConnector:
        num_layers = metadata.kv_shape[0]
        num_kv_heads = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_heads * head_size
        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            device=device,
        )

    def get_shape(self, num_tokens: int) -> torch.Size:
        return torch.Size([2, self.num_layers, num_tokens, self.hidden_dim_size])

    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        tensor = memory_obj.tensor
        if tensor is None:
            raise ValueError("memory_obj.tensor must be available")

        kvcaches = self._get_kvcaches(kwargs)
        block_ids, offsets = self._get_block_ids_and_offsets(
            kwargs,
            start,
            end,
            device=kvcaches[0].device,
        )
        packed_layers = [
            self._gather_layer_tokens(layer_cache, block_ids, offsets)
            for layer_cache in kvcaches
        ]
        packed = torch.stack(packed_layers, dim=1)
        tensor.copy_(packed.to(tensor.device), non_blocking=False)
        memory_obj.metadata.fmt = MemoryFormat.KV_2LTD

    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs) -> None:
        tensor = memory_obj.tensor
        if tensor is None:
            raise ValueError("memory_obj.tensor must be available")

        if memory_obj.metadata.fmt is not MemoryFormat.KV_2LTD:
            raise ValueError("RBLNLMCacheTensorConnector expects MemoryFormat.KV_2LTD")

        kvcaches = self._get_kvcaches(kwargs)
        device = kvcaches[0].device
        block_ids, offsets = self._get_block_ids_and_offsets(
            kwargs,
            start,
            end,
            device=device,
        )
        packed = tensor.to(device)
        for layer_idx, layer_cache in enumerate(kvcaches):
            self._scatter_layer_tokens(
                packed[:, layer_idx],
                layer_cache,
                block_ids,
                offsets,
            )

    def batched_from_gpu(
        self,
        memory_objs,
        starts: list[int],
        ends: list[int],
        **kwargs,
    ) -> None:
        self._ensure_non_layerwise_inputs(memory_objs, starts, ends)
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def batched_to_gpu(
        self,
        memory_objs=None,
        starts: list[int] | None = None,
        ends: list[int] | None = None,
        **kwargs,
    ) -> None:
        if memory_objs is None or starts is None or ends is None:
            raise ValueError("memory_objs, starts, and ends are required")
        self._ensure_non_layerwise_inputs(memory_objs, starts, ends)
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.to_gpu(memory_obj, start, end, **kwargs)

    def _get_kvcaches(self, kwargs: dict) -> list[torch.Tensor]:
        if "kvcaches" not in kwargs:
            raise ValueError("'kvcaches' should be provided in kwargs")

        kvcaches = kwargs["kvcaches"]
        if isinstance(kvcaches, dict):
            normalized = list(kvcaches.values())
        elif isinstance(kvcaches, Sequence):
            normalized = list(kvcaches)
        else:
            raise TypeError("kvcaches must be a sequence or dict of tensors")

        if len(normalized) == 0:
            raise ValueError("kvcaches should not be empty")

        self.kvcaches = normalized
        return normalized

    def _get_block_ids_and_offsets(
        self,
        kwargs: dict,
        start: int,
        end: int,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs")

        slot_mapping = kwargs["slot_mapping"]
        token_slots = torch.as_tensor(
            slot_mapping[start:end], dtype=torch.long, device=device
        )
        if torch.any(token_slots < 0):
            raise ValueError(
                "RBLNLMCacheTensorConnector received negative slot_mapping entries"
            )

        block_size = self._get_block_size(self.kvcaches)
        block_ids = torch.div(token_slots, block_size, rounding_mode="floor")
        offsets = token_slots.remainder(block_size)
        return block_ids, offsets

    def _get_block_size(self, kvcaches: list[torch.Tensor] | None) -> int:
        if kvcaches is None or len(kvcaches) == 0:
            raise ValueError("kvcaches must be initialized before computing block size")
        layer_cache = kvcaches[0]
        self._validate_layer_cache_shape(layer_cache)
        return int(layer_cache.shape[4])

    def _validate_layer_cache_shape(self, layer_cache: torch.Tensor) -> None:
        if layer_cache.ndim != 6:
            raise ValueError(
                "Expected RBLN KV cache with 6 dimensions, got "
                f"shape {tuple(layer_cache.shape)}"
            )
        if layer_cache.shape[0] != 2:
            raise ValueError(
                "Expected first KV cache dimension to be 2 (K/V), got "
                f"shape {tuple(layer_cache.shape)}"
            )
        if layer_cache.shape[3] != 1:
            raise ValueError(
                "Expected singleton grouping dimension in RBLN KV cache, got "
                f"shape {tuple(layer_cache.shape)}"
            )
        hidden_dim = int(layer_cache.shape[2] * layer_cache.shape[5])
        if hidden_dim != self.hidden_dim_size:
            raise ValueError(
                f"Hidden dim mismatch: expected {self.hidden_dim_size}, "
                f"got {hidden_dim}"
            )

    def _gather_layer_tokens(
        self,
        layer_cache: torch.Tensor,
        block_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_layer_cache_shape(layer_cache)
        key = layer_cache[0, block_ids, :, 0, offsets, :].reshape(
            block_ids.numel(),
            self.hidden_dim_size,
        )
        value = layer_cache[1, block_ids, :, 0, offsets, :].reshape(
            block_ids.numel(),
            self.hidden_dim_size,
        )
        return torch.stack([key, value], dim=0)

    def _scatter_layer_tokens(
        self,
        packed_layer: torch.Tensor,
        layer_cache: torch.Tensor,
        block_ids: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        self._validate_layer_cache_shape(layer_cache)
        num_heads = int(layer_cache.shape[2])
        head_size = int(layer_cache.shape[5])
        layer_cache[0, block_ids, :, 0, offsets, :] = packed_layer[0].reshape(
            block_ids.numel(),
            num_heads,
            head_size,
        )
        layer_cache[1, block_ids, :, 0, offsets, :] = packed_layer[1].reshape(
            block_ids.numel(),
            num_heads,
            head_size,
        )

    def _ensure_non_layerwise_inputs(
        self,
        memory_objs,
        starts: list[int],
        ends: list[int],
    ) -> None:
        if len(starts) != len(ends):
            raise ValueError("starts and ends must have the same length")
        if len(memory_objs) != len(starts):
            raise ValueError("memory_objs, starts, and ends must have the same length")
        if len(memory_objs) > 0 and isinstance(memory_objs[0], list):
            raise NotImplementedError(
                "Layerwise batched transfers are not supported"
                " in RBLN CPU compatibility mode"
            )
