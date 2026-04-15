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
from dataclasses import dataclass, replace
from typing import Any

import torch
from vllm.model_executor.models.utils import extract_layer_index


def _storage_key(tensor: torch.Tensor) -> tuple[int, int]:
    """Return a hashable key that identifies the tensor's underlying storage.

    Meta tensors have no real storage, so we use the storage object identity
    which is shared across views of the same base tensor.
    """
    storage = tensor.untyped_storage()
    if tensor.device.type == "meta":
        return (id(storage), storage.nbytes())
    return (storage.data_ptr(), storage.nbytes())


@dataclass(frozen=True)
class KVCacheViewInfo:
    base_index: int = -1
    view_dtype: torch.dtype | None = None
    view_shape: tuple[int, ...] | None = None
    permute_order: tuple[int, ...] | None = None
    select_index: int | None = None


def build_kv_cache_base_bindings(
    kv_cache_bases_by_layer: dict[str, torch.Tensor],
    kv_cache_view_infos_by_layer: dict[str, KVCacheViewInfo],
    num_attn_module: int = 1,
) -> tuple[list[torch.Tensor], list[KVCacheViewInfo]]:
    """Deduplicate KV cache inputs by storage and capture per-layer view info."""
    base_tensors: list[torch.Tensor] = []
    view_infos: list[KVCacheViewInfo] = []
    base_index_by_storage: dict[tuple[int, int], int] = {}

    layer_names = _get_ordered_layer_names(
        kv_cache_view_infos_by_layer, num_attn_module
    )
    for layer_name in layer_names:
        base_tensor = kv_cache_bases_by_layer[layer_name]
        storage_key = _storage_key(base_tensor)
        base_index = base_index_by_storage.get(storage_key)
        if base_index is None:
            base_index = len(base_tensors)
            base_index_by_storage[storage_key] = base_index
            base_tensors.append(base_tensor)

        view_infos.append(
            replace(
                kv_cache_view_infos_by_layer[layer_name],
                base_index=base_index,
            )
        )

    return base_tensors, view_infos


def materialize_kv_cache_view(
    base_tensors: list[torch.Tensor], view_info: KVCacheViewInfo
) -> torch.Tensor:
    """Rebuild a layer-local KV cache view from a deduplicated base tensor."""
    tensor = base_tensors[view_info.base_index]
    if view_info.view_dtype is not None:
        tensor = tensor.view(view_info.view_dtype)
    if view_info.view_shape is not None:
        tensor = tensor.view(view_info.view_shape)
    if view_info.permute_order is not None:
        tensor = tensor.permute(*view_info.permute_order)
    if view_info.select_index is not None:
        tensor = tensor.select(0, view_info.select_index)
    return tensor


def attach_kv_cache_bindings(
    attn_metadata: Any,
    kv_caches: list[torch.Tensor] | None,
    kv_cache_bases: list[torch.Tensor] | None,
    kv_cache_view_infos: list[KVCacheViewInfo] | None,
) -> None:
    """Attach either per-layer KV views or deduplicated bases to metadata."""
    if kv_cache_bases and kv_cache_view_infos:
        attn_metadata.kv_caches = None
        attn_metadata.kv_cache_view_infos = kv_cache_view_infos
        return

    attn_metadata.kv_caches = kv_caches
    attn_metadata.kv_cache_view_infos = None


def build_kv_cache_forward_context_kwargs(
    kv_cache_bases: list[torch.Tensor] | None,
) -> dict[str, tuple[torch.Tensor, ...]]:
    if not kv_cache_bases:
        return {}
    return {"kv_cache_bases": tuple(kv_cache_bases)}


def validate_shared_attention_kv_cache_contiguity(
    kv_caches: dict[str, torch.Tensor],
    kv_cache_bases_by_layer: dict[str, torch.Tensor],
    kv_cache_view_infos_by_layer: dict[str, KVCacheViewInfo],
) -> None:
    layers_by_storage: dict[tuple[int, int], list[str]] = defaultdict(list)

    for layer_name in kv_cache_view_infos_by_layer:
        base_tensor = kv_cache_bases_by_layer.get(layer_name)
        kv_cache = kv_caches.get(layer_name)
        if base_tensor is None or kv_cache is None:
            continue
        storage_key = _storage_key(base_tensor)
        layers_by_storage[storage_key].append(layer_name)

    for layer_names in layers_by_storage.values():
        if len(layer_names) <= 1:
            continue

        non_contiguous = [
            layer_name
            for layer_name in layer_names
            if not kv_caches[layer_name].is_contiguous()
        ]
        if not non_contiguous:
            continue

        layer_summaries = ", ".join(
            (
                f"{layer_name}(shape={tuple(kv_caches[layer_name].shape)}, "
                f"stride={tuple(kv_caches[layer_name].stride())})"
            )
            for layer_name in non_contiguous
        )
        raise ValueError(
            "Shared attention KV cache tensors must be contiguous. "
            f"Non-contiguous layers: {layer_summaries}"
        )


def _get_ordered_layer_names(
    kv_caches: dict[str, torch.Tensor], num_attn_module: int
) -> list[str]:
    index_to_names: dict[int, list[str]] = defaultdict(list)
    for layer_name in kv_caches:
        layer_index = extract_layer_index(layer_name, num_attn_module)
        index_to_names[layer_index].append(layer_name)

    ordered_layer_names: list[str] = []
    for layer_index in sorted(index_to_names):
        ordered_layer_names.extend(index_to_names[layer_index])
    return ordered_layer_names
