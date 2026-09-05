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
"""Per-layer compile support.

A decoder layer compiled on its own hashes like every other layer of the same
shape only if nothing layer-specific reaches its graph. Two things normally do,
both through the read site Dynamo uses to name graph inputs and constants: the
KV cache (``attn_metadata.kv_caches[layer_index]``) and everything reached via
``forward_context.*[layer_name]`` (the attention metadata, and the Attention
layer whose k/v scales become ``get_attr`` constants). ``layer_slot`` is a
single, layer-agnostic read site for all of them: the model runner binds the
current layer's values before calling its compiled region, and the attention
patch reads them from here, so every layer's names come out as ``layer_slot_*``.
"""

from collections.abc import Callable
from types import FunctionType
from typing import Any

import torch
from vllm.model_executor.layers.attention import Attention


class _LayerSlot:
    __slots__ = ("attn_layers", "attn_metadatas", "kv_caches", "_index")

    def __init__(self) -> None:
        self.attn_layers: list[Attention] | None = None
        self.attn_metadatas: list[Any] | None = None
        self.kv_caches: list[torch.Tensor] | None = None
        self._index: dict[str, int] = {}

    def bind(
        self,
        attn_layers: list[Attention],
        attn_metadatas: list[Any],
        kv_caches: list[torch.Tensor],
    ) -> None:
        self.attn_layers = attn_layers
        self.attn_metadatas = attn_metadatas
        self.kv_caches = kv_caches
        self._index = {attn.layer_name: i for i, attn in enumerate(attn_layers)}

    def clear(self) -> None:
        self.attn_layers = None
        self.attn_metadatas = None
        self.kv_caches = None
        self._index = {}

    @property
    def bound(self) -> bool:
        return self.kv_caches is not None

    def attn_layer_for(self, layer_name: str) -> Attention:
        assert self.attn_layers is not None
        return self.attn_layers[self._index[layer_name]]

    def attn_metadata_for(self, layer_name: str) -> Any:
        assert self.attn_metadatas is not None
        return self.attn_metadatas[self._index[layer_name]]

    def kv_cache_for(self, layer_name: str) -> torch.Tensor:
        assert self.kv_caches is not None
        return self.kv_caches[self._index[layer_name]]


layer_slot = _LayerSlot()


def with_own_code(fn: Callable) -> Callable:
    """A copy of ``fn`` with its own code object. Dynamo caches graphs per code
    object and the tensor-only guard filter cannot tell one layer's closure from
    another's, so each layer region needs its own cache entry."""
    clone = FunctionType(
        fn.__code__.replace(),
        fn.__globals__,
        fn.__name__,
        fn.__defaults__,
        fn.__closure__,
    )
    clone.__kwdefaults__ = fn.__kwdefaults__
    clone.__qualname__ = fn.__qualname__
    return clone


def group_layers(
    layers: list[torch.nn.Module], size: int
) -> list[list[torch.nn.Module]]:
    """Consecutive layers of one class, at most `size` per group. A group's graph
    hashes like every other full group of that class, so grouping trades compile
    time against per-region dispatch and layout cost."""
    groups: list[list[torch.nn.Module]] = []
    for layer in layers:
        if groups and len(groups[-1]) < size and type(groups[-1][0]) is type(layer):
            groups[-1].append(layer)
        else:
            groups.append([layer])
    return groups


def single_attention(layer: torch.nn.Module) -> Attention:
    attns = [m for m in layer.modules() if isinstance(m, Attention)]
    if len(attns) != 1:
        raise NotImplementedError(
            "layerwise compile expects exactly one Attention per decoder layer, "
            f"found {len(attns)} in {type(layer).__name__}"
        )
    return attns[0]
