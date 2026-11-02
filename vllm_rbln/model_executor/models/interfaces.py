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

"""Model capabilities the RBLN vLLM path needs beyond upstream's interfaces."""

from collections.abc import Callable
from typing import ClassVar, Literal, Protocol, runtime_checkable

from torch import nn
from typing_extensions import TypeIs


@runtime_checkable
class SupportsRBLNMultiModal(Protocol):
    """A multimodal model whose ``forward()`` merges image embeddings itself.

    vLLM merges multimodal embeddings outside ``forward()``: the runner calls
    ``embed_input_ids(input_ids, multimodal_embeddings=..., is_multimodal=...)``
    and hands the result in as ``inputs_embeds``. The RBLN compiled graph takes a
    fixed set of inputs, so the runner instead stages the per-token embeddings as
    ``mm_embeds`` ``[B, L, H]`` and a 0/1 ``mm_mask`` ``[B, L, 1]``, and the model
    selects between them and its text embeddings inside the graph.

    Only a model declaring this takes the runner's multimodal path. An upstream
    multimodal model accepts the two kwargs through ``**kwargs`` and ignores
    them, which would run the encoder for nothing and answer as if text-only.
    """

    merges_mm_embeds_in_graph: ClassVar[Literal[True]] = True

    def compile_mm_encoder(self, compile_fn: Callable[[nn.Module], nn.Module]) -> None:
        """Compile the multimodal encoder with the runner's compile wrapper. A
        rank that owns no encoder returns without calling ``compile_fn``."""
        ...


def supports_rbln_multimodal(model: object) -> TypeIs[SupportsRBLNMultiModal]:
    return getattr(model, "merges_mm_embeds_in_graph", False) is True


__all__ = ["SupportsRBLNMultiModal", "supports_rbln_multimodal"]
