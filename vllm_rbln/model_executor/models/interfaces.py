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
from typing import TYPE_CHECKING, ClassVar, Literal, Protocol, runtime_checkable

from torch import nn
from typing_extensions import TypeIs
from vllm.model_executor.models.interfaces import supports_multimodal_embeddings

if TYPE_CHECKING:
    from vllm.config import VllmConfig


@runtime_checkable
class SupportsRBLNMultiModal(Protocol):
    """A multimodal model whose ``forward()`` merges image embeddings itself.

    vLLM merges multimodal embeddings outside ``forward()``: the runner calls
    ``embed_input_ids(input_ids, multimodal_embeddings=..., is_multimodal=...)``
    and hands the result in as ``inputs_embeds``. The RBLN compiled graph takes a
    fixed set of inputs, so the runner instead stages the per-token embeddings as
    ``mm_embeds`` ``[B, L, H]`` and a 0/1 ``mm_mask`` ``[B, L, 1]``, and the model
    selects between them and its text embeddings inside the graph. Where the mask
    is 0, ``mm_embeds`` holds finite leftovers rather than zeros: select, not add.

    Only a model declaring this takes the runner's multimodal path. An upstream
    multimodal model accepts the two kwargs through ``**kwargs`` and ignores
    them, which would run the encoder for nothing and answer as if text-only.
    """

    merges_mm_embeds_in_graph: ClassVar[Literal[True]] = True
    mm_encoder_patch_buckets: list[int]
    """Resolved, sorted, unique patch counts used by encoder warmup."""

    @classmethod
    def resolve_encoder_token_budgets(cls, vllm_config: "VllmConfig") -> list[int]:
        """The sorted encoder token budgets to compile: the configured ones, or
        the model's own choice when none are, refused if they cannot hold what
        the processor emits. The platform calls this once at start-up with image
        or video on and stores the result as ``encoder_token_budgets``, so given
        that stored list it must return it unchanged."""
        ...

    def compile_mm_encoder(self, compile_fn: Callable[[nn.Module], nn.Module]) -> None:
        """Compile the multimodal encoder with the runner's compile wrapper. A
        rank that owns no encoder returns without calling ``compile_fn``."""
        ...

    def warmup_mm_encoder(self) -> None:
        """Run every encoder graph once; return if this rank owns no encoder."""
        ...


def supports_rbln_multimodal(model: object) -> TypeIs[SupportsRBLNMultiModal]:
    return getattr(model, "merges_mm_embeds_in_graph", False) is True


def supports_rbln_multimodal_embeddings(model: object) -> bool:
    # Scheduling needs the same opt-in before the draft is instantiated.
    model_cls = model if isinstance(model, type) else type(model)
    opted_in = getattr(model_cls, "supports_rbln_multimodal_embeddings", False) is True
    return opted_in and supports_multimodal_embeddings(model_cls)


@runtime_checkable
class SupportsRBLNMmPrefix(Protocol):
    """An mm prefix LM (upstream's ``is_mm_prefix_lm``) the RBLN vLLM path serves.

    Each multimodal embedding range attends both ways on sliding layers only. A
    prefill chunk never splits a range, and each row reads keys up to its range's
    end.
    """

    @classmethod
    def max_mm_prefix_range_len(cls, vllm_config: "VllmConfig") -> int:
        """The most positions one embeds range can span; the longest prefill chunk
        must hold it. A function of config alone, so the frontend, which has no
        model, refuses a request whose range the compiled graphs cannot run."""
        ...


def supports_rbln_mm_prefix(model: object) -> TypeIs[SupportsRBLNMmPrefix]:
    return isinstance(model, SupportsRBLNMmPrefix)


__all__ = [
    "SupportsRBLNMmPrefix",
    "SupportsRBLNMultiModal",
    "supports_rbln_mm_prefix",
    "supports_rbln_multimodal",
    "supports_rbln_multimodal_embeddings",
]
