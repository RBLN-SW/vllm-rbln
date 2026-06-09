# Copyright 2025 Rebellions Inc. All rights reserved.
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
from typing import TYPE_CHECKING, Any

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionMetadata, AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_rbln.patches import register_patch
from vllm_rbln.v1.attention.backends.flash_attention import RBLNFlashAttentionMetadata
from vllm_rbln.v1.attention.kv_cache_bindings import materialize_kv_cache_view
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

if TYPE_CHECKING:
    from vllm.model_executor.layers.attention import MLAAttention

attention_original_init = Attention.__init__
attention_original_forward = Attention.forward


# NOTE(RBLN) - To represent kv cache as model input,
# modify attention instead of using the attention layer's embedded
# kv cache (self.kv_cache); use attention metadata's kv cache.
# attention metadata's kv cache must equal the attention layer's
# embedded kv cache.
def _resolve_kv_cache(
    attn_metadata: RBLNFlashAttentionMetadata, layer_index: int
) -> torch.Tensor:
    """Resolve the KV cache for a given layer, either from deduplicated
    base tensors (for torch.export compatibility) or from the direct list."""
    kv_cache_view_infos = getattr(attn_metadata, "kv_cache_view_infos", None)
    forward_context = get_forward_context()
    additional_kwargs = getattr(forward_context, "additional_kwargs", {}) or {}
    kv_cache_bases = additional_kwargs.get("kv_cache_bases")
    if kv_cache_bases and kv_cache_view_infos:
        assert layer_index < len(kv_cache_view_infos)
        return materialize_kv_cache_view(
            kv_cache_bases, kv_cache_view_infos[layer_index]
        )

    assert attn_metadata.kv_caches is not None
    assert layer_index < len(attn_metadata.kv_caches)
    return attn_metadata.kv_caches[layer_index]


def _get_attention_context(
    layer_name: str,
) -> tuple[Any, "Attention | MLAAttention", torch.Tensor]:
    """Extract attention context for a given layer."""
    forward_context = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    attn_metadata: AttentionMetadata
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]
    elif isinstance(attn_metadata_raw, list):
        # list[dict[str, AttentionMetadata]]: used in speculative decoding
        # where [0] is the base-model (non-speculative) metadata dict.
        attn_metadata = attn_metadata_raw[0][layer_name]
    else:
        attn_metadata = attn_metadata_raw
    attn_layer: Attention | MLAAttention = forward_context.no_compile_layers[layer_name]
    kv_cache = _resolve_kv_cache(attn_metadata, attn_layer.layer_index)
    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    return attn_metadata, attn_layer, kv_cache


@register_patch(
    target="vllm.model_executor.layers.attention.Attention.forward",
    reason="To preserve RBLN-specific attention output shape.",
)
def patched_attention_forward(
    self: Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
) -> torch.Tensor:
    if not output_shape:
        output_shape = query.shape
        output = attention_original_forward(self, query, key, value, output_shape)
        return output.view(output_shape)
    return attention_original_forward(self, query, key, value, output_shape)


@register_patch(
    target=(
        "vllm.model_executor.layers.attention.attention.unified_attention_with_output"
    ),
    reason=(
        "RBLN needs unified_attention_with_output to preserve the KV-cache "
        "dummy dependency and resolve KV cache from attention metadata or "
        "deduplicated KV-cache base tensors."
    ),
)
@maybe_transfer_kv_layer
def patched_unified_attention_with_output(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output: torch.Tensor,
    layer_name: str,
    output_scale: torch.Tensor | None = None,
    output_block_scale: torch.Tensor | None = None,
    kv_cache_dummy_dep: torch.Tensor | None = None,
) -> None:
    # kv_cache_dummy_dep is not used but accepting it creates a data dependency
    # that ensures torch.compile preserves ordering between KV cache update and
    # attention forward.
    del kv_cache_dummy_dep
    attn_metadata, attn, kv_cache = _get_attention_context(layer_name)

    attn.impl.forward(
        attn,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output,
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


@register_patch(
    target="vllm.model_executor.layers.attention.attention.Attention.__init__",
    reason=(
        "RBLN needs Attention initialization to record a pipeline-adjusted "
        "layer index so external KV-cache bindings can resolve the matching "
        "per-layer cache tensor."
    ),
)
def patched_attention_init(self: Attention, *args, **kwargs) -> None:
    attention_original_init(self, *args, **kwargs)

    # NOTE(RBLN): Layer index is required to use external binding KV cache.
    self.layer_index = extract_layer_index(self.layer_name)

    # NOTE(RBLN): Consider PP
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    if model_config is not None:
        start, _ = model_config.get_layers_start_end_indices(
            vllm_config.parallel_config
        )
        self.layer_index -= start


@register_patch(
    target=(
        "vllm.model_executor.layers.attention.attention.Attention.get_kv_cache_spec"
    ),
    reason=(
        "RBLN needs Attention KV-cache specs to use RBLN sliding-window "
        "metadata while preserving upstream full-attention spec fields."
    ),
)
def patched_get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
    # Block size may get updated after model loading, refresh it
    block_size = vllm_config.cache_config.block_size
    # Should not be called for enc-dec or encoder-only attention.
    assert self.attn_type == AttentionType.DECODER
    if self.sliding_window is not None:
        if vllm_config.model_config.use_mla:
            raise NotImplementedError(
                "MLA is not supported with sliding window attention."
            )
        return RBLNSlidingWindowSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            dtype=self.kv_cache_torch_dtype,
            sliding_window=self.sliding_window,
        )
    else:
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=self.num_kv_heads,
            head_size=self.head_size,
            head_size_v=self.head_size_v,
            dtype=self.kv_cache_torch_dtype,
        )
