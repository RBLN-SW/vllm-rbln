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

import torch
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
)
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_rbln.patches import register_patch
from vllm_rbln.v1.attention.backends.flash_attention import RBLNFlashAttentionMetadata
from vllm_rbln.v1.attention.kv_cache_bindings import materialize_kv_cache_view
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

attention_original_init = Attention.__init__


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


@register_patch(
    target="vllm.model_executor.layers.attention.attention.unified_attention",
    reason=(
        "RBLN needs unified_attention to resolve KV cache from attention "
        "metadata or deduplicated KV-cache base tensors instead of the "
        "attention layer's embedded KV cache."
    ),
)
@maybe_transfer_kv_layer
def patched_unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    attn_metadata, self, _, _ = get_attention_context(layer_name)

    # NOTE(RBLN) - To represent kv cache as model input,
    # modify attention instead of using the attention layer's embedded
    # kv cache (self.kv_cache); use attention metadata's kv cache.
    # attention metadata's kv cache must equal the attention layer's
    # embedded kv cache.
    kv_cache = _resolve_kv_cache(attn_metadata, self.layer_index)

    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)
    return output


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
    attn_metadata, self, _, _ = get_attention_context(layer_name)

    # NOTE(RBLN) - To represent kv cache as model input,
    # modify attention instead of using the attention layer's embedded
    # kv cache (self.kv_cache); use attention metadata's kv cache.
    # attention metadata's kv cache must equal the attention layer's
    # embedded kv cache.
    kv_cache = _resolve_kv_cache(attn_metadata, self.layer_index)
    self.impl.forward(
        self,
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
