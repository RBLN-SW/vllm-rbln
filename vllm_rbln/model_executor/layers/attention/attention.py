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

import torch
import vllm.model_executor.layers.attention.attention as vllm_attn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.models.utils import extract_layer_index
from vllm.v1.attention.backend import AttentionType
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheSpec

from vllm_rbln.v1.attention.kv_cache_bindings import materialize_kv_cache_view
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

# ---------------------------------------------------------------------------
# Snapshots of upstream implementations (used by RBLN overrides)
# ---------------------------------------------------------------------------
_upstream_init = Attention.__init__
_upstream_forward = Attention.forward
_upstream_get_kv_cache_spec = Attention.get_kv_cache_spec


def _rbln_attention_init(self, *args, **kwargs) -> None:
    _upstream_init(self, *args, **kwargs)

    # NOTE(jiwoo.park) layer index is required to use external binding KV cache.
    self.layer_index = extract_layer_index(self.layer_name)

    # NOTE(RBLN) - consider PP
    vllm_config = get_current_vllm_config()
    model_config = vllm_config.model_config
    if model_config is not None:
        start, _end = model_config.get_layers_start_end_indices(
            vllm_config.parallel_config
        )
        self.layer_index -= start


def _rbln_attention_forward(
    self,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    output_shape: torch.Size | None = None,
) -> torch.Tensor:
    """Preserve RBLN's [batch, seq, hidden] attention output.

    RBLN may pass q/k/v as [batch, seq, hidden].
    vLLM otherwise infers output shape as [num_tokens, hidden]
    from the first dimension, so request the original
    [batch, seq, hidden] shape explicitly.
    """
    if (
        query.dim() == 3
        and query.shape[-1] == self.num_heads * self.head_size
        and output_shape is None
    ):
        output_shape = query.shape
        output = _upstream_forward(self, query, key, value, output_shape)
        return output.reshape(output_shape)

    return _upstream_forward(self, query, key, value, output_shape)


def _resolve_kv_cache(attn_metadata, layer_index: int) -> torch.Tensor:
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


def _rbln_get_attention_context(layer_name: str):
    """Resolve attention context without reading the layer's KV cache tensor.

    vLLM's upstream helper reads ``attn_layer.kv_cache`` from no_compile_layers,
    which can be a meta tensor in the RBLN torch.compile/export path. RBLN uses
    KV cache tensors from attention metadata instead, so this helper returns
    the same layer and metadata context while deliberately leaving kv_cache as
    None.
    """
    forward_context = get_forward_context()
    attn_metadata_raw = forward_context.attn_metadata
    if isinstance(attn_metadata_raw, dict):
        attn_metadata = attn_metadata_raw[layer_name]
    elif isinstance(attn_metadata_raw, list):
        attn_metadata = attn_metadata_raw[0][layer_name]
    else:
        attn_metadata = attn_metadata_raw

    attn_layer = forward_context.no_compile_layers[layer_name]

    slot_mapping = forward_context.slot_mapping
    assert isinstance(slot_mapping, dict), (
        f"Expected slot_mapping to be a dict, got {type(slot_mapping)}. "
    )
    layer_slot_mapping = slot_mapping.get(layer_name)
    return attn_metadata, attn_layer, None, layer_slot_mapping


def _rbln_unified_attention_with_output(
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
    attn_metadata, self, _kv_cache, _ = _rbln_get_attention_context(layer_name)

    # NOTE(RBLN) - To represent kv cache as model input,
    # modify attention instead of using the attention layer's embedded
    # kv cache (self.kv_cache); use attention metadata's kv cache.
    # attention metadata's kv cache must equal the attention layer's
    # embedded kv cache.
    kv_cache = _resolve_kv_cache(attn_metadata, self.layer_index)

    # Attention.forward flattens q/k/v to [num_tokens, heads, head_size];
    # restore the [batch, q_len, hidden] layout expected by the RBLN impl.
    num_tokens = query.shape[0]
    q_len = attn_metadata.max_query_len
    batch = attn_metadata.seq_lens.shape[0]
    need = batch * q_len
    query = query[:need].reshape(batch, q_len, -1)
    key = key[:need].reshape(batch, q_len, -1)
    value = value[:need].reshape(batch, q_len, -1)

    self.impl.forward(
        self,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output=output[:need].reshape(batch, q_len, -1),
        output_scale=output_scale,
        output_block_scale=output_block_scale,
    )


def _rbln_get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
    # Block size may get updated after model loading, refresh it
    block_size = vllm_config.cache_config.block_size
    # Should not be called for enc-dec or encoder-only attention.
    assert self.attn_type == AttentionType.DECODER
    if self.sliding_window is not None:
        assert not vllm_config.model_config.use_mla, (
            "MLA is not supported for slidingwindow"
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


vllm_attn.unified_attention_with_output = maybe_transfer_kv_layer(
    _rbln_unified_attention_with_output
)

Attention.__init__ = _rbln_attention_init
Attention.forward = _rbln_attention_forward
Attention.get_kv_cache_spec = _rbln_get_kv_cache_spec
