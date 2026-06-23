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
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention import mla_attention as _mla_attention_mod
from vllm.model_executor.layers.attention.attention import Attention
from vllm.model_executor.layers.attention.kv_transfer_utils import (
    maybe_transfer_kv_layer,
)
from vllm.model_executor.layers.attention.mla_attention import MLAAttention
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


# ---------------------------------------------------------------------------
# MLAAttention overrides for RBLN
# ---------------------------------------------------------------------------
class _RBLNNoOpMLAPrefillBackend:
    def __init__(self, *args, **kwargs) -> None:
        pass


def _rbln_get_mla_prefill_backend(vllm_config) -> type[_RBLNNoOpMLAPrefillBackend]:
    return _RBLNNoOpMLAPrefillBackend


_mla_attention_mod.get_mla_prefill_backend = _rbln_get_mla_prefill_backend

_original_mla_attention_init = MLAAttention.__init__


def _rbln_mla_attention_init(self, *args, **kwargs) -> None:
    _original_mla_attention_init(self, *args, **kwargs)

    self.layer_index = extract_layer_index(self.layer_name)
    vllm_config = get_current_vllm_config()
    parallel_config = vllm_config.parallel_config
    model_config = vllm_config.model_config
    start, _end = model_config.get_layers_start_end_indices(parallel_config)
    self.layer_index -= start


def _rbln_mla_attention_forward(
    self,
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    output_shape: torch.Size | None = None,
) -> torch.Tensor:
    if self.calculate_kv_scales:
        torch.ops.vllm.maybe_calc_kv_scales(q, kv_c_normed, k_pe, self.layer_name)

    if self.use_direct_call:
        forward_context: ForwardContext = get_forward_context()
        attn_metadata = forward_context.attn_metadata
        if isinstance(attn_metadata, dict):
            attn_metadata = attn_metadata[self.layer_name]

        # NOTE(RBLN): Use attn_metadata's KV cache instead of self.kv_cache
        # so that KV caches appear as explicit model inputs for compilation.
        self_kv_cache = _resolve_kv_cache(attn_metadata, self.layer_index)

        if self.attn_backend.accept_output_buffer:
            output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
            self.impl.forward(
                self,
                q,
                kv_c_normed,
                k_pe,
                self_kv_cache,
                attn_metadata,
                output=output,
            )
            return output
        return self.impl.forward(
            self, q, kv_c_normed, k_pe, self_kv_cache, attn_metadata
        )

    if self.attn_backend.accept_output_buffer:
        output = torch.empty(output_shape, dtype=q.dtype, device=q.device)
        torch.ops.vllm.unified_mla_attention_with_output(
            q,
            kv_c_normed,
            k_pe,
            output,
            self.layer_name,
        )
        return output
    return torch.ops.vllm.unified_mla_attention(
        q,
        kv_c_normed,
        k_pe,
        self.layer_name,
    )


_original_mla_process_weights = MLAAttention.process_weights_after_loading


def _rbln_mla_process_weights(self, act_dtype: torch.dtype) -> None:
    _original_mla_process_weights(self, act_dtype)
    # RBLN uses 4D weights for batched matmul: [1, N, P, L] / [1, N, L, V]
    if hasattr(self, "W_UK_T"):
        self.W_UK_T = self.W_UK_T.unsqueeze(0)
    if hasattr(self, "W_UV"):
        self.W_UV = self.W_UV.unsqueeze(0)


MLAAttention.__init__ = _rbln_mla_attention_init
MLAAttention.forward = _rbln_mla_attention_forward
MLAAttention.process_weights_after_loading = _rbln_mla_process_weights
