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

# RBLN overrides for Multi-head Latent Attention (MLA). These mirror the
# non-MLA Attention overrides in ``patches/attention.py`` (KV cache resolved
# from attention metadata as a graph input, pipeline-adjusted layer index) and
# reuse its shared helpers.

import torch
from vllm.forward_context import ForwardContext, get_forward_context
from vllm.model_executor.layers.attention.mla_attention import MLAAttention

from vllm_rbln.patches import register_patch
from vllm_rbln.patches.attention import (
    _record_pipeline_layer_index,
    _resolve_kv_cache,
)

mla_attention_original_init = MLAAttention.__init__
mla_attention_original_process_weights = MLAAttention.process_weights_after_loading


class _RBLNNoOpMLAPrefillBackend:
    """No-op stand-in for vLLM's MLA chunked-prefill backend."""

    def __init__(self, *args, **kwargs) -> None:
        pass


@register_patch(
    target=(
        "vllm.model_executor.layers.attention.mla_attention.get_mla_prefill_backend"
    ),
    reason=(
        "The upstream MLA chunked-prefill backend requires FlashAttention/"
        "FlashInfer, which RBLN does not provide. RBLN handles prefill via a "
        "custom kernel and never uses MLAAttention.prefill_backend, so return a "
        "no-op backend to keep MLAAttention.__init__ from importing an "
        "unavailable kernel. MLAAttention.__init__ resolves the name from the "
        "mla_attention module globals at call time (after patches are applied)."
    ),
)
def patched_get_mla_prefill_backend(vllm_config) -> type[_RBLNNoOpMLAPrefillBackend]:
    return _RBLNNoOpMLAPrefillBackend


@register_patch(
    target="vllm.model_executor.layers.attention.mla_attention.MLAAttention.__init__",
    reason=(
        "RBLN needs MLAAttention initialization to record a pipeline-adjusted "
        "layer index so external KV-cache bindings can resolve the matching "
        "per-layer latent cache tensor (same requirement as `Attention.__init__`)."
    ),
)
def patched_mla_attention_init(self: MLAAttention, *args, **kwargs) -> None:
    mla_attention_original_init(self, *args, **kwargs)

    # NOTE(RBLN): Layer index is required to use external binding KV cache.
    _record_pipeline_layer_index(self)


@register_patch(
    target="vllm.model_executor.layers.attention.mla_attention.MLAAttention.forward",
    reason=(
        "RBLN resolves the latent KV cache from attention metadata (a graph "
        "input) instead of the layer's embedded cache, mirroring the non-MLA "
        "unified_attention_with_output override, so the KV cache enters the "
        "compiled graph as an input rather than a baked constant."
    ),
)
def patched_mla_attention_forward(
    self: MLAAttention,
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


@register_patch(
    target=(
        "vllm.model_executor.layers.attention.mla_attention."
        "MLAAttention.process_weights_after_loading"
    ),
    reason=(
        "RBLN uses 4-D weights for batched matmul in the MLA impl, so add a "
        "leading batch dim to the absorbed W_UK_T / W_UV projections after "
        "upstream weight processing."
    ),
)
def patched_mla_process_weights(self: MLAAttention, act_dtype: torch.dtype) -> None:
    mla_attention_original_process_weights(self, act_dtype)
    # RBLN uses 4D weights for batched matmul: [1, N, P, L] / [1, N, L, V]
    if hasattr(self, "W_UK_T"):
        self.W_UK_T = self.W_UK_T.unsqueeze(0)
    if hasattr(self, "W_UV"):
        self.W_UV = self.W_UV.unsqueeze(0)
