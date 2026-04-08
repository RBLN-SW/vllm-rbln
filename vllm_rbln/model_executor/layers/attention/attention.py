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
import vllm.model_executor.layers.attention.attention as vllm_attn
from vllm.config import VllmConfig, get_current_vllm_config
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

from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

# ---------------------------------------------------------------------------
# Snapshots of upstream implementations (used by RBLN overrides)
# ---------------------------------------------------------------------------
_upstream_init = Attention.__init__
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


def _rbln_unified_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    layer_name: str,
) -> torch.Tensor:
    attn_metadata, self, _kv_cache, _ = get_attention_context(layer_name)

    # NOTE(RBLN) - To represent kv cache as model input,
    # modify attention instead of using the attention layer's embedded
    # kv cache (self.kv_cache); use attention metadata's kv cache.
    # attention metadata's kv cache must equal the attention layer's
    # embedded kv cache.
    assert attn_metadata.kv_caches is not None
    assert self.layer_index < len(attn_metadata.kv_caches)
    kv_cache = attn_metadata.kv_caches[self.layer_index]

    output = self.impl.forward(self, query, key, value, kv_cache, attn_metadata)
    return output


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
    attn_metadata, self, _kv_cache, _ = get_attention_context(layer_name)

    # NOTE(RBLN) - To represent kv cache as model input,
    # modify attention instead of using the attention layer's embedded
    # kv cache (self.kv_cache); use attention metadata's kv cache.
    # attention metadata's kv cache must equal the attention layer's
    # embedded kv cache.
    assert attn_metadata.kv_caches is not None
    assert self.layer_index < len(attn_metadata.kv_caches)
    kv_cache = attn_metadata.kv_caches[self.layer_index]
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


vllm_attn.unified_attention = maybe_transfer_kv_layer(_rbln_unified_attention)
vllm_attn.unified_attention_with_output = maybe_transfer_kv_layer(
    _rbln_unified_attention_with_output
)

Attention.__init__ = _rbln_attention_init
Attention.get_kv_cache_spec = _rbln_get_kv_cache_spec
