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

import inspect
from functools import wraps

import torch
import vllm.model_executor.layers.attention.attention as vllm_attn
from vllm.config import VllmConfig, get_current_vllm_config
from vllm.distributed.kv_transfer import (
    get_kv_transfer_group,
    has_kv_transfer_group,
    is_v1_kv_transfer_group,
)
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.attention.attention import (
    Attention,
    get_attention_context,
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
    kv_cache = _resolve_kv_cache(attn_metadata, self.layer_index)

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


# NOTE(RBLN) fsw-inference#325: with a v1 KV connector + the device-tensor
# (export) compile path, some rebel-compiler/driver versions split the attention
# forward into multiple distinct-named executors, so rebel's nameless
# `create_runtime` aborts warm-up with "All executors must have the same name".
#
# Refined fix (vs the upstream fix B that `torch._dynamo.disable`s the whole
# wrapper and graph-breaks every attention layer): isolate the two connector
# hook calls behind `@torch._dynamo.disable` helpers. dynamo graph-breaks only
# at the wait/save calls; the attention `func` itself stays inside the traced
# graph and is still compiled onto the NPU. The hooks are only reached on the
# runtime path (connector has bound metadata), so normal serving and warm-up
# (dummy metadata is None -> early return) incur no graph break at all.
#
# This mirrors the upstream change to
# `vllm/model_executor/layers/attention/kv_transfer_utils.py`; we replicate
# `maybe_transfer_kv_layer` here so the fix ships with vllm-rbln instead of
# patching the vendored vllm file.


@torch._dynamo.disable
def _kv_transfer_wait(connector, layer_name):
    connector.wait_for_layer_load(layer_name)


@torch._dynamo.disable
def _kv_transfer_save(connector, layer_name, kv_cache, attn_metadata):
    connector.save_kv_layer(layer_name, kv_cache, attn_metadata)


def _rbln_maybe_transfer_kv_layer(func):
    """Like vllm's `maybe_transfer_kv_layer`, but the connector wait/save hooks
    are routed through `torch._dynamo.disable` helpers so only they graph-break
    (keeping the attention op in the compiled graph)."""
    sig = inspect.signature(func)
    param_names = list(sig.parameters.keys())
    try:
        layer_name_index = param_names.index("layer_name")
    except ValueError as e:
        raise TypeError(
            f"Function {func.__name__} must have a 'layer_name' parameter"
        ) from e

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not has_kv_transfer_group() or not is_v1_kv_transfer_group():
            return func(*args, **kwargs)

        layer_name = args[layer_name_index]
        attn_metadata, _, kv_cache, _ = get_attention_context(layer_name)
        connector = get_kv_transfer_group()
        if attn_metadata is None or not connector.has_connector_metadata():
            return func(*args, **kwargs)

        # Wait for KV layer on entry (graph-break: keep connector hook out of
        # the traced attention graph).
        _kv_transfer_wait(connector, layer_name)

        # Execute attention — stays in the graph for NPU compile.
        result = func(*args, **kwargs)

        # Save KV cache layer on exit (graph-break).
        _kv_transfer_save(connector, layer_name, kv_cache, attn_metadata)

        return result

    return wrapper


vllm_attn.unified_attention = _rbln_maybe_transfer_kv_layer(_rbln_unified_attention)
vllm_attn.unified_attention_with_output = _rbln_maybe_transfer_kv_layer(
    _rbln_unified_attention_with_output
)

Attention.__init__ = _rbln_attention_init
Attention.get_kv_cache_spec = _rbln_get_kv_cache_spec
