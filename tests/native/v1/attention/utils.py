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

"""Builders for the native v1/attention tests: a kv-cache spec, the builder,
and the CommonAttentionMetadata input that build() reads (CPU, no mocks)."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import torch


def make_attention_spec(
    *,
    block_size: int = 16,
    num_kv_heads: int = 4,
    head_size: int = 64,
    dtype: torch.dtype = torch.float16,
    sliding_window: int | None = None,
    appends_kv: bool = True,
) -> Any:
    """A FullAttentionSpec, or a sliding-window spec when sliding_window is set:
    upstream's, or RBLN's shift-layout one when appends_kv is False."""
    from vllm.v1.kv_cache_interface import FullAttentionSpec, SlidingWindowSpec

    from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec

    kwargs = dict(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        dtype=dtype,
    )
    if sliding_window is None:
        return FullAttentionSpec(**kwargs)
    spec_cls = SlidingWindowSpec if appends_kv else RBLNSlidingWindowSpec
    return spec_cls(**kwargs, sliding_window=sliding_window)


def make_common_attn_metadata(
    *,
    num_reqs: int,
    query_start_loc: torch.Tensor,
    seq_lens: torch.Tensor,
    block_table_tensor: torch.Tensor,
) -> SimpleNamespace:
    """The four fields build() reads (the real type has ~20; build() touches
    only these)."""
    return SimpleNamespace(
        num_reqs=num_reqs,
        query_start_loc=query_start_loc,
        seq_lens=seq_lens,
        block_table_tensor=block_table_tensor,
    )


def make_builder(
    vllm_config: Any,
    *,
    sliding_window: int | None = None,
    appends_kv: bool = True,
    spec: Any = None,
    layer_names: tuple[str, ...] = ("model.layers.0.self_attn",),
    device: str = "cpu",
) -> Any:
    """Construct the real builder. is_causal is read from
    VLLM_RBLN_FLASH_CAUSAL_ATTN at __init__, so set it before calling."""
    from vllm.config import set_current_vllm_config

    from vllm_rbln.v1.attention.backends.flash_attention import (
        RBLNFlashAttentionMetadataBuilder,
    )

    if spec is None:
        spec = make_attention_spec(sliding_window=sliding_window, appends_kv=appends_kv)
    with set_current_vllm_config(vllm_config):
        return RBLNFlashAttentionMetadataBuilder(
            spec, list(layer_names), vllm_config, torch.device(device)
        )


def make_impl(vllm_config: Any, **overrides) -> Any:
    """Construct RBLNFlashAttentionImpl with valid defaults; override to probe a
    single __init__ guard."""
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backend import AttentionType

    from vllm_rbln.v1.attention.backends.flash_attention import RBLNFlashAttentionImpl

    kwargs = dict(
        num_heads=8,
        head_size=128,  # in the supported set
        scale=1.0,
        num_kv_heads=8,  # divides num_heads
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        sinks=None,
    )
    kwargs.update(overrides)
    with set_current_vllm_config(vllm_config):
        return RBLNFlashAttentionImpl(**kwargs)


def make_mla_impl(vllm_config: Any, **overrides) -> Any:
    """Construct RBLNFlashAttnMLAImpl with valid defaults; override to probe a
    single __init__ guard."""
    from vllm.config import set_current_vllm_config
    from vllm.v1.attention.backend import AttentionType

    from vllm_rbln.v1.attention.backends.mla.flashattn_mla import RBLNFlashAttnMLAImpl

    kwargs = dict(
        num_heads=8,
        head_size=576,  # the only supported MLA head size
        scale=1.0,
        num_kv_heads=1,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        q_lora_rank=None,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        qk_head_dim=192,
        v_head_dim=128,
        kv_b_proj=None,
    )
    kwargs.update(overrides)
    with set_current_vllm_config(vllm_config):
        return RBLNFlashAttnMLAImpl(**kwargs)
