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

from vllm_rbln import envs

from ..ops import triton_sliding_window_attention_naive  # noqa: F401


def sliding_window_attention_naive_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.sliding_window_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                cache_seq_len,
                cache_offset,
                scale,
                block_tables,
                scale,  # dummy
            )
        else:
            return torch.ops.rbln_custom_ops.sliding_window_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                cache_seq_len,
                cache_offset,
                scale,
                block_tables,
                scale,  # dummy
                sinks,
            )

    raise NotImplementedError


def sliding_window_attention_naive_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    cache_seq_len: torch.Tensor,
    cache_offset: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.sliding_window_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                cache_seq_len,
                cache_offset,
                scale,
                block_tables,
                scale,  # dummy
            )
        else:
            return torch.ops.rbln_custom_ops.sliding_window_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                cache_seq_len,
                cache_offset,
                scale,
                block_tables,
                scale,  # dummy
                attn_mask,
                sinks,
            )

    raise NotImplementedError
