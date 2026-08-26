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

from . import triton_flash_causal_attention  # noqa: F401


def flash_causal_attention_naive_prefill(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor | None = None,
    k_quantize_scale: torch.Tensor | None = None,
    v_quantize_scale: torch.Tensor | None = None,
    cache_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.flash_causal_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy
            )
        else:
            return torch.ops.rbln_custom_ops.flash_causal_attention_naive_prefill(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy,
                sinks,
                k_quantize_scale,
                v_quantize_scale,
                cache_dtype,
            )

    raise NotImplementedError


def flash_causal_attention_naive_decode(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    scale: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    sinks: torch.Tensor | None = None,
    k_quantize_scale: torch.Tensor | None = None,
    v_quantize_scale: torch.Tensor | None = None,
    cache_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        if envs.VLLM_RBLN_USE_CUSTOM_KERNEL:
            return torch.ops.rbln_triton_ops.flash_causal_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy,
            )
        else:
            return torch.ops.rbln_custom_ops.flash_causal_attention_naive_decode(
                q,
                k,
                v,
                kv_cache,
                scale,
                seq_idx,
                block_tables,
                scale,  # dummy,
                sinks,
                k_quantize_scale,
                v_quantize_scale,
                cache_dtype,
            )

    raise NotImplementedError
