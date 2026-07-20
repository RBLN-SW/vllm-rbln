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

# Paged flash causal MLA custom ops. These are stubs for torch.compile /
# torch.export tracing; the actual kernel is provided by the RBLN runtime at
# execution time.

import torch

from vllm_rbln import envs


def paged_flash_causal_mla_naive_prefill(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        return torch.ops.rbln_custom_ops.paged_flash_causal_mla_naive_prefill(
            q,
            kv_c_normed,
            k_pe,
            kv_cache,
            seq_idx,
            block_tables,
            scale,
        )

    raise NotImplementedError


def paged_flash_causal_mla_naive_decode(
    q: torch.Tensor,
    kv_c_normed: torch.Tensor,
    k_pe: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    block_tables: torch.Tensor,
    scale: torch.Tensor,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        return torch.ops.rbln_custom_ops.paged_flash_causal_mla_naive_decode(
            q,
            kv_c_normed,
            k_pe,
            kv_cache,
            seq_idx,
            block_tables,
            scale,
        )

    raise NotImplementedError
