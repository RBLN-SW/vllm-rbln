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


def sliding_window_attention_v1(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    kv_cache: torch.Tensor,
    seq_idx: torch.Tensor,
    scale: torch.Tensor,
    block_tables: torch.Tensor,
    window_size: int,
    sinks: torch.Tensor | None = None,
) -> torch.Tensor:
    if envs.VLLM_RBLN_COMPILE_MODEL:
        return torch.ops.rbln_custom_ops.sliding_window_attention_v1(
            q,
            k,
            v,
            kv_cache,
            seq_idx,
            scale,
            block_tables,
            window_size,
            True,  # is_causal
            None,  # attn_mask: derived from the window by the converter
            sinks,
        )

    raise NotImplementedError
