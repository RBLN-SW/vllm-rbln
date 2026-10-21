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


def pack_attention_sinks(
    q: torch.Tensor, sinks: torch.Tensor | None, *, is_prefill: bool
) -> torch.Tensor:
    if sinks is None:
        return q
    if is_prefill and q.shape[0] != 1:
        raise NotImplementedError("Triton sink prefill requires batch size one")
    if q.shape[-1] != 64:
        raise NotImplementedError("Triton sinks require head size 64")
    if sinks.numel() != q.shape[1] * q.shape[2]:
        raise ValueError("Expected one sink per query head")
    # The carrier keeps the op ABI fixed; padding must contribute no softmax mass.
    sink = sinks.float().reshape(1, q.shape[1], q.shape[2], 1, 1)
    padding = torch.full(
        (*sink.shape[:-1], 63), float("-inf"), dtype=torch.float32, device=q.device
    )
    aligned = torch.cat((sink, padding), dim=-1)
    aligned = aligned.expand(q.shape[0], q.shape[1], q.shape[2], q.shape[-2], 64)
    return torch.cat((q.float(), aligned), dim=-1)
