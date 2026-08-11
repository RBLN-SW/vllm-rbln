# Copyright 2026 Rebellions Inc. All rights reserved.

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
from vllm.v1.sample.metadata import SamplingMetadata

GREEDY_TEMPERATURE = 0

GREEDY_TOP_K = 1
GREEDY_TOP_P = 1.0


def build_op_top_k_top_p(
    sampling_metadata: SamplingMetadata,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Build the per-request `top_k` / `top_p` inputs of an RBLN sampling op.
    """
    if sampling_metadata.all_greedy:
        return (
            torch.full((batch_size,), GREEDY_TOP_K, dtype=torch.int32, device=device),
            torch.full((batch_size,), GREEDY_TOP_P, dtype=torch.float32, device=device),
        )

    top_k = (
        sampling_metadata.top_k
        if sampling_metadata.top_k is not None
        # vLLM stores `top_k=0` (unset) as `vocab_size`, which disables top-k.
        else torch.full((batch_size,), vocab_size, dtype=torch.int32, device=device)
    )
    top_p = (
        sampling_metadata.top_p
        if sampling_metadata.top_p is not None
        else torch.ones(batch_size, dtype=torch.float32, device=device)
    )
    assert top_k.shape == (batch_size,)
    assert top_p.shape == (batch_size,)
    if sampling_metadata.all_random:
        return top_k, top_p

    assert sampling_metadata.temperature is not None
    is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    return (
        torch.where(is_greedy, top_k.new_full((), GREEDY_TOP_K), top_k),
        torch.where(is_greedy, top_p.new_full((), GREEDY_TOP_P), top_p),
    )
