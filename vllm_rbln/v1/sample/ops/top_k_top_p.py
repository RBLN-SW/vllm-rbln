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
# NOTE(RBLN): The (top_k, top_p) pair that the compiler's `rbln::argmax`
# converter feeds to `contrib_top_k_top_p_sample`. Handing it to a sampling op
# narrows a row to its single most probable token, so the op draws that row's
# argmax. See `build_op_top_k_top_p`.
GREEDY_TOP_K = 1
GREEDY_TOP_P = 0.0


def build_op_top_k_top_p(
    sampling_metadata: SamplingMetadata,
    batch_size: int,
    vocab_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build the per-request `top_k` / `top_p` inputs of an RBLN sampling op.

    Both tensors are always materialized, and greedy requests carry
    `(GREEDY_TOP_K, GREEDY_TOP_P)` instead of their own values.

    The RBLN sampling ops have no greedy kernel: they draw one token per row
    under that row's top-k/top-p. `sampling_metadata` on its own cannot tell an
    op which rows are greedy, because vLLM rewrites a greedy request's params to
    `top_k=0` (stored as `vocab_size`) and `top_p=1.0` -- exactly what a random
    request that uses neither top-k nor top-p carries. Narrowing a greedy row to
    its single most probable token restores the distinction, and makes the
    row's outcome its argmax.

    Materializing both tensors also keeps the compiled op's graph signature
    fixed. `None` and a tensor are different inputs, so a batch that gains or
    loses its last top-k request would otherwise force a recompile.
    """
    if sampling_metadata.all_greedy:
        return (
            torch.full((batch_size,), GREEDY_TOP_K, dtype=torch.int32, device=device),
            torch.full((batch_size,), GREEDY_TOP_P, dtype=torch.float32, device=device),
        )

    # vLLM already allocates these as int32 / float32 on the batch's device, so
    # they feed the op as they are; only a missing one has to be built.
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
