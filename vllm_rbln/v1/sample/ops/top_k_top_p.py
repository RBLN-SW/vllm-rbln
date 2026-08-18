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
) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    """
    Build the per-request `top_k` / `top_p` inputs of an RBLN sampling op.

    batch      | filters (random rows)    | top_k             | top_p
    -----------+--------------------------+-------------------+----------------
    all random | none (pure multinomial)  | None              | None
               | top-k only               | metadata tensor   | None
               | top-p only               | None              | metadata tensor
               | top-k and top-p          | metadata tensor   | metadata tensor
    mixed      | none (pure multinomial)  | 1 / vocab_size    | None
               | top-k only               | 1 / metadata      | None
               | top-p only               | 1 / vocab_size    | 1.0 / metadata
               | top-k and top-p          | 1 / metadata      | 1.0 / metadata

    In the mixed rows, `a / b` reads: `a` at greedy rows, `b` at random rows.
    `top_k` is never `None` there, because a greedy row is encoded as
    `top_k == 1`.

    An all-greedy batch never reaches this function:
    - rejection sampler: `rbln_rejection_sample_greedy` bakes in
      `top_k` = scalar 1 and `top_p` = None.
    - normal sampler: `rbln::argmax` carries the same encoding internally.
    """
    assert not sampling_metadata.all_greedy

    top_k = sampling_metadata.top_k
    top_p = sampling_metadata.top_p
    assert top_k is None or top_k.shape == (batch_size,)
    assert top_p is None or top_p.shape == (batch_size,)

    if sampling_metadata.all_random:
        return top_k, top_p

    assert sampling_metadata.temperature is not None
    is_greedy = sampling_metadata.temperature == GREEDY_TEMPERATURE
    if top_k is None:
        # vLLM stores `top_k=0` (unset) as `vocab_size`, which disables top-k.
        top_k = torch.full((batch_size,), vocab_size, dtype=torch.int32, device=device)
    top_k = torch.where(is_greedy, top_k.new_full((), GREEDY_TOP_K), top_k)
    # `top_p` needs no rewrite: vLLM already pins a greedy row's top_p to
    # GREEDY_TOP_P (1.0).
    return top_k, top_p
