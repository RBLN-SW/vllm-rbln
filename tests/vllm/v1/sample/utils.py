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

# Copied from tests.v1.sample.utils: https://github.com/vllm-project/vllm/blob/v0.13.0/tests/v1/sample/utils.py

import torch


def create_allowed_token_ids(
    batch_size: int,
    vocab_size: int,
    num_allowed_token_ids: int,
    device: torch.device,
) -> torch.Tensor | None:
    mask: torch.Tensor | None = None
    for i in range(batch_size):
        if i % 2 == 1:
            continue
        if mask is None:
            mask = torch.zeros(
                (batch_size, vocab_size), dtype=torch.bool, device=device
            )
        start = min(i, vocab_size - 1)
        end = min(i + num_allowed_token_ids, vocab_size - 1)
        mask[i, start:end] = True
    return mask
