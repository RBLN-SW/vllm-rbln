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
import torch


def batched_count_greater_than(x: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """
    Counts elements in each row of x that are greater than the corresponding
    value in values.  Use torch.compile to generate an optimized kernel for
    this function. otherwise, it will create additional copies of the input
    tensors and cause memory issues.

    Args:
        x (torch.Tensor): A 2D tensor of shape (batch_size, n_elements).
        values (torch.Tensor): A 2D tensor of shape (batch_size, 1).

    Returns:
        torch.Tensor: A 1D tensor of shape (batch_size,) with the counts.
    """
    torch._check(x.shape[0] >= 1)
    torch._check(x.shape[0] == values.shape[0])
    return (x >= values).sum(-1)
