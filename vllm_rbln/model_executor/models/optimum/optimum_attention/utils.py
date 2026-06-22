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


def pad_tensor2tensor(
    original_tensor: torch.Tensor,
    rows: int,
    cols: int,
    pad_value: int = 0,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = original_tensor.dtype
    valid_nums = original_tensor.shape[0]
    padded = torch.full((rows, cols), pad_value, dtype=dtype)
    padded[:valid_nums] = original_tensor
    return padded


def pad_tensors2tensor(
    original_tensors: list[torch.Tensor],
    rows: int,
    cols: int,
    pad_value: int = 0,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = original_tensors[0].dtype
    valid_nums = len(original_tensors)
    padded = torch.full((rows, cols), pad_value, dtype=dtype)
    original_tensor = torch.cat(original_tensors)
    padded[:valid_nums] = original_tensor
    return padded


def pad_list22dtensor(
    original_list: list[int],
    rows: int,
    cols: int,
    pad_value: int = 0,
    dtype: torch.dtype = None,
) -> torch.Tensor:
    if dtype is None:
        dtype = torch.int16
    valid_nums = len(original_list)
    padded = torch.full((rows, cols), pad_value, dtype=dtype)
    original_tensor = torch.tensor(original_list, dtype=dtype).unsqueeze(1)
    padded[:valid_nums] = original_tensor
    return padded
