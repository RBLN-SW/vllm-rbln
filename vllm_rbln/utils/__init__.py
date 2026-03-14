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

from typing import Union

import torch


def pad(
    x: torch.Tensor, dim: int, target_len: int, pad_value: Union[int, float] = 0
) -> torch.Tensor:
    """Pad along the given dimension to target_len using pad_value."""
    current = x.size(dim)
    if current >= target_len:
        # NOTE: dynamo distinguishes views and non-views for inputs,
        # so ensure that the output is always a non-view.
        # However, RBLN device tensors do not support clone()/copy_()
        # on views, so skip the non-view conversion for RBLN
        # devices (the dynamo concern does not apply there).
        if x._base is None:
            return x
        if x.device.type in ("cpu"):
            return x.clone()
        return x

    pad_shape = list(x.shape)
    pad_shape[dim] = target_len - current
    pad = torch.full(pad_shape, pad_value, dtype=x.dtype, device=x.device)
    return torch.cat([x, pad], dim=dim)


def pad_speculative_draft_tokens(
    input_ids: torch.Tensor,
    num_scheduled_tokens: torch.Tensor,
    max_len: int | None = None,
) -> torch.Tensor:
    """
    Pad per-request draft tokens to a uniform length (max across requests)
    by inserting zeros.

    Assumes `input_ids` is a 1D concatenation of per-request draft tokens
    in request order.
    Example 1:
      input_ids = [3925, 3823, 1694, 477] or [13, 18, 19, 20]
      num_scheduled_tokens = [1, 3]
    returns:
      [3925, 0, 0, 3823, 1694, 477] or [13, 0, 0, 18, 19, 20]

    Example 2:
      input_ids = [3363, 315, 11]
      num_scheduled_tokens = [2, 1]
      max_len = 3
    returns:
      [3363, 315, 0, 11, 0, 0]
    """
    if input_ids.ndim != 1:
        raise ValueError(f"input_ids must be 1D, got shape={tuple(input_ids.shape)}")

    if num_scheduled_tokens.ndim != 1:
        raise ValueError(
            f"num_scheduled_tokens must be 1D, got shape={num_scheduled_tokens.shape}"
        )

    num_reqs = num_scheduled_tokens.numel()
    max_sched = num_scheduled_tokens.max().item()

    if max_len is not None:
        if max_len < max_sched:
            raise ValueError(
                f"max_len({max_len}) must be >= max(num_scheduled_tokens)({max_sched})"
            )
        max_sched = max_len

    # Create flattened destination indices
    req_indices = torch.repeat_interleave(
        torch.arange(num_reqs, device=num_scheduled_tokens.device), num_scheduled_tokens
    )
    token_offsets = (
        torch.arange(input_ids.numel(), device=num_scheduled_tokens.device)
        - num_scheduled_tokens.cumsum(0)[req_indices]
        + num_scheduled_tokens[req_indices]
    )
    dest_indices = req_indices * max_sched + token_offsets

    # Scatter input tokens into padded output
    out = torch.zeros(
        num_reqs * max_sched, device=input_ids.device, dtype=input_ids.dtype
    )
    out.index_copy_(0, dest_indices, input_ids)

    return out
