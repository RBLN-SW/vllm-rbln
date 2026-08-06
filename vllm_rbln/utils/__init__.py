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

import os
from typing import Union

import torch

# Build the padded tensor directly instead of concatenating a pad block onto it.
#
# `torch.cat` on an RBLN device tensor -- and with
# VLLM_RBLN_USE_DEVICE_TENSOR=1 the drafter's hidden-state buffer is one -- costs
# far more than the bytes involved. On the step-200 trace of Qwen3-1.7B + eagle3
# (DP4, num_spec 3) `aten::cat` totals 1.12 ms/step, and inside a 0.46 ms
# `aten::cat` the only children are `narrow`/`slice`/`empty` adding up to 8 us:
# the remaining 0.45 ms is work the profiler never sees. It is the backend
# handling a device tensor, not a memcpy.
#
# `empty` + `copy_` + `fill_` writes the same values with one allocation instead
# of two, and one pass over the payload instead of two -- `cat` copies both
# operands, this copies only `x` and fills the tail. It also keeps the result a
# non-view, which the branch below exists to guarantee. Measured 1.12 -> 0.89
# ms/step, consistent across all four DP workers.
#
# Only `dim == 0` is rewritten; that is what the callers here use.
_PAD_NO_CAT = os.getenv("VLLM_RBLN_PAD_NO_CAT", "1") == "1"


def cat_last_dim(tensors: list[torch.Tensor]) -> torch.Tensor:
    """Concatenate along the last dimension.

    Kept as a named helper because the `empty` + `copy_` rewrite that works for
    `pad` above does NOT work here, and that is worth recording rather than
    rediscovering.

    `_PAD_NO_CAT` replaces a `dim=0` concat, where the writes are contiguous. The
    last dimension is different: `out[:, off:off + w].copy_(t)` is a strided
    write, and on an RBLN device tensor that costs more than letting `torch.cat`
    do it. On EAGLE3's three aux hidden states, region-normalised on the `0/2`
    decode step: 0.53 -> 1.30 ms/step, a 2.3x regression.

    So the generalisation "cat is expensive on device tensors" is wrong -- it is
    axis-dependent.
    """
    return torch.cat(tensors, dim=-1)


def pad(
    x: torch.Tensor, dim: int, target_len: int, pad_value: Union[int, float] = 0
) -> torch.Tensor:
    """Pad along the given dimension to target_len using pad_value."""
    current = x.size(dim)
    if current >= target_len:
        # NOTE: dynamo distinguishes views and non-views for inputs,
        # so ensure that the output is always a non-view.
        return x if x._base is None else x.clone()

    if _PAD_NO_CAT and dim == 0:
        out = torch.empty(
            (target_len,) + tuple(x.shape[1:]), dtype=x.dtype, device=x.device
        )
        out[:current].copy_(x)
        out[current:].fill_(pad_value)
        return out

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
