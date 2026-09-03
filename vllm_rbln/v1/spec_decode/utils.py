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

import numpy as np
import torch

# Upstream replaces both of the computations below with one Triton kernel each,
# and its CPU backend ships C++ versions of those kernels that are present in
# the wheel we run. We call neither. Those kernels reach for `at::parallel_for`
# unconditionally, without the `GRAIN_SIZE` guard that PyTorch's own aten ops
# use to stay on the calling thread for small work, so they pay the intra-op
# thread pool's start-up cost no matter how few elements they touch -- a cost
# that does not shrink with the batch, and that dominates the work itself at the
# batch sizes a decode bucket gives us. Measured against the op chain they
# replace, they lose; numpy wins at every bucket we compile for.
#
# In the drafter's own call path most of these are already host-resident --
# `cu_num_draft_tokens` comes from `torch.from_numpy` and `query_start_loc` is
# the runner's numpy-backed int32 buffer -- so the only crossing that path pays
# is the sampler's output. Both results are consumed on the host too.


def _host(t: torch.Tensor) -> "np.ndarray":
    """numpy needs host memory, and a caller may hold any of these on either
    side: the sampler's output is device-resident, the rest are the runner's
    numpy-backed buffers."""
    return t.numpy() if t.device.type == "cpu" else t.cpu().numpy()


def eagle_prepare_next_token_padded(
    # [bs, num_sampled_tokens_per_req]
    sampled_token_ids: torch.Tensor,
    # [bs], bool
    discard_request_mask: torch.Tensor,
    # [bs]
    backup_next_token_ids: torch.Tensor,
    vocab_size: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes the number of valid (1 + accepted) tokens for each request,
    and the corresponding "next" token id to sample from during speculative decoding.
    This is the "last accepted token" from the sampled tokens, or the backup token if no
    tokens were accepted or if the request is marked as discarded.
    """
    sampled = _host(sampled_token_ids)
    discard = _host(discard_request_mask)
    backup = _host(backup_next_token_ids)
    num_reqs, num_tokens = sampled.shape

    is_valid = (sampled != -1) & (sampled < vocab_size)
    valid_count = is_valid.sum(axis=1).astype(np.int32)

    # Index of the last valid column per row: argmax on the reversed mask finds
    # the first valid from the right. Rows with nothing valid index column 0,
    # whose value `valid_count == 0` then discards.
    last_valid_index = np.where(
        is_valid.any(axis=1),
        (num_tokens - 1) - is_valid[:, ::-1].argmax(axis=1),
        0,
    )
    last_valid_token = sampled[np.arange(num_reqs), last_valid_index]

    next_token_ids = np.where(valid_count > 0, last_valid_token, backup)
    next_token_ids = np.where(discard, backup, next_token_ids)
    valid_count = np.where(discard, 0, valid_count)

    return (
        torch.from_numpy(next_token_ids.astype(np.int32)),
        torch.from_numpy(valid_count.astype(np.int32)),
    )


def eagle_prepare_inputs_padded(
    # [num_reqs]
    cu_num_draft_tokens: torch.Tensor,
    # [num_reqs]
    valid_sampled_tokens_count: torch.Tensor,
    # [num_reqs + 1]
    query_start_loc: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    This function computes the token index to sample for each request, taking into
    account the number of draft tokens and the number of valid sampled tokens
    (which is one more than the number of accepted tokens). It also returns the
    number of rejected tokens for each request to match upstream's padded EAGLE
    input preparation contract.
    """
    cu_draft = _host(cu_num_draft_tokens)
    valid = _host(valid_sampled_tokens_count)
    qsl = _host(query_start_loc)

    # `cu_num_draft_tokens` is an inclusive cumulative sum, so the per-request
    # count is its first difference. Widen first: the difference is taken in
    # place and int32 inputs would wrap on a pathological cumsum.
    num_draft_tokens = cu_draft.astype(np.int64)
    num_draft_tokens[1:] -= cu_draft[:-1]

    num_rejected_tokens = np.where(
        num_draft_tokens > 0, num_draft_tokens + 1 - valid, 0
    ).astype(np.int32)
    token_indices_to_sample = (qsl[1:] - 1 - num_rejected_tokens).astype(np.int32)

    return (
        torch.from_numpy(token_indices_to_sample),
        torch.from_numpy(num_rejected_tokens),
    )
