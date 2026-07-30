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

import torch

# Leave the drafter's logits at draft-vocabulary width (32k) and map only the
# winning id back to the target vocabulary. With this off, every call builds a
# target-vocabulary (200k) tensor and scatters into it, as upstream does.
# Read by both the producer (patches/llama_eagle3.py) and the consumer
# (v1/spec_decode/eagle.py); they must agree.
NARROW_LOGITS = os.getenv("VLLM_RBLN_EAGLE3_NARROW_LOGITS", "0") == "1"

# Skip the drafter's DP shape rendezvous when its result provably cannot change
# the drafter's execution. See `EagleProposer._probe_dp_rendezvous_need` for the
# argument and for the two preconditions that are checked at load time.
SKIP_DP_RENDEZVOUS = os.getenv("VLLM_RBLN_EAGLE3_SKIP_DP_RENDEZVOUS", "0") == "1"

# Fold the aux-state projection into the first drafter forward, and keep the
# drafter's chosen id at int32 inside the compiled region.
#
# Both are named as "not included, still unverified end to end" in
# 3dd29cc3 (perf/eagle3-device-argmax-and-pad): "an int32 device path that
# avoids the converter's trailing i64 cast, and folding the aux-state
# projection into the first drafter forward. Both look right on the dumped
# MLIR but the paired step-time numbers are not in yet."
#
# The projection is its own compiled graph today (region 3/0, 1.17 ms/step
# measured), so folding it removes one graph launch. A 1-layer 212M drafter
# costs 1.0-1.5 ms per graph, which is mostly launch rather than compute.
FUSE_FIRST_FORWARD = os.getenv("VLLM_RBLN_EAGLE3_FUSE_FIRST_FORWARD", "0") == "1"



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
    _, num_tokens = sampled_token_ids.shape

    is_valid = (sampled_token_ids != -1) & (sampled_token_ids < vocab_size)
    valid_count = is_valid.sum(dim=1).to(torch.int32)

    token_offsets = torch.arange(num_tokens, device=sampled_token_ids.device)
    last_valid_index = torch.where(
        is_valid, token_offsets, torch.tensor(-1, device=sampled_token_ids.device)
    ).amax(dim=1)

    last_valid_token = (
        torch.where(
            token_offsets == last_valid_index.unsqueeze(1),
            sampled_token_ids,
            torch.zeros_like(sampled_token_ids),
        )
        .sum(dim=1)
        .to(torch.int32)
    )

    has_valid = valid_count > 0
    next_token_ids = torch.where(has_valid, last_valid_token, backup_next_token_ids)
    next_token_ids = torch.where(
        discard_request_mask, backup_next_token_ids, next_token_ids
    )
    valid_count = torch.where(
        discard_request_mask, torch.zeros_like(valid_count), valid_count
    )

    return next_token_ids, valid_count


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
    num_draft_tokens = cu_num_draft_tokens - torch.nn.functional.pad(
        cu_num_draft_tokens[:-1], (1, 0)
    )

    has_draft = num_draft_tokens > 0
    num_rejected_tokens = torch.where(
        has_draft,
        num_draft_tokens + 1 - valid_sampled_tokens_count,
        torch.zeros_like(valid_sampled_tokens_count),
    ).to(torch.int32)
    token_indices_to_sample = (query_start_loc[1:] - 1 - num_rejected_tokens).to(
        torch.int32
    )

    return token_indices_to_sample, num_rejected_tokens
