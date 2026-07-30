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

# Run `eagle_prepare_next_token_padded` on the host instead of letting each of
# its integer ops fall back there one at a time.
#
# The tensors are (batch, num_spec + 1) -- 1x4 at max_num_seqs 1 -- but the
# eleven integer ops on them have no fp16 device path, so nine of them fall back
# to the host implementation and each one is its own device round trip. Copying
# the three inputs across once, doing the same arithmetic in the same order, and
# copying the two results back is 2 transfers instead of 9 round trips. Measured
# 0.55 ms/step in the `drafter/pre: next_token_ids` scope.
HOST_NEXT_TOKEN = os.getenv("VLLM_RBLN_EAGLE3_HOST_NEXT_TOKEN", "0") == "1"

# Log the drafter's proposed ids for the first N steps, so that a change claimed
# to be an equivalence can be checked against a baseline run rather than against
# acceptance statistics.
#
# Acceptance is a poor gate for this. A mis-indexed per-iteration input makes the
# drafter attend with the wrong sequence length, which does not crash and does
# not collapse acceptance -- it degrades it by a few percent, which is inside the
# arm-to-arm spread. Diffing the ids themselves is exact: with temperature 0 and
# the same cache state the sequence is deterministic, so any single differing id
# means the transform is not an equivalence.
DRAFT_ID_LOG_STEPS = int(os.getenv("VLLM_RBLN_EAGLE3_DRAFT_ID_LOG_STEPS", "0"))

# Run the drafter's chain as one compiled graph instead of three.
#
# A decode step launches four graphs today: the target's verify forward, the
# drafter's first pass (qlen num_spec+1, aux projection folded in), and the
# drafter's loop forward twice (qlen 1). The three drafter graphs exist because
# `dynamic=False` compiles per input shape and the iterations chain through
# host-visible buffers with a metadata rebuild in between.
#
# Only `seq_lens` differs between iterations. `block_tables` is loop-invariant
# because `num_lookahead_tokens = num_spec` pre-allocates the draft positions;
# `attn_masks` is None under VLLM_RBLN_FLASH_CAUSAL_ATTN; the sliding-window
# fields are None (the EAGLE3 head has no window); and the KV write derives its
# slot from `seq_lens` inside the attention kernel, so there is no slot mapping to
# thread through.
#
# The metadata reaches attention through the forward context rather than as an
# argument, so the unroll hands `set_forward_context` a LIST of per-iteration
# dicts and advances `patches.attention.set_draft_unroll_index` before each
# in-graph call. Upstream already defines that list shape for speculative
# decoding (it reads `[0]`), and the getter is one we own, so nothing in vLLM
# changes.
#
# Verify with `equiv_check.sh`, never with acceptance: a mis-indexed `seq_lens`
# makes iterations 2 and 3 attend without seeing the tokens iteration 1 wrote,
# which neither crashes nor collapses acceptance -- it costs a few percent, which
# is inside the arm-to-arm spread.
UNROLL_DRAFTER = os.getenv("VLLM_RBLN_EAGLE3_UNROLL_DRAFTER", "0") == "1"



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
    if HOST_NEXT_TOKEN and sampled_token_ids.device.type != "cpu":
        dev = sampled_token_ids.device
        nti, vc = eagle_prepare_next_token_padded(
            sampled_token_ids.cpu(),
            discard_request_mask.cpu(),
            backup_next_token_ids.cpu(),
            vocab_size,
        )
        return nti.to(dev), vc.to(dev)

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
