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
# Default off: measured a regression in both regimes on MiniMax-M2.5 DP4+EP.
#
# A fixed-prompt low-variance bench (prefix hits ~73%, so almost pure decode)
# resolves 0.03 ms across server instances and 0.15-0.26 ms across runs, which is
# 20-100x tighter than the SWE-bench agent harness (3.2-6.3 ms). Against that:
#
#   pure decode          FUSE on is +0.33 ms TPOT
#   prefill every round  FUSE on is +0.70 ms TPOT  (matched 0%-cache pair: +1.01)
#
# Folding the aux projection into the drafter's first forward was supposed to pay
# for itself on prefill, and it does not -- it is worse there than in decode. The
# same bench puts the other three flags at DEVICE_ARGMAX -3.94, NARROW_LOGITS
# -1.38, SKIP_DP_RENDEZVOUS -0.87, so this is the only one that does not earn its
# place.
#
# The code stays because the measurement covers one model and one topology, and
# because `FUSE_PREFILL` splits the two halves for anyone re-measuring. But it is
# also where `_fold_combine`, `_aux_width`, the warmup/serving shape match and two
# preflight checks come from, and three arm failures traced back to it
# (`code=201` runtime recompile, a `dummy_run` prefill mismatch, a cold-cache
# widen). Removing it outright is worth considering.
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

# Log the first N prefill steps' request/token shape.
#
# `_preprocess`'s prefill branch reshapes the whole buffer with
# `view(num_reqs, -1)`, which assumes every request in the step carries the SAME
# number of tokens. At `max_num_batched_tokens=512` the scheduler generally
# spends the whole budget on one request's chunk, so `num_reqs == 1` and the
# assumption holds by accident. Raising the budget to 2048 lets several requests
# share a prefill step with unequal chunk lengths, and then the reshape mixes
# tokens across request boundaries.
#
# That is a hypothesis, and this exists to test it before any fix: a prefill step
# with `num_reqs > 1` and `num_input_tokens % num_reqs != 0` proves the uniform
# view wrong. Measured symptom it would explain -- mnbt 2048 + EAGLE3 produces
# RepeatedFormatError on 26/26 SWE-bench instances (0/26 at mnbt 512, and 0/26
# with spec-decode off at mnbt 2048) with accepted/draft at 0.097.
PREFILL_SHAPE_LOG = int(os.getenv("VLLM_RBLN_EAGLE3_PREFILL_SHAPE_LOG", "0"))

# Reproduce the pre-fix warmup, where `dummy_run` compiled the narrow input while
# serving handed the folded wide one, forcing a runtime recompile.
#
# For measurement only. That state works exactly once per fresh cache -- the
# second start fails with `code=201 INIT_INTERNAL (Seed address mismatch)` -- so
# it is not a shippable configuration. It exists because the fix is not provably
# performance-neutral: it changes which graphs the cache holds (the buggy path
# leaves an unused narrow graph behind), and the device memory layout that
# follows moves even graphs that were not touched. Measured `graph 1/1`, whose
# code and inputs are identical either way, differs by 0.32 ms/step between the
# two. Comparing them needs both.
WARMUP_SKIP_FOLD = os.getenv("VLLM_RBLN_EAGLE3_WARMUP_SKIP_FOLD", "0") == "1"

# Whether FUSE_FIRST_FORWARD also folds the projection on prefill steps.
#
# Decode and prefill are separable: the decode graph pads to the batch bucket
# while the prefill graph takes the whole buffer, so each can fold independently.
# Splitting them is what isolates the prefill half, whose only evidence so far is
# `d/first: combine` going 46.58 -> 0.00 in region 0/0 -- a number from one region
# that says nothing about what it costs elsewhere. `aux_cat` looked the same way
# and turned out to be a regression.
FUSE_PREFILL = os.getenv("VLLM_RBLN_EAGLE3_FUSE_PREFILL", "1") == "1"



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
