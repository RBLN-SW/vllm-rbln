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
# Operating-point dependent -- the sign flips with max_num_seqs. Default ON,
# matching the reference config's `--max-num-seqs 4`.
#
# MiniMax-M2.5 DP4+EP, agent workload, wall-clock (the only metric that
# reproduces at this operating point: 1.8% across two runs of one config, versus
# 6.6% for TPOT and 9.8% for throughput):
#
#   max_num_seqs 4    FUSE off costs +18.2%   32.9 min -> 38.8 min
#
# A fixed-prompt bench at max_num_seqs 1 says the opposite -- FUSE on is +0.33 ms
# TPOT in pure decode and +0.70 ms with prefill forced every round. That bench
# resolves 0.03 ms and is right about what it measures; it just cannot see this.
# At max_num_seqs 4 prefill is 63% of steps and one step carries several
# requests, which no mode of the fixed-prompt bench reproduces.
#
# So: keep it on for deployment (max_num_seqs 4+), turn it off only for
# single-sequence work, and do not re-derive the sign from a decode-only
# measurement. An earlier commit here read "+0.33 ms regression, default off" on
# exactly that basis.
#
# The drafter unroll requires this OFF, and two attempts to lift that did not
# work -- see UNROLL_DRAFTER below. At max_num_seqs 4 the two are mutually
# exclusive and this one is worth more (+18.2% versus the unroll's -5.8%).
FUSE_FIRST_FORWARD = os.getenv("VLLM_RBLN_EAGLE3_FUSE_FIRST_FORWARD", "1") == "1"

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
# MEASURED (MiniMax-M2.5 DP4+EP, max_num_seqs 1):
#   -2.69 ms TPOT, 48.26 against a 50.95 baseline whose four runs have sd 0.12.
#   Equivalence is verified, not assumed: 24/24 draft token ids identical to the
#   non-unrolled path at temperature 0, and accepted/draft identical to four
#   decimals (1.3095).
#
# Default stays OFF because of what it costs. The unrolled graph holds more
# intermediate buffers: at gpu_memory_utilization 0.6 the server fails to start
# ("Not enough memory for 90 blocks of KV cache"), and at 0.55 the KV cache is
# 63,488 tokens against 92,160 for the rolled graph -- about 31% less. That trade
# is worth taking at low concurrency and probably is not at max_num_seqs 4+,
# where several sequences compete for the same cache.
#
# Requires FUSE_FIRST_FORWARD off, and that is what limits it. Two attempts to
# make them coexist both failed, differently:
#
#   1. As written, `model_wrapper` branches on `hidden_states.shape[-1]` to decide
#      whether to fold the aux projection. Inside one unrolled region only the
#      first copy is handed a wide tensor, so the copies trace different bodies
#      and accepted/draft comes out exactly 0.
#   2. Hoisting the fold above the first `model_wrapper` call -- so every copy
#      sees a hidden_size-wide input and the branch is uniformly false -- changes
#      the failure rather than removing it. The engine dies during the request
#      with a DP collective mismatch: one rank in `execute_dummy_batch` ->
#      `_determine_batch_padding` -> `all_reduce`, another in
#      `num_tokens_across_dp`, gloo reporting the peer closed.
#   3. That looked like the drafter's skipped shape rendezvous desynchronising
#      the ranks, so the pair was run again with `SKIP_DP_RENDEZVOUS=0`. It does
#      not help: accepted/draft is back to 0. The rendezvous was not the cause.
#
# `UNROLL_DRAFTER=1` with `FUSE_FIRST_FORWARD=0` works and is exact, so all three
# failures belong to the combination, not to the unroll. Three attempts, three
# different mechanisms, root cause not identified. Treating the unroll as
# max_num_seqs 1 only.
#
# The hoist is kept: it is a no-op when FUSE is off (the width test is already
# false) and it is the clearer structure. It is not a fix.
#
# Consequence: at max_num_seqs 4, where the fold is worth +18.2% wall, the unroll
# cannot be used. Measured there anyway, with the fold off: the unroll gives
# -5.8% (38.8 -> 36.6 min) but the fold's absence costs more, so the combination
# lands 14.7% behind the deployment config. It stays useful at max_num_seqs 1.
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

# Skip the target softmax on the all-greedy rejection path.
#
# `rejection_sample` takes `target_probs.argmax(dim=-1)` and, when every request
# is greedy, returns before the recovered-token and random branches that actually
# read the distribution. `argmax(softmax(x)) == argmax(x)`, so the float32 softmax
# over [num_tokens, vocab_size] -- 200064 wide on MiniMax-M2.5 -- is computed and
# thrown away every decode step.
#
# Measured -1.34 ms TPOT on MiniMax-M2.5 DP4+EP (50.86 vs 52.20 with everything
# else fixed). The fixed-prompt bench puts two instances of the same config
# 0.01 ms apart, so that is about 9 sigma. Default on.
#
# Equivalence is exact rather than empirical: softmax is monotonic, so
# `argmax(softmax(x)) == argmax(x)`, and nothing else in the all-greedy path
# reads the distribution -- `sample_recovered_tokens` and the random branch sit
# after the early return, and the logprobs path reads `target_logits` /
# `raw_target_logits`, never `target_probs`.
SKIP_GREEDY_SOFTMAX = os.getenv("VLLM_RBLN_SKIP_GREEDY_SOFTMAX", "1") == "1"

# Use `index_select` instead of advanced indexing for the remaining 1-D row
# gathers.
#
# `_draft_ids` already records the reason: the two are equivalent for a 1-D row
# selection, but only `index_select` takes the backend's native path. The same
# pattern is still left at the two `logits[...]` gathers in the rejection sampler
# -- each pulls rows out of a [num_tokens, vocab_size] tensor, 200064 wide on
# MiniMax-M2.5, twice per decode step -- and at the pair inside the drafter's
# compiled region.
#
# Both indices are 1-D int32 (`torch.zeros(batch_size, dtype=torch.int32)` in
# vLLM's SpecDecodeMetadata), which is exactly what `index_select` wants, and it
# returns a tensor with its own storage, so the in-place update of
# `target_logits` further down stays safe.
#
# Measured: no gain. 50.84 vs a 50.97 baseline, where two runs of that unchanged
# baseline sit 0.22 apart -- the difference is smaller than the noise, so there is
# nothing here.
#
# The estimate that motivated this came from `_draft_ids`, whose docstring is
# about a gather INSIDE the drafter's compiled region. These two are in the
# rejection sampler, which runs eager, so "only `index_select` takes the
# backend's native path" does not transfer -- an argument was carried across
# contexts where it does not hold.
#
# Kept behind a default-off flag rather than reverted: the change is a correct
# equivalence and costs nothing to re-measure on a backend where the eager path
# does differ. The matching change inside the drafter region was reverted
# outright because `token_indices_to_sample` is on CPU there and `index_select`
# would force a per-step host-to-device copy.
INDEX_SELECT_GATHER = os.getenv("VLLM_RBLN_INDEX_SELECT_GATHER", "0") == "1"

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
