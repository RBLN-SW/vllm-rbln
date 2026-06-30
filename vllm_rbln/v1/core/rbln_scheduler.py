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

"""RBLN scheduler with spec-decode query backfill.

Motivation
----------
Speculative decode on RBLN otherwise needs two decode query shapes per
step: ``num_spec_tokens + 1`` for the full-spec path and ``1`` for the
boundary-induced no-spec fallback (when ``remaining_in_block`` /
``remaining_in_maxlen`` cannot fit the full window, or a variable-length
proposer such as ngram returns fewer drafts). Two shapes force two compile
variants of the decode graph, a cross-DP ``step_no_spec_required``
OR-reduce to agree on the per-step shape, and runtime branching that the
``specialized_moe_decode`` path cannot reconcile in a single graph.

Key idea
--------
The scheduler unconditionally keeps every decode step's query window at
``num_spec_tokens + 1`` by back-filling the deficit with past positions
whose KV is already in the current block::

    slide_distance = max_spec_decode_len - min(old_n, effective_remaining)

The runner prepends ``slide_distance`` past tokens to ``input_ids`` /
``positions``; the model re-runs them through attention (an idempotent KV
re-write) and their logits are pruned from ``logits_indices`` so the
sampler never sees them. Result: a single ``(batch, num_spec_tokens + 1)``
full-spec decode graph for every step that can be padded in-block.

Cross-block exception (variable-length proposers only): right after entering
a new block, a short draft would need backfill from the PREVIOUS block,
which the single-block decode path can't express. That step falls back to
no-spec (query_len = 1) via the ``step_no_spec_required`` cross-DP OR-reduce
(any rank tripping it -> every rank drops to no-spec). Fixed-length
proposers (EAGLE/EAGLE3/MEDUSA) stay full-spec only: ``num_spec + 1 <=
block_size`` keeps their backfill in-block at every interior boundary. The
sole exception is the ``max_model_len`` edge -- decode is capped to
``max_model_len - 1`` (reserved final position), so a final partial block of
size ``m = max_model_len % block_size <= num_spec + 1`` squeezes even a full
draft into a cross-block window. That is unsupported for fixed length and a
runtime assert rejects it (align ``max_model_len`` to ``block_size``).

Decode input shaping pipeline
-----------------------------
In short: the **scheduler** unifies query length WITHIN a rank; the
**runner** then does cross-DP query-length reconciliation AND batch padding.
These three stages together make every DP rank feed the compiled decode
graph an identical ``(batch_bucket, query_len)`` tensor:

1. Scheduler (this file) -- intra-rank query-length unification. Backfill
   sets every decode req's query window to ``num_spec_tokens + 1`` (or 1 on
   the no-spec fallback) within this rank. BOTH paths that enter the decode
   batch go through ``_decide_spec_slide``: the running loop AND the
   ``WAITING_FOR_REMOTE_KVS`` promotion path (a remote-prefilled req joins
   decode carrying 0 drafts, so it needs the same backfill). After this stage
   a rank's flat ``input_ids`` is already ``num_reqs * query_len`` -- uniform
   and reshape-safe. (Prefill never mixes with decode in one rank: the
   no-mixed-batching eviction keeps a prefill step at ``num_reqs == 1``.)
2. Runner -- cross-DP reconciliation (query-length axis). ``any_prefill``
   (a DP PEER prefilling) selects the prefill-sized shape; otherwise
   ``pad_speculative_draft_tokens`` lifts each rank's uniform window up to
   ``max_tokens_per_req_across_dp`` (the cross-DP max of per-rank
   ``spec_decode_max_query_len``) so all ranks agree on ``query_len``. This
   stage assumes stage 1's intra-rank uniformity -- it does NOT equalize
   per-req lengths within a rank.
3. Runner -- batch padding (batch axis). ``input_ids.view(num_reqs, -1)``
   then ``pad(..., batch_bucket_size)`` pads the batch dimension up to the
   cross-DP max ``num_reqs`` bucket (e.g. 7 -> 8).

If stage 1 leaves a rank non-uniform (e.g. a promoted req that skipped
backfill) AND the draft dict is empty (so the runner's pad is gated off),
stage 3's ``view(num_reqs, -1)`` crashes on the non-divisible flat length --
the regression this scheduler's promotion-path backfill prevents.

Compile coverage & guards
-------------------------
Because ``step_no_spec_required`` can fire at runtime, warmup must compile
BOTH decode graphs for variable-length proposers: full-spec
``(batch, num_spec+1)`` and no-spec ``(batch, 1)``. A decode graph's compile
key has TWO axes -- ``input_ids`` shape AND ``max_pads_across_dp`` size
(``num_padded``) -- so warmup must reproduce the runtime no-spec graph on
both. Setting ``query_len = 1`` only fixes the first axis; ``num_padded`` is
derived from ``step_no_spec_required`` (via ``spec_decode_max_query_len``),
so the warmup dummy drives the SAME runtime path by emitting an
``RBLNSchedulerOutput(step_no_spec_required=True)`` rather than hand-mimicking
inputs. Otherwise the no-spec step hot-path recompiles at runtime.

Two runtime asserts protect the in-block invariant against silent KV
corruption: (1) no backfill window may cross a block boundary
(``slide <= tokens_used_in_block``), and (2) once no-spec is elected the
scrub must clear all slide and force every ``query_len = 1``.

Naming: the mechanism is **query backfill**. Some code-level identifiers
(``slide_distance``, ``spec_decode_slide_distance``, ...) still carry the
older ``slide``/``sliding`` naming for stability across modules, but they
refer to query backfill -- not a separate mechanism.
"""

import itertools
import time
from dataclasses import dataclass, field
from typing import Any

from vllm.distributed.ec_transfer.ec_connector.base import ECConnectorMetadata
from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorMetadata
from vllm.utils.hashing import get_hash_fn_by_name
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_utils import init_none_hash
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType, EngineCoreOutputs
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus
from vllm.v1.utils import record_function_or_nullcontext

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.rbln_kv_cache_manager import (
    KVCacheCopyOp,
    RBLNKVCacheManager,
    SubBlockMatch,
)

logger = init_logger(__name__)


@dataclass
class RBLNSchedulerOutput(SchedulerOutput):
    """SchedulerOutput extended with KV cache copy operations for sub-block
    prefix caching, a legacy boundary-induced no-spec flag, and per-request
    sliding-window slide distances for fixed-length spec decode.

    Naming: "sliding-window" and "query backfill" refer to the same
    mechanism. Code-level identifiers below (``spec_decode_slide_distance``,
    ``slide_distance``, etc.) keep the ``slide``/``sliding`` naming for
    stability; user-facing log prefixes and docs use ``backfill``.
    """

    kv_cache_copy_ops: list[KVCacheCopyOp] = field(default_factory=list)
    # NOTE(RBLN): Set True when at least one running decode req cannot reach
    # full spec via in-block query backfill — i.e. it just entered a new block
    # and a variable-length proposer's draft would need backfill from the
    # PREVIOUS block (cross-block, unsupported). The runner OR-reduces this
    # across DP: if ANY rank set it, EVERY rank drops to no-spec (query_len=1)
    # this step. Stays False for fixed-length proposers (their backfill is
    # always in-block) and for any step paddable in-block.
    step_no_spec_required: bool = False
    # NOTE(RBLN): Per-request sliding-window slide distance. For a decode
    # req whose remaining_in_block (or remaining_in_maxlen) cannot fit the
    # full num_spec_tokens+1 query window, the runner prepends
    # `slide_distance` past tokens (from request.output_token_ids) so the
    # entire query window stays within already-allocated KV slots. Past
    # positions' logits are discarded; past KV gets idempotently
    # re-written. Empty / missing key => slide_distance is 0 (no slide).
    spec_decode_slide_distance: dict[str, int] = field(default_factory=dict)


def is_prefill(request: Request) -> bool:
    return request.num_computed_tokens < request.num_tokens - 1


class RBLNScheduler(Scheduler):
    def __init__(
        self,
        *args,
        sub_block_size: int | None = None,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        # Replace the upstream KVCacheManager with RBLNKVCacheManager
        # when sub-block prefix caching is enabled.
        # Sub-block size equals the prefill chunk size (max_num_batched_tokens)
        # so that each prefill does not span multiple blocks.
        if sub_block_size is None and envs.VLLM_RBLN_SUB_BLOCK_CACHE:
            sub_block_size = self.scheduler_config.max_num_batched_tokens
        if (
            self.cache_config.enable_prefix_caching
            and sub_block_size
            and RBLNKVCacheManager.can_use_sub_block_caching(
                self.kv_cache_config, sub_block_size
            )
        ):
            hash_fn = get_hash_fn_by_name(self.cache_config.prefix_caching_hash_algo)
            init_none_hash(hash_fn)

            self.kv_cache_manager = RBLNKVCacheManager(
                kv_cache_config=self.kv_cache_config,
                max_model_len=self.max_model_len,
                hash_block_size=self.block_size,
                sub_block_size=sub_block_size,
                hash_fn=hash_fn,
                use_eagle=self.use_eagle,
                log_stats=self.log_stats,
                enable_kv_cache_events=self.enable_kv_cache_events,
                dcp_world_size=self.dcp_world_size,
                pcp_world_size=self.pcp_world_size,
                metrics_collector=self.kv_metrics_collector,
            )

            logger.info(
                "Sub-block prefix caching enabled: block_size=%d, sub_block_size=%d",
                self.block_size,
                sub_block_size,
            )
            if self.enable_kv_cache_events:
                logger.info(
                    "NOTE that KV cache events emit at sub_block_size granularity. "
                    "Cache-aware routers must set token processing block size to "
                    "sub_block_size=%d.",
                    sub_block_size,
                )

        # NOTE(RBLN): Blocks allocated for a running (decode) request that is
        # then evicted by the "disable mixed batching" path (see schedule())
        # are already committed in the KV-cache coordinator but never reach the
        # model runner, because the evicted request is dropped from this step's
        # output. On the next step `allocate_slots` returns an empty delta (the
        # coordinator already holds the block), so the block id would be lost
        # to the runner forever — leaving a stale block-id 0 in the request's
        # block table. This is observable for block-aligned prompts, whose
        # second block is allocated exactly on the prefill->decode transition
        # step that the eviction targets. We stash the dropped delta here and
        # re-emit it on the request's next scheduled step. Keyed by request_id.
        self._stranded_new_blocks: dict[str, KVCacheBlocks] = {}

    def _decide_spec_slide(self, request: Request, new_n: int) -> tuple[int, bool]:
        """Sliding-window backfill decision for one decode request's query
        window so it reaches ``num_spec_tokens + 1``.

        Returns ``(slide_distance, cross_block_no_spec)``:
          * ``(slide_distance > 0, False)`` -- in-block backfill: prepend this
            many already-committed past positions to pad the query window (full
            spec kept).
          * ``(0, True)`` -- the backfill window would cross the current KV
            block boundary, so full spec is impossible this step; the caller
            must elect no-spec (query_len=1). Fixed-length proposers must never
            reach this and fail loudly here.
          * ``(0, False)`` -- no backfill needed (window already full).

        Shared by the running loop and the WAITING_FOR_REMOTE_KVS promotion path
        so both unify decode query windows identically. ``new_n`` is the
        boundary-capped advance for this step (>= 1; == 1 for a promoted
        single-token decode).
        """
        desired_slide = (self.num_spec_tokens + 1) - new_n
        if desired_slide <= 0:
            return 0, False
        tokens_used_in_block = request.num_computed_tokens % self.block_size
        if desired_slide > tokens_used_in_block:
            # The backfill window reaches back into the PREVIOUS block, which the
            # single-block decode path cannot express. Variable-length proposers
            # (ngram/suffix) fall back to no-spec (query_len=1); fixed-length
            # proposers compile no no-spec graph and their DP-idle peers vote
            # num_spec+1, so they must stay full-spec-only and are rejected.
            assert self.vllm_config.speculative_config.method in (
                "ngram",
                "suffix",
            ), (
                "cross-block no-spec fallback fired for fixed-length proposer "
                f"'{self.vllm_config.speculative_config.method}'; "
                "fixed-length spec decode must stay full-spec-only. "
                "Cause: the final block is too short for a full speculative "
                "window once the scheduler reserves the last position (decode "
                "capped to max_model_len-1), i.e. 0 < max_model_len % block_size "
                "<= num_spec_tokens+1 and a request reached the final block. "
                "Align max_model_len to block_size (or keep "
                "max_model_len % block_size > num_spec_tokens+1)."
            )
            return 0, True
        return desired_slide, False

    def schedule(self) -> RBLNSchedulerOutput:
        # Copied from vllm.v1.core.sched.Scheduler.schedule: https://github.com/vllm-project/vllm/blob/v0.18.0/vllm/v1/core/sched/scheduler.py#L338-L927
        # The only differences are:
        # - Disable mixed batching
        # - Limit prefill batch size to 1
        # - Limit decode batch size to (max_num_seqs // pipeline_parallel_size)
        # - Do not schedule spec tokens when they are across block boundaries
        # Search for NOTE(RBLN) for details

        # NOTE(woosuk) on the scheduling algorithm:
        # There's no "decoding phase" nor "prefill phase" in the scheduler.
        # Each request just has the num_computed_tokens and
        # num_tokens_with_spec. num_tokens_with_spec =
        # len(prompt_token_ids) + len(output_token_ids) + len(spec_token_ids).
        # At each step, the scheduler tries to assign tokens to the requests
        # so that each request's num_computed_tokens can catch up its
        # num_tokens_with_spec. This is general enough to cover
        # chunked prefills, prefix caching, speculative decoding,
        # and the "jump decoding" optimization in the future.

        scheduled_new_reqs: list[Request] = []
        scheduled_resumed_reqs: list[Request] = []
        scheduled_running_reqs: list[Request] = []
        preempted_reqs: list[Request] = []

        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        # Encoder-related.
        scheduled_encoder_inputs: dict[str, list[int]] = {}
        encoder_compute_budget = self.max_num_encoder_input_tokens
        # Spec decode-related.
        scheduled_spec_decode_tokens: dict[str, list[int]] = {}

        # For logging.
        scheduled_timestamp = time.monotonic()

        self.kv_cache_manager.new_step_starts()

        # NOTE(RBLN): Set True below if any decode req cannot reach full spec
        # via in-block query backfill (cross-block, variable-length only); the
        # runner OR-reduces it across DP so every rank drops to no-spec that
        # step. Stays False when every backfill is in-block.
        step_no_spec_required = False

        # NOTE(RBLN): Per-request sliding-window slide distances. For a
        # running decode req whose remaining_in_block / remaining_in_maxlen
        # budget can't fit a full num_spec_tokens+1 query window, we record
        # how many past positions the runner must prepend to the query
        # window so it stays within already-allocated KV slots. Populated
        # below as we walk running reqs and detect boundary cases.
        spec_decode_slide_distance: dict[str, int] = {}

        # First, schedule the RUNNING requests.
        # NOTE(RBLN): Prioritize prefill requests. Given our constraint that the prefill
        # batch size fixed to 1 if any prefill request is running there must be exactly
        # one at the end of the list.
        req_index = (
            len(self.running) - 1
            if self.running and is_prefill(self.running[-1])
            else 0
        )
        while req_index < len(self.running) and token_budget > 0:
            request = self.running[req_index]

            if (
                request.num_output_placeholders > 0
                # This is (num_computed_tokens + 1) - (num_output_placeholders - 1).
                # Since output placeholders are also included in the computed tokens
                # count, we subtract (num_output_placeholders - 1) to remove any draft
                # tokens, so that we can be sure no further steps are needed even if
                # they are all rejected.
                and request.num_computed_tokens + 2 - request.num_output_placeholders
                >= request.num_prompt_tokens + request.max_tokens
            ):
                # Async scheduling: Avoid scheduling an extra step when we are sure that
                # the previous step has reached request.max_tokens. We don't schedule
                # partial draft tokens since this prevents uniform decode optimizations.
                req_index += 1
                continue

            num_new_tokens = (
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens
            )
            if 0 < self.scheduler_config.long_prefill_token_threshold < num_new_tokens:
                num_new_tokens = self.scheduler_config.long_prefill_token_threshold
            num_new_tokens = min(num_new_tokens, token_budget)

            # Make sure the input position does not exceed the max model len.
            # This is necessary when using spec decoding.
            num_new_tokens = min(
                num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
            )

            # Schedule encoder inputs.
            encoder_inputs_to_schedule = None
            external_load_encoder_input: list[int] = []
            new_encoder_compute_budget = encoder_compute_budget
            if request.has_encoder_inputs:
                (
                    encoder_inputs_to_schedule,
                    num_new_tokens,
                    new_encoder_compute_budget,
                    external_load_encoder_input,
                ) = self._try_schedule_encoder_inputs(
                    request,
                    request.num_computed_tokens,
                    num_new_tokens,
                    encoder_compute_budget,
                    shift_computed_tokens=1 if self.use_eagle else 0,
                )

            if self.need_mamba_block_aligned_split:
                num_new_tokens = self._mamba_block_aligned_split(
                    request, num_new_tokens
                )

            if num_new_tokens == 0:
                # The request cannot be scheduled because one of the following
                # reasons:
                # 1. No new tokens to schedule. This may happen when
                #    (1) PP>1 and we have already scheduled all prompt tokens
                #    but they are not finished yet.
                #    (2) Async scheduling and the request has reached to either
                #    its max_total_tokens or max_model_len.
                # 2. The encoder budget is exhausted.
                # 3. The encoder cache is exhausted.
                # 4. Insufficient budget for a block-aligned chunk in hybrid
                #    models with mamba cache mode \"align\".
                # NOTE(woosuk): Here, by doing `continue` instead of `break`,
                # we do not strictly follow the FCFS scheduling policy and
                # allow the lower-priority requests to be scheduled.
                req_index += 1
                continue

            # Schedule newly needed KV blocks for the request.
            with record_function_or_nullcontext("schedule: allocate_slots"):
                while True:
                    new_blocks = self.kv_cache_manager.allocate_slots(
                        request,
                        num_new_tokens,
                        num_lookahead_tokens=self.num_lookahead_tokens,
                        # NOTE(RBLN): Cache blocks only after scheduling is finalized.
                        delay_cache_blocks=True,
                    )

                    if new_blocks is not None:
                        # The request can be scheduled.
                        break

                    # The request cannot be scheduled.
                    # Preempt the lowest-priority request.
                    if self.policy == SchedulingPolicy.PRIORITY:
                        preempted_req = max(
                            self.running,
                            key=lambda r: (r.priority, r.arrival_time),
                        )
                        self.running.remove(preempted_req)
                        if preempted_req in scheduled_running_reqs:
                            preempted_req_id = preempted_req.request_id
                            scheduled_running_reqs.remove(preempted_req)
                            token_budget += num_scheduled_tokens.pop(preempted_req_id)
                            req_to_new_blocks.pop(preempted_req_id)
                            scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                            preempted_encoder_inputs = scheduled_encoder_inputs.pop(
                                preempted_req_id, None
                            )
                            if preempted_encoder_inputs:
                                # Restore encoder compute budget if the preempted
                                # request had encoder inputs scheduled in this step.
                                num_embeds_to_restore = sum(
                                    preempted_req.get_num_encoder_embeds(i)
                                    for i in preempted_encoder_inputs
                                )
                                encoder_compute_budget += num_embeds_to_restore
                            req_index -= 1
                    else:
                        preempted_req = self.running.pop()

                    self._preempt_request(preempted_req, scheduled_timestamp)
                    preempted_reqs.append(preempted_req)
                    if preempted_req == request:
                        # No more request to preempt. Cannot schedule this request.
                        break

            if new_blocks is None:
                # Cannot schedule this request.
                break

            # Schedule the request.
            scheduled_running_reqs.append(request)
            request_id = request.request_id
            req_to_new_blocks[request_id] = new_blocks
            num_scheduled_tokens[request_id] = num_new_tokens
            token_budget -= num_new_tokens
            req_index += 1

            # Speculative decode related.
            if request.spec_token_ids:
                num_scheduled_spec_tokens = (
                    num_new_tokens
                    + request.num_computed_tokens
                    - request.num_tokens
                    - request.num_output_placeholders
                )
                if num_scheduled_spec_tokens > 0:
                    spec_token_ids = request.spec_token_ids
                    if len(spec_token_ids) > num_scheduled_spec_tokens:
                        spec_token_ids = spec_token_ids[:num_scheduled_spec_tokens]
                    scheduled_spec_decode_tokens[request.request_id] = spec_token_ids

                # New spec tokens will be set in `update_draft_token_ids` before the
                # next step when applicable.
                request.spec_token_ids = []

            # NOTE(RBLN): Sliding-window per-request boundary decision.
            # When a decode req's remaining block / maxlen can no longer
            # hold a full num_spec_tokens+1 window, slide the query window
            # backward by `desired_slide` past positions whose KV is
            # already in the current block. The runner re-feeds those past
            # tokens (idempotent KV re-write) and drops their logits from
            # sampling, so the effective advance per step is capped at
            # `effective_remaining` while no KV write crosses the block
            # boundary. Drafts that would land past the boundary get
            # trimmed in `scheduled_spec_decode_tokens`. Peers in the
            # same batch that aren't near a boundary keep running full
            # spec — the decision is purely per-request.
            if not is_prefill(request) and self.num_spec_tokens > 0:
                # Unified sliding-window decision: pad every decode
                # req's query window to a fixed `num_spec_tokens + 1`,
                # regardless of how many drafts the proposer returned
                # or whether the req is near a block boundary.
                #
                # - Variable-length proposers (ngram, suffix decoding)
                #   that return fewer than num_spec_tokens drafts → the
                #   shortage is filled with past positions (sliding
                #   pads the length deficit, logits at past positions
                #   are discarded).
                # - Fixed-length proposers (MTP, EAGLE) → no padding
                #   needed off the boundary; sliding only fires when
                #   the boundary squeezes effective_remaining below
                #   num_spec_tokens + 1.
                # - Boundary + variable-length combine naturally:
                #   `new_n = min(old_n, effective_remaining)` caps
                #   actual advance, and `desired_slide` pads the rest.
                #
                # The runner sees a single decode shape
                # (batch, num_spec_tokens + 1) — no_spec compile
                # variants and runtime branching are eliminated.
                max_spec_decode_len = self.num_spec_tokens + 1
                tokens_used_in_block = request.num_computed_tokens % self.block_size
                remaining_in_block = self.block_size - tokens_used_in_block
                remaining_in_maxlen = self.max_model_len - request.num_computed_tokens
                effective_remaining = min(remaining_in_block, remaining_in_maxlen)

                req_id = request.request_id
                old_n = num_scheduled_tokens[req_id]
                # Cap actual advance by boundary, then pad query
                # window length via slide.
                new_n = min(old_n, effective_remaining)
                desired_slide = max_spec_decode_len - new_n

                if desired_slide > 0:
                    assert effective_remaining >= 1, (
                        f"effective_remaining must be >= 1; req {req_id} has "
                        f"remaining_in_block={remaining_in_block}, "
                        f"remaining_in_maxlen={remaining_in_maxlen}"
                    )

                # Unified sliding-window decision (shared with the
                # WAITING_FOR_REMOTE_KVS promotion path below).
                slide_distance, cross_block_no_spec = self._decide_spec_slide(
                    request, new_n
                )
                if cross_block_no_spec:
                    # Backfill window crosses the KV block boundary -> full spec
                    # impossible this step; elect no-spec (query_len=1) and let
                    # the runner's cross-DP OR-reduce scrub slide on every rank.
                    # (`_decide_spec_slide` already rejected fixed-length
                    # proposers, which must never reach a cross-block step.)
                    step_no_spec_required = True
                    logger.debug(
                        "spec-decode no-spec fallback (cross-block "
                        "backfill): req=%s num_computed=%d "
                        "tokens_used_in_block=%d slide=%d proposed_drafts=%d",
                        req_id,
                        request.num_computed_tokens,
                        tokens_used_in_block,
                        desired_slide,
                        max(old_n - 1, 0),
                    )
                elif slide_distance > 0:
                    # In-block backfill keeps full spec.
                    spec_decode_slide_distance[req_id] = slide_distance
                    # Diagnostic log so end-to-end runs make the
                    # sliding decision observable. `proposed_drafts` is
                    # what the proposer returned BEFORE any sliding-induced
                    # trim (= old_n - 1); compare against `kept_drafts`
                    # (= max(new_n - 1, 0)): proposed > kept ⇒ boundary
                    # forced some drafts out; equal ⇒ only length padding.
                    logger.debug(
                        "spec-decode backfill: req=%s "
                        "num_computed=%d remaining_in_block=%d "
                        "remaining_in_maxlen=%d slide=%d "
                        "advance=%d proposed_drafts=%d kept_drafts=%d",
                        req_id,
                        request.num_computed_tokens,
                        remaining_in_block,
                        remaining_in_maxlen,
                        slide_distance,
                        new_n,
                        max(old_n - 1, 0),
                        max(new_n - 1, 0),
                    )

                    if old_n > new_n:
                        token_budget += old_n - new_n
                        num_scheduled_tokens[req_id] = new_n
                        num_spec = (
                            new_n
                            + request.num_computed_tokens
                            - request.num_tokens
                            - request.num_output_placeholders
                        )
                        if num_spec > 0 and req_id in scheduled_spec_decode_tokens:
                            scheduled_spec_decode_tokens[req_id] = (
                                scheduled_spec_decode_tokens[req_id][:num_spec]
                            )
                        elif num_spec <= 0:
                            scheduled_spec_decode_tokens.pop(req_id, None)

            # Encoder-related.
            if encoder_inputs_to_schedule:
                scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                # Allocate the encoder cache.
                for i in encoder_inputs_to_schedule:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)
                encoder_compute_budget = new_encoder_compute_budget
            if external_load_encoder_input:
                for i in external_load_encoder_input:
                    self.encoder_cache_manager.allocate(request, i)
                    if self.ec_connector is not None:
                        self.ec_connector.update_state_after_alloc(request, i)

            # NOTE(RBLN): We restrict the decode batch size to
            # (max_num_seqs // pipeline_parallel_size) to prevent pipeline
            # bubbles.
            if len(scheduled_running_reqs) >= (
                self.max_num_running_reqs
                // self.vllm_config.parallel_config.pipeline_parallel_size
            ):
                break

        # NOTE(RBLN): The legacy retroactive trim using spec_decode_cap is
        # gone — under sliding-window scheduling each boundary-affected
        # req has already been adjusted in-line (num_scheduled_tokens /
        # scheduled_spec_decode_tokens trimmed when boundary detected, and
        # slide_distance recorded for the runner to prepend past tokens).
        # Reqs that fit a full num_spec_tokens+1 window are untouched.

        # Record the LoRAs in scheduled_running_reqs
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        # Next, schedule the WAITING requests.
        # NOTE(RBLN): We do not attempt to schedule a new prefill request when a running
        # prefill request is already scheduled.
        if (
            not preempted_reqs
            and self._pause_state == PauseState.UNPAUSED
            and not (scheduled_running_reqs and is_prefill(scheduled_running_reqs[0]))
        ):
            # NOTE(RBLN): refresh the token budget to determine whether we can schedule
            # new prefill requests into the running batch.
            prefill_token_budget = self.max_num_scheduled_tokens

            step_skipped_waiting = create_request_queue(self.policy)
            sub_block_match = None

            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break

                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None

                request = request_queue.peek_request()
                request_id = request.request_id

                promoted_from_waiting_for_remote_kvs = (
                    request.status == RequestStatus.WAITING_FOR_REMOTE_KVS
                )

                # try to promote blocked statuses while traversing skipped queue.
                if self._is_blocked_waiting_status(
                    request.status
                ) and not self._try_promote_blocked_waiting_request(request):
                    if request.status == RequestStatus.WAITING_FOR_REMOTE_KVS:
                        logger.debug(
                            "%s is still in WAITING_FOR_REMOTE_KVS state.",
                            request_id,
                        )
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                # Check that adding the request still respects the max_loras
                # constraint.
                if (
                    self.lora_config
                    and request.lora_request
                    and (
                        len(scheduled_loras) == self.lora_config.max_loras
                        and request.lora_request.lora_int_id not in scheduled_loras
                    )
                ):
                    # Scheduling would exceed max_loras, skip.
                    request_queue.pop_request()
                    step_skipped_waiting.prepend_request(request)
                    continue

                num_external_computed_tokens = 0
                load_kv_async = False
                connector_prefix_cache_queries, connector_prefix_cache_hits = 0, 0
                sub_block_match = None
                num_sub_block_tokens = 0

                # Get already-cached tokens.
                if request.num_computed_tokens == 0:
                    # Get locally-cached tokens (full-block matches only).
                    new_computed_blocks, num_new_local_computed_tokens = (
                        self.kv_cache_manager.get_computed_blocks(request)
                    )

                    # Get externally-cached tokens if using a KVConnector.
                    if self.connector is not None:
                        ext_tokens, load_kv_async = (
                            self.connector.get_num_new_matched_tokens(
                                request, num_new_local_computed_tokens
                            )
                        )

                        if ext_tokens is None:
                            # The request cannot be scheduled because
                            # the KVConnector couldn't determine
                            # the number of matched tokens.
                            request_queue.pop_request()
                            step_skipped_waiting.prepend_request(request)
                            continue

                        request.num_external_computed_tokens = ext_tokens
                        num_external_computed_tokens = ext_tokens

                        connector_prefix_cache_queries = (
                            request.num_tokens - num_new_local_computed_tokens
                        )
                        connector_prefix_cache_hits = num_external_computed_tokens

                    # NOTE(RBLN): Arbitrate between sub-block match and KV connector.
                    sub_block_match, num_sub_block_tokens = self._try_sub_block_match(
                        request,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_sub_block_tokens > 0 and num_external_computed_tokens > 0:
                        # Cancel the KV connector match in favor of the sub-block match
                        request.num_external_computed_tokens = 0
                        num_external_computed_tokens = 0
                        load_kv_async = False
                        connector_prefix_cache_hits = 0

                    # Total computed tokens (local + external).
                    num_computed_tokens = (
                        num_new_local_computed_tokens
                        + num_sub_block_tokens
                        + num_external_computed_tokens
                    )
                    assert num_computed_tokens <= request.num_tokens

                    # Track first scheduled prefill, not post-preemption repeats.
                    if request.prefill_stats is not None:
                        assert num_computed_tokens <= request.num_prompt_tokens
                        request.prefill_stats.set(
                            num_prompt_tokens=request.num_prompt_tokens,
                            num_local_cached_tokens=(
                                num_new_local_computed_tokens + num_sub_block_tokens
                            ),
                            num_external_cached_tokens=num_external_computed_tokens,
                        )
                else:
                    # KVTransfer: WAITING reqs have num_computed_tokens > 0
                    # after async KV recvs are completed.
                    new_computed_blocks = self.kv_cache_manager.empty_kv_cache_blocks
                    num_new_local_computed_tokens = 0
                    num_computed_tokens = request.num_computed_tokens

                encoder_inputs_to_schedule = None
                external_load_encoder_input = []
                new_encoder_compute_budget = encoder_compute_budget

                if load_kv_async:
                    # KVTransfer: loading remote KV, do not allocate for new work.
                    assert num_external_computed_tokens > 0
                    num_new_tokens = 0
                else:
                    # Number of tokens to be scheduled.
                    # We use `request.num_tokens` instead of
                    # `request.num_prompt_tokens` to consider the resumed
                    # requests, which have output tokens.
                    num_new_tokens = request.num_tokens - num_computed_tokens
                    threshold = self.scheduler_config.long_prefill_token_threshold
                    if 0 < threshold < num_new_tokens:
                        num_new_tokens = threshold

                    # chunked prefill has to be enabled explicitly to allow
                    # pooling requests to be chunked
                    if (
                        not self.scheduler_config.enable_chunked_prefill
                        and num_new_tokens > token_budget
                    ):
                        # If chunked_prefill is disabled,
                        # we can stop the scheduling here.
                        break

                    # NOTE(RBLN): Use prefill_token_budget instead of
                    # token_budget. Running decode requests may have already
                    # consumed part of token_budget, but they will be kicked
                    # out when this new prefill is scheduled (see the
                    # "disable mixed batching" block below), restoring the
                    # full budget. Using token_budget here would clip the
                    # first prefill chunk short (e.g. 127 instead of 128).
                    num_new_tokens = min(num_new_tokens, prefill_token_budget)
                    assert num_new_tokens > 0

                    if (
                        not promoted_from_waiting_for_remote_kvs
                        and len(scheduled_new_reqs) > 0
                    ):
                        # NOTE(RBLN): promoted_from_waiting_for_remote_kvs is False, so
                        # this waiting request needs local prefill (not remote prefill).
                        # scheduled_new_reqs is non-empty because a prior iteration of
                        # this waiting loop already added a request to the decode batch
                        # from remote prefill.
                        # In this case, we defer scheduling this local prefill request
                        # (waiting request) to the next step.
                        assert len(scheduled_resumed_reqs) == 0
                        break

                    # Schedule encoder inputs.
                    if request.has_encoder_inputs:
                        (
                            encoder_inputs_to_schedule,
                            num_new_tokens,
                            new_encoder_compute_budget,
                            external_load_encoder_input,
                        ) = self._try_schedule_encoder_inputs(
                            request,
                            num_computed_tokens,
                            num_new_tokens,
                            encoder_compute_budget,
                            shift_computed_tokens=1 if self.use_eagle else 0,
                        )
                        if num_new_tokens == 0:
                            # The request cannot be scheduled.
                            break

                if self.need_mamba_block_aligned_split:
                    num_new_tokens = self._mamba_block_aligned_split(
                        request,
                        num_new_tokens,
                        num_new_local_computed_tokens,
                        num_external_computed_tokens,
                    )
                    if num_new_tokens == 0:
                        break

                # Handles an edge case when P/D Disaggregation
                # is used with Spec Decoding where an
                # extra block gets allocated which
                # creates a mismatch between the number
                # of local and remote blocks.
                effective_lookahead_tokens = (
                    0 if request.num_computed_tokens == 0 else self.num_lookahead_tokens
                )

                # Determine if we need to allocate cross-attention blocks.
                num_encoder_tokens = 0
                if (
                    self.is_encoder_decoder
                    and request.has_encoder_inputs
                    and encoder_inputs_to_schedule
                ):
                    num_encoder_tokens = sum(
                        request.get_num_encoder_embeds(i)
                        for i in encoder_inputs_to_schedule
                    )

                # NOTE(RBLN): Even when chunked prefill is enabled, we should schedule
                # a new prefill request only if there is enough KV cache space to
                # accommodate the full token count. Therefore, we allocate based on
                # request.num_tokens - num_computed_tokens, not num_new_tokens.
                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    request.num_tokens - num_computed_tokens,
                    num_new_computed_tokens=(
                        num_new_local_computed_tokens + num_sub_block_tokens
                    ),
                    new_computed_blocks=new_computed_blocks,
                    num_lookahead_tokens=effective_lookahead_tokens,
                    num_external_computed_tokens=num_external_computed_tokens,
                    # NOTE(RBLN): Cache blocks only after scheduling is finalized.
                    delay_cache_blocks=True,
                    num_encoder_tokens=num_encoder_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.

                    # NOTE: we need to untouch the request from the encode cache
                    # manager
                    if request.has_encoder_inputs:
                        self.encoder_cache_manager.free(request)
                    break

                # NOTE(RBLN): Apply sub-block match now that blocks are
                # allocated (the destination block exists).
                if sub_block_match is not None:
                    self.kv_cache_manager.apply_sub_block_match(sub_block_match)
                    sub_block_match = None

                # KVTransfer: the connector uses this info to determine
                # if a load is needed. Note that
                # This information is used to determine if a load is
                # needed for this request.
                if self.connector is not None:
                    self.connector.update_state_after_alloc(
                        request,
                        self.kv_cache_manager.get_blocks(request_id),
                        num_external_computed_tokens,
                    )
                    if (
                        self.connector_prefix_cache_stats is not None
                        and connector_prefix_cache_queries != 0
                    ):
                        self.connector_prefix_cache_stats.record(
                            num_tokens=connector_prefix_cache_queries,
                            num_hits=connector_prefix_cache_hits,
                            preempted=request.num_preemptions > 0,
                        )

                request = request_queue.pop_request()
                if load_kv_async:
                    # If loading async, allocate memory and put request
                    # into the WAITING_FOR_REMOTE_KV state.
                    request.status = RequestStatus.WAITING_FOR_REMOTE_KVS
                    step_skipped_waiting.prepend_request(request)
                    # Set num_computed_tokens even though KVs are not yet loaded.
                    # request.num_computed_tokens will not be used anywhere until
                    # the request finished the KV transfer.
                    #
                    # If a transfer error is reported by the connector,
                    # request.num_computed_tokens will be re-set accordingly in
                    # _update_requests_with_invalid_blocks.
                    #
                    # When the transfer is finished, either successfully or not,
                    # request.num_computed_tokens will correctly reflect the number
                    # of computed tokens.
                    # _update_waiting_for_remote_kv will then cache
                    # only the successfully loaded tokens.
                    request.num_computed_tokens = num_computed_tokens
                    continue

                self.running.append(request)
                if self.log_stats:
                    request.record_event(
                        EngineCoreEventType.SCHEDULED, scheduled_timestamp
                    )
                if request.status == RequestStatus.WAITING:
                    scheduled_new_reqs.append(request)
                elif request.status == RequestStatus.PREEMPTED:
                    scheduled_resumed_reqs.append(request)
                else:
                    raise RuntimeError(f"Invalid request status: {request.status}")

                if self.lora_config and request.lora_request:
                    scheduled_loras.add(request.lora_request.lora_int_id)
                req_to_new_blocks[request_id] = self.kv_cache_manager.get_blocks(
                    request_id
                )
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                request.num_computed_tokens = num_computed_tokens
                # Encoder-related.
                if encoder_inputs_to_schedule:
                    scheduled_encoder_inputs[request_id] = encoder_inputs_to_schedule
                    # Allocate the encoder cache.
                    for i in encoder_inputs_to_schedule:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)
                    encoder_compute_budget = new_encoder_compute_budget
                # Allocate for external load encoder cache
                if external_load_encoder_input:
                    for i in external_load_encoder_input:
                        self.encoder_cache_manager.allocate(request, i)
                        if self.ec_connector is not None:
                            self.ec_connector.update_state_after_alloc(request, i)

                if promoted_from_waiting_for_remote_kvs:
                    # NOTE(RBLN): A request promoted from WAITING_FOR_REMOTE_KVS is
                    # decode-ready: its prompt KV was prefilled on the remote
                    # (producer) instance, so exactly one new token is decoded this
                    # step. A partial remote match (num_new_tokens > 1) would leave a
                    # local "remainder prefill" that must not join the decode batch;
                    # that is unsupported here -- fail loudly instead of silently
                    # corrupting the (batch, num_spec+1) decode shape downstream
                    # (input_ids.view(num_reqs, -1)).
                    assert num_new_tokens == 1, (
                        f"promoted remote-KV request {request_id} has "
                        f"num_new_tokens={num_new_tokens} (expected 1); a partial "
                        "remote KV match leaves a local prefill remainder that "
                        "cannot mix into the decode batch."
                    )
                    # This promotion path bypasses the running-loop spec/slide
                    # block above, so apply the SAME sliding-window backfill via
                    # the shared `_decide_spec_slide`. With spec decode on, every
                    # decode query window is unified to num_spec_tokens+1; a
                    # freshly promoted req carries 0 drafts, so without this
                    # backfill its window stays length 1 while running peers run
                    # at num_spec+1 and the runner would zero-pad it (garbage KV)
                    # instead of re-feeding real past tokens. num_new_tokens == 1
                    # (asserted) so the advance is never boundary-trimmed
                    # (new_n == 1).
                    if not is_prefill(request) and self.num_spec_tokens > 0:
                        slide_distance, cross_block_no_spec = self._decide_spec_slide(
                            request, new_n=1
                        )
                        if cross_block_no_spec:
                            step_no_spec_required = True
                        elif slide_distance > 0:
                            spec_decode_slide_distance[request_id] = slide_distance
                    # The scheduled new request is added as a decoding-phase req, so
                    # we can continue to schedule the next request.
                    continue

                # NOTE(RBLN): Reaching this point means that this request can now be
                # added to the running batch. However, since we do not support mixed
                # batching for now, we remove all currently scheduled running requests
                # from the scheduler output and run only this prefill request for the
                # current step. In the next step (or after this request’s prefill
                # completes if it cannot finish within a single step) this request will
                # be scheduled together with the other running requests in the decoding
                # phase.
                for req in scheduled_running_reqs:
                    evicted_blocks = req_to_new_blocks.pop(req.request_id)
                    num_scheduled_tokens.pop(req.request_id)
                    req.spec_token_ids = scheduled_spec_decode_tokens.pop(
                        req.request_id, []
                    )
                    scheduled_encoder_inputs.pop(req.request_id, None)

                    # NOTE(RBLN): The block delta allocated for this evicted
                    # request is already committed in the coordinator but is
                    # being dropped from this step's output. Stash it (merging
                    # with any delta stranded on a previous consecutive
                    # eviction) so it is re-emitted to the runner on the
                    # request's next scheduled step. Skip empty deltas.
                    if evicted_blocks is not None and any(
                        len(g) > 0 for g in evicted_blocks.get_block_ids()
                    ):
                        prev = self._stranded_new_blocks.get(req.request_id)
                        self._stranded_new_blocks[req.request_id] = (
                            prev + evicted_blocks
                            if prev is not None
                            else evicted_blocks
                        )

                scheduled_running_reqs.clear()
                token_budget = prefill_token_budget

                # NOTE(RBLN): we restrict the prefill batch size to 1 for now.
                break

            # NOTE(RBLN): Release any un-applied sub-block match from a
            # break path (budget exhausted, allocation failure, etc.).
            if sub_block_match is not None:
                assert isinstance(self.kv_cache_manager, RBLNKVCacheManager)
                self.kv_cache_manager.release_sub_block_match(sub_block_match)

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Check if the scheduling constraints are satisfied.
        total_num_scheduled_tokens = sum(num_scheduled_tokens.values())
        assert total_num_scheduled_tokens <= self.max_num_scheduled_tokens

        assert token_budget >= 0
        assert len(self.running) <= self.max_num_running_reqs
        # Since some requests in the RUNNING queue may not be scheduled in
        # this step, the total number of scheduled requests can be smaller than
        # len(self.running).
        assert len(scheduled_new_reqs) + len(scheduled_resumed_reqs) + len(
            scheduled_running_reqs
        ) <= len(self.running)

        # NOTE(RBLN): All allocate_slots calls above used delay_cache_blocks=True
        # so that scheduling decisions (spec_decode_cap trimming, prefill kicking
        # out running decodes) can adjust token counts without needing to undo
        # premature caching. Now that scheduling is finalized, cache blocks and
        # schedule sub-block indexing for all scheduled requests.
        for req in itertools.chain(
            scheduled_running_reqs, scheduled_new_reqs, scheduled_resumed_reqs
        ):
            self.kv_cache_manager.cache_blocks(
                req,
                # Cap at req.num_tokens to exclude unverified spec decode
                # draft tokens, matching the upstream allocate_slots behavior.
                min(
                    req.num_computed_tokens + num_scheduled_tokens[req.request_id],
                    req.num_tokens,
                ),
            )
            if isinstance(self.kv_cache_manager, RBLNKVCacheManager):
                self.kv_cache_manager.schedule_sub_block_indexing(req)

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request_id = self.running[0].request_id
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(any_request_id)
                )

        # Construct the scheduler output.
        if self.use_v2_model_runner:
            scheduled_new_reqs = scheduled_new_reqs + scheduled_resumed_reqs
            scheduled_resumed_reqs = []
            new_reqs_data = [
                NewRequestData.from_request(
                    req,
                    req_to_new_blocks[req.request_id].get_block_ids(),
                    req._all_token_ids,
                )
                for req in scheduled_new_reqs
            ]
        else:
            new_reqs_data = [
                NewRequestData.from_request(
                    req, req_to_new_blocks[req.request_id].get_block_ids()
                )
                for req in scheduled_new_reqs
            ]

        # NOTE(RBLN): Re-emit any block delta that was stranded when a running
        # request was evicted by the "disable mixed batching" path on a prior
        # step. The coordinator already holds these blocks, so this step's
        # `allocate_slots` returned an empty delta for them; prepending the
        # stranded delta (allocated earlier) ahead of this step's delta keeps
        # the runner's block table in sync (the runner appends deltas in
        # order). Only requests actually scheduled this step (cached/running)
        # carry their delta to the runner, so drain on emit.
        if self._stranded_new_blocks:
            for req in scheduled_running_reqs:
                stranded = self._stranded_new_blocks.pop(req.request_id, None)
                if stranded is not None:
                    req_to_new_blocks[req.request_id] = (
                        stranded + req_to_new_blocks[req.request_id]
                    )

        with record_function_or_nullcontext("schedule: make_cached_request_data"):
            cached_reqs_data = self._make_cached_request_data(
                scheduled_running_reqs,
                scheduled_resumed_reqs,
                num_scheduled_tokens,
                scheduled_spec_decode_tokens,
                req_to_new_blocks,
            )

        # Record the request ids that were scheduled in this step.
        self.prev_step_scheduled_req_ids.clear()
        self.prev_step_scheduled_req_ids.update(num_scheduled_tokens.keys())

        new_block_ids_to_zero = (
            (self.kv_cache_manager.take_new_block_ids() or None)
            if self.needs_kv_cache_zeroing
            else None
        )

        scheduler_output = RBLNSchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=scheduled_encoder_inputs,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=self.encoder_cache_manager.get_freed_mm_hashes(),
            new_block_ids_to_zero=new_block_ids_to_zero,
            step_no_spec_required=step_no_spec_required,
            spec_decode_slide_distance=spec_decode_slide_distance,
        )

        # Drain pending copy ops from the KV cache manager.
        # Source-block refs are kept alive until update_from_output(),
        # which runs after the model runner finishes (safe for async
        # scheduling / pipeline parallelism).
        if isinstance(self.kv_cache_manager, RBLNKVCacheManager):
            scheduler_output.kv_cache_copy_ops = (
                self.kv_cache_manager.drain_pending_copy_ops()
            )

        # NOTE(Kuntai): this function is designed for multiple purposes:
        # 1. Plan the KV cache store
        # 2. Wrap up all the KV cache load / save ops into an opaque object
        # 3. Clear the internal states of the connector
        if self.connector is not None:
            meta: KVConnectorMetadata = self.connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.kv_connector_metadata = meta

        # Build the connector meta for ECConnector
        if self.ec_connector is not None:
            ec_meta: ECConnectorMetadata = self.ec_connector.build_connector_meta(
                scheduler_output
            )
            scheduler_output.ec_connector_metadata = ec_meta

        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        return scheduler_output

    def _free_request(
        self, request: Request, delay_free_blocks: bool = False
    ) -> dict[str, Any] | None:
        # Drop any block delta stashed for re-emit; the request is finishing and
        # will never be scheduled again, so the stash would otherwise leak.
        self._stranded_new_blocks.pop(request.request_id, None)
        return super()._free_request(request, delay_free_blocks)

    def _preempt_request(self, request: Request, timestamp: float) -> None:
        # Preemption frees ALL of the request's blocks (kv_cache_manager.free),
        # including any block stashed for re-emit. The request lives on and may
        # later re-enter the decode batch, where the merge would otherwise
        # prepend a now-freed (and possibly reused) block id to its block table.
        # Drop the stash so the resumed request relies solely on the fresh
        # block ids sent on resume.
        self._stranded_new_blocks.pop(request.request_id, None)
        super()._preempt_request(request, timestamp)

    def update_from_output(
        self,
        scheduler_output: SchedulerOutput,
        model_runner_output: ModelRunnerOutput,
    ) -> dict[int, EngineCoreOutputs]:
        assert isinstance(scheduler_output, RBLNSchedulerOutput)
        result = super().update_from_output(scheduler_output, model_runner_output)

        if isinstance(self.kv_cache_manager, RBLNKVCacheManager):
            # Now that execute_model has written KV data and
            # super().update_from_output() has updated num_computed_tokens
            # (and freed finished requests), index sub-blocks for the
            # remaining running requests and release copy-op source refs.
            self.kv_cache_manager.do_pending_indexing()
            if scheduler_output.kv_cache_copy_ops:
                self.kv_cache_manager.release_copy_ops(
                    scheduler_output.kv_cache_copy_ops
                )

        return result

    def _try_sub_block_match(
        self,
        request: Request,
        num_local_computed_tokens: int,
        num_external_computed_tokens: int,
    ) -> tuple[SubBlockMatch | None, int]:
        """Discover a sub-block match and arbitrate against a KV connector.

        Returns ``(match, extra_tokens)``.
        When *match* is not ``None`` the caller must later pass it to
        ``kv_cache_manager.apply_sub_block_match`` or
        ``kv_cache_manager.release_sub_block_match``.
        """
        if not isinstance(self.kv_cache_manager, RBLNKVCacheManager):
            return None, 0

        match = self.kv_cache_manager.get_computed_blocks_sub_block(
            request, num_local_computed_tokens
        )
        if match is not None and match.num_tokens >= num_external_computed_tokens:
            # sub-block wins on ties (local copy is cheaper than remote load)
            return match, match.num_tokens

        # Connector provides better coverage, or no sub-block match at all.
        if match is not None:
            self.kv_cache_manager.release_sub_block_match(match)
        return None, 0
