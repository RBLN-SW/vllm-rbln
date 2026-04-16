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

import itertools
import time
from dataclasses import dataclass, field

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
    prefix caching."""

    kv_cache_copy_ops: list[KVCacheCopyOp] = field(default_factory=list)


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

        # NOTE(RBLN): spec_decode_cap prevents block boundary crossing caused by
        # runner-side padding. The runner pads all requests in the batch to the
        # maximum scheduled token length (max_spec_decode_len). If any request
        # is trimmed due to a block boundary, other requests with more tokens
        # would cause that trimmed request to be padded beyond its boundary.
        # spec_decode_cap propagates the tightest remaining_in_block constraint
        # to all subsequent requests so no request exceeds it.
        spec_decode_cap = self.block_size

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

            # NOTE(RBLN): Update spec_decode_cap with the tightest constraint
            # for this request: remaining space in the current block and remaining
            # tokens until max_model_len. Done here (after confirmed scheduling)
            # so that only actually scheduled requests affect the cap, and
            # num_new_tokens reflects all prior adjustments. Even single-token
            # decode requests must constrain the cap because the runner pads all
            # requests to max_spec_decode_len.
            if not is_prefill(request):
                tokens_used_in_block = request.num_computed_tokens % self.block_size
                remaining_in_block = self.block_size - tokens_used_in_block
                remaining_in_maxlen = self.max_model_len - request.num_computed_tokens
                spec_decode_cap = min(
                    remaining_in_block, remaining_in_maxlen, spec_decode_cap
                )

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

        # NOTE(RBLN): Retroactively apply spec_decode_cap to requests scheduled
        # before the cap was tightened. Only needed when spec decode is active
        # this step (scheduled_spec_decode_tokens non-empty), since that is when
        # the runner pads all requests to max_spec_decode_len. A request processed
        # earlier in the loop may have been allocated more tokens than
        # spec_decode_cap allows; if left uncorrected, the runner would pad a
        # constrained request up to that larger length, causing it to cross its
        # block boundary.
        if spec_decode_cap < self.block_size and scheduled_spec_decode_tokens:
            for req in scheduled_running_reqs:
                req_id = req.request_id
                old_n = num_scheduled_tokens[req_id]
                if old_n <= spec_decode_cap:
                    continue
                new_n = spec_decode_cap

                token_budget += old_n - new_n
                num_scheduled_tokens[req_id] = new_n

                # Re-trim spec tokens to match the reduced token count.
                num_spec = (
                    new_n
                    + req.num_computed_tokens
                    - req.num_tokens
                    - req.num_output_placeholders
                )
                if num_spec > 0:
                    if req_id in scheduled_spec_decode_tokens:
                        scheduled_spec_decode_tokens[req_id] = (
                            scheduled_spec_decode_tokens[req_id][:num_spec]
                        )
                else:
                    scheduled_spec_decode_tokens.pop(req_id, None)

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
                # Count the number of prefix cached tokens.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = num_computed_tokens
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

                # NOTE(RBLN): Reaching this point means that this request can now be
                # added to the running batch. However, since we do not support mixed
                # batching for now, we remove all currently scheduled running requests
                # from the scheduler output and run only this prefill request for the
                # current step. In the next step (or after this request’s prefill
                # completes if it cannot finish within a single step) this request will
                # be scheduled together with the other running requests in the decoding
                # phase.
                for req in scheduled_running_reqs:
                    req_to_new_blocks.pop(req.request_id)
                    num_scheduled_tokens.pop(req.request_id)
                    req.spec_token_ids = scheduled_spec_decode_tokens.pop(
                        req.request_id, []
                    )
                    scheduled_encoder_inputs.pop(req.request_id, None)

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
