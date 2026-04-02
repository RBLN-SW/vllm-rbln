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

import time
from collections import defaultdict
from dataclasses import dataclass, field

import torch
from vllm.config import VllmConfig
from vllm.distributed.kv_events import EventPublisherFactory
from vllm.multimodal import MULTIMODAL_REGISTRY, MultiModalRegistry
from vllm.v1.core.encoder_cache_manager import (
    EncoderCacheManager,
    EncoderDecoderCacheManager,
)
from vllm.v1.core.kv_cache_manager import KVCacheBlocks
from vllm.v1.core.kv_cache_metrics import KVCacheMetricsCollector
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.core.sched.output import NewRequestData, SchedulerOutput
from vllm.v1.core.sched.request_queue import SchedulingPolicy, create_request_queue
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine import EngineCoreEventType
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.perf import ModelMetrics
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus
from vllm.v1.structured_output import StructuredOutputManager
from vllm.v1.utils import record_function_or_nullcontext

from vllm_rbln.logger import init_logger
from vllm_rbln.v1.core.optimum_kv_cache_manager import RBLNKVCacheManager

logger = init_logger(__name__)


@dataclass
class RBLNSchedulerOutput(SchedulerOutput):
    """
    block_table_dict: dict[str, torch.Tensor]
        Mapping from request ID to outer block table tensor
        for both prefill and decode.
    cached_block_table: list[int]
        List of cached outer block table entries for prefill.
    cached_length: list[int]
        List of cached lengths for each outer block for prefill.
    dummy_block: int
        The index of dummy block for padding. It is required
        if the number of requests is less than the number of batch_size
        in decode phase.
    """

    block_table_dict: dict[str, torch.Tensor] = field(default_factory=dict)
    cached_block_table: list[int] = field(default_factory=list)
    cached_length: list[int] = field(default_factory=list)
    dummy_block: int | None = None


class RBLNOptimumScheduler(Scheduler):
    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
        structured_output_manager: StructuredOutputManager,
        block_size: int,
        mm_registry: MultiModalRegistry = MULTIMODAL_REGISTRY,
        include_finished_set: bool = False,
        log_stats: bool = False,
    ) -> None:
        self.vllm_config = vllm_config
        self.scheduler_config = vllm_config.scheduler_config
        self.cache_config = vllm_config.cache_config
        self.lora_config = vllm_config.lora_config
        self.kv_cache_config = kv_cache_config
        self.kv_events_config = vllm_config.kv_events_config
        self.parallel_config = vllm_config.parallel_config
        self.log_stats = log_stats
        self.observability_config = vllm_config.observability_config
        self.kv_metrics_collector: KVCacheMetricsCollector | None = None
        if self.observability_config.kv_cache_metrics:
            self.kv_metrics_collector = KVCacheMetricsCollector(
                self.observability_config.kv_cache_metrics_sample,
            )
        self.structured_output_manager = structured_output_manager
        self.is_encoder_decoder = vllm_config.model_config.is_encoder_decoder

        # include_finished_set controls whether a separate set of finished
        # request ids should be included in the EngineCoreOutputs returned
        # by update_from_outputs(). This is currently used in the multi-engine
        # case to track request lifetimes efficiently.
        self.finished_req_ids_dict: dict[int, set[str]] | None = (
            defaultdict(set) if include_finished_set else None
        )
        self.prev_step_scheduled_req_ids: set[str] = set()

        # Scheduling constraints.
        self.max_num_running_reqs = self.scheduler_config.max_num_seqs
        self.max_num_scheduled_tokens = (
            self.scheduler_config.max_num_scheduled_tokens
            if self.scheduler_config.max_num_scheduled_tokens
            else self.scheduler_config.max_num_batched_tokens
        )
        self.max_model_len = vllm_config.model_config.max_model_len
        self.enable_kv_cache_events = (
            self.kv_events_config is not None
            and self.kv_events_config.enable_kv_cache_events
        )
        # Create KVConnector for the Scheduler. Note that each Worker
        # will have a corresponding KVConnector with Role=WORKER.
        # KV Connector pushes/pull of remote KVs for P/D and offloading.
        self.connector = None
        self.connector_prefix_cache_stats: PrefixCacheStats | None = None
        self.recompute_kv_load_failures = True
        self.kv_event_publisher = EventPublisherFactory.create(
            self.kv_events_config,
            self.parallel_config.data_parallel_rank,
        )
        self.ec_connector = None
        num_gpu_blocks = self.cache_config.num_gpu_blocks
        assert num_gpu_blocks is not None and num_gpu_blocks > 0

        self.block_size = self.cache_config.block_size
        # req_id -> Request
        self.requests: dict[str, Request] = {}
        # Scheduling policy
        try:
            self.policy = SchedulingPolicy(self.scheduler_config.policy)
        except ValueError as e:
            raise ValueError(
                f"Unknown scheduling policy: {self.scheduler_config.policy}"
            ) from e
        # Priority queues for requests.
        self.waiting = create_request_queue(self.policy)
        # requests skipped in waiting flow due async deps or constraints.
        self.skipped_waiting = create_request_queue(self.policy)
        self.running: list[Request] = []

        # The request IDs that are finished in between the previous and the
        # current steps. This is used to notify the workers about the finished
        # requests so that they can free the cached states for those requests.
        # This is flushed at the end of each scheduling step.
        self.finished_req_ids: set[str] = set()

        # Counter for requests waiting for streaming input. Used to calculate
        # number of unfinished requests
        self.num_waiting_for_streaming_input: int = 0

        # KV Connector: requests in process of async KV loading or recving
        self.finished_recving_kv_req_ids: set[str] = set()
        self.failed_recving_kv_req_ids: set[str] = set()

        # Create the KV cache manager.
        if (
            self.vllm_config.additional_config is not None
            and "attn_block_size" in self.vllm_config.additional_config
        ):
            attn_block_size = self.vllm_config.additional_config["attn_block_size"]
        else:
            attn_block_size = None
        self.kv_cache_manager = RBLNKVCacheManager(
            kv_cache_config=kv_cache_config,
            max_model_len=self.max_model_len,
            enable_caching=self.cache_config.enable_prefix_caching,
            use_eagle=False,
            log_stats=self.log_stats,
            enable_kv_cache_events=False,
            dcp_world_size=1,
            pcp_world_size=1,
            hash_block_size=self.block_size,
            metrics_collector=self.kv_metrics_collector,
            attn_block_size=attn_block_size,
            max_num_seqs=self.max_num_running_reqs,
        )
        self.perf_metrics: ModelMetrics | None = None
        if self.log_stats and vllm_config.observability_config.enable_mfu_metrics:
            self.perf_metrics = ModelMetrics(vllm_config)
        # Encoder-related.
        # It is not used in RBLN.
        # But for reuse original functions(e.g. free_request) in vLLM,
        # we keep it here.
        encoder_compute_budget = 0
        encoder_cache_size = 0

        # NOTE(woosuk): Here, "encoder" includes the vision encoder (and
        # projector if needed) for MM models as well as encoder-decoder
        # transformers.
        self.max_num_encoder_input_tokens = encoder_compute_budget
        # NOTE: For the models without encoder (e.g., text-only models),
        # the encoder cache will not be initialized because cache size is 0
        # for these models.
        self.encoder_cache_manager = (
            EncoderDecoderCacheManager(cache_size=encoder_cache_size)
            if self.is_encoder_decoder
            else EncoderCacheManager(cache_size=encoder_cache_size)
        )

        self.use_pp = False
        self.use_v2_model_runner = False
        self._pause_state: PauseState = PauseState.UNPAUSED

    def schedule(self) -> RBLNSchedulerOutput:
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

        # req_to_new_block_ids: dict[str, tuple[list[int], ...]] = {}
        req_to_new_blocks: dict[str, KVCacheBlocks] = {}
        num_scheduled_tokens: dict[str, int] = {}
        token_budget = self.max_num_scheduled_tokens
        # https://github.com/vllm-project/vllm/pull/34125
        if self._pause_state == PauseState.PAUSED_ALL:
            # Do not schedule any requests when paused.
            token_budget = 0

        scheduled_spec_decode_tokens: dict[str, list[int]] = {}
        # For logging.
        scheduled_timestamp = time.monotonic()
        block_table_dict: dict[str, torch.Tensor] = {}
        cached_block_table: list[int] = []
        cached_length: list[int] = []
        dummy_block: int | None = None

        # NOTE The scheduling process is changed like below.
        # (1) vllm-rbln distinguishes
        #   between requests in the prefill and decode phases.
        #   If a request is in the prefill phase,
        #   it is given priority and processed exclusively (only one at a time).
        # (2) For (1), vllm-rbln schedules the requests WAITING -> RUNNING.
        #   In the vLLM, requests are scheduled RUNNING -> WAITING.

        req_index = 0
        # It is always empty in decode phase.
        new_computed_blocks = KVCacheBlocks(blocks=([],))
        # Record the LoRAs in scheduled_running_reqs
        # It is for checking the max_loras constraint.
        scheduled_loras: set[int] = set()
        if self.lora_config:
            scheduled_loras = set(
                req.lora_request.lora_int_id
                for req in scheduled_running_reqs
                if req.lora_request and req.lora_request.lora_int_id > 0
            )
            assert len(scheduled_loras) <= self.lora_config.max_loras

        self.kv_cache_manager.new_step_starts()

        # First, schedule the WAITING requests.
        if not preempted_reqs and self._pause_state == PauseState.UNPAUSED:
            step_skipped_waiting = create_request_queue(self.policy)
            while (self.waiting or self.skipped_waiting) and token_budget > 0:
                if len(self.running) == self.max_num_running_reqs:
                    break
                request_queue = self._select_waiting_queue_for_scheduling()
                assert request_queue is not None
                request = request_queue.peek_request()
                request_id = request.request_id
                # try to promote blocked statuses while traversing skipped queue.
                # NOTE(eunji): prefill request is allowed only one
                if req_index > 0:
                    break

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

                assert request.num_computed_tokens == 0

                # Number of tokens to be scheduled.
                # We use `request.num_tokens` instead of
                # `request.num_prompt_tokens` to consider the resumed
                # requests, which have output tokens.
                num_new_tokens = request.num_tokens

                num_new_tokens = min(num_new_tokens, token_budget)
                assert num_new_tokens > 0

                new_blocks = self.kv_cache_manager.allocate_slots(
                    request,
                    num_new_tokens,
                )

                if new_blocks is None:
                    # The request cannot be scheduled.
                    break

                # Get locally-cached tokens.
                # NOTE(eunji.lee):
                # Allocation -> Caching
                # Allocate blocks before cache lookup to prevent
                # prefix-cache hit blocks from being reused for new requests,
                # which would otherwise reduce the cache hit rate.
                # This is special logic
                # because we do not touch cache-hit blocks.
                new_computed_blocks, num_new_local_computed_tokens = (
                    self.kv_cache_manager.get_computed_blocks(request)
                )

                # Get the cached blocks for prefix caching.
                # using new_computed_blocks, num_new_local_computed_tokens
                if self.cache_config.enable_prefix_caching:
                    (
                        cached_block_table,
                        cached_length,
                    ) = self.kv_cache_manager.get_prefix_cached_blocks(
                        request,
                        new_computed_blocks,
                        num_new_local_computed_tokens,
                    )

                    # Update the block table to the return output.
                    self.update_block_table_dict(request, block_table_dict)

                # Request was already popped from self.waiting
                # unless it was re-added above due to new_blocks being None.
                request = request_queue.pop_request()

                req_index += 1
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
                num_scheduled_tokens[request.request_id] = num_new_tokens
                token_budget -= num_new_tokens
                request.status = RequestStatus.RUNNING
                # NOTE Setting num_computed_tokens to the number of tokens hit
                # by prefix caching may cause incorrect computation
                # of new_blocks during the decode phase.
                request.num_computed_tokens = 0
                # NOTE(fix): num_cached_tokens defaults to -1.
                # It is used for logging and metrics.
                if request.num_cached_tokens < 0:
                    request.num_cached_tokens = request.num_computed_tokens

            # re-queue requests skipped in this pass ahead of older skipped items.
            if step_skipped_waiting:
                self.skipped_waiting.prepend_requests(step_skipped_waiting)

        # Next, schedule the RUNNING requests.
        if req_index == 0:
            while req_index < len(self.running) and token_budget > 0:
                request = self.running[req_index]
                num_new_tokens = 1
                num_new_tokens = min(num_new_tokens, token_budget)

                # Make sure the input position
                # does not exceed the max model len.
                # This is necessary when using spec decoding.
                num_new_tokens = min(
                    num_new_tokens, self.max_model_len - 1 - request.num_computed_tokens
                )

                if num_new_tokens == 0:
                    # The request cannot be scheduled
                    # because one of the following reasons:
                    # 1. No new tokens to schedule. This may happen when
                    #    (1) PP>1 and we have already scheduled
                    #    all prompt tokens but they are not finished yet.
                    #    (2) Async scheduling and the request has reached
                    #    to either its max_total_tokens or max_model_len.
                    # 2. The encoder budget is exhausted.
                    # 3. The encoder cache is exhausted.
                    # NOTE(woosuk): Here, by doing `continue`
                    # instead of `break`,
                    # we do not strictly follow the FCFS scheduling policy and
                    # allow the lower-priority requests to be scheduled.
                    req_index += 1
                    continue
                with record_function_or_nullcontext("schedule: allocate_slots"):
                    while True:
                        new_blocks = self.kv_cache_manager.allocate_slots(
                            request, num_new_tokens
                        )
                        if new_blocks is not None:
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
                                token_budget += num_scheduled_tokens.pop(
                                    preempted_req_id
                                )
                                req_to_new_blocks.pop(preempted_req_id)
                                scheduled_spec_decode_tokens.pop(preempted_req_id, None)
                                req_index -= 1
                        else:
                            preempted_req = self.running.pop()
                        self._preempt_request(preempted_req, scheduled_timestamp)
                        preempted_reqs.append(preempted_req)
                        if preempted_req == request:
                            # No more request to preempt.
                            # Cannot schedule this request.
                            break
                if new_blocks is None:
                    # Cannot schedule this request.
                    break
                if self.cache_config.enable_prefix_caching:
                    self.update_block_table_dict(request, block_table_dict)
                # Schedule the request.
                scheduled_running_reqs.append(request)
                request_id = request.request_id
                req_to_new_blocks[request_id] = new_blocks
                num_scheduled_tokens[request_id] = num_new_tokens
                token_budget -= num_new_tokens
                req_index += 1

        # [skip] speculative decoding, encoder-related tasks

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

        # Get the longest common prefix among all requests in the running queue.
        # This can be potentially used for cascade attention.
        num_common_prefix_blocks = [0] * len(self.kv_cache_config.kv_cache_groups)
        with record_function_or_nullcontext("schedule: get_num_common_prefix_blocks"):
            if self.running:
                any_request = self.running[0]
                num_common_prefix_blocks = (
                    self.kv_cache_manager.get_num_common_prefix_blocks(
                        any_request.request_id
                    )
                )

        # Construct the scheduler output.
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

        # Calculate the dummy block index.
        if self.cache_config.enable_prefix_caching:
            num_decode_reqs = len(scheduled_running_reqs)
            if num_decode_reqs > 0 and num_decode_reqs < self.max_num_running_reqs:
                dummy_block = self.kv_cache_manager.get_dummy_block()

        scheduler_output = RBLNSchedulerOutput(
            scheduled_new_reqs=new_reqs_data,
            scheduled_cached_reqs=cached_reqs_data,
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=total_num_scheduled_tokens,
            scheduled_spec_decode_tokens=scheduled_spec_decode_tokens,
            scheduled_encoder_inputs=None,
            num_common_prefix_blocks=num_common_prefix_blocks,
            preempted_req_ids={req.request_id for req in preempted_reqs},
            # finished_req_ids is an existing state in the scheduler,
            # instead of being newly scheduled in this step.
            # It contains the request IDs that are finished in between
            # the previous and the current steps.
            finished_req_ids=self.finished_req_ids,
            free_encoder_mm_hashes=[],
            new_block_ids_to_zero=None,  # It is used for Mamba models
            block_table_dict=block_table_dict,
            cached_block_table=cached_block_table,
            cached_length=cached_length,
            dummy_block=dummy_block,
        )
        with record_function_or_nullcontext("schedule: update_after_schedule"):
            self._update_after_schedule(scheduler_output)
        # self._update_after_schedule(scheduler_output)
        return scheduler_output

    def update_block_table_dict(
        self, request: Request, block_table_dict: dict[str, torch.Tensor]
    ) -> None:
        request_id = request.request_id
        block_table = self.kv_cache_manager.get_block_table(request_id)
        block_table_dict[request_id] = block_table

    def _preempt_request(
        self,
        request: Request,
        timestamp: float,
    ) -> None:
        assert request.status == RequestStatus.RUNNING, (
            "Only running requests can be preempted"
        )
        preempted_blocks = self.kv_cache_manager.get_block_ids(request.request_id)[0]
        self.kv_cache_manager.free(request, preemption=True)
        if not self.cache_config.enable_prefix_caching:
            preempted_blocks = [block_idx - 1 for block_idx in preempted_blocks]
        logger.warning(
            "Request %s is preempted. Freed block(s): %s Already generated tokens: %d",
            request.request_id,
            preempted_blocks,
            len(request.output_token_ids),
        )
        request.status = RequestStatus.PREEMPTED
        request.num_computed_tokens = 0
        request.num_preemptions += 1
        if self.log_stats:
            request.record_event(EngineCoreEventType.PREEMPTED, timestamp)

        # Put the request back to the waiting queue.
        self.waiting.prepend_request(request)
