# Copyright 2025 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
from collections import defaultdict
from collections.abc import Iterator, Sequence
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.cache import CacheConfig
from vllm.distributed.parallel_state import (
    get_dp_group,
    get_pp_group,
    get_tp_group,
)
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.interfaces import (
    supports_realtime,
    supports_transcription,
)
from vllm.model_executor.models.interfaces_base import (
    is_pooling_model,
    is_text_generation_model,
)
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.tracing import instrument
from vllm.utils import length_from_prompt_token_ids_or_embeds
from vllm.utils.math_utils import cdiv
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionMetadata,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.utils import create_fast_prefill_custom_backend
from vllm.v1.core.sched.output import GrammarOutput, NewRequestData
from vllm.v1.kv_cache_interface import (
    EncoderOnlyAttentionSpec,
    KVCacheConfig,
    MambaSpec,
    UniformTypeKVCacheSpecs,
)
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    SamplerOutput,
)
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.utils import (
    AttentionGroup,
    AttentionSpec,
    KVCacheGroupSpec,
    KVCacheSpec,
    add_kv_sharing_layers_to_kv_cache_groups,
    bind_kv_cache,
)

from vllm_rbln import envs
from vllm_rbln.compilation.backends import rbln_backend
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.logger import init_logger
from vllm_rbln.utils import pad
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadataBuilder,
)
from vllm_rbln.v1.attention.kv_cache_bindings import (
    KVCacheViewInfo,
    attach_kv_cache_bindings,
    build_kv_cache_base_bindings,
    build_kv_cache_forward_context_kwargs,
    validate_shared_attention_kv_cache_contiguity,
)
from vllm_rbln.v1.worker.bucketing import get_bucketing_manager
from vllm_rbln.v1.worker.utils import get_kv_cache_names, prepare_kernel_block_sizes

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
PerLayerAttnMetadata: TypeAlias = AttnMetadataDict  #  | list[AttnMetadataDict]


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    aux_hidden_states: list[torch.Tensor] | None


class RBLNModelRunner:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ) -> None:
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        # self.offload_config = vllm_config.offload_config
        self.compilation_config = vllm_config.compilation_config
        # self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        # self.speculative_config = vllm_config.speculative_config
        # self.observability_config = vllm_config.observability_config

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = model_config.dtype

        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, model_config
        )

        self.max_model_len = model_config.max_model_len
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        self.dcp_world_size = self.parallel_config.decode_context_parallel_size

        # Model-related.
        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # TODO(RBLN): Multi-modal data support
        # TODO(RBLN): Async scheduling

        # NOTE(RBLN): Compilation context for marking the KV cache address as static.
        from rebel.compile_context import CompileContext

        self.compile_context = CompileContext(
            use_weight_sharing=True, use_global_ctx=True
        )
        self.runtime_holder: list = []

        # Sampler
        if envs.VLLM_RBLN_SAMPLER:
            from vllm_rbln.v1.sample import RBLNSampler

            self.sampler = RBLNSampler(
                logprobs_mode=self.model_config.logprobs_mode,
                seed=self.vllm_config.model_config.seed,
                compile_context=self.compile_context,
            )
            logger.info("Using RBLN sampler.")
        else:
            self.sampler = Sampler(self.model_config.logprobs_mode)

        # Lazy initialization
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        self.kv_cache_bases: list[torch.Tensor] = []
        self.kv_cache_view_infos: list[KVCacheViewInfo] = []
        # Initialize in initialize_kv_cache_tensors
        self.cross_layers_kv_cache: torch.Tensor | None = None
        self.cross_layers_attn_backend: type[AttentionBackend] | None = None
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []

        self.use_aux_hidden_state_outputs = False
        # TODO(RBLN): Set up speculative decoding

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        self.num_prompt_logprobs: dict[str, int] = {}

        # Input Batch
        logits_processors = model_config.logits_processors
        custom_logitsprocs: Sequence[str | type[LogitsProcessor]] = (
            tuple(logits_processors) if logits_processors is not None else ()
        )
        placeholder_block_size = (
            self.cache_config.block_size or CacheConfig.DEFAULT_BLOCK_SIZE
        )
        self._init_block_sizes = [placeholder_block_size]
        self._init_kernel_block_sizes = [placeholder_block_size]
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=model_config.get_vocab_size(),
            block_sizes=[cache_config.block_size],
            kernel_block_sizes=[cache_config.block_size],
            is_spec_decode=False,
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                False,
                custom_logitsprocs,
            ),
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=False,
            cp_kv_cache_interleave_size=parallel_config.cp_kv_cache_interleave_size,
        )

        # Persistent buffers
        self.input_ids = torch.zeros(self.max_num_tokens, dtype=torch.int32)
        self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64)
        self.positions_np = self.positions.numpy()
        self.query_start_loc = torch.zeros(self.max_num_reqs + 1, dtype=torch.int32)
        self.seq_lens = torch.zeros(self.max_num_tokens, dtype=torch.int32)
        self.discard_request_mask = torch.zeros(self.max_num_reqs, dtype=torch.bool)
        # self.num_decode_draft_tokens
        # self.num_accepted_tokens

        # None in the first PP rank. The rest are after load_model
        self.intermediate_tensors: IntermediateTensors | None = None

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(
            max(self.max_num_reqs + 1, self.max_model_len, self.max_num_tokens),
            dtype=np.int64,
        )

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device
            )

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # TODO(RBLN): Cached outputs.

        # Ephemeral state transferred between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None
        # self.kv_connector_output: KVConnectorOutput | None = None

        # NOTE(RBLN): Initialize bucketing manager
        self.bucketing_manager = get_bucketing_manager(
            envs.VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY,
            max_batch_size=self.max_num_reqs,
            min_batch_size=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_MIN,
            step=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_STEP,
            limit=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT,
            manual_buckets=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_MANUAL_BUCKETS,
        )
        logger.info(
            "Using %s. Decode batch buckets: %s",
            type(self.bucketing_manager).__name__,
            self.bucketing_manager.decode_batch_buckets,
        )

        self.specialized_moe_decode = False

    def _get_positions(self, num_tokens: Any):
        assert not isinstance(num_tokens, int)
        return self.positions[:num_tokens]

    def _may_reorder_batch(self, scheduler_output: "SchedulerOutput") -> None:
        # NOTE(RBLN): Unlike upstream GPUModelRunner, we do not split mixed batches
        # into decode / extend / prefill regions here. The RBLN execution path assumes
        # a homogeneous batch phase and therefore does not use scheduler_output-based
        # phase classification. Instead, we perform a stable sort by current sequence
        # length (num_tokens_no_spec, descending).
        if (
            not envs.VLLM_RBLN_SORT_BATCH
            or len(self.kv_cache_config.kv_cache_groups) == 0
        ):
            return

        if (n := len(self.input_batch.req_ids)) < 2:
            return

        sorted_indices = np.argsort(
            -self.input_batch.num_tokens_no_spec[:n], kind="stable"
        )
        if np.array_equal(sorted_indices, np.arange(n)):
            return

        src_to_dst = {
            int(src): dst for dst, src in enumerate(sorted_indices) if src != dst
        }

        for src in tuple(src_to_dst):
            dst = src_to_dst[src]
            while src != dst:
                self.input_batch.swap_states(src, dst)
                next_dst = src_to_dst.get(dst, dst)
                src_to_dst[dst] = dst
                dst = next_dst

    def _update_states(self, scheduler_output: "SchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output."""
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)

        # Remove the finished requests from the persistent batch.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids

        # Remove the unscheduled requests from the persistent batch.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids - resumed_req_ids)
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            if req_id in self.requests:
                # For streaming case only.
                req_state = self._update_streaming_request(req_id, new_req_data)
                reqs_to_add.append(req_state)
                continue

            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if (
                sampling_params
                and sampling_params.sampling_type == SamplingType.RANDOM_SEED
            ):
                generator = torch.Generator(device=self.device)
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                prompt_embeds=new_req_data.prompt_embeds,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )
            self.requests[req_id] = req_state

            if sampling_params and sampling_params.prompt_logprobs is not None:
                self.num_prompt_logprobs[req_id] = (
                    self.input_batch.vocab_size
                    if sampling_params.prompt_logprobs == -1
                    else sampling_params.prompt_logprobs
                )

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            num_output_tokens = req_data.num_output_tokens[i]
            req_index = self.input_batch.req_id_to_index.get(req_id)

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                assert req_data.new_token_ids
                # Non-async scheduling with PP: The scheduler sends
                # sampled token ids back because there's no direct communication
                # between the first-stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (
                    num_computed_tokens + len(new_token_ids) - req_state.num_tokens
                )
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(new_token_ids[-num_new_tokens:])
            elif num_output_tokens < len(req_state.output_token_ids):
                # Some output tokens were discarded due to a sync-KV-load
                # failure. Align the cached state.
                del req_state.output_token_ids[num_output_tokens:]
                if req_index is not None:
                    end_idx = (
                        self.input_batch.num_prompt_tokens[req_index]
                        + num_output_tokens
                    )
                    self.input_batch.num_tokens_no_spec[req_index] = end_idx

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids, new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert req_index is None
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = num_computed_tokens
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index, start_token_index:end_token_index
                ] = new_token_ids
                self.input_batch.num_tokens_no_spec[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            self.input_batch.update_req_spec_token_ids(req_state, scheduled_spec_tokens)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
            self.input_batch.update_req_spec_token_ids(request, scheduled_spec_tokens)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _update_streaming_request(
        self, req_id: str, new_req_data: NewRequestData
    ) -> CachedRequestState:
        """Updates streaming session request from `scheduled_new_reqs`

        Removes the request from InputBatch (if present), updates the cached
        state, and prepares it for re-addition to the batch.
        """
        self.input_batch.remove_request(req_id)
        req_state = self.requests[req_id]

        req_state.prompt_token_ids = new_req_data.prompt_token_ids
        req_state.mm_features = new_req_data.mm_features
        req_state.prompt_embeds = new_req_data.prompt_embeds
        req_state.sampling_params = new_req_data.sampling_params
        req_state.pooling_params = new_req_data.pooling_params
        req_state.block_ids = new_req_data.block_ids
        req_state.num_computed_tokens = new_req_data.num_computed_tokens
        req_state.num_prompt_tokens = length_from_prompt_token_ids_or_embeds(
            req_state.prompt_token_ids, req_state.prompt_embeds
        )

        # Clear `output_token_ids` as previous output tokens are now part of
        # `prompt_token_ids`.
        req_state.output_token_ids.clear()

        return req_state

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: np.dtype | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Get the cumulative sum and batched arange of the given array.
        # E.g., [2, 5, 3] -> ([2, 7, 10], [0, 1, 0, 1, 2, 3, 4, 0, 1, 2])
        # Equivalent to but faster than:
        # np.concatenate([np.arange(n) for n in num_tokens])
        """
        # Step 1. [2, 5, 3] -> [2, 7, 10]
        cu_num_tokens = np.cumsum(num_tokens, dtype=cumsum_dtype)
        total_num_tokens = cu_num_tokens[-1]
        # Step 2. [2, 7, 10] -> [0, 0, 2, 2, 2, 2, 2, 7, 7, 7]
        cumsums_offsets = np.repeat(cu_num_tokens - num_tokens, num_tokens)
        # Step 3. [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        arange = self.arange_np[:total_num_tokens] - cumsums_offsets

        return cu_num_tokens, arange

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, None]:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions_np[:total_num_scheduled_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices],
            arange,
            out=positions_np,
        )

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (
            positions_np + req_indices * self.input_batch.token_ids_cpu.shape[1]
        )
        token_indices_tensor = torch.from_numpy(token_indices)

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(
            self.input_batch.token_ids_cpu_tensor.flatten(),
            0,
            token_indices_tensor,
            out=self.input_ids[:total_num_scheduled_tokens],
        )

        self.input_batch.block_table.compute_slot_mapping(req_indices, positions_np)

        # Prepare the attention metadata.
        query_start_loc_np = self.query_start_loc.numpy()
        query_start_loc_np[0] = 0
        query_start_loc_np[1 : num_reqs + 1] = cu_num_tokens
        query_start_loc_np[num_reqs + 1 :].fill(cu_num_tokens[-1])

        seq_lens_np = self.seq_lens.numpy()
        seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + num_scheduled_tokens
        )
        seq_lens_np[num_reqs:].fill(0)

        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returningj
        discard_request_mask_np = self.discard_request_mask.numpy()
        discard_request_mask_np[:num_reqs] = seq_lens_np[:num_reqs] < num_tokens_np

        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        assert not use_spec_decode
        # NOTE(woosuk): Due to chunked prefills, the batch may contain
        # partial requests. While we should not sample any token
        # from these partial requests, we do so for simplicity.
        # We will ignore the sampled tokens from the partial requests.
        # TODO: Support prompt logprobs.
        logits_indices = self.query_start_loc[1 : num_reqs + 1] - 1
        spec_decode_metadata = None

        # TODO(RBLN): Hot-Swap lora model

        return (
            logits_indices,
            spec_decode_metadata,
        )

    def _build_attention_metadata(
        self,
        num_tokens: int,
        num_reqs: int,
        max_query_len: int,
        num_tokens_padded: int | None = None,
        num_reqs_padded: int | None = None,
        logits_indices: torch.Tensor | None = None,
        use_spec_decode: bool = False,
    ) -> tuple[PerLayerAttnMetadata, CommonAttentionMetadata | None]:
        """
        :return: tuple[attn_metadata, spec_decode_common_attn_metadata]
        """
        if len(kv_cache_groups := self.kv_cache_config.kv_cache_groups) == 0:
            return {}, None

        num_tokens_padded = num_tokens_padded or num_tokens
        num_reqs_padded = num_reqs_padded or num_reqs

        attn_metadata: PerLayerAttnMetadata = {}

        assert not use_spec_decode

        def _get_block_table(kv_cache_gid: int):
            blk_table = self.input_batch.block_table[kv_cache_gid]
            blk_table_tensor = blk_table.get_cpu_tensor()[:num_reqs]

            return blk_table_tensor

        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc[: num_reqs + 1],
            query_start_loc_cpu=self.query_start_loc[: num_reqs + 1],
            seq_lens=self.seq_lens[:num_reqs],
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=max_query_len,
            max_seq_len=self.seq_lens[:num_reqs].max().item(),
            block_table_tensor=_get_block_table(0),
            slot_mapping=torch.tensor(0),  # dummy
            causal=True,
        )

        if logits_indices is not None and self.cache_config.kv_sharing_fast_prefill:
            cm_base.num_logits_indices = logits_indices.size(0)
            cm_base.logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices
            )

        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        spec_decode_common_attn_metadata = None
        for kv_cache_gid, _ in enumerate(kv_cache_groups):
            cm = copy(cm_base)  # shallow copy

            if kv_cache_gid > 0:
                cm.block_table_tensor = _get_block_table(kv_cache_gid)

            for attn_gid in range(len(self.attn_groups[kv_cache_gid])):
                attn_group = self.attn_groups[kv_cache_gid][attn_gid]
                builder = attn_group.get_metadata_builder(0)
                assert isinstance(builder, RBLNFlashAttentionMetadataBuilder)

                attn_metadata_i = builder.build(
                    common_attn_metadata=cm,
                    positions=self.positions,
                    is_prefill=self.is_prefill,
                    batch_pad=num_reqs_padded,
                )

                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        self._attach_kv_cache_bindings(attn_metadata)

        return attn_metadata, spec_decode_common_attn_metadata

    # TODO(RBLN): Enabling spec decode
    # def _calc_spec_decode_metadata(...)

    def _prepare_kv_sharing_fast_prefill(
        self,
        logits_indices: torch.Tensor,
    ) -> torch.Tensor:
        assert self.kv_sharing_fast_prefill_logits_indices is not None
        num_logits = logits_indices.shape[0]
        assert num_logits > 0
        self.kv_sharing_fast_prefill_logits_indices[:num_logits].copy_(logits_indices)
        # There might have leftover indices in logits_indices[num_logits:]
        # from previous iterations, whose values may be greater than the
        # batch size in the current iteration. To ensure indices are always
        # valid, we fill the padded indices with the last index.
        self.kv_sharing_fast_prefill_logits_indices[num_logits:].fill_(
            logits_indices[-1].item()
        )
        return self.kv_sharing_fast_prefill_logits_indices[:num_logits]

    def get_model(self) -> torch.nn.Module:
        if not hasattr(self, "model"):
            raise ValueError("Cannot get model before model has been initialized")
        return self.model

    def get_supported_generation_tasks(self) -> list[GenerationTask]:
        model = self.get_model()
        supported_tasks = list[GenerationTask]()

        if is_text_generation_model(model):
            supported_tasks.append("generate")

        if supports_transcription(model):
            if model.supports_transcription_only:
                return ["transcription"]

            supported_tasks.append("transcription")

        if supports_realtime(model):
            supported_tasks.append("realtime")

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if "score" in supported_tasks:
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)
            if num_labels != 1:
                supported_tasks.remove("score")
                logger.debug_once("Score API is only enabled for num_labels == 1.")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def _preprocess(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_input_tokens: int,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        torch.Tensor | None,
        torch.Tensor | None,
        torch.Tensor,
        IntermediateTensors | None,
    ]:
        """
        :return: tuple[input_ids, inputs_embeds, positions, intermediate_tensors]
        """
        # For text-only models
        input_ids = self.input_ids[:num_input_tokens]
        inputs_embeds = None

        positions = self.positions[:num_input_tokens]

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None

        input_ids = input_ids.view(num_reqs, -1)
        positions = positions.view(num_reqs, -1)

        if self.is_prefill:
            input_ids = pad(input_ids, -1, self.max_num_tokens)
            positions = pad(positions, -1, self.max_num_tokens)
        else:
            input_ids = pad(input_ids, 0, num_reqs_padded)
            positions = pad(positions, 0, num_reqs_padded)

        return (
            input_ids,
            inputs_embeds,
            positions,
            intermediate_tensors,
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            return self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )

        raise NotImplementedError

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
    ) -> tuple[
        dict[str, int],
        LogprobsLists,
        list[list[int]],
        dict[str, LogprobsTensors | None],
        list[str],
        dict[str, int],
    ]:
        """
        :return: tuple[num_nans_in_logits, logprobs_lists, valid_sampled_token_ids,
                    prompt_logprobs_dict, req_ids_output_copy,
                    req_id_to_index_output_copy]
        """
        num_nans_in_logits: dict[str, int] = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            raise NotImplementedError
            num_nans_in_logits = self._get_nans_in_logits(logits)

        num_reqs = self.input_batch.num_reqs
        discard_sampled_tokens_req_indices = np.nonzero(
            self.discard_request_mask.numpy()[:num_reqs]
        )[0]
        for i in discard_sampled_tokens_req_indices:
            gen = self.input_batch.generators.get(int(i))
            if gen is not None:
                gen.set_offset(gen.get_offset() - 4)

        # Copy some objects so they don't get modified after returning.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = None

        # Get the valid generated tokens.
        max_gen_len = sampled_token_ids.shape[-1]
        if max_gen_len == 1:
            # No spec decode tokens.
            valid_sampled_token_ids: list[list[int]] = sampled_token_ids.tolist()
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                valid_sampled_token_ids[int(i)].clear()

            if logprobs_tensors is not None:
                logprobs_lists = logprobs_tensors.tolists()
        else:
            # Includes spec decode tokens.
            valid_sampled_token_ids, logprobs_lists = RejectionSampler.parse_output(
                sampled_token_ids,
                self.input_batch.vocab_size,
                discard_sampled_tokens_req_indices,
                logprobs_tensors=logprobs_tensors,
            )

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            sampled_ids = valid_sampled_token_ids[req_idx]
            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}"
            )

            self.input_batch.token_ids_cpu[req_idx, start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> ModelRunnerOutput | IntermediateTensors | None:
        if self.execute_model_state is not None:
            raise RuntimeError(
                "State error: sample_tokens() must be called "
                "after execute_model() returns None."
            )

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        with record_function_or_nullcontext("rbln_model_runner: preprocess"):
            # Update persistent batch states.
            self._update_states(scheduler_output)

            if not num_scheduled_tokens:
                return EMPTY_MODEL_RUNNER_OUTPUT

            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect "
                    "logprobs for prompt tokens, tokens, please disable "
                    "it when the requests need prompt logprobs"
                )

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            logits_indices, spec_decode_metadata = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            num_reqs_padded = self._determine_batch_padding(num_reqs)
            num_tokens_padded, num_tokens_across_dp = None, None

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0

            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_tokens_unpadded,
                    num_tokens_padded=num_tokens_padded,
                    max_query_len=int(num_scheduled_tokens_np.max()),
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                )
            )

            (
                input_ids,
                inputs_embeds,
                positions,
                intermediate_tensors,
            ) = self._preprocess(
                num_reqs, num_reqs_padded, num_tokens_unpadded, intermediate_tensors
            )

        token_indices: torch.Tensor | None = None
        if self.is_prefill and self.use_wrapped_compute_logits:
            token_indices = logits_indices

        # Run the model.
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_scheduled_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                num_padded_tokens=num_tokens_padded,
                **build_kv_cache_forward_context_kwargs(self.kv_cache_bases),
            ),
            record_function_or_nullcontext("rbln_model_runner: forward"),
        ):
            model_output = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                token_indices=token_indices,
            )

        with record_function_or_nullcontext("rbln_model_runner: postprocess"):
            hidden_states, aux_hidden_states, logits = model_output
            assert aux_hidden_states is None

            if not get_pp_group().is_last_rank:
                # Return the intermediate tensors.
                assert isinstance(hidden_states, IntermediateTensors)
                return hidden_states

            sample_hidden_states = hidden_states
            assert self.use_wrapped_compute_logits
            if not self.is_prefill:
                logits = logits[:num_scheduled_tokens]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
        )
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput:
        if self.execute_model_state is None:
            output = copy(EMPTY_MODEL_RUNNER_OUTPUT)
            return output

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
        ) = self.execute_model_state
        self.execute_model_state = None  # Clear ephemeral state

        # TODO(RBLN): structured output bitmasks if present.
        if grammar_output is not None:
            # NOTE(RBLN): `xgr.apply_token_bitmask_inplace` requires logits
            # to be float32 dtype for CPU tensors
            origin_dtype = logits.dtype
            logits = logits.to(torch.float32)
            apply_grammar_bitmask(
                scheduler_output, grammar_output, self.input_batch, logits
            )
            logits = logits.to(origin_dtype)

        with record_function_or_nullcontext("rbln_model_runner: sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        # TODO(RBLN): need to process draft tokens

        with record_function_or_nullcontext("rbln_model_runner: bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
            )

        with record_function_or_nullcontext("rbln_model_runner: ModelRunnerOutput"):
            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                num_nans_in_logits=num_nans_in_logits,
            )

        return output

    @instrument(span_name="Loading (NPU)")
    def load_model(self) -> None:
        logger.info(
            "Starting to load model %s...",
            self.model_config.model,
        )

        model_loader = get_model_loader(self.load_config)
        self.model = model_loader.load_model(
            vllm_config=self.vllm_config, model_config=self.model_config
        )
        if hasattr(self.model, "logits_processor"):
            self.logits_processor = self.model.logits_processor
        else:
            self.logits_processor = None
        # TODO(RBLN): load lora
        # TODO(RBLN): load drafter

        if self.use_aux_hidden_state_outputs:
            raise NotImplementedError

        # NOTE(RBLN): This wrapper is designed to be compiled by torch.compile.
        # It handles the forward pass of the underlying model and computes
        # the logits from the hidden_states if necessary.
        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
            token_indices: torch.Tensor | None = None,
        ):
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
            )

            logits = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output
                aux_hidden_states = None

            if (
                get_pp_group().is_last_rank
                and self.use_wrapped_compute_logits
                and self.logits_processor is not None
            ):
                if token_indices is not None:
                    hidden_states = hidden_states[:, token_indices]
                logits = self.model.compute_logits(hidden_states)
                logits = logits.view(-1, logits.size(-1))

            return hidden_states, aux_hidden_states, logits

        if self.model_config.enforce_eager or not envs.VLLM_RBLN_COMPILE_MODEL:
            self.model_executable = model_wrapper
            self.compute_logits = self.model.compute_logits
        else:
            # NOTE(RBLN): Refer to pytorch 2.5 release notes.
            # To prevent nn.modules parameters to be modmel input, set false.
            # If this flag is set, nn.modules parameters are treated as model input.
            torch._dynamo.config.inline_inbuilt_nn_modules = False
            torch._dynamo.config.cache_size_limit = 64

            self.model_executable = self._compile(model_wrapper)
            self.compute_logits = self._compile(self.model.compute_logits)

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        if not (num_prompt_logprobs_dict := self.num_prompt_logprobs):
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, LogprobsTensors | None] = {}

        # Since prompt logprobs are a rare feature, prioritize simple,
        # maintainable loop over optimal performance.
        completed_prefill_reqs = []
        for req_id, num_prompt_logprobs in num_prompt_logprobs_dict.items():
            num_tokens = num_scheduled_tokens.get(req_id)
            if num_tokens is None:
                # This can happen if the request was preempted in prefill stage.
                continue

            # Get metadata for this request.
            request = self.requests[req_id]
            if request.prompt_token_ids is None:
                # Prompt logprobs is incompatible with prompt embeddings
                continue

            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True
            )

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                in_progress_dict[req_id] = logprobs_tensors

            # Determine number of logits to retrieve.
            start_idx = request.num_computed_tokens
            start_tok = start_idx + 1
            num_remaining_tokens = num_prompt_tokens - start_tok
            if num_tokens <= num_remaining_tokens:
                # This is a chunk, more tokens remain.
                # In the == case, there are no more prompt logprobs to produce
                # but we want to defer returning them to the next step where we
                # have new generated tokens to return.
                num_logits = num_tokens
            else:
                # This is the last chunk of prompt tokens to return.
                num_logits = num_remaining_tokens
                completed_prefill_reqs.append(req_id)
                prompt_logprobs_dict[req_id] = logprobs_tensors

            if num_logits <= 0:
                # This can happen for the final chunk if we prefilled exactly
                # (num_prompt_tokens - 1) tokens for this request in the prior
                # step. There are no more prompt logprobs to produce.
                continue

            # Get the logits corresponding to this req's prompt tokens.
            # If this is a partial request (i.e. chunked prefill),
            # then there is prompt logprob generated for each index.
            req_idx = self.input_batch.req_id_to_index[req_id]
            offset = self.query_start_loc[req_idx].item()
            prompt_hidden_states = hidden_states[offset : offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok : start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks, _ = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        return prompt_logprobs_dict

    def _get_nans_in_logits(
        self,
        logits: torch.Tensor | None,
    ) -> dict[str, int]:
        try:
            if logits is None:
                return {req_id: 0 for req_id in self.input_batch.req_ids}

            num_nans_in_logits = {}
            num_nans_for_index = logits.isnan().sum(dim=-1).numpy()
            for req_id in self.input_batch.req_ids:
                req_index = self.input_batch.req_id_to_index[req_id]
                num_nans_in_logits[req_id] = (
                    int(num_nans_for_index[req_index])
                    if num_nans_for_index is not None and req_index < logits.shape[0]
                    else 0
                )
            return num_nans_in_logits
        except IndexError:
            return {}

    @torch.inference_mode()
    def _dummy_run(
        self, num_reqs: int, num_tokens_per_req: int, is_prefill: bool
    ) -> None:
        """
        Run a dummy forward pass to warm up for the model.
        """
        num_tokens = num_tokens_per_req * num_reqs
        assert num_tokens <= self.max_num_tokens
        assert num_reqs <= self.max_num_reqs

        num_scheduled_tokens_list = [num_tokens_per_req] * num_reqs
        num_scheduled_tokens = np.array(num_scheduled_tokens_list, dtype=np.int32)
        num_tokens_unpadded = int(num_scheduled_tokens.sum())

        num_reqs_padded = num_reqs
        num_tokens_padded, num_tokens_across_dp = None, None

        seq_lens_np = self.seq_lens.numpy()
        seq_lens_np[:num_reqs] = num_scheduled_tokens
        seq_lens_np[num_reqs:] = 0

        self.input_batch.num_tokens_no_spec[:num_reqs] = num_scheduled_tokens

        cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
        query_start_loc_np = self.query_start_loc.numpy()
        query_start_loc_np[1 : num_reqs + 1] = cum_num_tokens

        attn_metadata, _ = self._build_attention_metadata(
            num_tokens=num_tokens_unpadded,
            num_tokens_padded=num_tokens_padded,
            max_query_len=num_tokens_per_req,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
        )

        input_ids = self.input_ids[:num_tokens_unpadded]
        inputs_embeds = None
        positions = self.positions[:num_tokens_unpadded]
        token_indices: torch.Tensor | None = None
        if self.use_wrapped_compute_logits and is_prefill:
            token_indices = torch.arange(
                num_tokens_per_req - 1,
                num_reqs * num_tokens_per_req,
                num_tokens_per_req,
                device=input_ids.device,
                dtype=torch.int32,
            )

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = self.model.make_empty_intermediate_tensors(
                batch_size=num_tokens_unpadded,
                dtype=self.model_config.dtype,
                device=self.device,
            )
            intermediate_tensors = IntermediateTensors(
                {
                    k: v.view(num_reqs_padded, num_tokens_per_req, -1)
                    for k, v in intermediate_tensors.items()
                }
            )

        # NOTE(RBLN): Clone tensors to make tensors non-view tensors.
        input_ids = input_ids.view(num_reqs, num_tokens_per_req).clone()
        positions = positions.view(num_reqs, num_tokens_per_req).clone()

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_tokens_padded,
            **build_kv_cache_forward_context_kwargs(self.kv_cache_bases),
        ):
            _ = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                token_indices=token_indices,
            )

        self.input_batch.num_tokens_no_spec[:num_reqs] = 0

    @torch.inference_mode()
    def _dummy_sampler_run(self, num_reqs: int) -> None:
        logits = torch.randn(
            (num_reqs, self.model_config.get_vocab_size()),
            device=self.device,
            dtype=self.dtype,
        )

        dummy_tensors = lambda v: torch.full((num_reqs,), v, device=self.device)
        dummy_metadata = SamplingMetadata(
            temperature=None,
            all_greedy=True,
            all_random=False,
            top_p=None,
            top_k=None,
            generators={},
            max_num_logprobs=None,
            no_penalties=True,
            prompt_token_ids=None,
            frequency_penalties=dummy_tensors(0.1),
            presence_penalties=dummy_tensors(0.1),
            repetition_penalties=dummy_tensors(0.1),
            output_token_ids=[],
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
            spec_token_ids=[[] for _ in range(num_reqs)],
        )

        _ = self.sampler(
            logits=logits,
            sampling_metadata=dummy_metadata,
        )

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]], set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names
            )
            attn_backends = {}
            attn_backend_layers = defaultdict(list)
            # Dedupe based on full class name; this is a bit safer than
            # using the class itself as the key because when we create dynamic
            # attention backend subclasses (e.g. ChunkedLocalAttention) unless
            # they are cached correctly, there will be different objects per
            # layer.
            for layer_name in kv_cache_group_spec.layer_names:
                attn_backend = layers[layer_name].get_attn_backend()

                if layer_name in self.kv_sharing_fast_prefill_eligible_layers:
                    attn_backend = create_fast_prefill_custom_backend(
                        "FastPrefill",
                        attn_backend,  # type: ignore[arg-type]
                    )

                full_cls_name = attn_backend.full_cls_name()
                layer_kv_cache_spec = kv_cache_group_spec.kv_cache_spec
                if isinstance(layer_kv_cache_spec, UniformTypeKVCacheSpecs):
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(
                    attn_backend, layer_kv_cache_spec
                )
                attn_backend_layers[key].append(layer_name)
            return (
                {attn_backends[k]: v for k, v in attn_backend_layers.items()},
                set(group_key.attn_backend for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend, kv_cache_spec), layer_names in attn_backends_map.items():
                attn_group = AttentionGroup(
                    attn_backend,
                    layer_names,
                    kv_cache_spec,
                    kv_cache_group_id,
                )

                attn_groups.append(attn_group)
            return attn_groups

        attention_backend_maps = []
        attention_backend_list = []
        for kv_cache_group_spec in kv_cache_config.kv_cache_groups:
            attn_backends = get_attn_backends_for_group(kv_cache_group_spec)
            attention_backend_maps.append(attn_backends[0])
            attention_backend_list.append(attn_backends[1])

        for i, attn_backend_map in enumerate(attention_backend_maps):
            self.attn_groups.append(create_attn_groups(attn_backend_map, i))

    def initialize_metadata_builders(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes)
                    else None,
                    num_metadata_builders=1,  # not use ubatching
                )

        # RBLN(TODO): Initialize drafter attention backend

    def may_reinitialize_input_batch(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        what it was originally created with. This happens when the final
        block size (determined after model loading) differs from the
        placeholder used during __init__, or when there are multiple
        KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        block_sizes = []
        max_num_blocks = []
        max_model_len = self.max_model_len
        for kv_cache_group in kv_cache_config.kv_cache_groups:
            assert not isinstance(
                kv_cache_group.kv_cache_spec, EncoderOnlyAttentionSpec
            )
            assert not isinstance(kv_cache_group.kv_cache_spec, MambaSpec)
            block_size = kv_cache_group.kv_cache_spec.block_size
            block_sizes.append(block_size)
            max_num_blocks_per_req = cdiv(max_model_len, block_size)
            max_num_blocks.append(max_num_blocks_per_req)

        if (
            block_sizes != self._init_block_sizes
            or kernel_block_sizes != self._init_kernel_block_sizes
        ):
            self._init_block_sizes = block_sizes
            self._init_kernel_block_sizes = kernel_block_sizes
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max_model_len,
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                max_num_blocks_per_req=max_num_blocks,
                is_spec_decode=False,
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=False,
            )

        assert self._init_block_sizes == block_sizes, (
            f"InputBatch block_sizes {self._init_block_sizes} != "
            f"kv_cache block_sizes {block_sizes}"
        )
        assert self._init_kernel_block_sizes == kernel_block_sizes, (
            f"InputBatch kernel_block_sizes {self._init_kernel_block_sizes} "
            f"!= kv_cache kernel_block_sizes {kernel_block_sizes}"
        )

    def _allocate_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig
    ) -> dict[str, torch.Tensor]:
        """
        Initializes the KV cache buffer with the correct size. The buffer needs
        to be reshaped to the desired shape before being used by the models.

        Args:
            kv_cache_config: The KV cache config
        Returns:
            dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_cache_raw_tensors: dict[str, torch.Tensor] = {}
        for kv_cache_tensor in kv_cache_config.kv_cache_tensors:
            device = "cpu" if not envs.VLLM_RBLN_COMPILE_MODEL else "meta"
            tensor = torch.zeros(kv_cache_tensor.size, dtype=torch.int8, device=device)
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys()), (
            "Some layers are not correctly initialized"
        )
        return kv_cache_raw_tensors

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> tuple[
        dict[str, torch.Tensor],
        dict[str, torch.Tensor],
        dict[str, KVCacheViewInfo],
    ]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Tuple of (kv_caches, kv_cache_base_tensors, kv_cache_view_infos):
            - kv_caches: layer name -> reshaped+permuted KV cache tensor
            - kv_cache_base_tensors: layer name -> typed base tensor (pre-permute)
            - kv_cache_view_infos: layer name -> view transformation metadata
        """
        kv_caches: dict[str, torch.Tensor] = {}
        kv_cache_base_tensors: dict[str, torch.Tensor] = {}
        kv_cache_view_infos: dict[str, KVCacheViewInfo] = {}
        for group in self._kv_cache_spec_attn_group_iterator():
            kv_cache_spec = group.kv_cache_spec
            attn_backend = group.backend
            if group.kv_cache_group_id == len(kernel_block_sizes):
                # There may be a last group for layers without kv cache.
                continue
            kernel_block_size = kernel_block_sizes[group.kv_cache_group_id]
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                raw_tensor = kv_cache_raw_tensors[layer_name]
                assert raw_tensor.numel() % kv_cache_spec.page_size_bytes == 0
                num_blocks = raw_tensor.numel() // kv_cache_spec.page_size_bytes
                if isinstance(kv_cache_spec, AttentionSpec):
                    num_blocks_per_kv_block = (
                        kv_cache_spec.block_size // kernel_block_size
                    )
                    kernel_num_blocks = num_blocks * num_blocks_per_kv_block

                    kv_cache_shape = attn_backend.get_kv_cache_shape(
                        kernel_num_blocks,
                        kernel_block_size,
                        kv_cache_spec.num_kv_heads,
                        kv_cache_spec.head_size,
                        cache_dtype_str=self.cache_config.cache_dtype,
                    )
                    dtype = kv_cache_spec.dtype
                    try:
                        kv_cache_stride_order = attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(
                        kv_cache_shape[i] for i in kv_cache_stride_order
                    )
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    # Keep the deduped base in a backend-native multidimensional
                    # shape so export/Relay never sees a giant flat dimension.
                    typed_base = (
                        kv_cache_raw_tensors[layer_name]
                        .view(dtype)
                        .view(kv_cache_shape)
                    )
                    kv_caches[layer_name] = typed_base.permute(*inv_order)
                    kv_cache_base_tensors[layer_name] = typed_base
                    kv_cache_view_infos[layer_name] = KVCacheViewInfo(
                        view_shape=kv_cache_shape,
                        permute_order=tuple(inv_order),
                    )
                else:
                    raise NotImplementedError

        return kv_caches, kv_cache_base_tensors, kv_cache_view_infos

    def initialize_kv_cache_tensors(
        self, kv_cache_config: KVCacheConfig, kernel_block_sizes: list[int]
    ) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        # TODO(RBLN): add uniform kv cache case for kv connector

        # General case
        kv_cache_raw_tensors = self._allocate_kv_cache_tensors(kv_cache_config)

        # Change the memory buffer to the desired shape
        kv_caches, kv_cache_bases_by_layer, kv_cache_view_infos = (
            self._reshape_kv_cache_tensors(
                kv_cache_config,
                kv_cache_raw_tensors,
                kernel_block_sizes,
            )
        )

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items():
            logger.debug("%s reuses KV cache of %s", layer_name, target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]
            kv_cache_bases_by_layer[layer_name] = kv_cache_bases_by_layer[
                target_layer_name
            ]
            if target_layer_name in kv_cache_view_infos:
                kv_cache_view_infos[layer_name] = kv_cache_view_infos[target_layer_name]

        validate_shared_attention_kv_cache_contiguity(
            kv_caches,
            kv_cache_bases_by_layer,
            kv_cache_view_infos,
        )

        num_attn_module = (
            2 if self.model_config.hf_config.model_type == "longcat_flash" else 1
        )
        self._update_kv_cache_base_bindings(
            kv_cache_bases_by_layer,
            kv_cache_view_infos,
            num_attn_module,
        )
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )

        if not self.model_config.enforce_eager and envs.VLLM_RBLN_COMPILE_MODEL:
            kv_cache_names = get_kv_cache_names(kv_caches, num_attn_module)
            for kv_cache, name in zip(self.kv_caches, kv_cache_names):
                self.compile_context.mark_static_address(kv_cache, name)

        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
        self, kv_cache_config: KVCacheConfig
    ) -> None:
        """
        Add layers that re-use KV cache to KV cache group of its target layer.
        Mapping of KV cache tensors happens in `initialize_kv_cache_tensors()`
        """
        if not self.shared_kv_cache_layers:
            # No cross-layer KV sharing, return
            return

        add_kv_sharing_layers_to_kv_cache_groups(
            self.shared_kv_cache_layers,
            kv_cache_config.kv_cache_groups,
            self.runner_only_attn_layers,
        )

        if self.cache_config.kv_sharing_fast_prefill:
            # In You Only Cache Once (https://arxiv.org/abs/2405.05254) or other
            # similar KV sharing setups, only the layers that generate KV caches
            # are involved in the prefill phase, enabling prefill to early exit.
            attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(layer_name)
                else:
                    break

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize KV cache based on `kv_cache_config`."""
        if envs.VLLM_RBLN_SUB_BLOCK_CACHE and (
            len(kv_cache_config.kv_cache_groups) > 1
        ):
            raise NotImplementedError(
                "Sub-block prefix caching does not support "
                "multi-group KV caches yet.  "
                "Set VLLM_RBLN_SUB_BLOCK_CACHE=false to disable."
            )

        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group, but the attention
        # backends for that group only supports block_size 64, we will return
        # kernel_block_size 64 and split the 256-token-block to 4 blocks with 64
        # tokens each.
        kernel_block_sizes = prepare_kernel_block_sizes(
            kv_cache_config, self.attn_groups
        )
        self._kernel_block_sizes = kernel_block_sizes

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        _ = self.initialize_kv_cache_tensors(kv_cache_config, kernel_block_sizes)

        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.cache_config.num_cpu_blocks = 0

        total_gb = sum(t.size for t in kv_cache_config.kv_cache_tensors) / 1024**3
        logger.info(
            "KV cache initialized: blocks=%d, groups=%d, tensors=%d, total=%.3f GiB",
            kv_cache_config.num_blocks,
            len(kv_cache_config.kv_cache_groups),
            len(kv_cache_config.kv_cache_tensors),
            total_gb,
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        layer_type = cast(type[Any], AttentionLayerBase)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
        for layer_name, attn_module in attn_layers.items():
            if isinstance(attn_module, Attention) and (
                kv_tgt_layer := attn_module.kv_sharing_target_layer_name
            ):
                # The layer doesn't need its own KV cache and will use that of
                # the target layer. We skip creating a KVCacheSpec for it, so
                # that KV cache management logic will act as this layer does
                # not exist, and doesn't allocate KV cache for the layer. This
                # enables the memory saving of cross-layer kv sharing, allowing
                # a given amount of memory to accommodate longer context lengths
                # or enable more requests to be processed simultaneously.
                self.shared_kv_cache_layers[layer_name] = kv_tgt_layer
                continue
            # Skip modules that don't need KV cache (eg encoder-only attention)
            if spec := attn_module.get_kv_cache_spec(self.vllm_config):
                kv_cache_spec[layer_name] = spec

        return kv_cache_spec

    ####################################################################################################
    # Only RBLN-Specific Methods
    ####################################################################################################

    @property
    def is_prefill(self) -> bool:
        num_computed_tokens = self.input_batch.num_computed_tokens_cpu[0]
        num_tokens_no_spec = self.input_batch.num_tokens_no_spec[0]
        return bool(num_computed_tokens < (num_tokens_no_spec - 1))

    @property
    def is_last_prefill(self) -> bool:
        num_computed_tokens = self.input_batch.num_computed_tokens_cpu[0]
        num_prompt_tokens = self.input_batch.num_prompt_tokens[0]
        return bool((num_computed_tokens + self.max_num_tokens) >= num_prompt_tokens)

    @property
    def use_wrapped_compute_logits(self) -> bool:
        return True

    def _attach_kv_cache_bindings(
        self, attn_metadata: PerLayerAttnMetadata | None
    ) -> None:
        if attn_metadata is None:
            return
        for attn_metadatum in attn_metadata.values():
            attach_kv_cache_bindings(
                attn_metadatum,
                self.kv_caches,
                self.kv_cache_bases,
                self.kv_cache_view_infos,
            )

    def _compile(self, model: torch.nn.Module):
        tp = get_tp_group()
        pp = get_pp_group()
        dp = get_dp_group()

        process_group_dict = {}
        process_group_dict[tp.device_group.group_name] = tp.ranks
        process_group_dict[tp.cpu_group.group_name] = tp.ranks
        process_group_dict[pp.device_group.group_name] = pp.ranks
        process_group_dict[pp.cpu_group.group_name] = pp.ranks
        process_group_dict[dp.device_group.group_name] = dp.ranks
        process_group_dict[dp.cpu_group.group_name] = dp.ranks

        options = {
            "compile_context": self.compile_context,
            "tensor_parallel_size": envs.VLLM_RBLN_TP_SIZE,
            "process_group_dict": process_group_dict,
            "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe,
            "_runtime_holder": self.runtime_holder,
        }
        if envs.VLLM_RBLN_COMPILE_STRICT_MODE:
            options["mode"] = "strict"
        if not envs.VLLM_DISABLE_COMPILE_CACHE:
            options["cache_dir"] = os.path.join(envs.VLLM_CACHE_ROOT, "rbln")

        return torch.compile(
            model,
            backend=rbln_backend,
            options=copy(options),
            dynamic=False,
        )

    def _determine_batch_padding(
        self,
        num_reqs_unpadded: int,
    ) -> int:
        num_reqs_padded = (
            self.bucketing_manager.find_decode_batch_bucket(num_reqs_unpadded)
            if not self.is_prefill
            else num_reqs_unpadded
        )

        # TODO(RBLN): Determine pads across dp
        return num_reqs_padded

    def _update_kv_cache_base_bindings(
        self,
        kv_cache_bases_by_layer: dict[str, torch.Tensor],
        kv_cache_view_infos_by_layer: dict[str, KVCacheViewInfo],
        num_attn_module: int,
    ) -> None:
        if not kv_cache_view_infos_by_layer:
            self.kv_cache_bases = []
            self.kv_cache_view_infos = []
            return

        kv_cache_bases, kv_cache_view_infos = build_kv_cache_base_bindings(
            kv_cache_bases_by_layer,
            kv_cache_view_infos_by_layer,
            num_attn_module=num_attn_module,
        )
        # If no deduplication occurred (each layer has its own unique base),
        # the new system adds overhead without benefit — disable it.
        if len(kv_cache_bases) == len(kv_cache_view_infos):
            self.kv_cache_bases = []
            self.kv_cache_view_infos = []
            return
        self.kv_cache_bases = kv_cache_bases
        self.kv_cache_view_infos = kv_cache_view_infos

    def warmup_model(self) -> None:
        logger.info("Compile and warming up model.")
        # 1. prefill
        self._dummy_run(
            1,
            self.max_num_tokens,
            True,
        )

        # 2. decode
        for size in self.bucketing_manager.decode_batch_buckets:
            self._dummy_run(
                size,
                1,  # query_len
                False,
            )

        # 3. compute_logits
        if not self.use_wrapped_compute_logits:
            for size in self.bucketing_manager.batch_buckets:
                hidden_states = torch.randn(
                    (size, self.model_config.get_hidden_size()),
                    device=self.device,
                    dtype=self.dtype,
                )
                _ = self.compute_logits(hidden_states)

        # 4. sampler
        for size in range(self.max_num_reqs):
            self._dummy_sampler_run(size)
