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

from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import nullcontext
from copy import copy, deepcopy
from typing import Any, NamedTuple, TypeAlias, cast

import numpy as np
import torch
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.config.cache import CacheConfig
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.layers.attention import Attention
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.models.interfaces import (
    supports_eagle3,
    supports_realtime,
    supports_transcription,
)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling,
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
    DraftTokenIds,
    LogprobsLists,
    LogprobsTensors,
    ModelRunnerOutput,
    PoolerOutput,
    SamplerOutput,
)
from vllm.v1.sample.logits_processor import LogitsProcessors, build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
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
from vllm_rbln.compilation import (
    build_process_group_dict,
    compile,
    create_compile_context,
    set_compile_stage,
)
from vllm_rbln.forward_context import RBLNDPMetadata, set_forward_context
from vllm_rbln.logger import init_logger
from vllm_rbln.platform import HAS_TORCH_RBLN, USE_DEVICE_TENSOR
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
from vllm_rbln.v1.core.rbln_kv_cache_manager import KVCacheCopyOp
from vllm_rbln.v1.core.rbln_scheduler import RBLNSchedulerOutput
from vllm_rbln.v1.sample.rbln_rejection_sampler import RBLNRejectionSampler
from vllm_rbln.v1.spec_decode.eagle import RBLNEagleProposer
from vllm_rbln.v1.spec_decode.medusa import RBLNMedusaProposer
from vllm_rbln.v1.worker.bucketing import get_bucketing_manager
from vllm_rbln.v1.worker.input_stager import InputLayout, InputStager, StagedModelInputs
from vllm_rbln.v1.worker.metrics_v2 import PerformanceContext, ProfileSection
from vllm_rbln.v1.worker.utils import prepare_kernel_block_sizes

logger = init_logger(__name__)

AttnMetadataDict: TypeAlias = dict[str, AttentionMetadata]
PerLayerAttnMetadata: TypeAlias = AttnMetadataDict  #  | list[AttnMetadataDict]


def _copy_pooler_output(
    raw_pooler_output: PoolerOutput, finished_mask: list[bool]
) -> list[torch.Tensor | None]:
    num_reqs = len(finished_mask)

    if isinstance(raw_pooler_output, torch.Tensor):
        if raw_pooler_output.shape[0] != num_reqs:
            raise ValueError(
                "Pooler output batch size does not match finished mask size: "
                f"{raw_pooler_output.shape[0]} != {num_reqs}"
            )

        num_finished = sum(finished_mask)
        if num_finished == 0:
            return [None] * num_reqs
        if num_finished == num_reqs:
            return list(raw_pooler_output)

        # partial finished
        finished_indices = [i for i, include in enumerate(finished_mask) if include]
        index_tensor = torch.tensor(
            finished_indices, device=raw_pooler_output.device, dtype=torch.long
        )
        finished_outputs = raw_pooler_output.index_select(0, index_tensor)
        partial_pooler_output: list[torch.Tensor | None] = [None] * num_reqs
        for i, out in zip(finished_indices, finished_outputs):
            partial_pooler_output[i] = out
        return partial_pooler_output

    assert isinstance(raw_pooler_output, list)
    if len(raw_pooler_output) != num_reqs:
        raise ValueError(
            "Pooler output batch size does not match finished mask size: "
            f"{len(raw_pooler_output)} != {num_reqs}."
        )

    pooler_output: list[torch.Tensor | None] = [None] * num_reqs
    for i, (out, include) in enumerate(zip(raw_pooler_output, finished_mask)):
        if include and out is not None:
            pooler_output[i] = out
    return pooler_output


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: RBLNSchedulerOutput
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
        self.speculative_config = vllm_config.speculative_config
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

        # Pooling models
        self.is_pooling_model = model_config.runner_type == "pooling"

        # These will be overridden in load_model()
        self.max_model_len = model_config.max_model_len
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        self.dcp_world_size = self.parallel_config.decode_context_parallel_size

        # Model-related.
        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn

        # TODO(RBLN): Multi-modal data support
        # TODO(RBLN): Async scheduling

        # NOTE(RBLN): Compilation context for marking the KV cache address as static.
        self.compile_context = (
            create_compile_context(use_weight_sharing=True, use_global_ctx=True)
            if not USE_DEVICE_TENSOR
            else None
        )
        self.runtime_holder: list = []

        # Sampler
        if envs.VLLM_RBLN_SAMPLER:
            from vllm_rbln.v1.sample import RBLNSampler

            self.sampler = RBLNSampler(
                logprobs_mode=self.model_config.logprobs_mode,
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
        # Set up speculative decoding.
        # NOTE(Jiayi): We put the entire draft model on the last PP rank.
        # This is not ideal if there are many layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: (
                RBLNEagleProposer
                | RBLNMedusaProposer
                | NgramProposer
                | SuffixDecodingProposer
            )
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            elif self.speculative_config.method == "suffix":
                self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.method == "medusa":
                self.drafter = RBLNMedusaProposer(self.vllm_config, self.device)
            elif self.speculative_config.use_eagle():
                self.drafter = RBLNEagleProposer(self.vllm_config, self.device, self)
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = (
                        self.drafter.eagle3_use_aux_hidden_state
                    )
            else:
                raise ValueError(
                    "Unsupported speculative decoding method: "
                    f"{self.speculative_config.method}"
                )
            self.rejection_sampler = RBLNRejectionSampler(
                self.sampler, self.compile_context
            )

        self.num_spec_tokens = 0
        if self.speculative_config:
            self.num_spec_tokens = self.speculative_config.num_speculative_tokens
            draft_config = self.speculative_config.draft_model_config
            if draft_config is not None and draft_config.max_model_len is not None:
                self.effective_drafter_max_model_len = draft_config.max_model_len
            else:
                self.effective_drafter_max_model_len = self.max_model_len

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
            num_spec_tokens=self.num_spec_tokens,
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=parallel_config.cp_kv_cache_interleave_size,
        )

        # Persistent buffers
        self.input_ids = torch.zeros(self.max_num_tokens, dtype=torch.int32)
        self.positions = torch.zeros(self.max_num_tokens, dtype=torch.int64)
        self.query_start_loc = self._make_buffer(
            self.max_num_reqs + 1, dtype=torch.int32
        )
        self.seq_lens = torch.zeros(self.max_num_tokens, dtype=torch.int32)
        self.discard_request_mask = torch.zeros(self.max_num_reqs, dtype=torch.bool)
        self.input_stager = InputStager(self.device)

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

        # Cached outputs.
        self._draft_token_ids: list[list[int]] | torch.Tensor | None = None

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

        self.specialized_moe_decode = (
            parallel_config.data_parallel_size > 1
            and envs.VLLM_RBLN_SPECIALIZE_MOE_DECODE
        )

        self.performance_ctx = PerformanceContext("runner")

    def _get_positions(self, num_tokens: Any):
        assert not isinstance(num_tokens, int)
        return self.positions[:num_tokens]

    def _make_buffer(
        self, *size: int | torch.SymInt, dtype: torch.dtype, numpy: bool = True
    ) -> CpuGpuBuffer:
        return CpuGpuBuffer(
            *size,
            dtype=dtype,
            device=self.device,
            pin_memory=self.pin_memory,
            with_numpy=numpy,
        )

    def _init_model_kwargs(self):
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if (
                param.extra_kwargs is not None
                and (token_types := param.extra_kwargs.get("compressed_token_type_ids"))
                is not None
            ):
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        # TODO(RBLN): RBLN keeps seq_lens on CPU for now. If this becomes a device
        # tensor, switch to a CPU-side cached length buffer to avoid device copy.
        seq_lens = self.seq_lens[:num_reqs].tolist()
        token_type_ids = []

        for i in range(num_reqs):
            seq_len_i = seq_lens[i]
            pos = token_type_id_requests.get(i, seq_len_i)
            ids = (torch.arange(seq_len_i) >= pos).int()
            token_type_ids.append(ids)

        token_type_ids_cpu = torch.empty(sum(seq_lens), dtype=torch.int32)
        torch.cat(token_type_ids, out=token_type_ids_cpu)
        model_kwargs["token_type_ids"] = token_type_ids_cpu
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: RBLNSchedulerOutput) -> None:
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

    def _update_states(self, scheduler_output: RBLNSchedulerOutput) -> None:
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
                generator = torch.Generator(device="cpu")
                generator.manual_seed(sampling_params.seed)
            else:
                generator = None

            if self.is_pooling_model:
                assert pooling_params is not None
                task = pooling_params.task
                assert task is not None, "You did not set `task` in the API"

                model = cast(VllmModelForPooling, self.get_model())
                to_update = model.pooler.get_pooling_updates(task)
                to_update.apply(pooling_params)

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
        scheduler_output: RBLNSchedulerOutput,
        num_scheduled_tokens: np.ndarray,
    ) -> tuple[torch.Tensor, SpecDecodeMetadata | None, np.ndarray, int]:
        assert scheduler_output.total_num_scheduled_tokens > 0
        assert (num_reqs := self.input_batch.num_reqs) > 0
        logical_num_tokens = num_scheduled_tokens

        # NOTE(RBLN): Build the fixed full-spec query only when the scheduler
        # actually kept draft tokens. Unsafe boundary cases and zero-draft
        # ngram/suffix steps clear scheduled_spec_decode_tokens and run with
        # the logical query length, usually qlen=1.
        use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0
        if use_spec_decode and self.num_spec_tokens > 0 and not self.is_prefill:
            target_query_len = self.num_spec_tokens + 1
            query_lengths = np.full(num_reqs, target_query_len, dtype=np.int32)
            backfill = query_lengths - logical_num_tokens

            assert np.all(backfill >= 0), (
                f"query_lengths={query_lengths}, "
                f"logical_num_tokens={logical_num_tokens}"
            )
        else:
            query_lengths = logical_num_tokens
            backfill = np.zeros_like(logical_num_tokens)

        total_query_tokens = int(query_lengths.sum())

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs], query_lengths)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(query_lengths)

        # Get positions.
        positions_np = self.positions.numpy()[:total_query_tokens]
        np.add(
            self.input_batch.num_computed_tokens_cpu[req_indices]
            - backfill[req_indices],
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
            out=self.input_ids[:total_query_tokens],
        )

        # Prepare the attention metadata.
        query_start_loc_np = self.query_start_loc.np
        query_start_loc_np[0] = 0
        query_start_loc_np[1 : num_reqs + 1] = cu_num_tokens
        query_start_loc_np[num_reqs + 1 :].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()

        seq_lens_np = self.seq_lens.numpy()
        seq_lens_np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] + logical_num_tokens
        )
        seq_lens_np[num_reqs:].fill(0)

        num_tokens = [self.requests[r].num_tokens for r in self.input_batch.req_ids]
        num_tokens_np = np.array(num_tokens, dtype=np.int32)

        # Record which requests should not be sampled,
        # so that we could clear the sampled tokens before returningj
        discard_request_mask_np = self.discard_request_mask.numpy()
        discard_request_mask_np[:num_reqs] = seq_lens_np[:num_reqs] < num_tokens_np

        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = self.query_start_loc.cpu[1 : num_reqs + 1] - 1
            spec_decode_metadata = None
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            # For chunked prefills, use -1 as mask rather than 0, as guided
            # decoding may rollback speculative tokens.
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for (
                req_id,
                draft_token_ids,
            ) in scheduler_output.scheduled_spec_decode_tokens.items():
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]
                ):
                    num_decode_draft_tokens[req_idx] = len(draft_token_ids)
            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens
            )
            logits_indices = spec_decode_metadata.logits_indices

        # TODO(RBLN): Hot-Swap lora model

        return (
            logits_indices,
            spec_decode_metadata,
            query_lengths,
            total_query_tokens,
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

        def _get_block_table(kv_cache_gid: int):
            blk_table = self.input_batch.block_table[kv_cache_gid]
            blk_table_tensor = blk_table.get_cpu_tensor()[:num_reqs]

            return blk_table_tensor

        cm_base = CommonAttentionMetadata(
            query_start_loc=self.query_start_loc.gpu[: num_reqs + 1],
            query_start_loc_cpu=self.query_start_loc.cpu[: num_reqs + 1],
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

            if self.speculative_config and spec_decode_common_attn_metadata is None:
                if isinstance(self.drafter, RBLNEagleProposer):
                    if self.drafter.kv_cache_gid == kv_cache_gid:
                        spec_decode_common_attn_metadata = cm
                else:
                    spec_decode_common_attn_metadata = cm

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

    def _calc_spec_decode_metadata(
        self,
        num_draft_tokens: np.ndarray,
        cu_num_scheduled_tokens: np.ndarray,
    ) -> SpecDecodeMetadata:
        # Inputs:
        # cu_num_scheduled_tokens:  [  4, 104, 107, 207, 209]
        # num_draft_tokens:         [  3,   0,   2,   0,   1]
        # Outputs:
        # cu_num_draft_tokens:      [  3,   3,   5,   5,   6]
        # logits_indices:           [  0,   1,   2,   3, 103, 104, 105, 106,
        #                            206, 207, 208]
        # target_logits_indices:    [  0,   1,   2,   5,   6,   9]
        # bonus_logits_indices:     [  3,   4,   7,   8,  10]

        # Compute the logits indices.
        # [4, 1, 3, 1, 2]
        num_sampled_tokens = num_draft_tokens + 1

        # Step 1. cu_num_sampled_tokens: [4, 5, 8, 9, 11]
        # arange: [0, 1, 2, 3, 0, 0, 1, 2, 0, 0, 1]
        cu_num_sampled_tokens, arange = self._get_cumsum_and_arange(
            num_sampled_tokens, cumsum_dtype=np.int32
        )
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens
        )
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32
        )
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens
        )
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # Make tensors.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens)
        logits_indices = torch.from_numpy(logits_indices)
        target_logits_indices = torch.from_numpy(target_logits_indices)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids[logits_indices]
        draft_token_ids = draft_token_ids[target_logits_indices + 1]

        return SpecDecodeMetadata(
            draft_token_ids=draft_token_ids,
            num_draft_tokens=num_draft_tokens.tolist(),
            cu_num_draft_tokens=cu_num_draft_tokens,
            cu_num_sampled_tokens=cu_num_sampled_tokens,
            target_logits_indices=target_logits_indices,
            bonus_logits_indices=bonus_logits_indices,
            logits_indices=logits_indices,
        )

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

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        # kv_connector_output: KVConnectorOutput | None,
    ) -> ModelRunnerOutput:
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == len(self.input_batch.pooling_params), (
            "Either all or none of the requests in a batch must be pooling request"
        )

        hidden_states = hidden_states[:num_scheduled_tokens]
        seq_lens_cpu = self.seq_lens[:num_reqs]

        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(
            num_scheduled_tokens_np, seq_lens_cpu, device=hidden_states.device
        )

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata
        )

        finished_mask = [
            seq_len == prompt_len
            for seq_len, prompt_len in zip(seq_lens_cpu, pooling_metadata.prompt_lens)
        ]

        model_runner_output = ModelRunnerOutput(
            req_ids=self.input_batch.req_ids.copy(),
            req_id_to_index=self.input_batch.req_id_to_index.copy(),
            kv_connector_output=None,
        )

        if raw_pooler_output is None or not any(finished_mask):
            model_runner_output.pooler_output = [None] * num_reqs
            return model_runner_output

        model_runner_output.pooler_output = _copy_pooler_output(
            raw_pooler_output=raw_pooler_output,
            finished_mask=finished_mask,
        )

        return model_runner_output

    def _preprocess(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_input_tokens: int,
        logits_indices: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
    ) -> tuple[
        StagedModelInputs,
        dict[str, Any],
    ]:
        """
        :return: tuple[
            staged_model_inputs,
            model_kwargs,
        ]
        """
        # For text-only models
        input_ids = self.input_ids[:num_input_tokens].view(num_reqs, -1)
        positions = self.positions[:num_input_tokens].view(num_reqs, -1)
        inputs_embeds = None

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            assert intermediate_tensors is not None

        layout = InputLayout(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs if self.is_prefill else num_reqs_padded,
            query_len=input_ids.shape[1],
            query_len_padded=self.max_num_tokens
            if self.is_prefill
            else input_ids.shape[1],
        )
        staged_model_inputs = self.input_stager.stage(
            input_ids=input_ids,
            positions=positions,
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            token_indices=logits_indices
            if self.is_prefill and self.use_wrapped_compute_logits
            else None,
            layout=layout,
        )

        model_kwargs = {
            **self._init_model_kwargs(),
        }

        return (
            staged_model_inputs,
            model_kwargs,
        )

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        if self.is_intermediate_chunked_prefill:
            # NOTE(RBLN): During intermediate chunked prefill, skip sampling and return
            # empty tensor with expected shape for performance. The output is discarded
            # anyway through discard_request_mask.
            assert logits is not None
            return SamplerOutput(
                sampled_token_ids=torch.full(
                    (1, 1), -1, dtype=torch.int32, device=logits.device
                ),
                logprobs_tensors=None,
            )

        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        with self.performance_ctx.profile(
            section=ProfileSection.SAMPLER,
            token_count=logits.shape[0] if logits is not None else 0,
        ):
            if spec_decode_metadata is None:
                return self.sampler(
                    logits=logits,
                    sampling_metadata=sampling_metadata,
                )

            return self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                logits,
                sampling_metadata,
            )

    def _bookkeeping_sync(
        self,
        scheduler_output: RBLNSchedulerOutput,
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
            valid_sampled_token_ids, logprobs_lists = RBLNRejectionSampler.parse_output(
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
        scheduler_output: RBLNSchedulerOutput,
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

            # Process sub-block KV cache copy operations before the forward
            # pass so that partially cached blocks are populated.
            if scheduler_output.kv_cache_copy_ops:
                self._process_kv_cache_copy_ops(scheduler_output.kv_cache_copy_ops)

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

            (
                logits_indices,
                spec_decode_metadata,
                query_lengths,
                num_query_tokens,
            ) = self._prepare_inputs(
                scheduler_output,
                num_scheduled_tokens_np,
            )

            num_reqs_padded, num_tokens_padded, num_tokens_across_dp = (
                self._determine_batch_padding(num_reqs, num_query_tokens)
            )

            use_spec_decode = len(scheduler_output.scheduled_spec_decode_tokens) > 0

            attn_metadata, spec_decode_common_attn_metadata = (
                self._build_attention_metadata(
                    num_tokens=num_query_tokens,
                    num_tokens_padded=num_tokens_padded,
                    max_query_len=int(query_lengths.max()),
                    num_reqs=num_reqs,
                    num_reqs_padded=num_reqs_padded,
                    logits_indices=logits_indices,
                    use_spec_decode=use_spec_decode,
                )
            )

            (
                staged_model_inputs,
                model_kwargs,
            ) = self._preprocess(
                num_reqs,
                num_reqs_padded,
                num_query_tokens,
                logits_indices,
                intermediate_tensors,
            )

        # Run the model.
        with (
            set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_query_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                num_padded_tokens=num_tokens_padded,
                **build_kv_cache_forward_context_kwargs(self.kv_cache_bases),
            ),
            self.performance_ctx.profile(
                self.is_prefill,
                section=ProfileSection.MODEL,
                token_count=num_scheduled_tokens,
            ),
            record_function_or_nullcontext("rbln_model_runner: forward"),
        ):
            model_output = self.model_executable(
                **staged_model_inputs.as_kwargs(),
                **model_kwargs,
            )

        with record_function_or_nullcontext("rbln_model_runner: postprocess"):
            hidden_states, aux_hidden_states, logits = model_output

            if not get_pp_group().is_last_rank:
                # Return the intermediate tensors.
                assert isinstance(hidden_states, IntermediateTensors)
                return hidden_states

            if self.is_pooling_model:
                # Return the pooling output.
                return self._pool(
                    hidden_states.flatten(0, -2),
                    num_scheduled_tokens,
                    num_scheduled_tokens_np,
                )

            sample_hidden_states = hidden_states
            assert self.use_wrapped_compute_logits
            if not self.is_prefill:
                logits = logits[logits_indices]

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

        self._draft_token_ids = None

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("rbln_model_runner: draft"):
                self._draft_token_ids = self.propose_draft_token_ids(
                    scheduler_output,
                    sampled_token_ids,
                    self.input_batch.sampling_metadata,
                    hidden_states,
                    sample_hidden_states,
                    aux_hidden_states,
                    spec_decode_metadata,
                    spec_decode_common_attn_metadata,
                )

        spec_config = self.speculative_config
        propose_drafts_after_bookkeeping = False
        if spec_config is not None:
            input_fits_in_drafter = spec_decode_common_attn_metadata is not None and (
                spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
                <= self.effective_drafter_max_model_len
            )
            # TODO(RBLN): supports mtp and extract hidden states
            if spec_config.use_eagle():
                if input_fits_in_drafter:
                    propose_draft_token_ids(sampler_output.sampled_token_ids)
            else:
                propose_drafts_after_bookkeeping = input_fits_in_drafter

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

        if propose_drafts_after_bookkeeping:
            propose_draft_token_ids(valid_sampled_token_ids)

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

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        req_ids = self.input_batch.req_ids.copy()
        if not self.num_spec_tokens or not req_ids:
            return None
        draft_token_ids = (
            self._draft_token_ids.tolist()
            if isinstance(self._draft_token_ids, torch.Tensor)
            else self._draft_token_ids
        )
        return DraftTokenIds(req_ids, draft_token_ids)

    def propose_draft_token_ids(
        self,
        scheduler_output: RBLNSchedulerOutput,
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> list[list[int]] | torch.Tensor:
        assert (spec_config := self.speculative_config) is not None
        if spec_config.method == "ngram":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
            )
        elif spec_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(self.input_batch, sampled_token_ids)
        elif spec_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, RBLNMedusaProposer)

            if spec_decode_metadata is None:
                target_hidden_states = sample_hidden_states.view(
                    -1, self.drafter.hidden_size
                )
            else:
                batch_indices = torch.arange(
                    len(sampled_token_ids), device=sample_hidden_states.device
                )
                last_token_indices = torch.tensor(
                    [len(t) - 1 for t in sampled_token_ids],
                    device=sample_hidden_states.device,
                )
                target_hidden_states = sample_hidden_states[
                    batch_indices, last_token_indices
                ]

            if self.is_intermediate_chunked_prefill:
                draft_token_ids = torch.zeros(
                    target_hidden_states.shape[0],
                    spec_config.num_speculative_tokens,
                    device=target_hidden_states.device,
                    dtype=torch.int64,
                )
            else:
                draft_token_ids = self.drafter.propose(
                    target_hidden_states, sampling_metadata
                )
        elif spec_config.use_eagle():
            assert isinstance(self.drafter, RBLNEagleProposer)
            assert isinstance(sampled_token_ids, torch.Tensor)

            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    common_attn_metadata,
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_mask,
                )
            )

            target_hidden_states = hidden_states
            num_rejected_tokens: torch.Tensor | None = None
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
                target_token_ids = self.input_ids[:num_scheduled_tokens]
                target_positions = self.positions[:num_scheduled_tokens]
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h[:num_scheduled_tokens] for h in aux_hidden_states], dim=-1
                    )
                else:
                    target_hidden_states = hidden_states[:num_scheduled_tokens]
            else:
                (
                    common_attn_metadata,
                    token_indices_to_sample,
                    num_rejected_tokens,
                ) = self.drafter.prepare_inputs_padded(
                    common_attn_metadata,
                    spec_decode_metadata,
                    valid_sampled_tokens_count,
                )
                total_num_tokens = common_attn_metadata.num_actual_tokens
                target_token_ids = self.input_ids[:total_num_tokens]
                target_positions = self.positions[:total_num_tokens]
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h.view(-1, h.shape[-1]) for h in aux_hidden_states], dim=-1
                    )

            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                token_indices_to_sample=token_indices_to_sample,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=None,
                num_rejected_tokens=num_rejected_tokens,
            )

        return draft_token_ids

    @instrument(span_name="Loading (NPU)")
    def load_model(self) -> None:
        logger.info(
            "Starting to load model %s...",
            self.model_config.model,
        )

        model_loader = get_model_loader(self.load_config)
        offload_context = nullcontext
        if HAS_TORCH_RBLN and USE_DEVICE_TENSOR and not envs.VLLM_RBLN_DISABLE_OFFLOAD:
            offload_context = torch.rbln.offload
        with offload_context():
            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config
            )
        if hasattr(self.model, "logits_processor"):
            self.logits_processor = self.model.logits_processor
        elif self.model_config.is_multimodal_model and hasattr(
            self.model.get_language_model(), "logits_processor"
        ):
            self.logits_processor = self.model.get_language_model().logits_processor
        else:
            self.logits_processor = None
        # TODO(RBLN): load lora
        if hasattr(self, "drafter"):
            logger.info_once("Loading drafter model...")
            self.drafter.load_model(self.model)

        if self.use_aux_hidden_state_outputs:
            if not supports_eagle3(self.get_model()):
                raise RuntimeError(
                    "Model does not support EAGLE3 interface but "
                    "aux_hidden_state_outputs was requested"
                )
            aux_layers = self._get_eagle3_aux_layers_from_config()
            if aux_layers:
                logger.info(
                    "Using auxiliary layers from speculative config: %s",
                    aux_layers,
                )
            else:
                aux_layers = self.model.get_eagle3_default_aux_hidden_state_layers()

            self.model.set_aux_hidden_state_layers(aux_layers)

        # NOTE(RBLN): This wrapper is designed to be compiled by torch.compile.
        # It handles the forward pass of the underlying model and computes
        # the logits from the hidden_states if necessary.
        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: IntermediateTensors | None = None,
            inputs_embeds: torch.Tensor | None = None,
            token_indices: torch.Tensor | None = None,
            **kwargs,
        ):
            with offload_context():
                model_output = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **kwargs,
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
                    sample_hidden_states = hidden_states[:, token_indices]
                    # NOTE(RBLN): token_indices points to the last-token positions used
                    # for sampling. EAGLE needs the full hidden_states during prefill,
                    # so do not slice them here.
                    if not (
                        self.speculative_config and self.speculative_config.use_eagle()
                    ):
                        hidden_states = sample_hidden_states
                else:
                    sample_hidden_states = hidden_states
                logits = self.model.compute_logits(sample_hidden_states)
                logits = logits.view(-1, logits.size(-1))

            return hidden_states, aux_hidden_states, logits

        if self.model_config.enforce_eager or not envs.VLLM_RBLN_COMPILE_MODEL:
            self.model_executable = model_wrapper
            self.compute_logits = self.model.compute_logits
        else:
            process_group_dict = build_process_group_dict()
            self.model_executable = compile(
                model_wrapper,
                dynamic=False,
                compile_context=self.compile_context,
                num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
                model_trace_method="export" if USE_DEVICE_TENSOR else "",
                process_group_dict=process_group_dict,
                guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                runtime_holder=self.runtime_holder,
                mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
            )
            # NOTE(RBLN): We compile compute_logits separately to cover cases when
            # `self.use_wrapped_compute_logits` is `False`
            self.compute_logits = compile(
                self.model.compute_logits,
                dynamic=False,
                compile_context=self.compile_context,
                num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
                model_trace_method="export" if USE_DEVICE_TENSOR else "",
                process_group_dict=process_group_dict,
                guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                runtime_holder=self.runtime_holder,
                mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
            )

    def _get_eagle3_aux_layers_from_config(self) -> tuple[int, ...] | None:
        """Extract Eagle3 auxiliary layer indices from speculative config."""
        if not (self.speculative_config and self.speculative_config.draft_model_config):
            return None

        hf_config = self.speculative_config.draft_model_config.hf_config
        if not hasattr(hf_config, "eagle_aux_hidden_state_layer_ids"):
            return None

        layer_ids = hf_config.eagle_aux_hidden_state_layer_ids
        if layer_ids and isinstance(layer_ids, (list, tuple)):
            return tuple(layer_ids)

        return None

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        if not (num_prompt_logprobs_dict := self.num_prompt_logprobs):
            return {}

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
            prompt_token_ids = torch.tensor(request.prompt_token_ids)

            # Set up target LogprobsTensors object.
            logprobs_tensors = request.in_progress_prompt_logprobs_cpu
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1
                )
                request.in_progress_prompt_logprobs_cpu = logprobs_tensors

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
            offset = self.query_start_loc.cpu[req_idx].item()
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

            # Transfer
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True
            )
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs, non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True
            )

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            self.requests[req_id].in_progress_prompt_logprobs_cpu = None

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
        self,
        num_reqs: int,
        num_tokens_per_req: int,
        is_prefill: bool,
        *,
        num_tokens_padded: int | None = None,
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

        seq_lens_np = self.seq_lens.numpy()
        seq_lens_np[:num_reqs] = num_scheduled_tokens
        seq_lens_np[num_reqs:] = 0

        # NOTE(RBLN): self.is_prefill is derived from num_tokens_no_spec.
        # For decode warmup, keep it at 1 so multi-token speculative decode
        # query lengths are not misclassified as prefill.
        if is_prefill:
            self.input_batch.num_tokens_no_spec[:num_reqs] = num_scheduled_tokens
        else:
            self.input_batch.num_tokens_no_spec[:num_reqs] = 1

        num_reqs_padded, _num_tokens_padded, num_tokens_across_dp = (
            self._determine_batch_padding(num_reqs, num_tokens_unpadded)
        )
        num_tokens_padded = num_tokens_padded or _num_tokens_padded

        cum_num_tokens, _ = self._get_cumsum_and_arange(num_scheduled_tokens)
        query_start_loc_np = self.query_start_loc.np
        query_start_loc_np[1 : num_reqs + 1] = cum_num_tokens

        attn_metadata, _ = self._build_attention_metadata(
            num_tokens=num_tokens_unpadded,
            num_tokens_padded=num_tokens_padded,
            max_query_len=num_tokens_per_req,
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs_padded,
            use_spec_decode=self.speculative_config is not None,
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
        staged_model_input = self.input_stager.stage(
            input_ids=input_ids.view(num_reqs, num_tokens_per_req),
            positions=positions.view(num_reqs, num_tokens_per_req),
            intermediate_tensors=intermediate_tensors,
            inputs_embeds=inputs_embeds,
            token_indices=token_indices,
            layout=InputLayout(
                num_reqs=num_reqs,
                num_reqs_padded=num_reqs,
                query_len=num_tokens_per_req,
                query_len_padded=num_tokens_per_req,
            ),
        )

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_tokens_padded,
            **build_kv_cache_forward_context_kwargs(self.kv_cache_bases),
        ):
            _ = self.model_executable(**staged_model_input.as_kwargs())

        self.input_batch.num_tokens_no_spec[:num_reqs] = 0

    @torch.inference_mode()
    def _dummy_sampler_run(self, num_reqs: int) -> None:
        from vllm_rbln.v1.sample import WARM_UP_CONFIGS

        logits = torch.randn(
            (num_reqs, self.model_config.get_vocab_size()),
            device=self.device,
            dtype=self.dtype,
        )

        def dummy_float_tensor(buffer: torch.Tensor, value: float | None):
            if value is None:
                return None
            return buffer[:num_reqs].fill_(float(value)).to(self.device)

        def dummy_int_tensor(buffer: torch.Tensor, value: int | float | None):
            if value is None:
                return None
            return buffer[:num_reqs].fill_(int(value)).to(self.device)

        for config in WARM_UP_CONFIGS:
            dummy_metadata = SamplingMetadata(
                temperature=dummy_float_tensor(
                    self.input_batch.temperature, config.get("temperature")
                ),
                all_greedy=config.get("all_greedy", True),
                all_random=config.get("all_random", False),
                top_p=dummy_float_tensor(self.input_batch.top_p, config.get("top_p")),
                top_k=dummy_int_tensor(self.input_batch.top_k, config.get("top_k")),
                generators={},
                max_num_logprobs=None,
                no_penalties=config.get("no_penalties", True),
                prompt_token_ids=torch.zeros(
                    (num_reqs, 1), dtype=torch.long, device=self.device
                )
                if not config.get("no_penalties", True)
                else None,
                frequency_penalties=dummy_float_tensor(
                    self.input_batch.frequency_penalties,
                    config.get("frequency_penalties", 0.1),
                ),
                presence_penalties=dummy_float_tensor(
                    self.input_batch.presence_penalties,
                    config.get("presence_penalties", 0.1),
                ),
                repetition_penalties=dummy_float_tensor(
                    self.input_batch.repetition_penalties,
                    config.get("repetition_penalties", 0.1),
                ),
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

        # Initialize drafter attention backend
        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, RBLNEagleProposer)
            self.drafter.initialize_attn_backend(kv_cache_config, kernel_block_sizes)

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
                num_spec_tokens=self.num_spec_tokens,
                logitsprocs=self.input_batch.logitsprocs,
                logitsprocs_need_output_token_ids=self.input_batch.logitsprocs_need_output_token_ids,
                is_pooling_model=self.is_pooling_model,
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
            device = (
                "cpu"
                if not envs.VLLM_RBLN_COMPILE_MODEL
                else self.device
                if USE_DEVICE_TENSOR
                else "meta"
            )
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

        if (
            not USE_DEVICE_TENSOR
            and not self.model_config.enforce_eager
            and envs.VLLM_RBLN_COMPILE_MODEL
        ):
            assert len(kv_caches) == len(self.kv_caches)
            for k, v in kv_caches.items():
                self.compile_context.mark_static_address(v, k)

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
    def is_intermediate_chunked_prefill(self) -> bool:
        return self.is_prefill and bool(self.discard_request_mask[0])

    @property
    def use_wrapped_compute_logits(self) -> bool:
        return not self.is_pooling_model

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

    def _determine_batch_padding(
        self,
        num_reqs_unpadded: int,
        num_tokens_unpadded: int,
    ) -> tuple[int, int | None, torch.Tensor | None]:
        num_reqs_padded = (
            self.bucketing_manager.find_decode_batch_bucket(num_reqs_unpadded)
            if not self.is_prefill
            else num_reqs_unpadded
        )
        if self.parallel_config.data_parallel_size == 1:
            return num_reqs_padded, None, None

        num_tokens_across_dp, num_reqs_across_dp = (
            RBLNDPMetadata.num_tokens_and_reqs_across_dp(
                num_tokens_unpadded,
                num_reqs_unpadded,
                self.parallel_config.data_parallel_size,
                self.parallel_config.data_parallel_rank,
                self.is_prefill,
            )
        )
        if num_reqs_across_dp is None or not self.specialized_moe_decode:
            num_tokens_padded = self.max_num_tokens
        else:
            num_reqs_padded = self.bucketing_manager.find_decode_batch_bucket(
                int(torch.max(num_reqs_across_dp).item())
            )
            assert num_reqs_padded is not None
            assert torch.all(num_tokens_across_dp % num_reqs_across_dp == 0)
            tokens_per_req_across_dp = num_tokens_across_dp // num_reqs_across_dp
            max_tokens_per_req = int(torch.max(tokens_per_req_across_dp).item())
            num_tokens_padded = num_reqs_padded * max_tokens_per_req

        return num_reqs_padded, num_tokens_padded, num_tokens_across_dp

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

        with set_compile_stage("warmup"):
            # 1. prefill
            self._dummy_run(1, self.max_num_tokens, True)

            # 2. decode
            query_lens = [1]
            if self.speculative_config:
                query_lens.append(self.speculative_config.num_speculative_tokens + 1)
            for num_req in self.bucketing_manager.decode_batch_buckets:
                for query_len in query_lens:
                    self._dummy_run(num_req, query_len, False)

            if self.specialized_moe_decode:
                # NOTE(RBLN): Compile decode graph with prefill-sized padding to cover
                # the DP-asymmetric case (this rank decoding while another rank
                # prefills). The bit-encoded all_reduce in get_dp_padding forces
                # num_padded_tokens to max_num_tokens whenever any rank prefills,
                # which the small-bucket decode graphs from 2. decode above cannot
                # satisfy.
                for num_req in self.bucketing_manager.decode_batch_buckets:
                    for query_len in query_lens:
                        self._dummy_run(
                            num_req,
                            query_len,
                            False,
                            num_tokens_padded=self.max_num_tokens,
                        )
                    if self.speculative_config:
                        # Cover DP-asymmetric decode where a peer runs spec decode.
                        spec_query_len = (
                            self.speculative_config.num_speculative_tokens + 1
                        )
                        self._dummy_run(
                            num_req,
                            1,
                            False,
                            num_tokens_padded=num_req * spec_query_len,
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
            if not self.is_pooling_model:
                for size in range(1, self.max_num_reqs + 1):
                    self._dummy_sampler_run(size)

            # 5. drafter
            if self.speculative_config:
                if self.speculative_config.method == "medusa":
                    self.drafter.dummy_run()
                elif self.speculative_config.use_eagle():
                    # prefill
                    self.drafter.dummy_run(1, self.max_num_tokens, True)

                    # decode
                    for num_req in self.bucketing_manager.decode_batch_buckets:
                        self.drafter.dummy_run(num_req, 1, False)

    def _process_kv_cache_copy_ops(
        self,
        copy_ops: list[KVCacheCopyOp],
    ) -> None:
        use_runtime_kv_copy = (
            not USE_DEVICE_TENSOR
            and not self.model_config.enforce_eager
            and envs.VLLM_RBLN_COMPILE_MODEL
        )
        for op in copy_ops:
            if use_runtime_kv_copy:
                runtime = self.runtime_holder[0]
                runtime._copy_kv_cache(op.src_block_id, op.dst_block_id, op.num_tokens)
            else:
                for kv_cache in self.kv_caches:
                    src = op.src_block_id
                    dst = op.dst_block_id
                    nt = op.num_tokens
                    kv_cache[:, dst, :, :, :nt, :] = kv_cache[:, src, :, :, :nt, :]
