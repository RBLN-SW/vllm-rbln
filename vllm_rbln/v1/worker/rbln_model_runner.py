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

import contextlib
import itertools
import math
import os
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from copy import copy, deepcopy
from typing import TYPE_CHECKING, Any, NamedTuple, Optional, Union, cast

import numpy as np
import rebel
import torch
import torch.nn as nn
from vllm.attention.backends.abstract import (AttentionBackend, AttentionType,
                                              MultipleOf)
from vllm.attention.layer import Attention
from vllm.config import VllmConfig, get_layers_from_vllm_config
from vllm.distributed.eplb.eplb_state import EplbState
from vllm.distributed.kv_transfer import (get_kv_transfer_group,
                                          has_kv_transfer_group)
from vllm.distributed.kv_transfer.kv_connector.utils import copy_kv_blocks
from vllm.distributed.parallel_state import (get_dp_group, get_pp_group,
                                             get_tp_group)
from vllm.forward_context import set_forward_context
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.model_executor.model_loader import TensorizerLoader, get_model_loader
from vllm.model_executor.models.interfaces import supports_transcription
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling, is_pooling_model, is_text_generation_model)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams, SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils.math_utils import round_up
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.utils.torch_utils import get_dtype_size, kv_cache_dtype_str_to_dtype
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadataBuilder
from vllm.v1.attention.backends.utils import (
    CommonAttentionMetadata, create_fast_prefill_custom_backend)
from vllm.v1.core.sched.output import (CachedRequestData, NewRequestData,
                                       SchedulerOutput)
# yapf conflicts with isort for this block
# yapf: disable
from vllm.v1.kv_cache_interface import (AttentionSpec, CrossAttentionSpec,
                                        EncoderOnlyAttentionSpec,
                                        KVCacheConfig, KVCacheGroupSpec,
                                        KVCacheSpec, MambaSpec,
                                        UniformTypeKVCacheSpecs)
# yapf: enable
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             DraftTokenIds, KVConnectorOutput, LogprobsLists,
                             LogprobsTensors, ModelRunnerOutput, SamplerOutput)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.logits_processor.interface import LogitsProcessor
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.medusa import MedusaProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.ngram_proposer import NgramProposer
from vllm.v1.spec_decode.suffix_decoding import SuffixDecodingProposer
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import CpuGpuBuffer, record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch
from vllm.v1.worker.kv_connector_model_runner_mixin import (
    KVConnectorModelRunnerMixin)
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin
from vllm.v1.worker.utils import (AttentionGroup, MultiModalBudget,
                                  add_kv_sharing_layers_to_kv_cache_groups,
                                  bind_kv_cache)

import vllm_rbln.rbln_envs as envs
import vllm_rbln.utils as rbln_utils
from vllm_rbln.forward_context import RBLNDPMetadata
from vllm_rbln.logger import init_logger
from vllm_rbln.lora.inputs import LoRAInputs
from vllm_rbln.lora.mask import LoRAMask
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadataBuilder)
from vllm_rbln.v1.kv_cache import RBLNSlidingWindowSpec
from vllm_rbln.v1.sample import RBLNSampler
from vllm_rbln.v1.worker.bucketing import get_bucketing_manager_class
from vllm_rbln.worker.metrics import PerformanceTracker

if TYPE_CHECKING:
    from vllm.model_executor.model_loader.tensorizer import TensorizerConfig
    from vllm.v1.core.sched.output import GrammarOutput

logger = init_logger(__name__)


# Wrapper for ModelRunnerOutput to support overlapped execution.
class AsyncRBLNModelRunnerOutput(AsyncModelRunnerOutput):

    def __init__(
        self,
        model_runner_output: ModelRunnerOutput,
        sampled_token_ids: torch.Tensor,
        invalid_req_indices: list[int],
        async_output_copy_stream: torch.cuda.Stream,
    ):
        self._model_runner_output = model_runner_output
        self._invalid_req_indices = invalid_req_indices

        # Event on the copy stream so we can synchronize the non-blocking copy.
        # self._async_copy_ready_event = torch.cuda.Event()

        # Keep a reference to the device tensor to avoid it being
        # deallocated until we finish copying it to the host.
        self._sampled_token_ids = sampled_token_ids

        # Initiate the copy on a separate stream, but do not synchronize it.
        # default_stream = torch.cuda.current_stream()
        # with torch.cuda.stream(async_output_copy_stream):
        #     async_output_copy_stream.wait_stream(default_stream)
        #     self._sampled_token_ids_cpu = self._sampled_token_ids.to(
        #         'cpu', non_blocking=True)
        #     self._async_copy_ready_event.record()

    def get_output(self) -> ModelRunnerOutput:
        """Copy the device tensors to the host and return a ModelRunnerOutput.

        This function blocks until the copy is finished.
        """
        self._async_copy_ready_event.synchronize()

        # Release the device tensor once the copy has completed
        del self._sampled_token_ids

        valid_sampled_token_ids = self._sampled_token_ids_cpu.tolist()
        for i in self._invalid_req_indices:
            valid_sampled_token_ids[i].clear()

        output = self._model_runner_output
        output.sampled_token_ids = valid_sampled_token_ids
        return output


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    spec_decode_metadata: SpecDecodeMetadata | None
    spec_decode_common_attn_metadata: CommonAttentionMetadata | None
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor | None
    aux_hidden_states: list[torch.Tensor] | None
    kv_connector_output: KVConnectorOutput | None


class DummyRunState(NamedTuple):
    """Input state for dummy run."""

    attn_metadata: dict[int, dict[str, Any]]
    num_input_tokens: int
    input_ids: dict[int, torch.Tensor]
    positions: dict[int, torch.Tensor]


class RBLNModelRunner(LoRAModelRunnerMixin, KVConnectorModelRunnerMixin):

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        self.speculative_config = vllm_config.speculative_config
        self.observability_config = vllm_config.observability_config

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
        set_cpu_offload_max_bytes(
            int(self.cache_config.cpu_offload_gb * 1024**3))
        assert self.cache_config.cpu_offload_gb == 0, "Not support cpu offload"

        model_config = self.model_config
        cache_config = self.cache_config
        scheduler_config = self.scheduler_config
        parallel_config = self.parallel_config
        self.device = device
        self.pin_memory = is_pin_memory_available()
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config)

        self.is_pooling_model = (model_config.runner_type == 'pooling')
        self.is_multimodal_raw_input_only_model = (
            model_config.is_multimodal_raw_input_only_model)

        self.max_model_len = model_config.max_model_len
        self.dcp_world_size = self.parallel_config.decode_context_parallel_size
        self.max_num_tokens = scheduler_config.max_num_batched_tokens
        self.max_num_reqs = scheduler_config.max_num_seqs

        # Model-related.
        self.num_query_heads = model_config.get_num_attention_heads(
            parallel_config)
        self.inputs_embeds_size = model_config.get_inputs_embeds_size()
        self.attention_chunk_size = model_config.attention_chunk_size
        # Only relevant for models using ALiBi (e.g, MPT)
        self.use_alibi = model_config.uses_alibi

        self.cascade_attn_enabled = not self.model_config.disable_cascade_attn
        assert not self.cascade_attn_enabled, "Not support cascade attention"

        # Multi-modal data support
        self.mm_registry = MULTIMODAL_REGISTRY
        self.uses_mrope = model_config.uses_mrope
        assert not self.uses_mrope, "RBLN does not support M-RoPE."

        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config)
        if envs.VLLM_RBLN_DISABLE_MM:
            self.supports_mm_inputs = False

        if self.model_config.is_encoder_decoder:
            # Maximum length of the encoder input, only for encoder-decoder
            # models.
            self.max_encoder_len = scheduler_config.max_num_encoder_input_tokens
        else:
            self.max_encoder_len = 0

        # Sampler
        self.use_rbln_sampler = envs.VLLM_RBLN_SAMPLER
        if self.use_rbln_sampler:
            logger.info("Using RBLN sampler: %s", self.use_rbln_sampler)
            sampler = RBLNSampler(
                logprobs_mode=self.model_config.logprobs_mode,
                seed=self.vllm_config.model_config.seed,
            )
        else:
            logger.info("Using default vLLM sampler.")
            sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)
        self.sampler = sampler

        # Lazy initialization
        self.compute_logits_model: nn.Module

        self.eplb_state: Optional[EplbState] = None
        """
        State of the expert parallelism load balancer.

        Will be lazily initialized when the model is loaded.
        """

        # Lazy initializations
        # self.model: nn.Module  # Set after load_model
        # Initialize in initialize_kv_cache
        self.kv_caches: list[torch.Tensor] = []
        # indexes: [kv_cache_group_id][attn_group]
        self.attn_groups: list[list[AttentionGroup]] = []
        # self.kv_cache_config: KVCacheConfig

        # mm_hash ->  encoder_output
        self.encoder_cache: dict[str, torch.Tensor] = {}

        self.use_aux_hidden_state_outputs = False
        # Set up speculative decoding.
        # NOTE(Jiayi): currently we put the entire draft model on
        # the last PP rank. This is not ideal if there are many
        # layers in the draft model.
        if self.speculative_config and get_pp_group().is_last_rank:
            self.drafter: (NgramProposer | SuffixDecodingProposer
                           | EagleProposer | MedusaProposer)
            if self.speculative_config.method == "ngram":
                self.drafter = NgramProposer(self.vllm_config)
            # elif self.speculative_config.method == "suffix":
            #     self.drafter = SuffixDecodingProposer(self.vllm_config)
            elif self.speculative_config.use_eagle():
                self.drafter = EagleProposer(self.vllm_config, self.device,
                                             self)  # type: ignore
                if self.speculative_config.method == "eagle3":
                    self.use_aux_hidden_state_outputs = \
                        self.drafter.eagle3_use_aux_hidden_state
            elif self.speculative_config.method == "medusa":
                self.drafter = MedusaProposer(
                    vllm_config=self.vllm_config,
                    device=self.device)  # type: ignore
            else:
                raise ValueError("Unknown speculative decoding method: "
                                 f"{self.speculative_config.method}")
            self.rejection_sampler = RejectionSampler(self.sampler)

        self.num_spec_tokens = 0
        if self.speculative_config:
            self.num_spec_tokens = \
                self.speculative_config.num_speculative_tokens

        # Request states.
        self.requests: dict[str, CachedRequestState] = {}
        # NOTE(rob): num_prompt_logprobs only includes reqs
        # that are currently in the prefill phase.
        self.num_prompt_logprobs: dict[str, int] = {}

        # Input Batch
        # NOTE(Chen): Ideally, we should initialize the input batch inside
        # `initialize_kv_cache` based on the kv cache config. However, as in
        # https://github.com/vllm-project/vllm/pull/18298, due to some unknown
        # reasons, we have to initialize the input batch before `load_model`,
        # quantization + weight offloading will fail otherwise. As a temporary
        # solution, we initialize the input batch here, and re-initialize it
        # in `initialize_kv_cache` if the block_sizes here is different from
        # the block_sizes in the kv cache config.
        logits_processors = model_config.logits_processors
        custom_logitsprocs: Sequence[Union[str, type[LogitsProcessor]]] = (
            tuple(logits_processors) if logits_processors is not None else ())
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoer
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits
            # processor uses output token ids so we set this conservatively.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.
            cp_kv_cache_interleave_size,
        )

        self.use_async_scheduling = self.scheduler_config.async_scheduling
        # self.async_output_copy_stream = torch.cuda.Stream() if \
        #     self.use_async_scheduling else None

        # Cache the device properties.
        self._init_device_properties()

        # Persistent buffers for graphs.
        self.input_ids = self._make_buffer(self.max_num_tokens,
                                           dtype=torch.int32)
        self.positions = self._make_buffer(self.max_num_tokens,
                                           dtype=torch.int64)
        self.query_start_loc = self._make_buffer(self.max_num_reqs + 1,
                                                 dtype=torch.int32)
        self.seq_lens = self._make_buffer(self.max_num_reqs, dtype=torch.int32)
        # Because inputs_embeds may be bfloat16 and we don't need a numpy
        # version of this tensor, avoid a RuntimeError by not creating a
        # numpy buffer.
        self.inputs_embeds = self._make_buffer(self.max_num_tokens,
                                               self.inputs_embeds_size,
                                               dtype=self.dtype,
                                               numpy=False)
        self.discard_request_mask = self._make_buffer(self.max_num_reqs,
                                                      dtype=torch.bool)
        self.num_decode_draft_tokens = self._make_buffer(self.max_num_reqs,
                                                         dtype=torch.int32)
        self.num_accepted_tokens = self._make_buffer(self.max_num_reqs,
                                                     dtype=torch.int64)

        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            # NOTE: `mrope_positions` is implemented with one additional dummy
            # position on purpose to make it non-contiguous so that it can work
            # with torch compile.
            # See detailed explanation in https://github.com/vllm-project/vllm/pull/12128#discussion_r1926431923

            # NOTE: When M-RoPE is enabled, position ids are 3D regardless of
            # the modality of inputs. For text-only inputs, each dimension has
            # identical position IDs, making M-RoPE functionally equivalent to
            # 1D-RoPE.
            # See page 5 of https://arxiv.org/abs/2409.12191
            self.mrope_positions = self._make_buffer(
                (3, self.max_num_tokens + 1), dtype=torch.int64)

        # None in the first PP rank. The rest are set after load_model.
        self.prefill_intermediate_tensors: Optional[IntermediateTensors] = None
        self.decode_intermediate_tensors: dict[int, IntermediateTensors] = {}

        # OPTIMIZATION: Cache the tensors rather than creating them every step.
        # Keep in int64 to avoid overflow with long context
        self.arange_np = np.arange(max(self.max_num_reqs + 1,
                                       self.max_model_len,
                                       self.max_num_tokens),
                                   dtype=np.int64)

        # Layer pairings for cross-layer KV sharing.
        # If an Attention layer `layer_name` is in the keys of this dict, it
        # means this layer will perform attention using the keys and values
        # from the KV cache of `shared_kv_cache_layers[layer_name]`.
        self.shared_kv_cache_layers: dict[str, str] = {}
        self.kv_sharing_fast_prefill_eligible_layers: set[str] = set()

        self.kv_sharing_fast_prefill_logits_indices = None
        if self.cache_config.kv_sharing_fast_prefill:
            self.kv_sharing_fast_prefill_logits_indices = torch.zeros(
                self.max_num_tokens, dtype=torch.int32, device=self.device)

        self.uniform_decode_query_len = 1 if not self.speculative_config else \
            1 + self.speculative_config.num_speculative_tokens

        self.mm_budget = MultiModalBudget(
            self.model_config,
            self.scheduler_config,
            self.mm_registry,
        ) if self.supports_mm_inputs else None

        self.reorder_batch_threshold: Optional[int] = None

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        self.runner_only_attn_layers: set[str] = set()

        # Cached outputs.
        self._draft_token_ids: Optional[Union[list[list[int]],
                                              torch.Tensor]] = None
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_model_len, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory,
        )

        self.valid_sampled_token_count_event: torch.Event | None = None

        # Ephemeral state transferred between
        # execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None

        self.max_batch_size = (self.scheduler_config.max_num_seqs //
                               self.parallel_config.pipeline_parallel_size)
        self.max_num_seqs = self.scheduler_config.max_num_seqs
        self.max_prefill_batch_size = 1
        self.max_num_batched_tokens = (
            self.scheduler_config.max_num_batched_tokens)

        bucketing_manager_class = get_bucketing_manager_class(
            envs.VLLM_RBLN_DECODE_BATCH_BUCKET_STRATEGY)
        self.bucketing_manager = bucketing_manager_class(
            max_batch_size=self.max_batch_size,
            min_batch_size=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_MIN,
            step=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_STEP,
            limit=envs.VLLM_RBLN_DECODE_BATCH_BUCKET_LIMIT,
        )
        logger.info("Using %s bucketing manager",
                    bucketing_manager_class.__name__)
        logger.info("decode batch buckets: %s",
                    self.bucketing_manager.decode_batch_buckets)

        self.performance_tracker = None

        self.dummy_run_state: DummyRunState | None = None

        self.specialized_moe_decode = parallel_config.data_parallel_size > 1 \
            and envs.VLLM_RBLN_SPECIALIZE_MOE_DECODE

    def _enable_performance_tracker(self):
        if envs.VLLM_RBLN_METRICS:
            self.performance_tracker = PerformanceTracker()
            self.performance_tracker.register_cleanup()

    def _get_positions(self, num_tokens: Any):
        if isinstance(num_tokens, int):
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, :num_tokens]
            # if self.uses_xdrope_dim > 0:
            #     return self.xdrope_positions.gpu[:, :num_tokens]
            return self.positions.gpu[:num_tokens]
        else:
            if self.uses_mrope:
                return self.mrope_positions.gpu[:, num_tokens]
            # if self.uses_xdrope_dim > 0:
            #     return self.xdrope_positions.gpu[:, num_tokens]
            return self.positions.gpu[num_tokens]

    def _make_buffer(self,
                     *size: Union[int, torch.SymInt],
                     dtype: torch.dtype,
                     numpy: bool = True) -> CpuGpuBuffer:
        return CpuGpuBuffer(*size,
                            dtype=dtype,
                            device=self.device,
                            pin_memory=self.pin_memory,
                            with_numpy=numpy)

    def _init_model_kwargs(self, num_tokens: int):
        model_kwargs = dict[str, Any]()

        if not self.is_pooling_model:
            return model_kwargs

        num_reqs = self.input_batch.num_reqs
        pooling_params = self.input_batch.get_pooling_params()

        token_type_id_requests = dict[int, Any]()
        for i, param in enumerate(pooling_params):
            if param.extra_kwargs is not None and \
            (token_types := param.extra_kwargs.get(
                "compressed_token_type_ids")) is not None:
                token_type_id_requests[i] = token_types

        if len(token_type_id_requests) == 0:
            return model_kwargs

        seq_lens = self.seq_lens.gpu[:num_reqs]
        token_type_ids = []

        for i in range(num_reqs):
            pos = token_type_id_requests.get(i, seq_lens[i])
            ids = (torch.arange(seq_lens[i]) >= pos).int()
            token_type_ids.append(ids)

        model_kwargs["token_type_ids"] = torch.concat(token_type_ids).to(
            device=self.device)
        return model_kwargs

    def _may_reorder_batch(self, scheduler_output: SchedulerOutput) -> None:
        """
        Update the order of requests in the batch based on the attention
        backend's needs. For example, some attention backends (namely MLA) may
        want to separate requests based on if the attention computation will be
        compute-bound or memory-bound.

        Args:
            scheduler_output: The scheduler output.
        """
        # Attention free models have zero kv_cache_goups, however models
        # like Mamba are also attention free but use the kv_cache for
        # keeping its internal state. This is why we check the number
        # of kv_cache groups instead of solely checking
        # for self.model_config.is_attention_free.
        if not envs.VLLM_RBLN_SORT_BATCH:
            return
        if len(self.kv_cache_config.kv_cache_groups) == 0:
            return

        orig_indices = np.arange(len(self.input_batch.req_ids))
        sorted_order = np.argsort(self.input_batch.num_tokens[orig_indices] *
                                  (-1),
                                  kind="stable")
        src_indices = orig_indices[sorted_order]
        src_dest_map = {
            int(src): int(dst)
            for src, dst in zip(src_indices, orig_indices)
        }

        for src in src_dest_map:
            dst = src_dest_map[src]
            while src != dst:
                self.input_batch.swap_states(src, dst)
                # Mark dst as done by updating its destination to itself
                next_dst = src_dest_map.get(dst, dst)
                src_dest_map[dst] = dst
                dst = next_dst

    # Note: used for model runner override.
    def _init_device_properties(self) -> None:
        pass

    # Note: used for model runner override.
    def _sync_device(self) -> None:
        pass

    def _update_states(self, scheduler_output: SchedulerOutput) -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input GPU tensors for the model.

        The SamplingMetadata is updated and copied to the GPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            self.requests.pop(req_id, None)
            self.num_prompt_logprobs.pop(req_id, None)
        # Remove the finished requests from the persistent batch.
        # NOTE(woosuk): There could be an edge case where finished_req_ids and
        # scheduled_req_ids overlap. This happens when a request is aborted and
        # then resubmitted with the same ID. In this case, we treat them as two
        # distinct requests - clearing the cached states for the first request
        # and handling the second as a new request.
        for req_id in scheduler_output.finished_req_ids:
            self.input_batch.remove_request(req_id)

        # Free the cached encoder outputs.
        for mm_hash in scheduler_output.free_encoder_mm_hashes:
            self.encoder_cache.pop(mm_hash, None)

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        unscheduled_req_ids = cached_req_ids - scheduled_req_ids
        # NOTE(woosuk): The persistent batch optimization assumes that
        # consecutive batches contain mostly the same requests. If batches
        # have low request overlap (e.g., alternating between two distinct
        # sets of requests), this optimization becomes very inefficient.
        for req_id in unscheduled_req_ids:
            self.input_batch.remove_request(req_id)

        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            if sampling_params and \
                sampling_params.sampling_type == SamplingType.RANDOM_SEED:
                generator = torch.Generator(device=self.device)
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
                    if sampling_params.prompt_logprobs == -1 else
                    sampling_params.prompt_logprobs)

            # TODO(jiwoo.park) We don't support M-RoPE yet.
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            if self.uses_mrope:
                self._init_mrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs
        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_data.resumed_from_preemption[i]

            # Update the cached states.
            req_state.num_computed_tokens = num_computed_tokens

            if not is_last_rank:
                # When using PP, the scheduler sends the sampled tokens back,
                # because there's no direct communication between the first-
                # stage worker and the last-stage worker.
                new_token_ids = req_data.new_token_ids[i]
                # Add the sampled token(s) from the previous step (if any).
                # This doesn't include "unverified" tokens like spec tokens.
                num_new_tokens = (num_computed_tokens + len(new_token_ids) -
                                  req_state.num_tokens)
                if num_new_tokens == 1:
                    # Avoid slicing list in most common case.
                    req_state.output_token_ids.append(new_token_ids[-1])
                elif num_new_tokens > 0:
                    req_state.output_token_ids.extend(
                        new_token_ids[-num_new_tokens:])

            # Update the block IDs.
            if not resumed_from_preemption:
                if new_block_ids is not None:
                    # Append the new blocks to the existing block IDs.
                    for block_ids, new_ids in zip(req_state.block_ids,
                                                  new_block_ids):
                        block_ids.extend(new_ids)
            else:
                assert new_block_ids is not None
                # The request is resumed from preemption.
                # Replace the existing block IDs with the new ones.
                req_state.block_ids = new_block_ids

            req_index = self.input_batch.req_id_to_index.get(req_id)
            if req_index is None:
                # The request is not in the persistent batch.
                # The request was either preempted and resumed later, or was not
                # scheduled in the previous step and needs to be added again.
                reqs_to_add.append(req_state)
                continue

            # Update the persistent batch.
            self.input_batch.num_computed_tokens_cpu[req_index] = (
                num_computed_tokens)
            if new_block_ids is not None:
                self.input_batch.block_table.append_row(
                    new_block_ids, req_index)

            # For the last rank, we don't need to update the token_ids_cpu
            # because the sampled tokens are already cached.
            if not is_last_rank:
                # Add new_token_ids to token_ids_cpu.
                start_token_index = num_computed_tokens
                end_token_index = num_computed_tokens + len(new_token_ids)
                self.input_batch.token_ids_cpu[
                    req_index,
                    start_token_index:end_token_index] = new_token_ids
                self.input_batch.num_tokens_no_spec[
                    req_index] = end_token_index
                self.input_batch.num_tokens[req_index] = end_token_index

            # Add spec_token_ids to token_ids_cpu.
            spec_token_ids = (
                scheduler_output.scheduled_spec_decode_tokens.get(req_id, ()))
            if spec_token_ids:
                num_spec_tokens = len(spec_token_ids)
                start_index = self.input_batch.num_tokens_no_spec[req_index]
                end_token_index = start_index + num_spec_tokens
                self.input_batch.token_ids_cpu[
                    req_index, start_index:end_token_index] = spec_token_ids
                # NOTE(woosuk): `num_tokens` here may include spec tokens.
                self.input_batch.num_tokens[req_index] += num_spec_tokens

            # When speculative decoding is used with structured output,
            # the scheduler can drop draft tokens that do not
            # conform to the schema. This can result in
            # scheduler_output.scheduled_spec_decode_tokens being empty,
            # even when speculative decoding is enabled.
            self.input_batch.spec_token_ids[req_index].clear()
            self.input_batch.spec_token_ids[req_index].extend(spec_token_ids)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)

        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()
        # Allow attention backend to reorder the batch, potentially
        self._may_reorder_batch(scheduler_output)
        # Refresh batch metadata with any pending updates.
        self.input_batch.refresh_metadata()

    def _update_states_after_model_execute(
            self, output_token_ids: torch.Tensor) -> None:
        """Update the cached states after model execution.

        This is used for MTP/EAGLE for hybrid models, as in linear attention,
        only the last token's state is kept. In MTP/EAGLE, for draft tokens
        the state are kept util we decide how many tokens are accepted for
        each sequence, and a shifting is done during the next iteration
        based on the number of accepted tokens.
        """
        if not self.model_config.is_hybrid or not self.speculative_config:
            return

        # Find the number of accepted tokens for each sequence.
        num_accepted_tokens = (torch.cat(
            [
                output_token_ids,
                torch.full((output_token_ids.size(0), 1),
                           -1,
                           device=output_token_ids.device),
            ],
            dim=1) == -1).int().argmax(-1).cpu().numpy()
        for i, num_tokens in enumerate(num_accepted_tokens):
            self.input_batch.num_accepted_tokens_cpu[i] = num_tokens

    def _get_cumsum_and_arange(
        self,
        num_tokens: np.ndarray,
        cumsum_dtype: Optional[np.dtype] = None,
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

    def _prepare_input_ids(self, scheduler_output: "SchedulerOutput",
                           total_num_scheduled_tokens: int,
                           cu_num_tokens: np.ndarray) -> None:
        """Prepare the input IDs for the current batch.

        Carefully handles the `prev_sampled_token_ids` which can be cached
        from the previous engine iteration, in which case those tokens on the
        GPU need to be copied into the corresponding slots into input_ids."""

        if self.input_batch.prev_sampled_token_ids is None:
            # Normal scheduling case
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            return

        # Async scheduling case, where some decode requests from the previous
        # iteration won't have entries in input_ids_cpu and need to be copied
        # on the GPU from prev_sampled_token_ids.
        prev_req_id_to_index = self.input_batch.prev_req_id_to_index
        assert prev_req_id_to_index is not None
        sample_flattened_indices: list[int] = []
        spec_flattened_indices: list[int] = []
        prev_common_req_indices: list[int] = []
        prev_draft_token_indices: list[int] = []
        indices_match = True
        max_flattened_index = -1
        total_num_spec_tokens = 0
        scheduled_spec_tokens = scheduler_output.scheduled_spec_decode_tokens

        for req_id, cur_index in self.input_batch.req_id_to_index.items():
            if (prev_index := prev_req_id_to_index.get(req_id)) is not None:
                prev_common_req_indices.append(prev_index)
                # We need to compute the flattened input_ids index of the
                # last token in each common request.
                draft_len = len(scheduled_spec_tokens.get(req_id, ()))
                total_num_spec_tokens += draft_len
                flattened_index = cu_num_tokens[cur_index].item() - 1
                # example: cu_num_tokens = [2, 5, 8], draft_tokens = [1, 2, 2]
                # sample_flattened_indices = [0, 2, 5]
                # spec_flattened_indices = [1,   3, 4,    6, 7]
                sample_flattened_indices.append(flattened_index - draft_len)
                spec_flattened_indices.extend(
                    range(flattened_index - draft_len + 1,
                          flattened_index + 1))
                start = prev_index * self.num_spec_tokens
                # prev_draft_token_indices is used to find which draft_tokens_id
                # should be copied to input_ids
                # example: prev draft_tokens_id [[1,2], [3,4], [5, 6]]
                # flatten draft_tokens_id [1,2,3,4,5,6]
                # draft_len of each request [1, 2, 1]
                # then prev_draft_token_indices is [0,   2, 3,   4]
                prev_draft_token_indices.extend(range(start,
                                                      start + draft_len))
                indices_match &= prev_index == flattened_index
                max_flattened_index = max(max_flattened_index, flattened_index)
        num_commmon_tokens = len(sample_flattened_indices)
        total_without_spec = total_num_scheduled_tokens - total_num_spec_tokens
        if num_commmon_tokens < total_without_spec:
            # If not all requests are decodes from the last iteration,
            # We need to copy the input_ids_cpu to the GPU first.
            self.input_ids.copy_to_gpu(total_num_scheduled_tokens)
            # if self.enable_prompt_embeds:
            #     self.inputs_embeds.copy_to_gpu(total_num_scheduled_tokens)
            #     self.is_token_ids.copy_to_gpu(total_num_scheduled_tokens)
        if num_commmon_tokens == 0:
            # No requests in common with the previous iteration
            # So input_ids.cpu will have all the input ids.
            return
        if indices_match and max_flattened_index == (num_commmon_tokens - 1):
            # Common-case optimization: the batch is unchanged
            # and no reordering happened.
            # The indices are both the same permutation of 0..N-1 so
            # we can copy directly using a single slice.
            self.input_ids.gpu[:num_commmon_tokens].copy_(
                self.input_batch.prev_sampled_token_ids[:num_commmon_tokens,
                                                        0],
                non_blocking=True,
            )
            # if self.enable_prompt_embeds:
            #     self.is_token_ids.gpu[:num_commmon_tokens] = True
            return
        # Upload the index tensors asynchronously
        # so the scatter can be non-blocking.
        sampled_tokens_index_tensor = torch.tensor(
            sample_flattened_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)
        prev_common_req_indices_tensor = torch.tensor(
            prev_common_req_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)
        self.input_ids.gpu.scatter_(
            dim=0,
            index=sampled_tokens_index_tensor,
            src=self.input_batch.prev_sampled_token_ids[
                prev_common_req_indices_tensor, 0],
        )

        # Scatter the draft tokens after the sampled tokens are scattered.
        if self._draft_token_ids is None or not spec_flattened_indices:
            return

        assert isinstance(self._draft_token_ids, torch.Tensor)
        draft_tokens_index_tensor = torch.tensor(
            spec_flattened_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)
        prev_draft_token_indices_tensor = torch.tensor(
            prev_draft_token_indices,
            dtype=torch.int64,
            pin_memory=self.pin_memory).to(self.device, non_blocking=True)

        # because input_ids dtype is torch.int32,
        # so convert draft_token_ids to torch.int32 here.
        draft_token_ids = self._draft_token_ids.to(dtype=torch.int32)
        self._draft_token_ids = None

        self.input_ids.gpu.scatter_(
            dim=0,
            index=draft_tokens_index_tensor,
            src=draft_token_ids.flatten()[prev_draft_token_indices_tensor],
        )

    def _get_encoder_seq_lens(
        self,
        scheduler_output: "SchedulerOutput",
        kv_cache_spec: KVCacheSpec,
        num_reqs: int,
    ) -> Optional[np.ndarray]:
        if not isinstance(kv_cache_spec, CrossAttentionSpec):
            return None

        # Build encoder_seq_lens array mapping request indices to
        # encoder lengths for inputs scheduled in this batch
        encoder_seq_lens = np.zeros(num_reqs, dtype=np.int32)
        for req_id in scheduler_output.scheduled_encoder_inputs:
            req_index = self.input_batch.req_id_to_index[req_id]
            encoder_seq_lens[req_index] = self.max_encoder_len

        return encoder_seq_lens

    def _prepare_inputs(
        self,
        scheduler_output: SchedulerOutput,
        num_scheduled_tokens: np.ndarray,
        num_padded_tokens: int | None = None,
    ) -> tuple[dict[str, Any], torch.Tensor, Optional[SpecDecodeMetadata],
               np.ndarray, Optional[CommonAttentionMetadata], int]:
        """
        :return: tuple[
            attn_metadata: layer-to-attention_metadata mapping,
            logits_indices, spec_decode_metadata
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        self.input_batch.block_table.commit_block_table(num_reqs)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(self.input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)

        # Calculate M-RoPE positions.
        # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
        if self.uses_mrope:
            self._calc_mrope_positions(scheduler_output)

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * self.input_batch.token_ids_cpu.shape[1])

        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(self.input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=self.input_ids.cpu[:total_num_scheduled_tokens])

        self.input_batch.block_table.compute_slot_mapping(
            req_indices, positions_np)
        self.input_batch.block_table.commit_slot_mapping(
            total_num_scheduled_tokens)

        # Prepare the attention metadata.
        self.query_start_loc.np[0] = 0
        self.query_start_loc.np[1:num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        self.query_start_loc.np[num_reqs + 1:].fill(cu_num_tokens[-1])
        self.query_start_loc.copy_to_gpu()
        query_start_loc = self.query_start_loc.gpu[:num_reqs + 1]

        self.seq_lens.np[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        # Fill unused with 0 for full cuda graph mode.
        self.seq_lens.np[num_reqs:].fill(0)
        self.seq_lens.copy_to_gpu()

        # Copy the tensors to the GPU.
        # TODO(jiwoo.park) Currently, this code is meaningless.(overhead)
        # The input_ids may be padded by chunk size and max batch size.
        self._prepare_input_ids(scheduler_output, total_num_scheduled_tokens,
                                cu_num_tokens)

        if self.uses_mrope:
            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            self.mrope_positions.gpu[:, :total_num_scheduled_tokens].copy_(
                self.mrope_positions.cpu[:, :total_num_scheduled_tokens],
                non_blocking=True)
        else:
            # Common case (1D positions)
            self.positions.copy_to_gpu(total_num_scheduled_tokens)

        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if not use_spec_decode:
            # NOTE(woosuk): Due to chunked prefills, the batch may contain
            # partial requests. While we should not sample any token
            # from these partial requests, we do so for simplicity.
            # We will ignore the sampled tokens from the partial requests.
            # TODO: Support prompt logprobs.
            logits_indices = query_start_loc[1:] - 1
            num_draft_tokens = None
            spec_decode_metadata = None
            num_sampled_tokens = np.ones(num_reqs, dtype=np.int32)
        else:
            # Get the number of draft tokens for each request.
            # Iterate over the dictionary rather than all requests since not all
            # requests have draft tokens.
            num_draft_tokens = np.zeros(num_reqs, dtype=np.int32)
            num_decode_draft_tokens = np.full(num_reqs, -1, dtype=np.int32)
            for req_id, draft_token_ids in (
                    scheduler_output.scheduled_spec_decode_tokens.items()):
                req_idx = self.input_batch.req_id_to_index[req_id]
                num_draft_tokens[req_idx] = len(draft_token_ids)
                num_decode_draft_tokens[req_idx] = (len(draft_token_ids) if (
                    self.input_batch.num_computed_tokens_cpu[req_idx]
                    >= self.input_batch.num_prompt_tokens[req_idx]) else -1)

            spec_decode_metadata = self._calc_spec_decode_metadata(
                num_draft_tokens, cu_num_tokens)
            logits_indices = spec_decode_metadata.logits_indices
            num_sampled_tokens = num_draft_tokens + 1

            self.num_decode_draft_tokens.np[:num_reqs] = num_draft_tokens
            self.num_decode_draft_tokens.np[num_reqs:].fill(-1)
            self.num_decode_draft_tokens.copy_to_gpu()

        logits_indices_padded = None
        if self.cache_config.kv_sharing_fast_prefill:
            logits_indices_padded = self._prepare_kv_sharing_fast_prefill(
                logits_indices)

        attn_metadata: dict[str, Any] = {}

        # Used in the below loop.
        query_start_loc_cpu = self.query_start_loc.cpu[:num_reqs + 1]
        seq_lens = self.seq_lens.gpu[:num_reqs]
        max_seq_len = self.seq_lens.np[:num_reqs].max().item()
        seq_lens_cpu = self.seq_lens.cpu[:num_reqs]
        num_computed_tokens_cpu = (
            self.input_batch.num_computed_tokens_cpu_tensor[:num_reqs])
        spec_decode_common_attn_metadata = None
        if use_spec_decode:
            self.num_accepted_tokens.np[:num_reqs] = (
                self.input_batch.num_accepted_tokens_cpu[:num_reqs])
            self.num_accepted_tokens.np[num_reqs:].fill(1)
            self.num_accepted_tokens.copy_to_gpu()

        max_num_scheduled_tokens = int(num_scheduled_tokens.max())
        batch_bucket_size = \
            self.bucketing_manager.find_decode_batch_bucket(num_reqs)

        (batch_bucket_size, num_padded_tokens, num_tokens_across_dp) = \
            self.get_dp_padding(total_num_scheduled_tokens, batch_bucket_size,
                                num_padded_tokens, bool(self.is_prefills()[0]))
        # Prepare the attention metadata for each KV cache group and make layers
        # in the same group share the same metadata.
        for kv_cache_group_id, kv_cache_group_spec in enumerate(
                self.kv_cache_config.kv_cache_groups):
            encoder_seq_lens = self._get_encoder_seq_lens(
                scheduler_output, kv_cache_group_spec.kv_cache_spec, num_reqs)

            if isinstance(kv_cache_group_spec.kv_cache_spec,
                          EncoderOnlyAttentionSpec):
                # Encoder-only layers do not have KV cache, so we need to
                # create a dummy block table and slot mapping for them.
                blk_table_tensor = torch.zeros(
                    (num_reqs, 1),
                    dtype=torch.int32,
                    device=self.device,
                )
                slot_mapping = torch.zeros(
                    (total_num_scheduled_tokens, ),
                    dtype=torch.int64,
                    device=self.device,
                )
                num_common_prefix_blocks = 0
            else:
                blk_table = self.input_batch.block_table[kv_cache_group_id]
                blk_table_tensor = blk_table.get_device_tensor(num_reqs)
                slot_mapping = blk_table.slot_mapping.gpu[:
                                                          total_num_scheduled_tokens]

                # Fill unused with -1. Needed for reshape_and_cache in full cuda
                # graph mode. `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
                slot_mapping[total_num_scheduled_tokens:
                             total_num_scheduled_tokens].fill_(-1)
                blk_table_tensor[num_reqs:total_num_scheduled_tokens].fill_(-1)

            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc_cpu,
                seq_lens=seq_lens,
                seq_lens_cpu=seq_lens_cpu,
                num_computed_tokens_cpu=num_computed_tokens_cpu,
                num_reqs=num_reqs,
                num_actual_tokens=total_num_scheduled_tokens,
                max_query_len=max_num_scheduled_tokens,
                max_seq_len=max_seq_len,
                block_table_tensor=blk_table_tensor,
                slot_mapping=slot_mapping,
                logits_indices_padded=logits_indices_padded,
                num_logits_indices=logits_indices.size(0),
                causal=True,
                encoder_seq_lens=encoder_seq_lens,
            )

            if self.speculative_config and \
                spec_decode_common_attn_metadata is None:
                spec_decode_common_attn_metadata = common_attn_metadata

            for attn_group in self.attn_groups[kv_cache_group_id]:
                # Prepare for cascade attention if enabled & beneficial.
                common_prefix_len = 0
                builder = attn_group.get_metadata_builder()
                if self.cascade_attn_enabled:
                    common_prefix_len = self._compute_cascade_attn_prefix_len(
                        num_scheduled_tokens,
                        num_common_prefix_blocks,
                        kv_cache_group_spec.kv_cache_spec,
                        builder,
                    )

                extra_attn_metadata_args = {}
                if use_spec_decode and isinstance(builder,
                                                  GDNAttentionMetadataBuilder):
                    extra_attn_metadata_args = dict(
                        num_accepted_tokens=self.num_accepted_tokens.
                        gpu[:num_reqs],
                        num_draft_tokens=self.num_draft_tokens.gpu[:num_reqs],
                    )

                if isinstance(builder, RBLNFlashAttentionMetadataBuilder):
                    extra_attn_metadata_args["num_tokens"] = \
                        self.input_batch.num_tokens_no_spec
                    extra_attn_metadata_args["positions"] = self.positions.cpu
                    extra_attn_metadata_args["batch_pad"] = batch_bucket_size
                attn_metadata_i = builder.build(
                    common_prefix_len=common_prefix_len,
                    common_attn_metadata=common_attn_metadata,
                    **extra_attn_metadata_args)

                for layer_name in attn_group.layer_names:
                    attn_metadata[layer_name] = attn_metadata_i

        # Hot-Swap lora model
        if self.lora_config:
            assert (
                np.sum(num_sampled_tokens)
                <= self.vllm_config.scheduler_config.max_num_batched_tokens)
            self.set_active_loras(self.input_batch, num_scheduled_tokens,
                                  num_sampled_tokens)

        return (attn_metadata, logits_indices, spec_decode_metadata,
                num_scheduled_tokens, spec_decode_common_attn_metadata,
                max_num_scheduled_tokens, batch_bucket_size, num_padded_tokens,
                num_tokens_across_dp)

    def _compile_model(self, model):
        TP = get_tp_group()
        PP = get_pp_group()
        DP = get_dp_group()

        process_group_dict = {}
        process_group_dict[TP.device_group.group_name] = TP.ranks
        process_group_dict[TP.cpu_group.group_name] = TP.ranks
        process_group_dict[PP.device_group.group_name] = PP.ranks
        process_group_dict[PP.cpu_group.group_name] = PP.ranks
        process_group_dict[DP.device_group.group_name] = DP.ranks
        process_group_dict[DP.cpu_group.group_name] = DP.ranks

        options = {
            "compile_context": self.compile_context,
            "tensor_parallel_size": envs.VLLM_RBLN_TP_SIZE,
            "process_group_dict": process_group_dict,
            "guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe,
        }
        if not envs.VLLM_DISABLE_COMPILE_CACHE:
            logger.info("Once the model is compiled for the first time, "
                        "the cached compiled binary will be reused.")
            options["cache_dir"] = os.path.join(envs.VLLM_CACHE_ROOT, 'rbln')
        if envs.VLLM_RBLN_COMPILE_STRICT_MODE:
            options["mode"] = "strict"

        # compile compute_logits
        self.compute_logits = torch.compile(
            self.compute_logits,
            backend="rbln",
            options=copy(options),
            dynamic=False,
        )

        compiled_model = torch.compile(
            model,
            backend="rbln",
            options=copy(options),
            dynamic=False,
        )

        return compiled_model

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
            num_sampled_tokens, cumsum_dtype=np.int32)
        # Step 2. [0, 0, 0, 0, 103, 104, 104, 104, 206, 207, 207]
        logits_indices = np.repeat(
            cu_num_scheduled_tokens - num_sampled_tokens, num_sampled_tokens)
        # Step 3. [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208]
        logits_indices += arange

        # Compute the bonus logits indices.
        bonus_logits_indices = cu_num_sampled_tokens - 1

        # Compute the draft logits indices.
        # cu_num_draft_tokens: [3, 3, 5, 5, 6]
        # arange: [0, 1, 2, 0, 1, 0]
        cu_num_draft_tokens, arange = self._get_cumsum_and_arange(
            num_draft_tokens, cumsum_dtype=np.int32)
        # [0, 0, 0, 5, 5, 9]
        target_logits_indices = np.repeat(
            cu_num_sampled_tokens - num_sampled_tokens, num_draft_tokens)
        # [0, 1, 2, 5, 6, 9]
        target_logits_indices += arange

        # TODO: Optimize the CPU -> GPU copy.
        cu_num_draft_tokens = torch.from_numpy(cu_num_draft_tokens).to(
            self.device, non_blocking=True)
        cu_num_sampled_tokens = torch.from_numpy(cu_num_sampled_tokens).to(
            self.device, non_blocking=True)
        logits_indices = torch.from_numpy(logits_indices).to(self.device,
                                                             non_blocking=True)
        target_logits_indices = torch.from_numpy(target_logits_indices).to(
            self.device, non_blocking=True)
        bonus_logits_indices = torch.from_numpy(bonus_logits_indices).to(
            self.device, non_blocking=True)

        # Compute the draft token ids.
        # draft_token_indices:      [  1,   2,   3, 105, 106, 208]
        draft_token_ids = self.input_ids.gpu[logits_indices]
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

    def get_model(self) -> nn.Module:
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

        return supported_tasks

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if (self.scheduler_config.enable_chunked_prefill
                and "encode" in supported_tasks):
            supported_tasks.remove("encode")

            logger.debug_once("Chunked prefill is not supported with "
                              "encode task which using ALL pooling. "
                              "Please turn off chunked prefill by "
                              "`--no-enable-chunked-prefill` before using it.")

        if "score" in supported_tasks:
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)
            if num_labels != 1:
                supported_tasks.remove("score")
                logger.debug_once(
                    "Score API is only enabled for num_labels == 1.")

        return supported_tasks

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def sync_and_slice_intermediate_tensors(
        self,
        batch_size: int,
        seq_len: int,
        intermediate_tensors: Optional[IntermediateTensors],
        sync_self: bool,
    ) -> IntermediateTensors:
        # FIXME - RBLN does not support intermediate tensor slicing
        tp = self.vllm_config.parallel_config.tensor_parallel_size
        enabled_sp = self.compilation_config.pass_config. \
            enable_sequence_parallelism

        if intermediate_tensors is None:
            # for warm_up, from empty dummy intermediate tensors
            assert self.intermediate_tensors is not None
            assert batch_size > 0
            assert seq_len > 0
            num_tokens = batch_size * seq_len
            if enabled_sp:
                # When sequence parallelism is enabled, we always pad num_tokens
                # to be a multiple of tensor_parallel_size (tp) earlier
                assert num_tokens % tp == 0
            residual_scatter = tp > 1 and enabled_sp \
                and num_tokens % tp == 0
            assert not enabled_sp, "RBLN warm_up = !sp(sequence_parallel)"
            assert not residual_scatter, "RBLN warm_up = !residual_scatter"
            assert not sync_self, "RBLN warm_up = !sync self(from dummy)"
            return IntermediateTensors({
                k: v.reshape((batch_size, seq_len, -1))
                for k, v in self.intermediate_tensors.items()
            })
        else:
            # for execution, from input intermediate tensors
            assert batch_size == -1
            assert seq_len == -1
            assert sync_self, "RBLN execute = sync self(from input)"
            return IntermediateTensors({
                k: v
                for k, v in intermediate_tensors.items()
            })

    def get_dp_padding(
        self,
        num_tokens: int,
        batch_bucket_size: int,
        num_padded_tokens: int | None = None,
        is_prefill: bool = False
    ) -> tuple[int, Optional[int], Optional[torch.Tensor]]:
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        dp_rank = self.vllm_config.parallel_config.data_parallel_rank

        if dp_size == 1:
            assert num_padded_tokens is None, \
                "num_padded_tokens should not be applied for non-DP case"
            return batch_bucket_size, num_padded_tokens, None

        if num_padded_tokens is not None:
            assert self.specialized_moe_decode, \
                "num_padded_tokens is only supported when " \
                "specialized MOE decode is enabled"
            assert num_padded_tokens == self.max_num_batched_tokens, \
                "num_padded_tokens should be equal to max_num_batched_tokens"
            assert not is_prefill, \
                "num_padded_tokens is only supported for decode stage"
            num_tokens_across_dp_cpu = RBLNDPMetadata.num_tokens_across_dp(
                num_tokens, dp_size, dp_rank)
            return (batch_bucket_size, num_padded_tokens,
                    num_tokens_across_dp_cpu)

        num_tokens_across_dp_cpu, max_decode_tokens = \
            RBLNDPMetadata.num_tokens_across_dp_with_max_decode_tokens(
            num_tokens, dp_size, dp_rank, is_prefill)

        any_prefill = max_decode_tokens is None
        if any_prefill or not self.specialized_moe_decode:
            num_padded_tokens = self.max_num_batched_tokens
        else:
            batch_bucket_size = self.bucketing_manager.find_decode_batch_bucket(
                max_decode_tokens)
            num_padded_tokens = batch_bucket_size

        return batch_bucket_size, num_padded_tokens, num_tokens_across_dp_cpu

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
        kv_connector_output: Optional[KVConnectorOutput],
    ) -> ModelRunnerOutput:
        assert self.input_batch.num_reqs ==\
            len(self.input_batch.pooling_params), \
        "Either all or none of the requests in" \
        " a batch must be pooling request"

        hidden_states = hidden_states[:num_scheduled_tokens]
        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(num_scheduled_tokens_np.tolist(),
                                              device=hidden_states.device)
        seq_lens_cpu = self.seq_lens.cpu[:self.input_batch.num_reqs]

        # Pooling models D2H & synchronize occurs in pooler.py:build_output
        raw_pooler_output = self.model.pooler(
            hidden_states=hidden_states, pooling_metadata=pooling_metadata)

        pooler_output: list[Optional[torch.Tensor]] = []
        for raw_output, seq_len, prompt_len in zip(
                raw_pooler_output, seq_lens_cpu, pooling_metadata.prompt_lens):

            output = raw_output.data if seq_len == prompt_len else None
            pooler_output.append(output)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
            kv_connector_output=kv_connector_output,
        )

    def _preprocess(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[int, Optional[torch.Tensor], Optional[torch.Tensor],
               Optional[torch.Tensor], torch.Tensor,
               Optional[IntermediateTensors], dict[str, Any]]:

        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens

        # Pad tokens to multiple of tensor_parallel_size when
        # enabled collective fusion for SP
        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        if self.compilation_config.pass_config. \
            enable_sequence_parallelism and tp_size > 1:
            num_input_tokens = round_up(num_scheduled_tokens, tp_size)
        else:
            num_input_tokens = num_scheduled_tokens

        # Padding for DP
        # NOTE(RBLN): RBLN handles DP padding in _prepare_inputs

        # _prepare_inputs may reorder the batch, so we must gather multi
        # modal outputs after that to ensure the correct order
        if (self.supports_mm_inputs and get_pp_group().is_first_rank
                and not self.model_config.is_encoder_decoder):
            # Run the multimodal encoder if any.
            self._execute_mm_encoder(scheduler_output)
            mm_embeds = self._gather_mm_embeddings(scheduler_output)

            # NOTE(woosuk): To unify token ids and soft tokens (vision
            # embeddings), we always use embeddings (rather than token ids)
            # as input to the multimodal model, even when the input is text.
            inputs_embeds_scheduled = self.model.get_input_embeddings(
                input_ids=self.input_ids.gpu[:num_scheduled_tokens],
                multimodal_embeddings=mm_embeds or None,
            )

            # TODO(woosuk): Avoid the copy. Optimize.
            self.inputs_embeds.gpu[:num_scheduled_tokens].copy_(
                inputs_embeds_scheduled)

            input_ids = None
            inputs_embeds = self.inputs_embeds.gpu[:num_input_tokens]
            model_kwargs = {
                **self._init_model_kwargs(num_scheduled_tokens),
                **self._extract_mm_kwargs(scheduler_output),
            }
        else:
            # For text-only models, we use token ids as input.
            # While it is possible to use embeddings as input just like the
            # multimodal models, it is not desirable for performance since
            # then the embedding layer is not included in the CUDA graph.
            input_ids = self.input_ids.gpu[:num_input_tokens]
            inputs_embeds = None
            model_kwargs = self._init_model_kwargs(num_input_tokens)
        if self.uses_mrope:
            positions = self.mrope_positions.gpu[:, :num_input_tokens]
        else:
            positions = self.positions.gpu[:num_input_tokens]

        if (self.model_config.is_encoder_decoder
                and scheduler_output.scheduled_encoder_inputs):
            encoder_inputs = self._extract_encoder_inputs(scheduler_output)
            model_kwargs.update(encoder_inputs)

        return (
            num_input_tokens,
            input_ids,
            inputs_embeds,
            positions,
            model_kwargs,
        )

    def _sample(
            self, logits: Optional[torch.Tensor],
            spec_decode_metadata: Optional[SpecDecodeMetadata]
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        if spec_decode_metadata is None:
            sampler_output = self.sampler(
                logits=logits,
                sampling_metadata=sampling_metadata,
            )
        else:
            sampler_output = self.rejection_sampler(
                spec_decode_metadata,
                None,  # draft_probs
                logits,
                sampling_metadata,
            )
            self._update_states_after_model_execute(
                sampler_output.sampled_token_ids)

        return sampler_output

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        return self.model.compute_logits(hidden_states)

    @torch.inference_mode()
    def warm_up_model(self) -> None:
        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)

        logger.info("Warm up prefill graph")
        prefill_seq_len = (self.scheduler_config.max_num_batched_tokens
                           if self.scheduler_config.enable_chunked_prefill else
                           self.model_config.max_model_len)
        dummy_prefill_requests = []
        dummy_prefill_num_scheduled_tokens = {}
        self._add_dummy_requests(
            requests=dummy_prefill_requests,
            num_scheduled_tokens=dummy_prefill_num_scheduled_tokens,
            total_tokens=prefill_seq_len,
            num_computed_tokens=0,
            num_kv_cache_groups=num_kv_cache_groups,
            sampling_params=None if self.is_pooling_model else SamplingParams(
                temperature=0.0),
            pooling_params=PoolingParams(
                task=self.get_supported_pooling_tasks()[0])
            if self.is_pooling_model else None,
        )
        so, cso = self._make_dummy_scheduler_outputs(
            dummy_prefill_requests, dummy_prefill_num_scheduled_tokens,
            num_kv_cache_groups)
        self._execute_dummy_requests(so, cso,
                                     self.prefill_intermediate_tensors)

        # compile decode graph
        for batch_bucket_size in self.bucketing_manager.decode_batch_buckets:
            decode_max_seq_len = self.max_model_len

            dummy_decode_requests = []
            dummy_decode_num_scheduled_tokens = {}
            num_speculative_tokens = \
            self.speculative_config.num_speculative_tokens \
                if self.speculative_config is not None else 0
            for _ in range(batch_bucket_size):
                self._add_dummy_requests(
                    requests=dummy_decode_requests,
                    num_scheduled_tokens=dummy_decode_num_scheduled_tokens,
                    total_tokens=decode_max_seq_len - 1 -
                    num_speculative_tokens,
                    num_computed_tokens=decode_max_seq_len - 1 -
                    num_speculative_tokens,
                    num_kv_cache_groups=num_kv_cache_groups,
                    sampling_params=None if self.is_pooling_model else
                    SamplingParams(temperature=0.0),
                    pooling_params=PoolingParams(
                        task=self.get_supported_pooling_tasks()[0])
                    if self.is_pooling_model else None,
                    num_speculative_tokens=num_speculative_tokens,
                )
            so, cso = self._make_dummy_scheduler_outputs(
                dummy_decode_requests, dummy_decode_num_scheduled_tokens,
                num_kv_cache_groups)
            current_intermediate_tensors = \
                self.decode_intermediate_tensors.get(batch_bucket_size)
            assert current_intermediate_tensors is not None

            if self.specialized_moe_decode:
                self._execute_dummy_requests(
                    so,
                    cso,
                    current_intermediate_tensors,
                    num_padded_tokens=self.max_num_batched_tokens)

            self._execute_dummy_requests(so, cso, current_intermediate_tensors)

    def _add_dummy_requests(
        self,
        requests: list[NewRequestData],
        num_scheduled_tokens: dict[str, int],
        total_tokens: int,
        num_computed_tokens: int,
        num_kv_cache_groups: int,
        sampling_params: Optional[SamplingParams] = None,
        pooling_params: Optional[PoolingParams] = None,
        num_speculative_tokens: int = 0,
    ) -> None:
        num_blocks = round_up(
            total_tokens,
            self.cache_config.block_size) // self.cache_config.block_size
        prompt_token_ids = list(range(total_tokens))
        # the dummy block maintained by BlockPool (null_block)
        null_block_id = 0

        req = NewRequestData(
            req_id=f"dummy_request_{len(requests)}",
            prompt_token_ids=prompt_token_ids,
            mm_features=[],
            sampling_params=sampling_params,
            pooling_params=pooling_params,
            block_ids=([null_block_id] * num_blocks, ) * num_kv_cache_groups,
            num_computed_tokens=num_computed_tokens,
            lora_request=None,
        )
        requests.append(req)
        num_scheduled_tokens[req.req_id] = (1 + num_speculative_tokens) \
            if total_tokens - num_computed_tokens == 0 \
                else total_tokens - num_computed_tokens

    def _make_dummy_scheduler_outputs(
            self, requests: list[NewRequestData],
            num_scheduled_tokens: dict[str, int], num_kv_cache_groups: int
    ) -> tuple[SchedulerOutput, SchedulerOutput]:
        sched_output = SchedulerOutput(
            scheduled_new_reqs=requests,
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens=num_scheduled_tokens,
            total_num_scheduled_tokens=sum(num_scheduled_tokens.values()),
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[0] * num_kv_cache_groups,
            finished_req_ids=set(),
            free_encoder_mm_hashes=[],
            kv_connector_metadata=None,
        )
        cleanup_sched_output = SchedulerOutput(
            scheduled_new_reqs=[],
            scheduled_cached_reqs=CachedRequestData.make_empty(),
            num_scheduled_tokens={},
            total_num_scheduled_tokens=0,
            scheduled_spec_decode_tokens={},
            scheduled_encoder_inputs={},
            num_common_prefix_blocks=[1] * num_kv_cache_groups,
            finished_req_ids={req.req_id
                              for req in requests},
            free_encoder_mm_hashes=[],
            kv_connector_metadata=None,
        )
        return sched_output, cleanup_sched_output

    def _execute_dummy_requests(
        self,
        sched_output: SchedulerOutput,
        cleanup_sched_output: SchedulerOutput,
        intermediate_tensors: IntermediateTensors,
        num_padded_tokens: int | None = None,
    ) -> None:
        if get_pp_group().is_first_rank:
            intermediate_tensors = None

        output = self.execute_model(sched_output, intermediate_tensors,
                                    num_padded_tokens)
        if output is None:
            self.sample_tokens(None)
        output = self.execute_model(cleanup_sched_output, intermediate_tensors,
                                    num_padded_tokens)
        if output is None:
            self.sample_tokens(None)

    def _update_dummy_states(self, scheduler_output: SchedulerOutput,
                             input_batch: InputBatch) -> None:
        reqs_to_add: list[CachedRequestState] = []
        # Add new requests to the cached states.
        for new_req_data in scheduler_output.scheduled_new_reqs:
            req_id = new_req_data.req_id
            sampling_params = new_req_data.sampling_params
            pooling_params = new_req_data.pooling_params

            generator = None

            req_state = CachedRequestState(
                req_id=req_id,
                prompt_token_ids=new_req_data.prompt_token_ids,
                mm_features=new_req_data.mm_features,
                sampling_params=sampling_params,
                pooling_params=pooling_params,
                generator=generator,
                block_ids=new_req_data.block_ids,
                num_computed_tokens=new_req_data.num_computed_tokens,
                output_token_ids=[],
                lora_request=new_req_data.lora_request,
            )

            reqs_to_add.append(req_state)

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            input_batch.add_request(request)

        # Refresh batch metadata with any pending updates.
        input_batch.refresh_metadata()

    def _prepare_dummy_inputs(
        self,
        scheduler_output: SchedulerOutput,
        input_batch: InputBatch,
    ) -> DummyRunState:
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        num_reqs = input_batch.num_reqs

        # OPTIMIZATION: Start copying the block table first.
        # This way, we can overlap the copy with the following CPU operations.
        input_batch.block_table.commit_block_table(num_reqs)

        # Get the number of scheduled tokens for each request.
        req_ids = input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        max_num_scheduled_tokens = max(tokens)

        # Get request indices.
        # E.g., [2, 5, 3] -> [0, 0, 1, 1, 1, 1, 1, 2, 2, 2]
        req_indices = np.repeat(self.arange_np[:num_reqs],
                                num_scheduled_tokens)

        # cu_num_tokens: [2, 5, 3] -> [2, 7, 10]
        # arange: [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        cu_num_tokens, arange = self._get_cumsum_and_arange(
            num_scheduled_tokens)

        # Get positions.
        positions_np = self.positions.np[:total_num_scheduled_tokens]
        np.add(input_batch.num_computed_tokens_cpu[req_indices],
               arange,
               out=positions_np)
        positions_np = positions_np.copy()
        positions = self.positions.cpu.clone()

        # Get token indices.
        # E.g., [0, 1, 0, 1, 2, 3, 4, 0, 1, 2]
        # -> [0, 1, M, M + 1, M + 2, M + 3, M + 4, 2 * M, 2 * M + 1, 2 * M + 2]
        # where M is the max_model_len.
        token_indices = (positions_np +
                         req_indices * input_batch.token_ids_cpu.shape[1])

        input_ids = self.input_ids.cpu.clone()
        # NOTE(woosuk): We use torch.index_select instead of np.take here
        # because torch.index_select is much faster than np.take for large
        # tensors.
        torch.index_select(input_batch.token_ids_cpu_tensor.flatten(),
                           0,
                           torch.from_numpy(token_indices),
                           out=input_ids[:total_num_scheduled_tokens])

        input_batch.block_table.compute_slot_mapping(req_indices, positions_np)
        input_batch.block_table.commit_slot_mapping(total_num_scheduled_tokens)

        query_start_loc_np = self.query_start_loc.np.copy()
        query_start_loc_np[0] = 0
        query_start_loc_np[1:num_reqs + 1] = cu_num_tokens
        # Note: pad query_start_loc to be non-decreasing, as kernels
        # like FlashAttention requires that
        query_start_loc_np[num_reqs + 1:].fill(cu_num_tokens[-1])
        query_start_loc = torch.tensor(query_start_loc_np,
                                       dtype=torch.int32)[:num_reqs + 1]

        seq_lens_np = self.seq_lens.np.copy()
        seq_lens_np[:num_reqs] = (
            input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)
        # Fill unused with 0 for full cuda graph mode.
        seq_lens_np[num_reqs:].fill(0)
        seq_lens = torch.tensor(seq_lens_np, dtype=torch.int32)[:num_reqs]
        max_seq_len = seq_lens_np[:num_reqs].max().item()

        # TODO: support spec_decode
        use_spec_decode = len(
            scheduler_output.scheduled_spec_decode_tokens) > 0
        if use_spec_decode:
            raise NotImplementedError(
                "Spec decode is not supported for DP dummy run")

        logits_indices = query_start_loc[1:] - 1

        logits_indices_padded = None

        # Used in the below loop.
        query_start_loc_cpu = query_start_loc
        seq_lens_cpu = seq_lens
        num_computed_tokens_cpu = (
            input_batch.num_computed_tokens_cpu_tensor[:num_reqs])

        attn_metadata_bucket: dict[int, dict[str, Any]] = {}
        input_ids_bucket: dict[int, torch.Tensor] = {}
        positions_bucket: dict[int, torch.Tensor] = {}

        for batch_bucket_size in self.bucketing_manager.decode_batch_buckets:
            attn_metadata: dict[str, Any] = {}
            # Prepare the attention metadata for each KV cache group and
            # make layers in the same group share the same metadata.
            for kv_cache_group_id, kv_cache_group_spec in enumerate(
                    self.kv_cache_config.kv_cache_groups):
                encoder_seq_lens = None

                if isinstance(kv_cache_group_spec.kv_cache_spec,
                              EncoderOnlyAttentionSpec):
                    raise NotImplementedError(
                        "Encoder-only attention is not supported for "
                        "DP dummy run")
                else:
                    blk_table = input_batch.block_table[kv_cache_group_id]
                    blk_table_tensor = blk_table.get_device_tensor(num_reqs)
                    slot_mapping = \
                        blk_table.slot_mapping.gpu[:total_num_scheduled_tokens]

                    # Fill unused with -1.
                    # Needed for reshape_and_cache in full cuda graph mode.
                    # `blk_table_tensor` -1 to match mamba PAD_SLOT_ID
                    slot_mapping[total_num_scheduled_tokens:
                                 total_num_scheduled_tokens].fill_(-1)
                    blk_table_tensor[
                        num_reqs:total_num_scheduled_tokens].fill_(-1)

                common_attn_metadata = CommonAttentionMetadata(
                    query_start_loc=query_start_loc,
                    query_start_loc_cpu=query_start_loc_cpu,
                    seq_lens=seq_lens,
                    seq_lens_cpu=seq_lens_cpu,
                    num_computed_tokens_cpu=num_computed_tokens_cpu,
                    num_reqs=num_reqs,
                    num_actual_tokens=total_num_scheduled_tokens,
                    max_query_len=max_num_scheduled_tokens,
                    max_seq_len=max_seq_len,
                    block_table_tensor=blk_table_tensor,
                    slot_mapping=slot_mapping,
                    logits_indices_padded=logits_indices_padded,
                    num_logits_indices=logits_indices.size(0),
                    causal=True,
                    encoder_seq_lens=encoder_seq_lens,
                )

                for attn_group in self.attn_groups[kv_cache_group_id]:
                    # Prepare for cascade attention if enabled & beneficial.
                    common_prefix_len = 0
                    builder = attn_group.get_metadata_builder()
                    if self.cascade_attn_enabled:
                        raise NotImplementedError(
                            "Cascade attention is not supported "
                            "for DP dummy run")

                    extra_attn_metadata_args = {}

                    if isinstance(builder, RBLNFlashAttentionMetadataBuilder):
                        extra_attn_metadata_args["num_tokens"] = \
                            input_batch.num_tokens
                        extra_attn_metadata_args["positions"] = positions
                        extra_attn_metadata_args[
                            "batch_pad"] = batch_bucket_size
                    attn_metadata_i = builder.build(
                        common_prefix_len=common_prefix_len,
                        common_attn_metadata=common_attn_metadata,
                        **extra_attn_metadata_args)

                    for layer_name in attn_group.layer_names:
                        attn_metadata[layer_name] = attn_metadata_i

            for attn_metadatum in attn_metadata.values():
                attn_metadatum.kv_caches = self.kv_caches
            num_input_tokens = total_num_scheduled_tokens
            input_ids = input_ids[:num_input_tokens]
            positions = positions[:num_input_tokens]

            input_ids = input_ids.view(num_reqs, -1).to(torch.long)
            positions = positions.view(num_reqs, -1)

            # decode batch padding
            input_ids = rbln_utils.pad(input_ids, 0, batch_bucket_size)
            positions = rbln_utils.pad(positions, -2, batch_bucket_size)

            attn_metadata_bucket[batch_bucket_size] = attn_metadata
            input_ids_bucket[batch_bucket_size] = input_ids
            positions_bucket[batch_bucket_size] = positions

        return DummyRunState(attn_metadata=attn_metadata_bucket,
                             num_input_tokens=num_input_tokens,
                             input_ids=input_ids_bucket,
                             positions=positions_bucket)

    def _prepare_dummy_input_batch(self) -> InputBatch:
        logits_processors = self.model_config.logits_processors
        custom_logitsprocs: Sequence[Union[str, type[LogitsProcessor]]] = (
            tuple(logits_processors) if logits_processors is not None else ())
        dummy_input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            # We need to use the encoder length for encoder-decoer
            # because of KV cache for cross-attention.
            max_model_len=max(self.max_model_len, self.max_encoder_len),
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[self.cache_config.block_size],
            kernel_block_sizes=[self.cache_config.block_size],
            is_spec_decode=bool(self.vllm_config.speculative_config),
            logitsprocs=build_logitsprocs(
                self.vllm_config,
                self.device,
                self.pin_memory,
                self.is_pooling_model,
                custom_logitsprocs,
            ),
            # We currently don't know whether a particular custom logits
            # processor uses output token ids so we set this conservatively.
            logitsprocs_need_output_token_ids=bool(custom_logitsprocs),
            is_pooling_model=self.is_pooling_model,
            cp_kv_cache_interleave_size=self.parallel_config.
            cp_kv_cache_interleave_size,
        )

        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in self.kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec,
                              EncoderOnlyAttentionSpec)
        ]
        kernel_block_sizes = self.kernel_block_sizes

        if block_sizes != [
                self.cache_config.block_size
        ] or kernel_block_sizes != [self.cache_config.block_size]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            dummy_input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.max_model_len, self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=dummy_input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config else 0),
            )

        return dummy_input_batch

    @torch.inference_mode()
    def prepare_dummy_run(self) -> None:
        # TODO: support spec_decode, pooling, mrope, lora,
        #       and encoder-only attention
        if self.is_pooling_model:
            raise NotImplementedError(
                "Pooling model is not supported for DP dummy run")
        if self.uses_mrope:
            raise NotImplementedError(
                "M-RoPE is not supported for DP dummy run")
        if self.lora_config:
            raise NotImplementedError("LoRA is not supported for DP dummy run")

        num_kv_cache_groups = len(self.kv_cache_config.kv_cache_groups)
        dummy_run_requests = []
        dummy_run_num_scheduled_tokens = {}
        self._add_dummy_requests(
            requests=dummy_run_requests,
            num_scheduled_tokens=dummy_run_num_scheduled_tokens,
            total_tokens=1,
            num_computed_tokens=1,
            num_kv_cache_groups=num_kv_cache_groups,
            sampling_params=None if self.is_pooling_model else SamplingParams(
                temperature=0.0),
            pooling_params=PoolingParams(
                task=self.get_supported_pooling_tasks()[0])
            if self.is_pooling_model else None,
        )
        dummy_run_scheduler_output, _ = self._make_dummy_scheduler_outputs(
            dummy_run_requests, dummy_run_num_scheduled_tokens,
            num_kv_cache_groups)

        dummy_input_batch = self._prepare_dummy_input_batch()
        self._update_dummy_states(dummy_run_scheduler_output,
                                  dummy_input_batch)
        self.dummy_run_state = self._prepare_dummy_inputs(
            dummy_run_scheduler_output, dummy_input_batch)

    @torch.inference_mode()
    def dummy_run(self) -> None:
        (attn_metadata, num_input_tokens, input_ids,
         positions) = self.dummy_run_state

        (batch_bucket_size, num_padded_tokens,
         num_tokens_across_dp) = self.get_dp_padding(
             num_input_tokens, self.bucketing_manager.decode_batch_buckets[0])

        attn_metadata = attn_metadata.get(batch_bucket_size)
        input_ids = input_ids.get(batch_bucket_size)
        positions = positions.get(batch_bucket_size)
        assert attn_metadata is not None \
            and input_ids is not None \
            and positions is not None, \
            "attn_metadata, input_ids, and positions should be defined" \
            f" for batch_bucket_size: {batch_bucket_size}"

        if get_pp_group().is_first_rank:
            intermediate_tensors = None
        else:
            intermediate_tensors = \
                self.decode_intermediate_tensors.get(batch_bucket_size)
            assert intermediate_tensors is not None

        with set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                num_padded_tokens=num_padded_tokens,
        ):
            token_indices = None
            inputs_embeds = None
            model_kwargs = dict[str, Any]({})

            _ = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                selected_token_indices=token_indices,
                inputs_embeds=inputs_embeds,
                **model_kwargs,
            )

    def _bookkeeping_sync(
        self, scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput, logits: Optional[torch.Tensor],
        hidden_states: torch.Tensor, num_scheduled_tokens: int
    ) -> tuple[
            dict[str, int],
            Optional[LogprobsLists],
            list[list[int]],
            dict[str, Optional[LogprobsTensors]],
            list[str],
            dict[str, int],
            list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        # TODO(woosuk): The following loop can be slow since it iterates over
        # the requests one by one. Optimize.
        discard_sampled_tokens_req_indices = []
        for i, req_id in enumerate(self.input_batch.req_ids):
            req_state = self.requests[req_id]
            seq_len = (req_state.num_computed_tokens +
                       scheduler_output.num_scheduled_tokens[req_id])
            if seq_len < req_state.num_tokens:
                # Ignore the sampled token for partial prefills.
                # Rewind the generator state as if the token was not sampled.
                # This relies on cuda-specific torch-internal impl details
                generator = self.input_batch.generators.get(i)
                if generator is not None:
                    generator.set_offset(generator.get_offset() - 4)
                # Record the index of the request that should not be sampled,
                # so that we could clear the sampled tokens before returning.
                discard_sampled_tokens_req_indices.append(i)

        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = \
            self.input_batch.req_id_to_index.copy()

        # NOTE: GPU -> CPU Sync happens here.
        # Move as many CPU operations as possible before this sync point.
        logprobs_tensors = sampler_output.logprobs_tensors
        logprobs_lists = logprobs_tensors.tolists() \
            if logprobs_tensors is not None else None

        # Compute prompt logprobs if needed.
        prompt_logprobs_dict = self._get_prompt_logprobs_dict(
            hidden_states[:num_scheduled_tokens],
            scheduler_output.num_scheduled_tokens,
        )

        num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
        sampled_token_ids = sampler_output.sampled_token_ids
        invalid_req_indices = []
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            if max_gen_len == 1:
                # No spec decode tokens.
                valid_sampled_token_ids = self._to_list(sampled_token_ids)
            else:
                # Includes spec decode tokens.
                valid_sampled_token_ids, _ = \
                    self.rejection_sampler.parse_output(
                        sampled_token_ids,
                        self.input_batch.vocab_size,
                    )
            # Mask out the sampled tokens that should not be sampled.
            for i in discard_sampled_tokens_req_indices:
                if i < len(valid_sampled_token_ids):
                    valid_sampled_token_ids[i].clear()
        else:
            valid_sampled_token_ids = []
            invalid_req_indices = list(discard_sampled_tokens_req_indices)
            invalid_req_indices_set = set(invalid_req_indices)
            assert sampled_token_ids.shape[-1] == 1

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            self.input_batch.prev_sampled_token_ids = \
                sampled_token_ids
            self.input_batch.prev_sampled_token_ids_invalid_indices = \
                invalid_req_indices_set
            self.input_batch.prev_req_id_to_index = {
                req_id: i
                for i, req_id in enumerate(self.input_batch.req_ids)
                if i not in invalid_req_indices_set
            }

        # Cache the sampled tokens in the model runner, so that the scheduler
        # doesn't need to send them back.
        # NOTE(woosuk): As an exception, when using PP, the scheduler sends
        # the sampled tokens back, because there's no direct communication
        # between the first-stage worker and the last-stage worker.
        req_ids = self.input_batch.req_ids
        for req_idx in range(num_sampled_tokens):
            if self.use_async_scheduling:
                sampled_ids = [-1] if \
                    req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]
            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + len(sampled_ids)
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)

        return (
            num_nans_in_logits,
            logprobs_lists,
            valid_sampled_token_ids,
            prompt_logprobs_dict,
            req_ids_output_copy,
            req_id_to_index_output_copy,
            invalid_req_indices,
        )

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
        num_padded_tokens: int | None = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors, None]:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("Preprocess"):
            self._update_states(scheduler_output)
            if not num_scheduled_tokens:
                if not has_kv_transfer_group():
                    # Return empty ModelRunnerOutput if there's no work to do.
                    return EMPTY_MODEL_RUNNER_OUTPUT
                return self.kv_connector_no_forward(scheduler_output,
                                                    self.vllm_config)
            if self.cache_config.kv_sharing_fast_prefill:
                assert not self.num_prompt_logprobs, (
                    "--kv-sharing-fast-prefill produces incorrect logprobs for "
                    "prompt tokens, tokens, please disable it when the requests"
                    " need prompt logprobs")

            num_reqs = self.input_batch.num_reqs
            req_ids = self.input_batch.req_ids
            tokens = [
                scheduler_output.num_scheduled_tokens[i] for i in req_ids
            ]
            num_scheduled_tokens_np = np.array(tokens, dtype=np.int32)
            # max_num_scheduled_tokens = int(num_scheduled_tokens_np.max())
            # num_tokens_unpadded = scheduler_output.total_num_scheduled_tokens

            # Prepare the decoder inputs.
            (attn_metadata, logits_indices, spec_decode_metadata,
             num_scheduled_tokens_np, spec_decode_common_attn_metadata,
             max_query_len, batch_bucket_size, num_padded_tokens,
             num_tokens_across_dp) = self._prepare_inputs(
                 scheduler_output, num_scheduled_tokens_np, num_padded_tokens)

            (
                num_input_tokens,
                input_ids,
                inputs_embeds,
                positions,
                model_kwargs,
            ) = self._preprocess(scheduler_output)

        # Padding for speculative decoding
        # in case of that all requests are not scheduled equally.
        num_scheduled_tokens_per_req = torch.tensor([
            scheduler_output.num_scheduled_tokens[i]
            for i in self.input_batch.req_ids
        ],
                                                    device=input_ids.device,
                                                    dtype=torch.int32)
        max_num_scheduled_tokens = torch.max(num_scheduled_tokens_per_req)

        if self.speculative_config is not None and not torch.all(
                num_scheduled_tokens_per_req == max_num_scheduled_tokens):
            input_ids = rbln_utils.pad_speculative_draft_tokens(
                input_ids, num_scheduled_tokens_per_req)
            positions = rbln_utils.pad_speculative_draft_tokens(
                positions, num_scheduled_tokens_per_req)

        # Run the model.
        # Use persistent buffers for CUDA graphs.
        with (set_forward_context(
                attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                num_padded_tokens=num_padded_tokens,
        ), record_function_or_nullcontext("Forward"),
              self.maybe_get_kv_connector_output(scheduler_output) as
              kv_connector_output):
            if attn_metadata is not None:
                for attn_metadatum in attn_metadata.values():
                    attn_metadatum.kv_caches = self.kv_caches

            # FIXME(jiwoo.park) This is a temporary workaround;
            # we must resolve the batch dimension.
            input_ids = input_ids.view(num_reqs, -1).to(torch.long)
            positions = positions.view(num_reqs, -1)

            is_prefills = self.is_prefills()

            token_indices = None
            if is_prefills[0]:
                # DO NOT include compute logits if lora_config is enabled
                token_indices = logits_indices

            # The prefill and decode cannot be mixed.
            assert len(is_prefills) > 0 and all(
                is_prefill == is_prefills[0]
                for is_prefill in is_prefills[:num_reqs])
            if is_prefills[0]:
                # prefill chunk padding
                max_seq_len = int(self.seq_lens.np[:num_reqs].max())
                prefill_size = (self.scheduler_config.max_num_batched_tokens
                                if self.scheduler_config.enable_chunked_prefill
                                else 1 << (math.ceil(math.log2(max_seq_len))))
                input_ids = rbln_utils.pad(input_ids, -1, prefill_size)
                positions = rbln_utils.pad(positions, -1, prefill_size)
            else:
                # decode batch padding
                input_ids = rbln_utils.pad(input_ids, 0, batch_bucket_size)
                positions = rbln_utils.pad(positions, -2, batch_bucket_size)

            if hasattr(rebel, "capture_reports"):
                capture_ctx = rebel.capture_reports()
            else:
                # use a dummy context manager that does nothing
                capture_ctx = contextlib.nullcontext()

            if self.lora_config is not None:
                lora_ids = [
                    self.requests[req_id].lora_request.lora_int_id
                    if self.requests[req_id].lora_request is not None else 0
                    for req_id in self.input_batch.req_ids
                ]

                lora_mask = create_lora_mask(
                    input_ids,
                    lora_ids,
                    self.lora_manager._adapter_manager.lora_index_to_id,
                    self.lora_config.max_loras,
                    self.lora_config.max_lora_rank,
                    self.lora_config.lora_dtype,
                    self.device,
                )
                sampler_indices_padded = create_sampler_indices_padded(
                    lora_ids,
                    self.lora_manager._adapter_manager.lora_index_to_id,
                    batch_bucket_size,
                    is_prefills[0],
                    self.lora_config.max_loras,
                    self.device,
                )
                LoRAMask.set_lora_mask(lora_mask)
                LoRAInputs.set_sampler_indices_padded(sampler_indices_padded)

            start_time = time.perf_counter()
            with capture_ctx as reports:
                if not self.use_wrapped_compute_logits():
                    model_output = self.model_executable(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs,
                    )
                else:
                    model_output = self.model_executable(
                        input_ids=input_ids,
                        positions=positions,
                        intermediate_tensors=intermediate_tensors,
                        selected_token_indices=token_indices,
                        inputs_embeds=inputs_embeds,
                        **model_kwargs,
                    )
            if self.performance_tracker is not None:
                # Record performance metrics
                end_time = time.perf_counter()
                execution_time = end_time - start_time
                host_time = None
                device_time = None
                ccl_time = None

                if reports is not None and len(reports) > 0:
                    host_time = reports[0].get('total_host', None)
                    device_time = reports[0].get('total_device', None)
                    ccl_time = reports[0].get('total_ccl', None)

                if is_prefills[0]:
                    self.performance_tracker.record_prefill(
                        execution_time,
                        num_scheduled_tokens,
                        host_time=host_time,
                        device_time=device_time,
                        ccl_time=ccl_time,
                        request_ids=self.input_batch.req_ids,
                    )
                else:
                    padded_decode = num_padded_tokens and \
                        num_padded_tokens != batch_bucket_size
                    self.performance_tracker.record_decode(
                        execution_time,
                        num_scheduled_tokens,
                        host_time=host_time,
                        device_time=device_time,
                        ccl_time=ccl_time,
                        padded_decode=padded_decode,
                        request_ids=self.input_batch.req_ids,
                    )

        with record_function_or_nullcontext("Postprocess"):
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states, logits = model_output
            else:
                hidden_states, logits = model_output
                aux_hidden_states = None
            sample_hidden_states = None

            # Broadcast PP output for external_launcher (torchrun)
            # to make sure we are synced across pp ranks
            # TODO: Support overlapping mirco-batches
            # https://github.com/vllm-project/vllm/issues/18019
            broadcast_pp_output = \
                self.parallel_config.distributed_executor_backend \
                == "external_launcher" and len(get_pp_group().ranks) > 0
            if not get_pp_group().is_last_rank:
                # For mid-pipeline stages, return intermediate tensors
                assert isinstance(hidden_states, IntermediateTensors)
                if not broadcast_pp_output:
                    hidden_states.kv_connector_output = kv_connector_output
                    return hidden_states
                # NOTE - DO NOT all_gather_group for RBLN pp
                get_pp_group().send_tensor_dict(hidden_states.tensors)
                logits = None
            else:
                # for last-pipeline stages, return hidden states
                if self.is_pooling_model:
                    return self._pool(hidden_states.flatten(0, -2),
                                      num_scheduled_tokens,
                                      num_scheduled_tokens_np,
                                      kv_connector_output)
                sample_hidden_states = hidden_states
                if not self.use_wrapped_compute_logits():
                    # DO NOT include compute logits

                    # FIXME(jiwoo.park) This is a temporary workaround;
                    # SHOULD resolve the batch dimension.
                    hidden_states = hidden_states.flatten(0, -2)

                    if is_prefills[0]:  # prefill
                        sample_hidden_states = hidden_states[logits_indices]
                        logits = self.compute_logits(sample_hidden_states)
                    else:  # decode
                        logits = self.compute_logits(hidden_states)
                        logits = logits[logits_indices]
                    if not envs.VLLM_RBLN_LOGITS_ALL_GATHER:
                        logits = self.logits_processor._gather_logits(logits)
                    logits = logits.view(-1, logits.size(-1))
                else:
                    selected_token_indices = logits_indices
                    assert selected_token_indices.dim() == 1
                    if is_prefills[0]:  # prefill
                        assert selected_token_indices.size(0) == 1
                        num_computed = self.input_batch.num_computed_tokens_cpu
                        num_prompted = self.input_batch.num_prompt_tokens
                        is_last_prefill = (num_computed +
                                           self.max_num_tokens) >= num_prompted
                        if not is_last_prefill[0]:
                            selected_token_indices = torch.tensor(
                                [], dtype=selected_token_indices.dtype)
                            # chunked prefill(#0~#N-1, intermediate)
                            # token_indices = torch.tensor([max_num_seqs-1])
                            # selected = torch.tensor([])
                            logits = logits[selected_token_indices]
                        else:
                            # chunked prefill(#N, final)
                            # token_indices = torch.tensor([last_seq_idx-1])
                            # selected_token_indices == token_indices
                            logits = logits
                    else:  # decode
                        # selected_token_indices is for valid decode tokens
                        # token_indices == None, selected = torch.tensor([0])
                        batch_indices = torch.arange(self.input_batch.num_reqs,
                                                     device=self.device)
                        if self.speculative_config is not None:
                            sample_hidden_states = hidden_states[
                                batch_indices, :self.speculative_config.
                                num_speculative_tokens + 1]
                        logits = logits[selected_token_indices]

            if broadcast_pp_output:
                model_output_broadcast_data = {
                    "logits": logits.contiguous(),
                } if logits is not None else {}
                model_output_broadcast_data = get_pp_group(
                ).broadcast_tensor_dict(model_output_broadcast_data,
                                        src=len(get_pp_group().ranks) - 1)
                assert model_output_broadcast_data is not None
                logits = model_output_broadcast_data["logits"]

        self.execute_model_state = ExecuteModelState(
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            kv_connector_output,
        )
        return None

    @torch.inference_mode()
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        if self.execute_model_state is None:
            # Nothing to do (PP non-final rank case), output isn't used.
            return None  # noqa

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            spec_decode_metadata,
            spec_decode_common_attn_metadata,
            hidden_states,
            sample_hidden_states,
            aux_hidden_states,
            kv_connector_output,
        ) = self.execute_model_state
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None and logits.shape[0] > 0:
            # NOTE(RBLN): `xgr.apply_token_bitmask_inplace` requires logits
            # to be float32 dtype for CPU tensors
            original_dtype = logits.dtype
            logits = logits.to(torch.float32)
            apply_grammar_bitmask(scheduler_output, grammar_output,
                                  self.input_batch, logits)
            logits = logits.to(original_dtype)

        with record_function_or_nullcontext("Sample"):
            sampler_output = self._sample(logits, spec_decode_metadata)

        def propose_draft_token_ids(sampled_token_ids):
            assert spec_decode_common_attn_metadata is not None
            with record_function_or_nullcontext("Draft"):
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
        use_padded_batch_for_eagle = (
            spec_config is not None and spec_config.use_eagle()
            and not spec_config.disable_padded_drafter_batch)
        effective_drafter_max_model_len = self.max_model_len
        if effective_drafter_max_model_len is None:
            effective_drafter_max_model_len = self.model_config.max_model_len
        if (spec_config is not None
                and spec_config.draft_model_config is not None
                and spec_config.draft_model_config.max_model_len is not None):
            effective_drafter_max_model_len = \
                spec_config.draft_model_config.max_model_len
        input_fits_in_drafter = spec_decode_common_attn_metadata and (
            spec_decode_common_attn_metadata.max_seq_len + self.num_spec_tokens
            <= effective_drafter_max_model_len)
        if use_padded_batch_for_eagle:
            assert spec_config is not None
            assert isinstance(self.drafter, EagleProposer)
            sampled_token_ids = sampler_output.sampled_token_ids
            if input_fits_in_drafter:
                # EAGLE speculative decoding can use the GPU sampled tokens as
                # inputs, and does not need to wait for bookkeeping to finish.
                propose_draft_token_ids(sampled_token_ids)

        with record_function_or_nullcontext("Bookkeep"):
            (
                num_nans_in_logits,
                logprobs_lists,
                valid_sampled_token_ids,
                prompt_logprobs_dict,
                req_ids_output_copy,
                req_id_to_index_output_copy,
                invalid_req_indices,
            ) = self._bookkeeping_sync(
                scheduler_output,
                sampler_output,
                logits,
                hidden_states,
                scheduler_output.total_num_scheduled_tokens,
            )

        if (self.speculative_config and not use_padded_batch_for_eagle
                and input_fits_in_drafter):
            # ngram and other speculative decoding methods use the sampled
            # tokens on the CPU, so they are run after bookkeeping.
            propose_draft_token_ids(valid_sampled_token_ids)

        # FIXME(jiwoo.park) EPLB is not supported in RBLN
        # with record_function_or_nullcontext("EPLB"):
        #     self.eplb_step()

        output = ModelRunnerOutput(
            req_ids=req_ids_output_copy,
            req_id_to_index=req_id_to_index_output_copy,
            sampled_token_ids=valid_sampled_token_ids,
            logprobs=logprobs_lists,
            prompt_logprobs_dict=prompt_logprobs_dict,
            pooler_output=[],
            kv_connector_output=kv_connector_output,
            num_nans_in_logits=num_nans_in_logits,
        )

        if not self.use_async_scheduling:
            return output

        return AsyncRBLNModelRunnerOutput(
            model_runner_output=output,
            sampled_token_ids=sampler_output.sampled_token_ids,
            invalid_req_indices=invalid_req_indices,
            async_output_copy_stream=self.async_output_copy_stream,
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        if self._draft_token_ids is None:
            return None
        req_ids = self.input_batch.req_ids
        if isinstance(self._draft_token_ids, torch.Tensor):
            draft_token_ids = self._draft_token_ids.tolist()
        else:
            draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return DraftTokenIds(req_ids, draft_token_ids)

    def _copy_valid_sampled_token_count(
            self, next_token_ids: torch.Tensor,
            valid_sampled_tokens_count: torch.Tensor) -> None:
        if self.valid_sampled_token_count_event is None:
            return

        default_stream = torch.cuda.current_stream()
        # Initialize a new stream to overlap the copy operation with
        # prepare_input of draft model.
        with torch.cuda.stream(self.valid_sampled_token_count_copy_stream):
            self.valid_sampled_token_count_copy_stream.wait_stream(
                default_stream)  # type: ignore
            counts = valid_sampled_tokens_count
            counts_cpu = self.valid_sampled_token_count_cpu
            counts_cpu[:counts.shape[0]].copy_(counts, non_blocking=True)
            self.valid_sampled_token_count_event.record()

        self.input_batch.prev_sampled_token_ids = next_token_ids.unsqueeze(1)

    def propose_draft_token_ids(
        self,
        scheduler_output: "SchedulerOutput",
        sampled_token_ids: torch.Tensor | list[list[int]],
        sampling_metadata: SamplingMetadata,
        hidden_states: torch.Tensor,
        sample_hidden_states: torch.Tensor,
        aux_hidden_states: list[torch.Tensor] | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
        common_attn_metadata: CommonAttentionMetadata,
    ) -> list[list[int]] | torch.Tensor:
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        spec_config = self.speculative_config
        assert spec_config is not None
        if spec_config.method == "ngram":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, NgramProposer)
            draft_token_ids = self.drafter.propose(
                sampled_token_ids,
                self.input_batch.req_ids,
                self.input_batch.num_tokens_no_spec,
                self.input_batch.token_ids_cpu,
                self.input_batch.spec_decode_unsupported_reqs,
            )
        elif spec_config.method == "suffix":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, SuffixDecodingProposer)
            draft_token_ids = self.drafter.propose(self.input_batch,
                                                   sampled_token_ids)
        elif spec_config.method == "medusa":
            assert isinstance(sampled_token_ids, list)
            assert isinstance(self.drafter, MedusaProposer)

            if sample_hidden_states.shape[0] == len(sampled_token_ids):
                # The input to the target model does not include draft tokens.
                hidden_states = sample_hidden_states
            else:
                indices = []
                offset = 0
                assert spec_decode_metadata is not None, (
                    "No spec decode metadata for medusa")
                for num_draft, tokens in zip(
                        spec_decode_metadata.num_draft_tokens,
                        sampled_token_ids,
                        strict=False):
                    indices.append(offset + len(tokens) - 1)
                    offset += num_draft + 1
                indices = torch.tensor(indices, device=self.device)
                hidden_states = sample_hidden_states[indices]

            hidden_states = hidden_states.reshape(-1, hidden_states.shape[-1])
            draft_token_ids = self.drafter.propose(
                target_hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )
        elif spec_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            assert spec_config.disable_padded_drafter_batch is not True, (
                "This option is not supported in vllm-rbln.")
            assert isinstance(
                sampled_token_ids,
                torch.Tensor), ("sampled_token_ids should be a torch.Tensor.")

            next_token_ids, valid_sampled_tokens_count = (
                self.drafter.prepare_next_token_ids_padded(
                    common_attn_metadata,
                    sampled_token_ids,
                    self.requests,
                    self.input_batch,
                    self.discard_request_mask.gpu,
                ))
            self._copy_valid_sampled_token_count(next_token_ids,
                                                 valid_sampled_tokens_count)

            target_hidden_states = hidden_states
            if spec_decode_metadata is None:
                token_indices_to_sample = None
                # input_ids can be None for multimodal models.
                target_token_ids = self.input_ids.gpu[:num_scheduled_tokens]
                target_positions = self._get_positions(num_scheduled_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h.view(-1, h.shape[-1]) for h in aux_hidden_states],
                        dim=-1)
            else:
                common_attn_metadata, token_indices_to_sample = (
                    self.drafter.prepare_inputs_padded(
                        common_attn_metadata,
                        spec_decode_metadata,
                        valid_sampled_tokens_count,
                    ))
                total_num_tokens = common_attn_metadata.num_actual_tokens
                # When padding the batch, token_indices is just a range
                target_token_ids = self.input_ids.gpu[:total_num_tokens]
                target_positions = self._get_positions(total_num_tokens)
                if self.use_aux_hidden_state_outputs:
                    assert aux_hidden_states is not None
                    target_hidden_states = torch.cat(
                        [h.view(-1, h.shape[-1]) for h in aux_hidden_states],
                        dim=-1)

            if self.supports_mm_inputs:
                mm_embed_inputs = self._gather_mm_embeddings(
                    scheduler_output,
                    shift_computed_tokens=1,
                )
            else:
                mm_embed_inputs = None

            draft_token_ids = self.drafter.propose(
                target_token_ids=target_token_ids,
                target_positions=target_positions,
                target_hidden_states=target_hidden_states,
                next_token_ids=next_token_ids,
                last_token_indices=token_indices_to_sample,
                sampling_metadata=sampling_metadata,
                common_attn_metadata=common_attn_metadata,
                mm_embed_inputs=mm_embed_inputs,
                kv_caches=self.kv_caches,
            )

        return draft_token_ids

    def load_model(self) -> None:
        logger.info("Starting to load model %s...", self.model_config.model)
        model_loader = get_model_loader(self.load_config)
        if not hasattr(self, "model"):
            logger.info("Loading model from scratch...")
            self.model = model_loader.load_model(
                vllm_config=self.vllm_config, model_config=self.model_config)
        else:
            logger.info(
                "Model was already initialized. Loading weights inplace...")
            model_loader.load_weights(self.model,
                                      model_config=self.model_config)

        self.model = self.get_model().eval()
        self.compute_logits_model = self.model
        if self.model_config.is_multimodal_model and hasattr(
                self.model.get_language_model(), "logits_processor"):
            self.compute_logits_model = self.model.get_language_model()
            self.logits_processor = self.model.get_language_model(
            ).logits_processor
        elif hasattr(self.model, "logits_processor"):
            self.logits_processor = self.model.logits_processor
        else:
            self.logits_processor = None

        logger.info("load_model = %s", self.model)
        logger.info("model_config.num_layers = %d",
                    self.model_config.get_num_layers(self.parallel_config))

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            intermediate_tensors: Optional[IntermediateTensors] = None,
            inputs_embeds: Optional[torch.Tensor] = None,
            selected_token_indices: Optional[torch.Tensor] = None,
        ) -> tuple[Union[torch.Tensor, IntermediateTensors],
                   Optional[torch.Tensor]]:
            """
            This wrapper function is designed to be compiled by torch.compile.
            It handles the forward pass of the underlying model and, computes
            the logits from the hidden states if necessary.
            """
            model_output = self.model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds)

            logits = None
            if self.use_aux_hidden_state_outputs:
                hidden_states, aux_hidden_states = model_output
            else:
                hidden_states = model_output

            if get_pp_group().is_last_rank \
                and self.lora_config is None \
                and (self.speculative_config is None \
                     or self.speculative_config.method \
                        not in ("eagle", "eagle3")) \
                and not self.is_pooling_model \
                and self.logits_processor is not None:

                # last rank create real model output
                if selected_token_indices is not None:
                    # aten::select -> adv_index -->
                    #     contrib_dynamic_take (tensor -> scalar)
                    # aten::index_select --> take -->
                    #     contrib_dynamic_take (tensor -> scalar)
                    hidden_states = hidden_states[:, selected_token_indices]
                logits = self.compute_logits_model.compute_logits(
                    hidden_states)
                logits = logits.view(-1, logits.size(-1))

            # non last rank create intermediate tensors, bypass it
            if self.use_aux_hidden_state_outputs:
                return hidden_states, aux_hidden_states, logits
            return hidden_states, logits

        if self.lora_config:
            self.model = self.load_lora_model(
                self.model,
                self.vllm_config,
                self.device,
            )
        if hasattr(self, "drafter"):
            logger.info("Loading drafter model...")
            self.drafter.load_model(self.model)
        if self.use_aux_hidden_state_outputs:
            self.model.set_aux_hidden_state_layers(
                self.model.get_eagle3_aux_hidden_state_layers())

        # FIXME - device specific communication buffer (CUDA)?
        # disable communication buffer for RBLN (NYI)
        # communication buffers for efficient communication
        # TODO - RBLN communication buffer can be defined
        # prepare_communication_buffer_for_model(self.model)

        if self.model_config.enforce_eager or not envs.VLLM_RBLN_COMPILE_MODEL:
            self.model_executable = model_wrapper
        else:
            # NOTE - refer to pytorch 2.5 release notes
            # torch.compile regional compilation without recompilations
            # To prevent nn.modules parameters to be model input, set false
            # if this flag is set, nn.modules parameters are treated
            # as model input
            torch._dynamo.config.inline_inbuilt_nn_modules = False
            # RBLN compile context to mark static address for kv cache tensor
            # if tensor is set to have static address,
            # similar to RBLN kv cache binding
            from rebel.compile_context import CompileContext

            self.compile_context = CompileContext(use_weight_sharing=True)
            compiled_graph = self._compile_model(model_wrapper)
            self.model_executable = compiled_graph

        distributed_executor_backend = \
            self.vllm_config.parallel_config.distributed_executor_backend
        if distributed_executor_backend == "ray":
            self._prepare_prefill_intermediate_tensors()
            for batch_bucket_size \
                in self.bucketing_manager.decode_batch_buckets:
                self._prepare_decode_intermediate_tensors(batch_bucket_size)
        else:
            with torch.inference_mode():
                self._prepare_prefill_intermediate_tensors()
                for batch_bucket_size \
                    in self.bucketing_manager.decode_batch_buckets:
                    self._prepare_decode_intermediate_tensors(
                        batch_bucket_size)

    def _prepare_prefill_intermediate_tensors(self) -> None:

        def _reshape(
                batch_size: int, seq_len: int,
                intermediate_tensors: IntermediateTensors
        ) -> IntermediateTensors:
            return IntermediateTensors({
                k: v.view(batch_size, seq_len, -1)
                for k, v in intermediate_tensors.items()
            })

        batch_size = self.max_prefill_batch_size
        seq_len = self.max_num_batched_tokens
        self.prefill_intermediate_tensors = _reshape(
            batch_size, seq_len,
            self.model.make_empty_intermediate_tensors(
                batch_size=batch_size * seq_len,
                dtype=self.model_config.dtype,
                device=self.device))

    def _prepare_decode_intermediate_tensors(self, batch_bucket_size) -> None:

        def _reshape(
                batch_size: int, seq_len: int,
                intermediate_tensors: IntermediateTensors
        ) -> IntermediateTensors:
            return IntermediateTensors({
                k: v.view(batch_size, seq_len, -1)
                for k, v in intermediate_tensors.items()
            })

        batch_size = batch_bucket_size
        seq_len = 1
        self.decode_intermediate_tensors[batch_bucket_size] = _reshape(
            batch_size, seq_len,
            self.model.make_empty_intermediate_tensors(
                batch_size=batch_size * seq_len,
                dtype=self.model_config.dtype,
                device=self.device))

    def save_tensorized_model(
        self,
        tensorizer_config: "TensorizerConfig",
    ) -> None:
        model = self.get_model()
        TensorizerLoader.save_model(
            model,
            tensorizer_config=tensorizer_config,
            model_config=self.model_config,
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, Optional[LogprobsTensors]]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
            return {}

        in_progress_dict = self.input_batch.in_progress_prompt_logprobs_cpu
        prompt_logprobs_dict: dict[str, Optional[LogprobsTensors]] = {}

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
            num_prompt_tokens = len(request.prompt_token_ids)
            prompt_token_ids = torch.tensor(request.prompt_token_ids).to(
                self.device, non_blocking=True)

            # Set up target LogprobsTensors object.
            logprobs_tensors = in_progress_dict.get(req_id)
            if not logprobs_tensors:
                # Create empty logprobs CPU tensors for the entire prompt.
                # If chunked, we'll copy in slice by slice.
                logprobs_tensors = LogprobsTensors.empty_cpu(
                    num_prompt_tokens - 1, num_prompt_logprobs + 1)
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
            offset = self.query_start_loc.np[req_idx].item()
            prompt_hidden_states = hidden_states[offset:offset + num_logits]
            logits = self.model.compute_logits(prompt_hidden_states)

            # Get the "target" tokens for each index. For prompt at index i,
            # the token at prompt index i+1 is the "sampled" token we want
            # to gather the logprob for.
            tgt_token_ids = prompt_token_ids[start_tok:start_tok + num_logits]

            # Compute prompt logprobs.
            logprobs = self.sampler.compute_logprobs(logits)
            token_ids, logprobs, ranks = self.sampler.gather_logprobs(
                logprobs, num_prompt_logprobs, tgt_token_ids)

            # Transfer GPU->CPU async.
            chunk_slice = slice(start_idx, start_idx + num_logits)
            logprobs_tensors.logprob_token_ids[chunk_slice].copy_(
                token_ids, non_blocking=True)
            logprobs_tensors.logprobs[chunk_slice].copy_(logprobs,
                                                         non_blocking=True)
            logprobs_tensors.selected_token_ranks[chunk_slice].copy_(
                ranks, non_blocking=True)

        # Remove requests that have completed prefill from the batch
        # num_prompt_logprobs_dict.
        for req_id in completed_prefill_reqs:
            del num_prompt_logprobs_dict[req_id]
            del in_progress_dict[req_id]

        # Must synchronize the non-blocking GPU->CPU transfers.
        if prompt_logprobs_dict:
            self._sync_device()

        return prompt_logprobs_dict

    @contextmanager
    def maybe_randomize_inputs(self, input_ids: torch.Tensor):
        """
        Randomize input_ids if VLLM_RANDOMIZE_DP_DUMMY_INPUTS is set.
        This is to help balance expert-selection
         - during profile_run
         - during DP rank dummy run
        """
        dp_size = self.vllm_config.parallel_config.data_parallel_size
        randomize_inputs = envs.VLLM_RANDOMIZE_DP_DUMMY_INPUTS and dp_size > 1
        if not randomize_inputs:
            yield
        else:
            import functools

            @functools.cache
            def rand_input_ids() -> torch.Tensor:
                return torch.randint_like(
                    self.input_ids.gpu,
                    low=0,
                    high=self.model_config.get_vocab_size(),
                    dtype=input_ids.dtype)

            logger.debug_once("Randomizing dummy data for DP Rank")
            input_ids.copy_(rand_input_ids()[:input_ids.size(0)],
                            non_blocking=True)
            yield
            input_ids.fill_(0)

    def initialize_attn_backend(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize the attention backends and attention metadata builders.
        """
        assert len(self.attn_groups) == 0, \
            "Attention backends are already initialized"

        class AttentionGroupKey(NamedTuple):
            attn_backend: type[AttentionBackend]
            kv_cache_spec: KVCacheSpec

        def get_attn_backends_for_group(
            kv_cache_group_spec: KVCacheGroupSpec,
        ) -> tuple[dict[AttentionGroupKey, list[str]],
                   set[type[AttentionBackend]]]:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(
                self.vllm_config, layer_type, kv_cache_group_spec.layer_names)
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
                    layer_kv_cache_spec = layer_kv_cache_spec.kv_cache_specs[
                        layer_name]
                key = (full_cls_name, layer_kv_cache_spec)
                attn_backends[key] = AttentionGroupKey(attn_backend,
                                                       layer_kv_cache_spec)
                attn_backend_layers[key].append(layer_name)
            return (
                {
                    attn_backends[k]: v
                    for k, v in attn_backend_layers.items()
                },
                set(group_key.attn_backend
                    for group_key in attn_backends.values()),
            )

        def create_attn_groups(
            attn_backends_map: dict[AttentionGroupKey, list[str]],
            kv_cache_group_id: int,
        ) -> list[AttentionGroup]:
            attn_groups: list[AttentionGroup] = []
            for (attn_backend,
                 kv_cache_spec), layer_names in attn_backends_map.items():
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

    def initialize_metadata_builders(self, kv_cache_config: KVCacheConfig,
                                     kernel_block_sizes: list[int]) -> None:
        """
        Create the metadata builders for all KV cache groups and attn groups.
        """
        for kv_cache_group_id in range(len(kv_cache_config.kv_cache_groups)):
            for attn_group in self.attn_groups[kv_cache_group_id]:
                attn_group.create_metadata_builders(
                    self.vllm_config,
                    self.device,
                    kernel_block_sizes[kv_cache_group_id]
                    if kv_cache_group_id < len(kernel_block_sizes) else None,
                    num_metadata_builders=1
                    if not self.parallel_config.enable_dbo else 2,
                )
        # Calculate reorder batch threshold (if needed)
        # Note (tdoublep): do this *after* constructing builders,
        # because some of them change the threshold at init time.
        self.calculate_reorder_batch_threshold()

    def calculate_reorder_batch_threshold(self) -> None:
        """
        Check that if any backends reorder batches; that the reordering
        is compatible (e.g., decode threshold is the same)
        """

        # check that if any backends reorder batches; that the reordering
        # is compatible (e.g., decode threshold is the same)
        # TODO(jiwoo.park): We need to implement reorder batch threshold
        pass

    @staticmethod
    def select_common_block_size(kv_manager_block_size: int,
                                 attn_groups: list[AttentionGroup]) -> int:
        """
        Select a block size that is supported by all backends and is a factor of
        kv_manager_block_size.

        If kv_manager_block_size is supported by all backends,
        return it directly. Otherwise, return the max supported size.

        Args:
            kv_manager_block_size: Block size of KV cache
            attn_groups: List of attention groups

        Returns:
            The selected block size

        Raises:
            ValueError: If no valid block size found
        """

        def block_size_is_supported(backends: list[type[AttentionBackend]],
                                    block_size: int) -> bool:
            """
            Check if the block size is supported by all backends.
            """
            for backend in backends:
                is_supported = False
                for supported_size in backend.get_supported_kernel_block_sizes(
                ):
                    if isinstance(supported_size, int):
                        if block_size == supported_size:
                            is_supported = True
                    elif isinstance(supported_size, MultipleOf):
                        if block_size % supported_size.base == 0:
                            is_supported = True
                    else:
                        raise ValueError(
                            f"Unknown supported size: {supported_size}")
                if not is_supported:
                    return False
            return True

        backends = [group.backend for group in attn_groups]

        # Case 1:
        # if the block_size of kv cache manager is supported by all backends,
        # return it directly
        if block_size_is_supported(backends, kv_manager_block_size):
            return kv_manager_block_size

        # Case 2:
        # otherwise, the block_size must be an `int`-format supported size of
        # at least one backend. Iterate over all `int`-format supported sizes in
        # descending order and return the first one that is supported by all
        # backends.
        # Simple proof:
        # If the supported size b is in MultipleOf(x_i) format for all attention
        # backends i, and b a factor of kv_manager_block_size, then
        # kv_manager_block_size also satisfies MultipleOf(x_i) for all i.
        # We will return kv_manager_block_size in case 1.
        all_int_supported_sizes = set(
            supported_size for backend in backends
            for supported_size in backend.get_supported_kernel_block_sizes()
            if isinstance(supported_size, int))

        for supported_size in sorted(all_int_supported_sizes, reverse=True):
            if kv_manager_block_size % supported_size != 0:
                continue
            if block_size_is_supported(backends, supported_size):
                return supported_size
        raise ValueError(f"No common block size for {kv_manager_block_size}. ")

    def may_reinitialize_input_batch(self, kv_cache_config: KVCacheConfig,
                                     kernel_block_sizes: list[int]) -> None:
        """
        Re-initialize the input batch if the block sizes are different from
        `[self.cache_config.block_size]`. This usually happens when there
        are multiple KV cache groups.

        Args:
            kv_cache_config: The KV cache configuration.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        """
        block_sizes = [
            kv_cache_group.kv_cache_spec.block_size
            for kv_cache_group in kv_cache_config.kv_cache_groups
            if not isinstance(kv_cache_group.kv_cache_spec,
                              EncoderOnlyAttentionSpec)
        ]

        if block_sizes != [
                self.cache_config.block_size
        ] or kernel_block_sizes != [self.cache_config.block_size]:
            assert self.cache_config.cpu_offload_gb == 0, (
                "Cannot re-initialize the input batch when CPU weight "
                "offloading is enabled. See https://github.com/vllm-project/vllm/pull/18298 "  # noqa: E501
                "for more details.")
            self.input_batch = InputBatch(
                max_num_reqs=self.max_num_reqs,
                max_model_len=max(self.max_model_len, self.max_encoder_len),
                max_num_batched_tokens=self.max_num_tokens,
                device=self.device,
                pin_memory=self.pin_memory,
                vocab_size=self.model_config.get_vocab_size(),
                block_sizes=block_sizes,
                kernel_block_sizes=kernel_block_sizes,
                is_spec_decode=bool(self.vllm_config.speculative_config),
                logitsprocs=self.input_batch.logitsprocs,
                is_pooling_model=self.is_pooling_model,
                num_speculative_tokens=(
                    self.vllm_config.speculative_config.num_speculative_tokens
                    if self.vllm_config.speculative_config else 0),
            )

    def _allocate_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig) -> dict[str, torch.Tensor]:
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
            tensor = torch.zeros(kv_cache_tensor.size,
                                 dtype=torch.int8,
                                 device="meta")
            for layer_name in kv_cache_tensor.shared_by:
                kv_cache_raw_tensors[layer_name] = tensor

        layer_names = set()
        for group in kv_cache_config.kv_cache_groups:
            for layer_name in group.layer_names:
                if layer_name in self.runner_only_attn_layers:
                    continue
                layer_names.add(layer_name)
        assert layer_names == set(kv_cache_raw_tensors.keys(
        )), "Some layers are not correctly initialized"
        return kv_cache_raw_tensors

    def _attn_group_iterator(self) -> Iterator[AttentionGroup]:
        return itertools.chain.from_iterable(self.attn_groups)

    def _kv_cache_spec_attn_group_iterator(self) -> Iterator[AttentionGroup]:
        if not self.kv_cache_config.kv_cache_groups:
            return
        for attn_groups in self.attn_groups:
            yield from attn_groups

    def _prepare_kernel_block_sizes(
            self, kv_cache_config: KVCacheConfig) -> list[int]:
        """
        Generate kernel_block_sizes that matches each block_size.

        For attention backends that support virtual block splitting,
        use the supported block sizes from the backend.
        For other backends (like Mamba), use the same block size (no splitting).

        Args:
            kv_cache_config: The KV cache configuration.

        Returns:
            list[int]: List of kernel block sizes for each cache group.
        """
        kernel_block_sizes = []
        for kv_cache_gid, kv_cache_group in enumerate(
                kv_cache_config.kv_cache_groups):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            if isinstance(kv_cache_spec, UniformTypeKVCacheSpecs):
                # All layers in the UniformTypeKVCacheSpecs have the same type,
                # Pick an arbitrary one to dispatch.
                kv_cache_spec = next(
                    iter(kv_cache_spec.kv_cache_specs.values()))
            if isinstance(kv_cache_spec, EncoderOnlyAttentionSpec):
                continue
            elif isinstance(kv_cache_spec, RBLNSlidingWindowSpec):
                kernel_block_sizes.append(kv_cache_spec.sliding_window)
            elif isinstance(kv_cache_spec, AttentionSpec):
                # This is an attention backend that supports virtual
                # block splitting. Get the supported block sizes from
                # all backends in the group.
                attn_groups = self.attn_groups[kv_cache_gid]
                kv_manager_block_size = kv_cache_group.kv_cache_spec.block_size
                selected_kernel_size = self.select_common_block_size(
                    kv_manager_block_size, attn_groups)
                kernel_block_sizes.append(selected_kernel_size)
            elif isinstance(kv_cache_spec, MambaSpec):
                # This is likely Mamba or other non-attention cache,
                # no splitting.
                kernel_block_sizes.append(kv_cache_spec.block_size)
            else:
                raise NotImplementedError(
                    f"unknown kv cache spec {kv_cache_group.kv_cache_spec}")
        return kernel_block_sizes

    def _reshape_kv_cache_tensors(
        self,
        kv_cache_config: KVCacheConfig,
        kv_cache_raw_tensors: dict[str, torch.Tensor],
        kernel_block_sizes: list[int],
    ) -> dict[str, torch.Tensor]:
        """
        Reshape the KV cache tensors to the desired shape and dtype.

        Args:
            kv_cache_config: The KV cache config
            kv_cache_raw_tensors: The KV cache buffer of each layer, with
                correct size but uninitialized shape.
            kernel_block_sizes: The kernel block sizes for each KV cache group.
        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """
        kv_caches: dict[str, torch.Tensor] = {}
        has_attn, has_mamba = False, False
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
                num_blocks = (raw_tensor.numel() //
                              kv_cache_spec.page_size_bytes)
                if isinstance(kv_cache_spec, AttentionSpec):
                    has_attn = True
                    num_blocks_per_kv_block = (kv_cache_spec.block_size //
                                               kernel_block_size)
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
                        kv_cache_stride_order = \
                            attn_backend.get_kv_cache_stride_order()
                        assert len(kv_cache_stride_order) == len(
                            kv_cache_shape)
                    except (AttributeError, NotImplementedError):
                        kv_cache_stride_order = tuple(
                            range(len(kv_cache_shape)))
                    # The allocation respects the backend-defined stride order
                    # to ensure the semantic remains consistent for each
                    # backend. We first obtain the generic kv cache shape and
                    # then permute it according to the stride order which could
                    # result in a non-contiguous tensor.
                    kv_cache_shape = tuple(kv_cache_shape[i]
                                           for i in kv_cache_stride_order)
                    # Maintain original KV shape view.
                    inv_order = [
                        kv_cache_stride_order.index(i)
                        for i in range(len(kv_cache_stride_order))
                    ]
                    kv_caches[layer_name] = kv_cache_raw_tensors[
                        layer_name].view(dtype).view(kv_cache_shape).permute(
                            *inv_order)
                elif isinstance(kv_cache_spec, MambaSpec):
                    has_mamba = True
                    raw_tensor = kv_cache_raw_tensors[layer_name]
                    state_tensors = []
                    storage_offset_bytes = 0
                    for (shape, dtype) in zip(kv_cache_spec.shapes,
                                              kv_cache_spec.dtypes):
                        dtype_size = get_dtype_size(dtype)
                        num_element_per_page = (
                            kv_cache_spec.page_size_bytes // dtype_size)
                        target_shape = (num_blocks, *shape)
                        stride = torch.empty(target_shape).stride()
                        target_stride = (num_element_per_page, *stride[1:])
                        assert storage_offset_bytes % dtype_size == 0
                        tensor = torch.as_strided(
                            raw_tensor.view(dtype),
                            size=target_shape,
                            stride=target_stride,
                            storage_offset=storage_offset_bytes // dtype_size,
                        )
                        state_tensors.append(tensor)
                        storage_offset_bytes += stride[0] * dtype_size

                    kv_caches[layer_name] = state_tensors
                else:
                    raise NotImplementedError

        if has_attn and has_mamba:
            self._update_hybrid_attention_mamba_layout(kv_caches)

        return kv_caches

    def initialize_kv_cache_tensors(
            self, kv_cache_config: KVCacheConfig,
            kernel_block_sizes: list[int]) -> dict[str, torch.Tensor]:
        """
        Initialize the memory buffer for KV cache.

        Args:
            kv_cache_config: The KV cache config
            kernel_block_sizes: The kernel block sizes for each KV cache group.

        Returns:
            Dict[str, torch.Tensor]: A map between layer names to their
            corresponding memory buffer for KV cache.
        """

        # Try creating KV caches optimized for kv-connector transfers
        cache_dtype = self.cache_config.cache_dtype
        if self.use_uniform_kv_cache(self.attn_groups, cache_dtype):
            kv_caches, cross_layers_kv_cache, attn_backend = (
                self.allocate_uniform_kv_caches(
                    kv_cache_config,
                    self.attn_groups,
                    cache_dtype,
                    self.device,
                    kernel_block_sizes,
                ))
            self.cross_layers_kv_cache = cross_layers_kv_cache
            self.cross_layers_attn_backend = attn_backend
        else:
            # Fallback to the general case
            # Initialize the memory buffer for KV cache
            kv_cache_raw_tensors = self._allocate_kv_cache_tensors(
                kv_cache_config)

            # Change the memory buffer to the desired shape
            kv_caches = self._reshape_kv_cache_tensors(kv_cache_config,
                                                       kv_cache_raw_tensors,
                                                       kernel_block_sizes)

        # Set up cross-layer KV cache sharing
        for layer_name, target_layer_name in self.shared_kv_cache_layers.items(
        ):
            logger.debug("%s reuses KV cache of %s", layer_name,
                         target_layer_name)
            kv_caches[layer_name] = kv_caches[target_layer_name]

        num_attn_module = (2 if self.model_config.hf_config.model_type
                           == "longcat_flash" else 1)
        bind_kv_cache(
            kv_caches,
            self.compilation_config.static_forward_context,
            self.kv_caches,
            num_attn_module,
        )
        if not self.model_config.enforce_eager and envs.VLLM_RBLN_COMPILE_MODEL:
            for kv_cache in self.kv_caches:
                self.compile_context.mark_static_address(kv_cache)

        return kv_caches

    def maybe_add_kv_sharing_layers_to_kv_cache_groups(
            self, kv_cache_config: KVCacheConfig) -> None:
        """
        Add layers that reuse KV cache to KV cache group of its target layer.
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
            attn_layers = get_layers_from_vllm_config(self.vllm_config,
                                                      Attention)
            for layer_name in reversed(attn_layers):
                if layer_name in self.shared_kv_cache_layers:
                    self.kv_sharing_fast_prefill_eligible_layers.add(
                        layer_name)
                else:
                    break

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """
        Initialize KV cache based on `kv_cache_config`.
        Args:
            kv_cache_config: Configuration for the KV cache, including the KV
            cache size of each layer
        """
        kv_cache_config = deepcopy(kv_cache_config)
        self.kv_cache_config = kv_cache_config
        self.may_add_encoder_only_layers_to_kv_cache_config()
        self.maybe_add_kv_sharing_layers_to_kv_cache_groups(kv_cache_config)
        self.initialize_attn_backend(kv_cache_config)
        # The kernel block size for all KV cache groups. For example, if
        # kv_cache_manager uses block_size 256 for a given group,
        # but the attention backends for that group only supports block_size 64,
        # we will return kernel_block_size 64 and split the 256-token-block to
        # 4 blocks with 64 tokens each.
        kernel_block_sizes = self._prepare_kernel_block_sizes(kv_cache_config)
        self.kernel_block_sizes = kernel_block_sizes

        # create metadata builders
        self.initialize_metadata_builders(kv_cache_config, kernel_block_sizes)

        # Reinitialize need to after initialize_attn_backend
        self.may_reinitialize_input_batch(kv_cache_config, kernel_block_sizes)
        kv_caches = self.initialize_kv_cache_tensors(kv_cache_config,
                                                     kernel_block_sizes)

        if self.speculative_config and self.speculative_config.use_eagle():
            assert isinstance(self.drafter, EagleProposer)
            # validate all draft model layers belong to the same kv cache
            # group
            self.drafter.validate_same_kv_cache_group(kv_cache_config)

        if has_kv_transfer_group():
            kv_transfer_group = get_kv_transfer_group()
            if self.cross_layers_kv_cache is not None:
                assert self.cross_layers_attn_backend is not None
                kv_transfer_group.register_cross_layers_kv_cache(
                    self.cross_layers_kv_cache, self.cross_layers_attn_backend)
            else:
                kv_transfer_group.register_kv_caches(kv_caches)
            kv_transfer_group.set_host_xfer_buffer_ops(copy_kv_blocks)

        if self.dcp_world_size > 1:
            layer_type = cast(type[Any], AttentionLayerBase)
            layers = get_layers_from_vllm_config(self.vllm_config, layer_type)
            for layer in layers.values():
                layer_impl = getattr(layer, "impl", None)
                if layer_impl is None:
                    continue
                assert layer_impl.need_to_return_lse_for_decode, (
                    "DCP requires attention impls to return"
                    " the softmax lse for decode, but the impl "
                    f"{layer_impl.__class__.__name__} "
                    "does not return the softmax lse for decode.")

        self.cache_config.num_gpu_blocks = kv_cache_config.num_blocks
        self.cache_config.num_cpu_blocks = 0

    def may_add_encoder_only_layers_to_kv_cache_config(self) -> None:
        """
        Add encoder-only layers to the KV cache config.
        """
        block_size = self.vllm_config.cache_config.block_size
        encoder_only_attn_specs: dict[AttentionSpec,
                                      list[str]] = defaultdict(list)
        attn_layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        for layer_name, attn_module in attn_layers.items():
            if attn_module.attn_type == AttentionType.ENCODER_ONLY:
                attn_spec: AttentionSpec = EncoderOnlyAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.kv_cache_dtype)
                encoder_only_attn_specs[attn_spec].append(layer_name)
                self.runner_only_attn_layers.add(layer_name)
        if len(encoder_only_attn_specs) > 0:
            assert len(
                encoder_only_attn_specs
            ) == 1, "Only support one encoder-only attention spec now"
            spec, layer_names = encoder_only_attn_specs.popitem()
            self.kv_cache_config.kv_cache_groups.append(
                KVCacheGroupSpec(layer_names=layer_names, kv_cache_spec=spec))

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
                    kv_tgt_layer := attn_module.kv_sharing_target_layer_name):
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

    def _to_list(self, sampled_token_ids: torch.Tensor) -> list[list[int]]:
        # This is a short term mitigation for issue mentioned in
        # https://github.com/vllm-project/vllm/issues/22754.
        # `tolist` would trigger a cuda wise stream sync, which
        # would block other copy ops from other cuda streams.
        # A cuda event sync would avoid such a situation. Since
        # this is in the critical path of every single model
        # forward loop, this has caused perf issue for a disagg
        # setup.
        pinned = self.sampled_token_ids_pinned_cpu[:sampled_token_ids.shape[0]]
        pinned.copy_(sampled_token_ids, non_blocking=True)
        # self.transfer_event.record()
        # self.transfer_event.synchronize()
        return pinned.tolist()

    def is_prefills(self) -> np.ndarray:
        return (self.input_batch.num_computed_tokens_cpu
                < self.input_batch.num_tokens_no_spec - 1)

    def use_wrapped_compute_logits(self) -> bool:
        return not (self.lora_config is not None or
                    (self.speculative_config is not None and
                     self.speculative_config.method in ("eagle", "eagle3")))


def create_lora_mask(
    input_ids: torch.Tensor,
    lora_ids: list[int],
    lora_index_to_id: list[int],
    max_loras: int,
    max_lora_rank: int,
    lora_dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    lora_mask = torch.zeros(input_ids.shape[0] * input_ids.shape[1],
                            max_loras * max_lora_rank,
                            dtype=lora_dtype,
                            device=device)
    ones = torch.ones(input_ids.shape[1],
                      max_lora_rank,
                      dtype=lora_dtype,
                      device=device)

    for i in range(len(lora_ids)):
        if lora_ids[i] == 0:
            continue

        lora_index = lora_index_to_id.index(lora_ids[i])
        start_row = i * input_ids.shape[1]
        start_col = lora_index * max_lora_rank
        lora_mask[start_row:start_row + input_ids.shape[1],
                  start_col:start_col + max_lora_rank] = ones

    return lora_mask


def create_sampler_indices_padded(
    lora_ids: list[int],
    lora_index_to_id: list[int],
    max_num_seqs: int,
    is_prefill: bool,
    max_loras: int,
    device: torch.device,
) -> torch.Tensor:
    if is_prefill:
        assert len(lora_ids
                   ) == 1, "Only single LoRA is supported during prefill phase"

    prompt_mapping: list[int] = [
        lora_index_to_id.index(lora_ids[i])
        if i < len(lora_ids) and lora_ids[i] > 0 else -1
        for i in range(len(lora_ids) if is_prefill else max_num_seqs)
    ]
    sampler_indices_padded = torch.tensor(prompt_mapping,
                                          dtype=torch.long,
                                          device=device)
    sampler_indices_padded = torch.where(sampler_indices_padded == -1,
                                         max_loras, sampler_indices_padded)
    sampler_indices_padded = torch.arange(
        0, len(sampler_indices_padded), dtype=torch.long,
        device=device) + (sampler_indices_padded * len(sampler_indices_padded))

    return sampler_indices_padded