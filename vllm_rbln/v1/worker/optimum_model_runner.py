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
import logging
import time
from typing import TYPE_CHECKING, NamedTuple, Optional, Union, cast

import numpy as np
import torch
import torch.distributed
import torch.nn as nn
from vllm.config import VllmConfig, set_current_vllm_config
from vllm.distributed.parallel_state import get_pp_group
from vllm.model_executor.models.interfaces import (SupportsMultiModal,
                                                   supports_transcription)
from vllm.model_executor.models.interfaces_base import (
    VllmModelForPooling, is_pooling_model, is_text_generation_model)
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.multimodal.inputs import (BatchedTensorInputs, MultiModalKwargs,
                                    MultiModalKwargsItem)
from vllm.multimodal.utils import group_mm_kwargs_by_modality
from vllm.sampling_params import SamplingType
from vllm.sequence import IntermediateTensors
from vllm.tasks import GenerationTask, PoolingTask, SupportedTask
from vllm.utils.import_utils import LazyLoader
from vllm.utils.jsontree import json_map_leaves
from vllm.utils.torch_utils import kv_cache_dtype_str_to_dtype
# from vllm.utils import LazyLoader, is_pin_memory_available)
from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
from vllm.v1.kv_cache_interface import (FullAttentionSpec, KVCacheConfig,
                                        KVCacheSpec)
# yapf: enable
from vllm.v1.outputs import (EMPTY_MODEL_RUNNER_OUTPUT, AsyncModelRunnerOutput,
                             LogprobsLists, LogprobsTensors, ModelRunnerOutput,
                             PoolerOutput, SamplerOutput)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.sampler import Sampler
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.structured_output.utils import apply_grammar_bitmask
from vllm.v1.utils import record_function_or_nullcontext
from vllm.v1.worker.gpu_input_batch import CachedRequestState
from vllm.v1.worker.lora_model_runner_mixin import LoRAModelRunnerMixin

import vllm_rbln.rbln_envs as envs
from vllm_rbln.logger import init_logger
from vllm_rbln.model_executor.model_loader.rbln_model_loader import (
    get_optimum_model)
from vllm_rbln.model_executor.models.optimum import ModelInputForRBLN
from vllm_rbln.utils.optimum.common import select_bucket_size
from vllm_rbln.utils.optimum.registry import get_rbln_model_info
from vllm_rbln.v1.core.optimum_scheduler import RBLNSchedulerOutput
from vllm_rbln.v1.sample import WARM_UP_CONFIGS, RBLNSampler
from vllm_rbln.v1.worker.optimum_input_batch import RBLNInputBatch
from vllm_rbln.worker.metrics import PerformanceTracker

if TYPE_CHECKING:
    import xgrammar as xgr
    from vllm.v1.core.sched.output import GrammarOutput, SchedulerOutput
else:
    xgr = LazyLoader("xgr", globals(), "xgrammar")

logger = init_logger(__name__)


class ExecuteModelState(NamedTuple):
    """Ephemeral cached state transferred between execute_model() and
    sample_tokens(), after execute_model() returns None."""

    scheduler_output: "SchedulerOutput"
    logits: torch.Tensor
    hidden_states: torch.Tensor
    sample_hidden_states: torch.Tensor
    is_prompt: bool


class RBLNOptimumModelRunner(LoRAModelRunnerMixin):
    input_batch: RBLNInputBatch

    def __init__(self, vllm_config: VllmConfig, device: torch.device):
        # FIXME: For RBLN support Enc-only model which based on enc-dec config.
        # When using an encoder-only model (such as T5EncoderModel)
        # with a config designed for enc-dec architectures,
        # it’s important to set the is_encoder_decoder flag to False.
        # This prevents the scheduler from applying text generation settings.
        _, model_cls_name = get_rbln_model_info(vllm_config.model_config)
        if model_cls_name in ["RBLNQwen3ForCausalLM"
                              ] and vllm_config.model_config.task == "embed":
            # NOTE The architecture of Qwen3-Embedding model in huggingface
            # is `Qwen3ForCausalLM`. But it have to be mapped to `Qwen3Model`
            # for optimum-rbln.
            vllm_config.model_config.hf_config.__dict__["architectures"] = [
                "Qwen3Model"
            ]

        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        # self.compilation_config = vllm_config.compilation_config
        self.lora_config = vllm_config.lora_config
        # self.load_config = vllm_config.load_config
        self.parallel_config = vllm_config.parallel_config
        self.scheduler_config = vllm_config.scheduler_config
        # self.speculative_config = vllm_config.speculative_config
        # self.prompt_adapter_config = vllm_config.prompt_adapter_config
        # self.observability_config = vllm_config.observability_config

        from vllm.model_executor.models.utils import set_cpu_offload_max_bytes
        set_cpu_offload_max_bytes(0)

        model_config = self.model_config
        cache_config = self.cache_config
        self.device = device
        self.pin_memory = False
        self.dtype = self.model_config.dtype
        self.kv_cache_dtype = kv_cache_dtype_str_to_dtype(
            cache_config.cache_dtype, self.model_config)
        # if cache_config.cache_dtype == "auto":
        #     self.kv_cache_dtype = self.dtype
        # else:
        #     self.kv_cache_dtype = STR_DTYPE_TO_TORCH_DTYPE[
        #         self.cache_config.cache_dtype]
        self.is_pooling_model = (model_config.runner_type == 'pooling')
        # When `is_multimodal_raw_input_only_model` is True, it means that
        # it extract multimodal raw inputs only and deliver as raw inputs to
        # the model.
        self.is_multimodal_raw_input_only_model = True

        self.vocab_size = self.model_config.get_vocab_size()
        self.max_model_len = self.model_config.max_model_len
        self.max_num_tokens = self.scheduler_config.max_num_batched_tokens
        self.max_num_reqs = self.scheduler_config.max_num_seqs
        self.inputs_embeds_size = self.model_config.get_inputs_embeds_size()

        # # Multi-modal data support
        # # NOTE There is a bug in vLLM MM registry internally in v0.10.X.
        # # As a workaround, VLLM_WORKER_MULTIPROC_METHOD should be set "spawn"
        # # in case of multi-modal encoder-decoder models.
        self.mm_registry = MULTIMODAL_REGISTRY
        # self.uses_mrope = model_config.uses_mrope
        self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
            model_config)

        # Sampler
        self.use_rbln_sampler = envs.VLLM_RBLN_SAMPLER
        if self.use_rbln_sampler:
            logger.info("Using RBLN sampler: %s", self.use_rbln_sampler)
            self.pooled_tensors: dict[int, torch.Tensor] = {}
            sampler = RBLNSampler(
                logprobs_mode=self.model_config.logprobs_mode,
                seed=self.vllm_config.model_config.seed,
            )
        else:
            logger.info("Using default vLLM sampler.")
            sampler = Sampler(logprobs_mode=self.model_config.logprobs_mode)
        self.sampler = sampler

        # Attention groups are not supported.
        self.attn_groups = []  # type: ignore

        # # Request states.
        self.requests: dict[str, CachedRequestState] = {}
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
        self.input_batch = RBLNInputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.get_vocab_size(),
            block_sizes=[cache_config.block_size],
            kernel_block_sizes=[cache_config.block_size
                                ],  # FIXME: why do we need this?
            is_spec_decode=False,  # No spec decode in optimum model runner
            logitsprocs=build_logitsprocs(
                self.vllm_config, self.device, self.pin_memory,
                self.is_pooling_model,
                self.vllm_config.model_config.logits_processors),
            is_pooling_model=self.is_pooling_model,
            use_rbln_sampler=self.use_rbln_sampler,
        )

        # FIXME enable async scheduling for optimum model runner
        self.use_async_scheduling = self.scheduler_config.async_scheduling
        self.enable_prefix_caching = cache_config.enable_prefix_caching
        self.seq_lens = np.zeros(self.max_num_reqs, dtype=np.int32)

        # self.uniform_decode_query_len = 1

        # Attention layers that are only in the KVCacheConfig of the runner
        # (e.g., KV sharing, encoder-only attention), but not in the
        # KVCacheConfig of the scheduler.
        # self.runner_only_attn_layers: set[str] = set()
        # D2H copy for sampled token ids (output)
        # unintentionally block all other copy operations
        # To prevent this, we use a pinned buffer for sampled token ids.
        # https://github.com/vllm-project/vllm/issues/22754
        self.sampled_token_ids_pinned_cpu = torch.empty(
            (self.max_model_len, 1),
            dtype=torch.int64,
            device="cpu",
            pin_memory=self.pin_memory)

        if envs.VLLM_RBLN_METRICS:
            self.performance_tracker = PerformanceTracker()
            self.performance_tracker.register_cleanup()

        # Ephemeral state transferred
        # between execute_model() and sample_tokens().
        self.execute_model_state: ExecuteModelState | None = None

        # FIXME async_scheduling?

    def load_model(self) -> None:
        with set_current_vllm_config(self.vllm_config, check_compile=False):
            self.model = get_optimum_model(vllm_config=self.vllm_config)
        self.use_optimum_lora = getattr(self.model.model.rbln_config,
                                        "use_lora", None)
        if self.lora_config and not self.use_optimum_lora:
            raise RuntimeError(
                "The compiled model is for LoRA."
                "Please compile the model with `rbln_lora_config`")
        if not self.lora_config and self.use_optimum_lora:
            raise RuntimeError("The model is compiled for LoRA."
                               "Please set `enable_lora=True` in vLLM.")

        if self.use_optimum_lora:
            self.valid_lora_ids = list(
                range(len(self.model.rbln_model_config.lora_config.adapters)))
        # NOTE(eunji.lee):
        # Set bucket sizes and pooled tensors for RBLN sampler
        # if use_multiple_decoder is True, use decoder_batch_sizes
        # otherwise, use max_num_seqs
        if self.use_rbln_sampler:
            self.prepare_rbln_sampler()

    def get_model(self) -> nn.Module:
        return self.model

    @torch.inference_mode()
    def execute_model(
        self,
        scheduler_output: "SchedulerOutput",
        intermediate_tensors: Optional[IntermediateTensors] = None,
    ) -> Union[ModelRunnerOutput, IntermediateTensors]:
        if self.execute_model_state is not None:
            raise RuntimeError("State error: sample_tokens() must be called "
                               "after execute_model() returns None.")
        num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        with record_function_or_nullcontext("rbln_model_runner: preprocess"):
            # with self.synchronize_input_prep():
            self._update_states(scheduler_output)
            if not num_scheduled_tokens:
                # FIXME If local block table exists in the model,
                # clear the local block table.
                # Because in the case of LLM (not AsyncLLMEngine),
                # `finished_request_ids` is provided separately
                # from new requests.
                # It is a temporary solution.
                if getattr(self.model, "attention_manager", None):
                    self.model.attention_manager.clear()
                # Return empty ModelRunnerOutput if there's no work to do.
                return EMPTY_MODEL_RUNNER_OUTPUT
            # Prepare the decoder inputs.
            model_input, num_scheduled_tokens_np = self._prepare_inputs(
                scheduler_output)

        with record_function_or_nullcontext("rbln_model_runner: forward"):
            start_time = time.perf_counter()
            # FIXME model_input must be modified to be padded
            hidden_states = self.model(model_input)
            sample_hidden_states = hidden_states.clone()
            end_time = time.perf_counter()
            if envs.VLLM_RBLN_METRICS:
                # Record performance metrics
                execution_time = end_time - start_time
                if model_input.is_prompt:
                    self.performance_tracker.record_prefill(
                        execution_time, num_scheduled_tokens)
                else:
                    self.performance_tracker.record_decode(
                        execution_time, num_scheduled_tokens)

        with record_function_or_nullcontext("rbln_model_runner: postprocess"):
            if self.is_pooling_model:
                return self._pool(hidden_states, num_scheduled_tokens,
                                  num_scheduled_tokens_np)
            # [batch_size, 1, vocab_size] -> [batch_size, vocab_size]
            hidden_states = hidden_states.squeeze(1)
            logits = self.model.compute_logits(hidden_states, None)
        self.execute_model_state = ExecuteModelState(
            scheduler_output=scheduler_output,
            logits=logits,
            hidden_states=hidden_states,
            sample_hidden_states=sample_hidden_states,
            is_prompt=model_input.is_prompt,
        )
        return None

    def mask_block_table(
        self,
        block_ids: torch.Tensor,
        num_blocks: int,
        *,
        pad_value: int = -1,
    ) -> torch.Tensor:
        """Mask (pad) unused block slots in-place.

        Sets entries beyond `num_blocks` to `pad_value`.
        Use `pad_value=0` for v1 (dummy block id 0), or pass your own padding.
        """
        if num_blocks < 0:
            raise ValueError("num_blocks must be >= 0")

        if block_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("block_ids must be int32 or int64")

        # In V1, block ID 0 is reserved as a dummy "null_block",
        # so valid blocks start from 1.
        # The compiler, however, expects valid blocks to start from 0.
        block_ids = block_ids - 1
        max_blocks = block_ids.size(-1)
        k = max(0, min(num_blocks, max_blocks))  # clamp to [0, max_blocks]
        if k < max_blocks:
            block_ids.narrow(-1, k, max_blocks - k).fill_(pad_value)

        return block_ids

    def _prepare_inputs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[ModelInputForRBLN, np.ndarray]:
        """
        :return: ModelInputForRBLN[
            input_tokens: Token IDs,
            input_positions: Position IDs,
            sampling_metadata, pooling_metadata: It is `None` in V1,
            multi_modal_kwargs: Batched multi-modal data,
            block_tables: [num_reqs, num_blocks_per_req] shaped tensor,
            running_requests_ids: RUNNING request IDs,
            finished_requests_ids: FINISHED request IDs in between
                the previous and the current steps,
            is_prompt: It is used only in V1
        ]
        """
        total_num_scheduled_tokens = scheduler_output.total_num_scheduled_tokens
        assert total_num_scheduled_tokens > 0
        num_reqs = self.input_batch.num_reqs
        assert num_reqs > 0

        # Get the number of scheduled tokens for each request.
        req_ids = self.input_batch.req_ids
        tokens = [scheduler_output.num_scheduled_tokens[i] for i in req_ids]
        num_scheduled_tokens = np.array(tokens, dtype=np.int32)
        num_prefill_reqs = len(scheduler_output.scheduled_new_reqs)
        num_decode_reqs = scheduler_output.scheduled_cached_reqs.num_reqs
        finished_requests_ids = scheduler_output.finished_req_ids
        is_prefill = False

        if num_prefill_reqs > 1 or (num_prefill_reqs >= 1
                                    and num_decode_reqs > 0):
            raise RuntimeError(
                "Prefill stage request cannot processed with other requests.")

        if num_prefill_reqs > 0 or \
            (num_decode_reqs == 1 and \
            scheduler_output.scheduled_cached_reqs.resumed_from_preemption[0]):
            is_prefill = True

        if is_prefill:
            input_ids, positions, block_tables, cached_block_tables, \
            cached_lengths, multi_modal_kwargs, running_request_ids \
                = self._prepare_prefill(scheduler_output)
        else:
            cached_block_tables = []
            cached_lengths = []
            input_ids, positions, block_tables, running_request_ids \
                = self._prepare_decode(scheduler_output)

        # Hot-Swap lora model
        if self.lora_config:
            self.set_active_loras(self.input_batch, is_prefill)

        # Set seq_lens
        self.seq_lens[:num_reqs] = (
            self.input_batch.num_computed_tokens_cpu[:num_reqs] +
            num_scheduled_tokens)

        # TODO interemediate_tensor should be set
        model_input = ModelInputForRBLN(
            input_tokens=input_ids,
            input_positions=positions,
            multi_modal_kwargs=multi_modal_kwargs if is_prefill else None,
            block_tables=block_tables,
            running_requests_ids=running_request_ids,
            finished_requests_ids=list(finished_requests_ids),
            cached_block_tables=cached_block_tables,
            cached_lengths=cached_lengths,
            is_prompt=is_prefill,
            dummy_block=scheduler_output.dummy_block)
        return model_input, num_scheduled_tokens

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """
        Generates the KVCacheSpec by parsing the kv cache format from each
        Attention module in the static forward context.
        Returns:
            KVCacheSpec: A dictionary mapping layer names to their KV cache
            format. Layers that do not need KV cache are not included.
        """

        # TODO It is temporary basic attention setting
        head_size = self.model_config.get_head_size()
        num_layers = self.model_config.get_num_layers(self.parallel_config)
        num_kv_heads = self.model_config.get_num_kv_heads(self.parallel_config)

        block_size = self.vllm_config.cache_config.block_size
        kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_idx in range(num_layers):
            kv_cache_spec[str(layer_idx)] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=self.kv_cache_dtype,
            )
        return kv_cache_spec

    def _extract_mm_kwargs(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> BatchedTensorInputs:
        if not scheduler_output or not self.is_multimodal_raw_input_only_model:
            return {}

        mm_kwargs = list[MultiModalKwargsItem]()
        for req in scheduler_output.scheduled_new_reqs:
            for feature in req.mm_features:
                if feature.data is not None:
                    mm_kwargs.append(feature.data)

        # Input all modalities at once
        model = cast(SupportsMultiModal, self.model)
        mm_kwargs_combined: BatchedTensorInputs = {}
        for _, _, mm_kwargs_group in group_mm_kwargs_by_modality(
                mm_kwargs,
                device=self.device,
                pin_memory=self.pin_memory,
                merge_by_field_config=model.merge_by_field_config,
        ):
            mm_kwargs_combined.update(mm_kwargs_group)

        return mm_kwargs_combined

    def _prepare_prefill(
        self,
        scheduler_output: "RBLNSchedulerOutput",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[int], list[int],
               Optional[MultiModalKwargs], list[str]]:
        input_tokens: list[list[int]] = []
        input_positions: list[list[int]] = []
        running_request_ids = []
        batched_mm_inputs: Optional[BatchedTensorInputs] = None

        num_blocks_per_req = self.input_batch.block_table.block_tables[
            0].num_blocks_per_row
        block_tables_cpu = self.input_batch.block_table.block_tables[
            0].get_cpu_tensor()
        cached_block_table = []
        cached_length = []

        if len(scheduler_output.scheduled_new_reqs) == 1:
            # New request started
            scheduled = scheduler_output.scheduled_new_reqs[0]
            req_id = scheduled.req_id
            req_index = self.input_batch.req_id_to_index[req_id]
            prompt_tokens = np.array(scheduled.prompt_token_ids)
            block_ids = scheduled.block_ids[0]
        elif scheduler_output.scheduled_cached_reqs.num_reqs == 1:
            # Preempted request resumed
            req_id = scheduler_output.scheduled_cached_reqs.req_ids[0]
            req_index = self.input_batch.req_id_to_index[req_id]
            logger.warning("Request %s is resumed.", req_id)
            num_token = int(self.input_batch.num_tokens[req_index])
            prompt_tokens = self.input_batch.token_ids_cpu[
                req_index][:num_token]
            block_ids = scheduler_output.scheduled_cached_reqs.new_block_ids[0]
        else:
            raise RuntimeError(
                "Prefill stage request cannot processed with other requests.")

        seq_len = len(prompt_tokens)
        input_positions = list(range(seq_len))
        num_blocks = num_blocks_per_req[req_index]
        if self.enable_prefix_caching:
            logger.debug(
                "Request %s is now scheduled. Prompt tokens: %s, "
                "Already generated tokens: %s, Allocated block(s): %s", req_id,
                len(self.requests[req_id].prompt_token_ids),
                len(self.requests[req_id].output_token_ids), block_ids)
            block_table = scheduler_output.block_table_dict[req_id]
            cached_block_table = scheduler_output.cached_block_table
            cached_length = scheduler_output.cached_length
            total_cached_length = sum(cached_length)
            if total_cached_length > 0:
                prompt_tokens = prompt_tokens[total_cached_length:]
                input_positions = input_positions[total_cached_length:]
                assert len(prompt_tokens) > 0, (
                    "The prompt tokens is empty after removing the "
                    "cached tokens.")
        else:
            block_table = block_tables_cpu[req_index]
            block_table = self.mask_block_table(block_table, num_blocks)
            logger.debug(
                "Request %s is now scheduled. Prompt tokens: %s, "
                "Already generated tokens: %s, Allocated block(s): %s", req_id,
                len(self.requests[req_id].prompt_token_ids),
                len(self.requests[req_id].output_token_ids),
                block_table.tolist())

        running_request_ids.append(req_id)

        if self.supports_mm_inputs:
            batched_mm_inputs = self._extract_mm_kwargs(scheduler_output)

        input_tokens = torch.tensor(prompt_tokens).unsqueeze(0)
        input_positions = torch.tensor(input_positions).unsqueeze(0)
        block_table = block_table.unsqueeze(0)
        # NOTE The cached_block_table is not unsqueezed for convenience.
        # It is used only for prefill
        return input_tokens, input_positions, block_table, cached_block_table, \
        cached_length, batched_mm_inputs, running_request_ids

    def _prepare_decode(
        self,
        scheduler_output: "SchedulerOutput",
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[str]]:
        input_tokens: list[list[int]] = []
        input_positions: list[list[int]] = []
        block_tables_list = []
        running_request_ids = []
        block_tables_cpu = self.input_batch.block_table.block_tables[
            0].get_cpu_tensor()
        num_blocks_per_req = self.input_batch.block_table.block_tables[
            0].num_blocks_per_row

        req_ids = self.input_batch.req_ids
        for req_id in req_ids:
            req_index = self.input_batch.req_id_to_index[req_id]
            input_position = int(
                self.input_batch.num_computed_tokens_cpu[req_index])
            input_tokens.append(
                [self.input_batch.token_ids_cpu[req_index][input_position]])
            input_positions.append([input_position])
            num_blocks = num_blocks_per_req[req_index]
            if self.enable_prefix_caching:
                block_tables_list.append(
                    scheduler_output.block_table_dict[req_id])
            else:
                block_table = block_tables_cpu[req_index]
                block_table = self.mask_block_table(block_table, num_blocks)
                block_tables_list.append(block_table)
            running_request_ids.append(req_id)

        input_tokens = torch.tensor(input_tokens)
        input_positions = torch.tensor(input_positions)
        block_tables = torch.stack(block_tables_list)

        return input_tokens, input_positions, block_tables, running_request_ids

    def _update_states(self, scheduler_output: "RBLNSchedulerOutput") -> None:
        """Update the cached states and the persistent batch with the scheduler
        output.

        The updated states are used by the `_prepare_inputs` function to create
        the input NPU tensors for the model.

        The SamplingMetadata is updated and copied to the NPU if there is a
        new/resumed/paused/finished request in the batch.
        """
        # Remove finished requests from the cached states.
        for req_id in scheduler_output.finished_req_ids:
            if logger.isEnabledFor(logging.DEBUG) and req_id in self.requests:
                if self.enable_prefix_caching:
                    block_ids = self.requests[req_id].block_ids[0]
                else:
                    block_ids = [
                        block_id - 1
                        for block_id in self.requests[req_id].block_ids[0]
                    ]
                logger.debug(
                    "Request %s is finished. Prompt tokens: %s, "
                    "Generated tokens: %s, Freed block(s): %s", req_id,
                    len(self.requests[req_id].prompt_token_ids),
                    len(self.requests[req_id].output_token_ids), block_ids)

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

        # Remove the unscheduled requests from the persistent batch.
        # NOTE(woosuk): The unscheduled requests are either preempted requests
        # or running requests that are not scheduled in this step. We remove
        # them from the persistent batch but keep their cached states since
        # they will be scheduled again sometime in the future.
        scheduled_req_ids = scheduler_output.num_scheduled_tokens.keys()
        cached_req_ids = self.input_batch.req_id_to_index.keys()
        resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
        # NOTE(zhuohan): cached_req_ids and resumed_req_ids are usually disjoint, # noqa: E501
        # so `(scheduled_req_ids - resumed_req_ids) == scheduled_req_ids` holds
        # apart from the forced-preemption case in reset_prefix_cache. And in
        # that case we include the resumed_req_ids in the unscheduled set so
        # that they get cleared from the persistent batch before being re-scheduled # noqa: E501
        # in the normal resumed request path.
        unscheduled_req_ids = cached_req_ids - (scheduled_req_ids -
                                                resumed_req_ids)
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

            if (sampling_params and sampling_params.sampling_type
                    == SamplingType.RANDOM_SEED):
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

            # Check lora_int_id is valid
            if new_req_data.lora_request and self.use_optimum_lora:
                lora_int_id = new_req_data.lora_request.lora_int_id
                if lora_int_id >= len(self.valid_lora_ids):
                    raise RuntimeError(
                        f"Invalid `lora_int_id`: {lora_int_id}. "
                        f"Valid `lora_int_ids` are {self.valid_lora_ids} "
                        "(must be consistent with the compiled model).")

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
                    if sampling_params.prompt_logprobs == -1 else
                    sampling_params.prompt_logprobs)

            # Only relevant for models using M-RoPE (e.g, Qwen2-VL)
            # if self.uses_mrope:
            #     self._init_mrope_positions(req_state)

            # Only relevant for models using XD-RoPE (e.g, HunYuan-VL)
            # if self.uses_xdrope_dim > 0:
            #     self._init_xdrope_positions(req_state)

            reqs_to_add.append(req_state)

        # Update the states of the running/resumed requests.
        is_last_rank = get_pp_group().is_last_rank
        req_data = scheduler_output.scheduled_cached_reqs

        # Wait until valid_sampled_tokens_count is copied to cpu,
        # then use it to update actual num_computed_tokens of each request.
        # valid_sampled_token_count = self._get_valid_sampled_token_count()

        for i, req_id in enumerate(req_data.req_ids):
            req_state = self.requests[req_id]
            num_computed_tokens = req_data.num_computed_tokens[i]
            new_block_ids = req_data.new_block_ids[i]
            resumed_from_preemption = req_id in req_data.resumed_req_ids
            req_index = self.input_batch.req_id_to_index.get(req_id)

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
                                                  new_block_ids,
                                                  strict=False):
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
            self.input_batch.num_computed_tokens_cpu[
                req_index] = num_computed_tokens
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

        # Add the new or resumed requests to the persistent batch.
        # The smaller empty indices are filled first.
        for request in reqs_to_add:
            self.input_batch.add_request(request)
        # Condense the batched states if there are gaps left by removed requests
        self.input_batch.condense()

        # Refresh batch metadata with any pending updates.
        use_padding = self.use_rbln_sampler and self.input_batch.num_reqs > 1
        if use_padding:
            # To pad sampling metadata for RBLN sampler
            self.bucket_size = select_bucket_size(self.input_batch.num_reqs,
                                                  self.bucket_sizes)
            self.input_batch.refresh_metadata_rbln(self.bucket_size)
        else:
            self.input_batch.refresh_metadata()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        pass

    def dummy_sampler_run(self):
        if not self.use_rbln_sampler:
            logger.info("Skip dummy sampler run since "
                        "it is only used in RBLN_SAMPLER=1")
            return

        def set_sampling_tensors(input_batch, **params):
            input_batch.temperature_cpu_tensor.fill_(params["temperature"])
            input_batch.temperature.fill_(params["temperature"])

            optional_keys = [
                ("top_p", input_batch.top_p_cpu_tensor, input_batch.top_p),
                ("top_k", input_batch.top_k_cpu_tensor, input_batch.top_k),
                ("frequency_penalties",
                 input_batch.frequency_penalties_cpu_tensor,
                 input_batch.frequency_penalties),
                ("presence_penalties",
                 input_batch.presence_penalties_cpu_tensor,
                 input_batch.presence_penalties),
                ("repetition_penalties",
                 input_batch.repetition_penalties_cpu_tensor,
                 input_batch.repetition_penalties),
            ]

            for key, cpu_tensor, dev_tensor in optional_keys:
                val = params.get(key)
                if val is not None:
                    cpu_tensor.fill_(val)
                    dev_tensor.fill_(val)

        def populate_reqs(input_batch, base_config, batch_size):
            for i in range(batch_size):
                req_id = f"dummy_request_{i}"
                input_batch._req_ids.append(req_id)
                input_batch.req_id_to_index[req_id] = i

                if base_config["all_greedy"]:
                    input_batch.greedy_reqs.add(req_id)
                elif base_config["all_random"]:
                    input_batch.random_reqs.add(req_id)

                for attr, req_set in [
                    ("top_p", input_batch.top_p_reqs),
                    ("top_k", input_batch.top_k_reqs),
                    ("frequency_penalties",
                     input_batch.frequency_penalties_reqs),
                    ("repetition_penalties",
                     input_batch.repetition_penalties_reqs),
                    ("presence_penalties",
                     input_batch.presence_penalties_reqs),
                ]:
                    if base_config.get(attr) is not None:
                        req_set.add(req_id)

        def clear_reqs(input_batch):
            input_batch._req_ids.clear()
            input_batch.req_id_to_index.clear()
            input_batch.greedy_reqs.clear()
            input_batch.random_reqs.clear()
            input_batch.top_p_reqs.clear()
            input_batch.top_k_reqs.clear()
            input_batch.frequency_penalties_reqs.clear()
            input_batch.repetition_penalties_reqs.clear()
            input_batch.presence_penalties_reqs.clear()

        def dummy_run_batches(base_config):
            for batch_size in self.bucket_sizes:
                input_batch = self.input_batch
                populate_reqs(input_batch, base_config, batch_size)

                metadata = input_batch._make_sampling_metadata()
                metadata.no_penalties = base_config["no_penalties"]
                metadata.all_greedy = base_config["all_greedy"]
                metadata.all_random = base_config["all_random"]

                if (not metadata.no_penalties
                        and metadata.prompt_token_ids is None):
                    metadata.prompt_token_ids = torch.zeros((batch_size, 1),
                                                            dtype=torch.long,
                                                            device="cpu")

                logger.debug(
                    "Running dummy compile with batch_size=%d, vocab_size=%d",
                    batch_size, input_batch.vocab_size)
                logger.debug("Sampling metadata: %s", metadata)

                with torch.inference_mode():
                    empty_logits = torch.empty(batch_size,
                                               input_batch.vocab_size,
                                               dtype=torch.float32)
                    _ = self.sampler(logits=empty_logits,
                                     sampling_metadata=metadata)

                clear_reqs(input_batch)

        for config in WARM_UP_CONFIGS:
            logger.info("Running dummy sampler config: %s", config["name"])

            set_sampling_tensors(
                self.input_batch,
                temperature=config["temperature"],
                top_p=config.get("top_p"),
                top_k=config.get("top_k"),
                frequency_penalties=config.get("frequency_penalties"),
                repetition_penalties=config.get("repetition_penalties"),
                presence_penalties=config.get("presence_penalties"),
            )

            dummy_run_batches(config)

    def set_active_loras(self, input_batch: RBLNInputBatch,
                         is_prefill: bool) -> None:
        num_reqs = self.input_batch.num_reqs
        req_lora_mapping_list = input_batch.request_lora_mapping[:
                                                                 num_reqs].tolist(
                                                                 )
        # Padding
        if not is_prefill and num_reqs < self.max_num_reqs:
            req_lora_mapping_list += [0] * (self.max_num_reqs - num_reqs)
        self.model.model.set_lora_int_ids(req_lora_mapping_list)

    def _pool(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        num_scheduled_tokens_np: np.ndarray,
    ) -> ModelRunnerOutput:
        num_reqs = self.input_batch.num_reqs
        assert num_reqs == \
            len(self.input_batch.pooling_params), \
        "Either all or none of the requests in" \
        " a batch must be pooling request"

        hidden_states = hidden_states[:num_scheduled_tokens]
        seq_lens_cpu = self.seq_lens[:num_reqs]

        pooling_metadata = self.input_batch.get_pooling_metadata()
        pooling_metadata.build_pooling_cursor(num_scheduled_tokens_np.tolist(),
                                              seq_lens_cpu,
                                              device=hidden_states.device)

        model = cast(VllmModelForPooling, self.model)
        raw_pooler_output: PoolerOutput = model.pooler(
            hidden_states=hidden_states,
            pooling_metadata=pooling_metadata,
        )
        raw_pooler_output = json_map_leaves(
            lambda x: x.to("cpu", non_blocking=True) if x is not None else x,
            raw_pooler_output,
        )

        pooler_output: list[torch.Tensor | None] = []
        for raw_output, seq_len, prompt_len in zip(
                raw_pooler_output,
                seq_lens_cpu,
                pooling_metadata.prompt_lens,
                strict=False):
            output = raw_output if seq_len == prompt_len else None
            pooler_output.append(output)

        return ModelRunnerOutput(
            req_ids=self.input_batch.req_ids,
            req_id_to_index=self.input_batch.req_id_to_index,
            sampled_token_ids=[],
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=pooler_output,
        )

    def _update_states_after_model_execute(
            self, output_token_ids: torch.Tensor) -> None:
        pass
        # This is used for MTP/EAGLE for hybrid models originally.
        # But it is not used in RBLN.

    def get_supported_pooling_tasks(self) -> list[PoolingTask]:
        model = self.get_model()
        if not is_pooling_model(model):
            return []

        supported_tasks = list(model.pooler.get_supported_tasks())

        if "score" in supported_tasks:
            num_labels = getattr(self.model_config.hf_config, "num_labels", 0)
            if num_labels != 1:
                supported_tasks.remove("score")
                logger.debug_once(
                    "Score API is only enabled for num_labels == 1.")

        return supported_tasks

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

    def get_supported_tasks(self) -> tuple[SupportedTask, ...]:
        tasks = list[SupportedTask]()

        if self.model_config.runner_type == "generate":
            tasks.extend(self.get_supported_generation_tasks())
        if self.model_config.runner_type == "pooling":
            tasks.extend(self.get_supported_pooling_tasks())

        return tuple(tasks)

    def _bookkeeping_sync(
        self,
        scheduler_output: "SchedulerOutput",
        sampler_output: SamplerOutput,
        logits: torch.Tensor | None,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: int,
        use_padding: bool,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> tuple[
            dict[str, int],
            LogprobsLists | None,
            list[list[int]],
            dict[str, LogprobsTensors | None],
            list[str],
            dict[str, int],
            list[int],
    ]:
        num_nans_in_logits = {}
        if envs.VLLM_COMPUTE_NANS_IN_LOGITS:
            num_nans_in_logits = self._get_nans_in_logits(logits)

        num_reqs = self.input_batch.num_reqs
        # Copy some objects so they don't get modified after returning.
        # This is important when using async scheduling.
        req_ids_output_copy = self.input_batch.req_ids.copy()
        req_id_to_index_output_copy = self.input_batch.req_id_to_index.copy()
        if use_padding:
            # Remove the padding from sampler_output
            num_sampled_tokens, sampled_token_ids, logprobs_tensors = \
                self.postprocess_sampler_output(sampler_output, num_reqs)
        else:
            num_sampled_tokens = sampler_output.sampled_token_ids.shape[0]
            sampled_token_ids = sampler_output.sampled_token_ids
            logprobs_tensors = sampler_output.logprobs_tensors

        invalid_req_indices = []
        cu_num_tokens: list[int] | None = None
        if not self.use_async_scheduling:
            # Get the valid generated tokens.
            max_gen_len = sampled_token_ids.shape[-1]
            assert max_gen_len == 1, \
                "No spec decode tokens. Max generation length must be 1."
            # No spec decode tokens.
            valid_sampled_token_ids = self._to_list(sampled_token_ids)
            # Mask out the sampled tokens that should not be sampled.
            # for i in discard_sampled_tokens_req_indices:
            #     valid_sampled_token_ids[int(i)].clear()
        else:
            valid_sampled_token_ids = []
            # FIXME: we need disacrd_sampled_tokens_req_indices for async scheduling # noqa: E501
            # invalid_req_indices = discard_sampled_tokens_req_indices.tolist()
            invalid_req_indices = []
            invalid_req_indices_set = set(invalid_req_indices)

            # Cache the sampled tokens on the GPU and avoid CPU sync.
            # These will be copied into input_ids in the next step
            # when preparing inputs.
            # With spec decoding, this is done in propose_draft_token_ids().
            if self.input_batch.prev_sampled_token_ids is None:
                assert sampled_token_ids.shape[-1] == 1
                self.input_batch.prev_sampled_token_ids = sampled_token_ids
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
                sampled_ids = [
                    -1
                ] if req_idx not in invalid_req_indices_set else None
            else:
                sampled_ids = valid_sampled_token_ids[req_idx]

            num_sampled_ids: int = len(sampled_ids) if sampled_ids else 0

            if not sampled_ids:
                continue

            start_idx = self.input_batch.num_tokens_no_spec[req_idx]
            end_idx = start_idx + num_sampled_ids
            assert end_idx <= self.max_model_len, (
                "Sampled token IDs exceed the max model length. "
                f"Total number of tokens: {end_idx} > max_model_len: "
                f"{self.max_model_len}")

            self.input_batch.token_ids_cpu[req_idx,
                                           start_idx:end_idx] = sampled_ids
            self.input_batch.is_token_ids[req_idx, start_idx:end_idx] = True
            self.input_batch.num_tokens_no_spec[req_idx] = end_idx
            self.input_batch.num_tokens[req_idx] = end_idx

            req_id = req_ids[req_idx]
            req_state = self.requests[req_id]
            req_state.output_token_ids.extend(sampled_ids)
        # FIXME padding handling
        logprobs_lists = (logprobs_tensors.tolists(cu_num_tokens)
                          if not self.use_async_scheduling
                          and logprobs_tensors is not None else None)

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
            invalid_req_indices,
        )

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

    @staticmethod
    def get_bucket_sizes(max_num_seqs: int) -> list[int]:
        """
        Get bucket sizes for RBLN sampler.
        NOTE:
        We don't pad the logits when num_reqs is 1
        to reduce the overhead of padding and unpadding.
        But bucket_sizes must contain 1 to warm up the sampler.

        NOTE:
        When the number of scheduled requests is only one,
        the input_batch is not refreshed.
        But if selected bucket_size is greater than 1, there is an misalignment
        between the logits and the sampling metadata.
        """
        bucket_sizes = [i for i in [1, 2, 4] if i <= max_num_seqs]
        if max_num_seqs >= 8:
            # Step size 8 for small batch sizes, up to 256(not included)
            bucket_sizes += list(range(8, min(max_num_seqs + 1, 256), 8))
        if max_num_seqs >= 256:
            # Step size 16 for larger batch sizes
            bucket_sizes += list(range(256, max_num_seqs + 1, 16))
        if max_num_seqs not in bucket_sizes:
            bucket_sizes.append(max_num_seqs)
        return bucket_sizes

    def post_process_logprobs_tensors(self, logprobs_tensors: LogprobsTensors,
                                      num_reqs: int) -> LogprobsTensors:
        # NOTE(eunji.lee):
        # This implementation is not efficient but kept for debugging purposes.
        # TODO: Modify this code in the next version when the shape of
        # logprobs_tensors changes.
        dict = {}
        for field_name in logprobs_tensors._fields:
            tensor = getattr(logprobs_tensors, field_name)
            dict[field_name] = tensor[:num_reqs]
        return LogprobsTensors(**dict)

    def prepare_rbln_sampler(self):
        # Set bucket sizes and pooled tensors for RBLN sampler
        # if use_multiple_decoder is True, use decoder_batch_sizes
        # otherwise, use max_num_seqs
        use_multiple_decoder = getattr(self.model.model.rbln_config,
                                       "use_multiple_decoder", False)
        if use_multiple_decoder:
            self.bucket_sizes = self.model.decoder_batch_sizes
        else:
            batch_size = self.vllm_config.scheduler_config.max_num_seqs
            self.bucket_sizes = tuple(self.get_bucket_sizes(batch_size))
        logger.debug("Bucket sizes for RBLN sampler: %s", self.bucket_sizes)
        with torch.inference_mode():
            for bucket_size in self.bucket_sizes:
                self.pooled_tensors[bucket_size] = torch.empty(
                    (bucket_size, self.model_config.get_vocab_size()),
                    dtype=self.model.dtype,
                )
        torch._dynamo.config.recompile_limit = len(
            self.bucket_sizes) * len(WARM_UP_CONFIGS)
        self.sampler = torch.compile(self.sampler,
                                     dynamic=False,
                                     fullgraph=False)

    @torch.inference_mode
    def sample_tokens(
        self, grammar_output: "GrammarOutput | None"
    ) -> ModelRunnerOutput | AsyncModelRunnerOutput | IntermediateTensors:
        # kv_connector_output = self.kv_connector_output
        # self.kv_connector_output = None
        assert self.execute_model_state is not None

        # Unpack ephemeral state.
        (
            scheduler_output,
            logits,
            hidden_states,
            sample_hidden_states,
            is_prompt,
        ) = self.execute_model_state
        use_padding = self.use_rbln_sampler and self.input_batch.num_reqs > 1
        # Clear ephemeral state.
        self.execute_model_state = None

        # Apply structured output bitmasks if present.
        if grammar_output is not None:
            apply_grammar_bitmask(scheduler_output, grammar_output,
                                  self.input_batch, logits)

        with record_function_or_nullcontext("rbln_model_runner: sample"):
            if use_padding:
                num_reqs = self.input_batch.num_reqs
                padded_logits = self.pooled_tensors[self.bucket_size]
                padded_logits[:num_reqs].copy_(logits)
            else:
                padded_logits = logits
            sampler_output = self._sample(padded_logits,
                                          spec_decode_metadata=None)
        self.input_batch.prev_sampled_token_ids = None

        with record_function_or_nullcontext("rbln_model_runner: bookkeep"):
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
                use_padding,
                spec_decode_metadata=None,
            )

        with record_function_or_nullcontext(
                "rbln_model_runner: ModelRunnerOutput"):
            output = ModelRunnerOutput(
                req_ids=req_ids_output_copy,
                req_id_to_index=req_id_to_index_output_copy,
                sampled_token_ids=valid_sampled_token_ids,
                logprobs=logprobs_lists,
                prompt_logprobs_dict=prompt_logprobs_dict,
                pooler_output=[],
                # kv_connector_output=kv_connector_output,
                num_nans_in_logits=num_nans_in_logits,
            )

        if not self.use_async_scheduling:
            return output
        # else: # FIXME
        # return

    def _sample(
        self,
        logits: torch.Tensor | None,
        spec_decode_metadata: SpecDecodeMetadata | None,
    ) -> SamplerOutput:
        # Sample the next token and get logprobs if needed.
        sampling_metadata = self.input_batch.sampling_metadata
        # Update output token ids with tokens sampled in last step
        # if async scheduling and required by current sampling params.
        self.input_batch.update_async_output_token_ids()
        return self.sampler(
            logits=logits,
            sampling_metadata=sampling_metadata,
        )

    def _get_prompt_logprobs_dict(
        self,
        hidden_states: torch.Tensor,
        num_scheduled_tokens: dict[str, int],
    ) -> dict[str, LogprobsTensors | None]:
        num_prompt_logprobs_dict = self.num_prompt_logprobs
        if not num_prompt_logprobs_dict:
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
        # if prompt_logprobs_dict:
        #     self._sync_device()

        return prompt_logprobs_dict

    def postprocess_sampler_output(
            self, sampler_output: SamplerOutput,
            num_reqs: int) -> tuple[int, torch.Tensor, LogprobsTensors]:
        dict = {}
        num_spec_decode_token = 1
        num_sampled_tokens = num_reqs
        logprobs_tensors = None
        sampled_token_ids = sampler_output.sampled_token_ids[:num_reqs]

        if sampler_output.logprobs_tensors is not None:
            for field_name in sampler_output.logprobs_tensors._fields:
                tensor = getattr(sampler_output.logprobs_tensors, field_name)
                dict[field_name] = tensor[:num_reqs * num_spec_decode_token]
            logprobs_tensors = LogprobsTensors(**dict)

        return num_sampled_tokens, sampled_token_ids, logprobs_tensors
