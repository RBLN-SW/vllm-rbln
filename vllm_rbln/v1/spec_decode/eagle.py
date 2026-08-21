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
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.model_executor.layers.fused_moe.runner.moe_runner import MoERunner
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

import vllm_rbln.envs as envs
from vllm_rbln.compilation import (
    build_process_group_dict,
    compile,
)
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.logger import init_logger
from vllm_rbln.platform import USE_DEVICE_TENSOR
from vllm_rbln.v1.attention.kv_cache_bindings import (
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)
from vllm_rbln.v1.spec_decode.utils import (
    eagle_prepare_inputs_padded,
    eagle_prepare_next_token_padded,
)
from vllm_rbln.v1.worker.dp_utils import determine_draft_batch_execution_and_padding
from vllm_rbln.v1.worker.input_stager import InputLayout, InputStager

if TYPE_CHECKING:
    from vllm_rbln.v1.worker.rbln_model_runner import RBLNModelRunner

logger = init_logger(__name__)


class RBLNEagleProposer(EagleProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "RBLNModelRunner",
    ):
        super().__init__(vllm_config, device, runner)

        if self.supports_mm_inputs:
            raise NotImplementedError
        if self.needs_extra_input_slots:
            raise NotImplementedError(
                "vllm-rbln does not support EAGLE extra input slots required for "
                "parallel drafting or draft-model speculative decoding yet."
            )
        draft_sample_method = self.speculative_config.draft_sample_method
        if draft_sample_method != "greedy":
            raise NotImplementedError(
                f"draft_sample_method={draft_sample_method!r} is not implemented yet."
            )

        self.runner = runner
        self.arange_cpu = torch.arange(self.arange.shape[0], dtype=torch.int32)
        self.input_stager = InputStager(device)
        # Populated from the draft model in `load_model`. None means the draft
        # head shares the target vocabulary, so `propose()` maps no ids.
        self.draft_id_to_target_id: torch.Tensor | None = None
        # Whether a draft forward joins the DP all-gather; see `load_model`.
        self.draft_has_moe = False

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # NOTE(RBLN): combine_hidden_states in eagle3 is fused
        # into the target model graph.
        assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens = target_token_ids.shape[0]

        assert self.runner is not None
        is_prefill = self.runner.is_prefill

        # Build attention metadata
        num_reqs = self.runner.input_batch.num_reqs
        batch_desc, num_tokens_across_dp = determine_draft_batch_execution_and_padding(
            cfg=self.runner.shape_config,
            status=self.runner.dp_status,
            dp_rank=self.dp_rank,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            is_prefill=is_prefill,
        )
        num_reqs_padded = batch_desc.num_reqs_padded
        num_padded_tokens = batch_desc.num_tokens_padded
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=target_positions,
                is_prefill=is_prefill,
                batch_pad=num_reqs_padded,
            )
            attach_kv_cache_bindings(
                attn_metadata,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        input_ids, positions, hidden_states, token_indices_to_sample_padded = (
            self._preprocess(
                num_reqs,
                num_reqs_padded,
                num_tokens,
                target_positions,
                target_hidden_states,
                is_prefill=is_prefill,
                token_indices_to_sample=token_indices_to_sample,
                target_token_ids=target_token_ids,
                next_token_ids=next_token_ids,
                cad=common_attn_metadata,
            )
        )
        inputs_embeds = None

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
            **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
        ):
            hidden_states, draft_ids = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                token_indices_to_sample=token_indices_to_sample_padded,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_tokens_ids = self._to_target_token_ids(draft_ids[:num_reqs])
            return draft_tokens_ids.view(-1, 1)

        assert token_indices_to_sample_padded is not None
        positions = target_positions[
            token_indices_to_sample_padded.to(target_positions.device)
        ]

        # `hidden_states` is deliberately not gathered here -- #821 moved
        # that gather inside `model_wrapper`. Doing it again would index an
        # already (num_reqs_padded, hidden_size) tensor with token-space
        # indices and raise on the first decode.
        draft_token_ids = self._to_target_token_ids(draft_ids[:num_reqs])

        if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types
        ):
            raise ValueError(
                f"Unsupported attention metadata type for speculative "
                "decoding with num_speculative_tokens > 1: "
                f"{type(attn_metadata)}. Supported types are: "
                f"{self.allowed_attn_types}"
            )

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        # it mutates seq_lens in place; detach from the runner's persistent buffer.
        common_attn_metadata.seq_lens = common_attn_metadata.seq_lens.clone()
        common_attn_metadata.num_actual_tokens = num_reqs
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange_cpu[: num_reqs + 1]
        common_attn_metadata.query_start_loc_cpu = self.arange_cpu[: num_reqs + 1]

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens

        batch_desc, num_tokens_across_dp = determine_draft_batch_execution_and_padding(
            cfg=self.runner.shape_config,
            status=self.runner.dp_status,
            dp_rank=self.dp_rank,
            num_reqs=num_reqs,
            num_tokens=num_reqs,
            is_prefill=False,
            first_pass=False,
        )
        num_reqs_padded = batch_desc.num_reqs_padded
        num_padded_tokens = batch_desc.num_tokens_padded
        for token_index in range(self.num_speculative_tokens - 1):
            self.input_ids[:num_reqs] = draft_token_ids_list[-1].int()
            positions = positions.view(-1) + 1

            exceeds_max_model_len = positions[:num_reqs] >= self.max_model_len
            common_attn_metadata.seq_lens += 1
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            per_layer_attn_metadata.clear()
            for attn_group in self.draft_attn_groups:
                attn_metadata = attn_group.get_metadata_builder().build(
                    common_attn_metadata=common_attn_metadata,
                    positions=positions,
                    is_prefill=False,
                    batch_pad=num_reqs_padded,
                )
                attach_kv_cache_bindings(
                    attn_metadata,
                    self.runner.kv_caches,
                    self.runner.kv_cache_bases,
                    self.runner.kv_cache_view_infos,
                )
                for layer_name in attn_group.layer_names:
                    per_layer_attn_metadata[layer_name] = attn_metadata

            staged_input_ids, staged_positions, staged_hidden_states, _ = (
                self._preprocess(
                    num_reqs,
                    num_reqs_padded,
                    num_reqs,
                    positions[:num_reqs],
                    hidden_states[:num_reqs],
                    is_prefill=False,
                )
            )

            # Run the model.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_reqs,
                num_tokens_across_dp=num_tokens_across_dp,
                num_padded_tokens=num_padded_tokens,
                **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
            ):
                hidden_states, draft_ids = self.model_executable(
                    input_ids=staged_input_ids,
                    positions=staged_positions,
                    hidden_states=staged_hidden_states,
                    inputs_embeds=inputs_embeds,
                    token_indices_to_sample=None,
                )
            # Mapped before the feed-back above: the draft head's input
            # embedding is in target space even when its output head is not.
            draft_token_ids = self._to_target_token_ids(draft_ids[:num_reqs])
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def _to_target_token_ids(self, draft_token_ids: torch.Tensor) -> torch.Tensor:
        """Map draft-vocabulary ids to target-vocabulary ids.

        `d2t` holds offsets, not absolute ids -- upstream scatters at
        `arange(draft_vocab_size) + d2t` -- hence `id + d2t[id]`. None means the
        draft head already predicts the target vocabulary, so ids pass through.

        Equivalent to upstream's scatter-then-argmax for a monotonic mapping:
        both pick the same winner, and on an exact tie both pick the lowest id.
        """
        d2t = self.draft_id_to_target_id
        if d2t is None:
            return draft_token_ids.long().clone()
        return draft_token_ids + d2t[draft_token_ids]

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens[i].item()
                )
                for i in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        batch_size = sampled_token_ids.shape[0]
        return eagle_prepare_next_token_padded(
            sampled_token_ids,
            discard_request_mask[:batch_size],
            backup_tokens_gpu[:batch_size],
            gpu_input_batch.vocab_size,
        )

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        """
        token_indices_to_sample, num_rejected_tokens = eagle_prepare_inputs_padded(
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
        )

        query_start_loc = common_attn_metadata.query_start_loc
        seq_lens = common_attn_metadata.seq_lens
        new_query_len_per_req = query_start_loc[1:] - query_start_loc[:-1]
        total_num_tokens = query_start_loc[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            seq_lens=seq_lens,
            query_start_loc_cpu=query_start_loc,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=seq_lens.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=torch.tensor(0),  # dummy,
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens,
        )

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)

        self.draft_id_to_target_id = getattr(self.model, "draft_id_to_target_id", None)

        # Fused MoE is the only reader of the step's padded token dimension, so a
        # draft without it runs no collective of its own -- which is what lets an
        # idle rank skip drafting entirely rather than draft a discarded result.
        self.draft_has_moe = any(
            isinstance(module, MoERunner) for module in self.model.modules()
        )

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            token_indices_to_sample: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
        ):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
            )
            if not self.model_returns_tuple():
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states.view(-1, self.hidden_size)
            sample_hidden_states = last_hidden_states.view(-1, self.hidden_size)

            if token_indices_to_sample is not None:
                hidden_states = hidden_states[token_indices_to_sample]
                sample_hidden_states = sample_hidden_states[token_indices_to_sample]

            if self.draft_id_to_target_id is not None:
                # NOTE(RBLN): upstream's `compute_logits` widens draft-vocab
                # logits to the target vocabulary by scattering into an `-inf`
                # row. Its index is input-independent, so the subgraph folds
                # into an anonymous constant that weight-free apply cannot
                # resolve by name; it then executes on placeholder indices and
                # that out-of-bounds write is the SIGSEGV in KV warmup. Stay in
                # draft-vocab space and map after the argmax instead
                # (`_to_target_token_ids`) -- no per-model patch needed.
                assert isinstance(
                    self.model, (Eagle3LlamaForCausalLM, Eagle3DeepseekV2ForCausalLM)
                )
                logits = self.model.logits_processor(
                    self.model.lm_head, sample_hidden_states
                )
            else:
                logits = self.model.compute_logits(sample_hidden_states)

            # NOTE(RBLN): the greedy pick belongs in the graph.
            # To support probabilistic sampling, we need to return
            # the logits too.
            return hidden_states, torch.ops.rbln.argmax(logits)

        if (
            self.vllm_config.speculative_config.enforce_eager
            or not envs.VLLM_RBLN_COMPILE_MODEL
        ):
            self.model_executable = model_wrapper
        else:
            self.model_executable = compile(
                model_wrapper,
                dynamic=False,
                fullgraph=True,
                compile_context=self.runner.compile_context,
                num_devices=envs.VLLM_RBLN_NUM_DEVICES_PER_LOCAL_RANK,
                model_trace_method="export" if USE_DEVICE_TENSOR else "",
                process_group_dict=build_process_group_dict(),
                guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                runtime_holder=self.runner.runtime_holder,
                mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
                use_static_output=True,
            )

    def _build_dummy_attn_metadata(
        self,
        num_reqs: int,
        num_tokens_per_req: int,
    ) -> CommonAttentionMetadata:
        num_tokens = num_tokens_per_req * num_reqs
        assert num_tokens <= self.max_num_tokens

        num_scheduled_tokens = np.array([num_tokens_per_req] * num_reqs, dtype=np.int32)
        seq_lens = torch.from_numpy(num_scheduled_tokens)

        cum_num_tokens, _ = self.runner._get_cumsum_and_arange(num_scheduled_tokens)
        query_start_loc = torch.zeros(num_reqs + 1, dtype=torch.int32)
        query_start_loc[1 : num_reqs + 1] = torch.from_numpy(cum_num_tokens)

        return CommonAttentionMetadata(
            query_start_loc=query_start_loc,
            query_start_loc_cpu=query_start_loc,
            seq_lens=seq_lens,
            num_reqs=num_reqs,
            num_actual_tokens=num_tokens,
            max_query_len=num_tokens_per_req,
            max_seq_len=seq_lens.max().item(),
            block_table_tensor=self.runner.input_batch.block_table[0].get_cpu_tensor()[
                :num_reqs
            ],
            slot_mapping=torch.tensor(0),  # dummy
            causal=True,
        )

    @torch.inference_mode()
    def dummy_run(
        self,
        num_reqs: int,
        num_tokens_per_req: int,
        is_prefill: bool,
        *,
        num_padded_tokens: int | None = None,
    ) -> None:
        status = self.runner.dp_status
        if (
            status is not None
            and status.is_idle[self.dp_rank]
            and not self.draft_has_moe
        ):
            # This rank has no work, so what a draft pass produces here is
            # discarded, and without fused MoE no busy rank is waiting inside a
            # collective for it.
            return

        num_tokens = num_tokens_per_req * num_reqs
        assert num_tokens <= self.max_num_tokens
        override_padded = num_padded_tokens

        common_attn_metadata = self._build_dummy_attn_metadata(
            num_reqs, num_tokens_per_req
        )
        batch_desc, num_tokens_across_dp = determine_draft_batch_execution_and_padding(
            cfg=self.runner.shape_config,
            status=self.runner.dp_status,
            dp_rank=self.dp_rank,
            num_reqs=num_reqs,
            num_tokens=num_tokens,
            is_prefill=is_prefill,
            pinned_num_tokens_padded=override_padded,
        )
        num_reqs_padded = batch_desc.num_reqs_padded
        num_padded_tokens = batch_desc.num_tokens_padded

        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=self.positions[:num_tokens],
                is_prefill=is_prefill,
                batch_pad=num_reqs_padded,
            )
            attach_kv_cache_bindings(
                attn_metadata,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        token_indices_to_sample = (
            torch.arange(num_reqs, device=self.device, dtype=torch.int32)
            * num_tokens_per_req
        )
        input_ids, positions, hidden_states, token_indices_to_sample_padded = (
            self._preprocess(
                num_reqs,
                num_reqs_padded,
                num_tokens,
                self.positions[:num_tokens],
                self.hidden_states[:num_tokens],
                is_prefill=is_prefill,
                token_indices_to_sample=token_indices_to_sample,
            )
        )
        inputs_embeds = None

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
            **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
        ):
            _, _ = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                token_indices_to_sample=token_indices_to_sample_padded,
            )

        if self.num_speculative_tokens == 1:
            return

        common_attn_metadata.num_actual_tokens = num_reqs
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange_cpu[: num_reqs + 1]
        common_attn_metadata.query_start_loc_cpu = self.arange_cpu[: num_reqs + 1]
        common_attn_metadata.seq_lens += 1

        batch_desc, num_tokens_across_dp = determine_draft_batch_execution_and_padding(
            cfg=self.runner.shape_config,
            status=self.runner.dp_status,
            dp_rank=self.dp_rank,
            num_reqs=num_reqs,
            num_tokens=num_reqs,
            is_prefill=False,
            first_pass=False,
            pinned_num_tokens_padded=override_padded,
        )
        num_reqs_padded = batch_desc.num_reqs_padded
        num_padded_tokens = batch_desc.num_tokens_padded
        per_layer_attn_metadata.clear()
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=self.positions[:num_reqs],
                is_prefill=False,
                batch_pad=num_reqs_padded,
            )
            attach_kv_cache_bindings(
                attn_metadata,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

        input_ids, positions, hidden_states, _ = self._preprocess(
            num_reqs,
            num_reqs_padded,
            num_reqs,
            self.positions[:num_reqs],
            self.hidden_states[:num_reqs],
            is_prefill=False,
        )

        for _ in range(self.num_speculative_tokens - 1):
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_reqs,
                num_tokens_across_dp=num_tokens_across_dp,
                num_padded_tokens=num_padded_tokens,
                **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
            ):
                _, _ = self.model_executable(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    inputs_embeds=inputs_embeds,
                    token_indices_to_sample=None,
                )

    def _preprocess(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_input_tokens: int,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        *,
        is_prefill: bool,
        token_indices_to_sample: torch.Tensor | None = None,
        target_token_ids: torch.Tensor | None = None,
        next_token_ids: torch.Tensor | None = None,
        cad: CommonAttentionMetadata | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if target_token_ids is not None:
            assert next_token_ids is not None
            assert num_input_tokens == target_token_ids.shape[0]

            if token_indices_to_sample is None:
                assert cad is not None
                token_indices_to_sample = cad.query_start_loc[1:] - 1
            token_indices_to_sample = token_indices_to_sample.to(self.device)

            self.input_ids[: num_input_tokens - 1] = target_token_ids[1:]
            self.input_ids[token_indices_to_sample] = next_token_ids

        input_ids = self.input_ids[:num_input_tokens].view(num_reqs, -1)
        positions = positions.view(-1)[:num_input_tokens].view(num_reqs, -1)
        hidden_states = hidden_states.view(-1, self.hidden_size)[
            :num_input_tokens
        ].view(num_reqs, -1, self.hidden_size)

        layout = InputLayout(
            num_reqs=num_reqs,
            num_reqs_padded=num_reqs if is_prefill else num_reqs_padded,
            query_len=input_ids.shape[1],
            query_len_padded=self.max_num_tokens if is_prefill else input_ids.shape[1],
        )
        staged = self.input_stager.stage(
            input_ids=input_ids,
            positions=positions,
            hidden_states=hidden_states,
            token_indices=token_indices_to_sample,
            layout=layout,
        )
        assert staged.hidden_states is not None

        return (
            staged.input_ids,
            staged.positions,
            staged.hidden_states,
            staged.token_indices,
        )
