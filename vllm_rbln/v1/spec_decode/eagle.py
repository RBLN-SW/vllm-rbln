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
from copy import copy
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dp_group, get_pp_group, get_tp_group
from vllm.model_executor.models.deepseek_eagle3 import Eagle3DeepseekV2ForCausalLM
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.eagle import EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

import vllm_rbln.envs as envs
from vllm_rbln.compilation.backends import rbln_backend
from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.logger import init_logger
from vllm_rbln.utils import pad
from vllm_rbln.v1.attention.kv_cache_bindings import (
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)
from vllm_rbln.v1.spec_decode.utils import (
    eagle_prepare_inputs_padded,
    eagle_prepare_next_token_padded,
)

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

        from rebel import CompileContext

        self.runner = runner
        self.compile_context = CompileContext(use_weight_sharing=True)

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
        if self.method == "eagle3":
            assert isinstance(
                self.model, (Eagle3LlamaForCausalLM, Eagle3DeepseekV2ForCausalLM)
            )
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        num_tokens, token_indices_to_sample = self.set_inputs_first_pass(
            target_token_ids=target_token_ids,
            next_token_ids=next_token_ids,
            target_positions=target_positions,
            target_hidden_states=target_hidden_states,
            token_indices_to_sample=token_indices_to_sample,
            cad=common_attn_metadata,
        )

        assert self.runner is not None
        is_prefill = self.runner.is_prefill

        # Build attention metadata
        num_reqs = self.runner.input_batch.num_reqs
        self.runner.bucketing_manager.find_decode_batch_bucket(num_reqs)
        num_reqs_padded = (
            self.runner.bucketing_manager.find_decode_batch_bucket(num_reqs)
            if not is_prefill
            else num_reqs
        )
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
                token_indices_to_sample,
                is_prefill,
            )
        )
        inputs_embeds = None

        num_padded_tokens, num_tokens_across_dp = None, None
        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
            **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
        ):
            hidden_states, logits = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                last_token_indices=token_indices_to_sample_padded,
            )

        if self.runner.is_intermediate_chunked_prefill:
            return torch.zeros(
                num_reqs,
                self.num_speculative_tokens,
                device=self.device,
                dtype=torch.int64,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_tokens_ids = logits[:num_reqs].argmax(dim=-1)
            return draft_tokens_ids.view(-1, 1)

        positions = target_positions[token_indices_to_sample_padded]

        draft_token_ids = logits[:num_reqs].argmax(dim=-1)

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

        common_attn_metadata.num_actual_tokens = num_reqs
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: num_reqs + 1]
        common_attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc

        # In padded drafter batch, we need to adjust the sequence lengths
        # to remove the "padding" (i.e. rejected tokens).
        # Only apply this adjustment when we have rejected tokens
        # (i.e., not the first proposal).
        if self.num_speculative_tokens > 1 and num_rejected_tokens is not None:
            common_attn_metadata.seq_lens -= num_rejected_tokens
            # Invalidate the CPU-side shadows to avoid H<>D sync.
            # common_attn_metadata._seq_lens_cpu = None
            # common_attn_metadata._num_computed_tokens_cpu = None

        num_reqs_padded = self.runner.bucketing_manager.find_decode_batch_bucket(
            num_reqs
        )
        for token_index in range(self.num_speculative_tokens - 1):
            # Update the inputs
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax returns int64 by default.
            self.input_ids[:num_reqs] = draft_token_ids_list[-1].int()
            positions = positions.view(-1) + 1
            self.positions[:num_reqs] = positions[:num_reqs]
            self.hidden_states[: hidden_states.shape[0]] = hidden_states

            exceeds_max_model_len = positions[:num_reqs] >= self.max_model_len
            common_attn_metadata.seq_lens += 1
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Rebuild attention metadata
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

            input_ids, positions, hidden_states, _ = self._preprocess(
                num_reqs,
                num_reqs_padded,
                num_reqs,
                None,
                False,
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
                hidden_states, logits = self.model_executable(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    inputs_embeds=inputs_embeds,
                    last_token_indices=None,
                )
            draft_token_ids = logits[:num_reqs].argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

    def set_inputs_first_pass(
        self,
        target_token_ids: torch.Tensor,
        next_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        cad: CommonAttentionMetadata,
    ) -> tuple[int, torch.Tensor]:
        if self.needs_extra_input_slots:
            raise NotImplementedError(
                "vllm-rbln does not support EAGLE extra input slots required for "
                "parallel drafting or draft-model speculative decoding yet."
            )

        if token_indices_to_sample is None:
            token_indices_to_sample = cad.query_start_loc[1:] - 1

        num_tokens = target_token_ids.shape[0]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        self.input_ids[token_indices_to_sample] = next_token_ids

        self._set_positions(num_tokens, target_positions)

        self.hidden_states[:num_tokens] = target_hidden_states.view(
            -1, self.hidden_size
        )[:num_tokens]

        return num_tokens, token_indices_to_sample

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

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        seq_lens_cpu = (
            common_attn_metadata._seq_lens_cpu
            if common_attn_metadata._seq_lens_cpu is not None
            else common_attn_metadata.seq_lens.cpu()
        )
        new_query_len_per_req = query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=common_attn_metadata._num_computed_tokens_cpu,
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=seq_lens_cpu.max().item(),
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

        def model_wrapper(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
            last_token_indices: torch.Tensor | None = None,
            inputs_embeds: torch.Tensor | None = None,
        ):
            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
            )
            last_hidden_states, hidden_states = ret_hidden_states

            hidden_states = hidden_states.view(-1, self.hidden_size)
            last_hidden_states = last_hidden_states.view(-1, self.hidden_size)
            sample_hidden_states = (
                last_hidden_states[last_token_indices]
                if last_token_indices is not None
                else last_hidden_states
            )
            logits = self.model.compute_logits(sample_hidden_states)

            return hidden_states, logits

        if (
            self.vllm_config.speculative_config.enforce_eager
            or not envs.VLLM_RBLN_COMPILE_MODEL
        ):
            self.model_executable = model_wrapper
        else:
            self.model_executable = self._compile(model_wrapper)

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
    ) -> None:
        if not is_prefill:
            num_tokens_per_req += self.num_speculative_tokens
        num_tokens = num_tokens_per_req * num_reqs
        assert num_tokens <= self.max_num_tokens

        common_attn_metadata = self._build_dummy_attn_metadata(
            num_reqs, num_tokens_per_req
        )

        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            attn_metadata = attn_group.get_metadata_builder().build(
                common_attn_metadata=common_attn_metadata,
                positions=self.positions[:num_tokens],
                is_prefill=is_prefill,
                batch_pad=num_reqs,
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
                num_reqs, num_reqs, num_tokens, token_indices_to_sample, is_prefill
            )
        )
        inputs_embeds = None

        num_padded_tokens, num_tokens_across_dp = None, None
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
                last_token_indices=token_indices_to_sample_padded,
            )

        if self.num_speculative_tokens == 1:
            return

        common_attn_metadata.num_actual_tokens = num_reqs
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: num_reqs + 1]
        common_attn_metadata.query_start_loc_cpu = common_attn_metadata.query_start_loc
        common_attn_metadata.seq_lens += 1

        num_reqs_padded = self.runner.bucketing_manager.find_decode_batch_bucket(
            num_reqs
        )
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
            None,
            False,
        )

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
                last_token_indices=None,
            )

    def _compile(self, model):
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
            "mode": "strict",
        }
        if not envs.VLLM_DISABLE_COMPILE_CACHE:
            options["cache_dir"] = os.path.join(envs.VLLM_CACHE_ROOT, "rbln")

        return torch.compile(
            model,
            backend=rbln_backend,
            options=copy(options),
            dynamic=False,
        )

    def _preprocess(
        self,
        num_reqs: int,
        num_reqs_padded: int,
        num_input_tokens: int,
        token_indices_to_sample: torch.Tensor | None,
        is_prefill: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if is_prefill:
            input_ids = self.input_ids.view(num_reqs_padded, -1)
            positions = self.positions.view(num_reqs_padded, -1)
            target_hidden_states = self.hidden_states
        else:
            input_ids = self.input_ids[:num_input_tokens].view(num_reqs, -1)
            positions = self.positions[:num_input_tokens].view(num_reqs, -1)
            target_hidden_states = self.hidden_states[:num_input_tokens].view(
                num_reqs, -1, self.hidden_size
            )
            input_ids = pad(input_ids, 0, num_reqs_padded)
            positions = pad(positions, 0, num_reqs_padded)
            target_hidden_states = pad(target_hidden_states, 0, num_reqs_padded)

        target_hidden_states = target_hidden_states.view(
            *input_ids.shape, -1
        )  # [B, L, H]
        token_indices_to_sample_padded = (
            pad(token_indices_to_sample, 0, num_reqs_padded)
            if token_indices_to_sample is not None
            else None
        )

        return (
            input_ids,
            positions,
            target_hidden_states,
            token_indices_to_sample_padded,
        )

    def initialize_kv_cache_tensors(self):
        if not self.speculative_config.enforce_eager and envs.VLLM_RBLN_COMPILE_MODEL:
            self.compile_context.mark_static_address(self.runner.kv_caches[-1])
