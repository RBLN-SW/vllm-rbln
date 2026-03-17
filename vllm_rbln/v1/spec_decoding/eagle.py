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

import numpy as np
import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.distributed.parallel_state import get_dp_group, get_pp_group, get_tp_group
from vllm.forward_context import set_forward_context
from vllm.v1.attention.backends.tree_attn import TreeAttentionMetadata
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID, EagleProposer
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

import vllm_rbln.rbln_envs as envs
import vllm_rbln.utils as rbln_utils
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadata,
)

logger = init_logger(__name__)


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
) -> torch.Tensor:
    """
    This function computes the token index to sample for each request, taking into
    account the number of draft tokens and the number of valid sampled tokens
    (which is one more than the number of accepted tokens)
    """
    num_draft_tokens = cu_num_draft_tokens - torch.nn.functional.pad(
        cu_num_draft_tokens[:-1], (1, 0)
    )

    has_draft = num_draft_tokens > 0
    num_rejected = has_draft * (num_draft_tokens + 1 - valid_sampled_tokens_count)

    return (query_start_loc[1:] - 1 - num_rejected).to(torch.int32)


class RBLNEagleProposer(EagleProposer):
    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        super().__init__(vllm_config, device, runner)

        # NOTE(RBLN): vllm-rbln does not use cudagraphs.
        self.use_cuda_graph = False

        # NOTE(RBLN): vllm-rbln uses only RBLNFlashAttentionMetadata
        self.allowed_attn_types = (RBLNFlashAttentionMetadata,)

        # TODO(RBLN): supports eagle/eagle3 with multi-modal.
        if self.supports_mm_inputs:
            raise NotImplementedError("Eagle is not supported with multi-modal.")

        # TODO(RBLN): Using a separate CompileContext for the draft model is a
        # temporary workaround. Since the base model's KV caches are managed
        # together within its CompileContext, using a separate one here causes
        # the draft model to redundantly allocate memory for the base model's
        # KV caches, resulting in unnecessary memory waste. This should be
        # revisited to properly share the base model's CompileContext.
        from rebel import CompileContext

        self.compile_context = CompileContext(use_weight_sharing=True)

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        last_token_indices: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    ) -> torch.Tensor:
        num_tokens = target_token_ids.shape[0]
        batch_size = next_token_ids.shape[0]

        if last_token_indices is None:
            last_token_indices = common_attn_metadata.query_start_loc[1:] - 1

        if self.method == "eagle3":
            # assert isinstance(self.model, Eagle3LlamaForCausalLM)
            target_hidden_states = self.model.combine_hidden_states(
                target_hidden_states
            )
            assert target_hidden_states.shape[-1] == self.hidden_size

        # Shift the input ids by one token.
        # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
        self.input_ids[: num_tokens - 1] = target_token_ids[1:]
        # Replace the last token with the next token.
        # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
        self.input_ids[last_token_indices] = next_token_ids

        assert self.runner is not None

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        # NOTE(RBLN): build attention metadata
        batch_bucket_size = self.runner.bucketing_manager.find_decode_batch_bucket(
            batch_size
        )
        num_padded_tokens = None
        num_tokens_across_dp = None
        extra_attn_metadata_args = {}
        extra_attn_metadata_args["num_tokens"] = (
            self.runner.input_batch.num_tokens_no_spec
        )
        extra_attn_metadata_args["positions"] = target_positions.cpu()
        extra_attn_metadata_args["batch_pad"] = batch_bucket_size
        attn_metadata = attn_metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
            **extra_attn_metadata_args,
        )
        # FIXME: support hybrid kv for draft model (remove separate indexer)
        if self.draft_indexer_metadata_builder:
            draft_indexer_metadata = (
                self.draft_indexer_metadata_builder.build_for_drafting(
                    common_attn_metadata=common_attn_metadata,
                    draft_index=0,
                )
            )
        else:
            draft_indexer_metadata = None
        # At this moment, we assume all eagle layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        for layer_name in self.indexer_layer_names:
            assert draft_indexer_metadata is not None
            per_layer_attn_metadata[layer_name] = draft_indexer_metadata

        # NOTE(RBLN): just set num_tokens to num_input_tokens
        num_input_tokens = num_tokens
        if self.supports_mm_inputs:
            mm_embeds, is_mm_embed = mm_embed_inputs or (None, None)

            self.inputs_embeds[:num_tokens] = self.model.embed_input_ids(
                self.input_ids[:num_tokens],
                multimodal_embeddings=mm_embeds,
                is_multimodal=is_mm_embed,
            )

            input_ids = None
            inputs_embeds = self.inputs_embeds[:num_input_tokens]
        else:
            # NOTE(RBLN): reshape tensors in the same way as the RBLN model runner.
            is_prefill = self.runner.is_prefills()[0]
            if is_prefill:
                input_ids = self.input_ids.view(batch_size, -1)
                positions = rbln_utils.pad(
                    target_positions.view(batch_size, -1), -1, input_ids.shape[-1]
                )
            else:
                input_ids = self.input_ids[:num_input_tokens].view(batch_size, -1)
                input_ids = rbln_utils.pad(input_ids, 0, batch_bucket_size)
                positions = target_positions.view(batch_size, -1)
                positions = rbln_utils.pad(positions, -2, batch_bucket_size)
            last_token_indices_padded = rbln_utils.pad(
                last_token_indices, 0, batch_bucket_size
            )
            hidden_states = target_hidden_states.view(*input_ids.shape, -1)

            inputs_embeds = None

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=num_padded_tokens,
        ):
            if per_layer_attn_metadata is not None:
                for attn_metadata in per_layer_attn_metadata.values():
                    attn_metadata.kv_caches = self.runner.kv_caches

            hidden_states, logits = self.model_executable(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
                last_token_indices=last_token_indices_padded,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            draft_tokens_ids = logits[:batch_size].argmax(dim=-1)
            return draft_tokens_ids.view(-1, 1)

        positions = (
            target_positions[:, last_token_indices]
            if self.uses_mrope
            else target_positions[last_token_indices]
        )
        if self.method in (
            "deepseek_mtp",
            "ernie_mtp",
            "longcat_flash_mtp",
            "pangu_ultra_moe_mtp",
        ):
            hidden_states = self.hidden_states[last_token_indices]
        else:
            hidden_states = hidden_states[last_token_indices]

        if isinstance(attn_metadata, TreeAttentionMetadata):
            # NOTE(RBLN): tree attention is not supported
            # # Draft using tree attention.
            # draft_token_ids_list = self.propose_tree(
            #     batch_size=batch_size,
            #     logits=logits,
            #     positions=positions,
            #     hidden_states=hidden_states,
            #     common_attn_metadata=common_attn_metadata,
            # )
            # # [batch_size, num_tree_tokens]
            # return torch.cat(draft_token_ids_list, dim=1)
            raise NotImplementedError("Tree attention is not supported")

        draft_token_ids = logits[:batch_size].argmax(dim=-1)

        # Generate the remaining draft tokens.
        draft_token_ids_list = [draft_token_ids]

        common_attn_metadata.num_actual_tokens = batch_size
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: batch_size + 1]
        common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
            self.token_arange_np[: batch_size + 1]
        ).clone()
        for _ in range(self.num_speculative_tokens - 1):
            # Update the inputs
            # cast to int32 is crucial when eagle model is compiled.
            # tensor.argmax returns int64 by default.
            input_ids = draft_token_ids_list[-1].int()
            positions = positions[:batch_size].view(-1)
            if self.uses_mrope:
                positions += 1
                # NOTE(woosuk): We should handle the case where the draft model
                # generates tokens beyond the max model length.
                # Since it is complex to remove such requests from the batch,
                # we keep them in the batch but adjust the position ids
                # and slot mappings to avoid the
                # out-of-range access during the model execution.
                # The draft tokens generated with this adjustment
                # should be ignored.
                exceeds_max_model_len = positions[0] >= self.max_model_len
                # Mask out the position ids that exceed the max model length.
                # Otherwise, we may get out-of-range error in RoPE.
                clamped_positions = torch.where(
                    exceeds_max_model_len.unsqueeze(0),
                    torch.zeros_like(positions),
                    positions,
                )
            else:
                positions += 1
                exceeds_max_model_len = positions >= self.max_model_len
                clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
            # For data integrity when async scheduling, we shouldn't use in place
            # operations in case they are modified in next step's `prepare_input`
            # of main model.
            # Increment the sequence lengths.
            common_attn_metadata.seq_lens += 1
            # For the requests that exceed the max model length, we set the
            # sequence length to 1 to minimize their overheads in attention.
            common_attn_metadata.seq_lens.masked_fill_(exceeds_max_model_len, 1)

            # Also update the CPU-side shadow; NOTE: this is hacky and should be
            # removed in when common_attn_metadata.seq_lens_cpu is deprecated.
            if common_attn_metadata._seq_lens_cpu is not None:
                common_attn_metadata._seq_lens_cpu += 1
            if common_attn_metadata._num_computed_tokens_cpu is not None:
                common_attn_metadata._num_computed_tokens_cpu += 1

            # Compute the slot mapping.
            if self.uses_mrope:
                # all dimensions of positions are the same
                block_numbers = clamped_positions[0] // self.block_size
            else:
                block_numbers = clamped_positions // self.block_size
            block_ids = common_attn_metadata.block_table_tensor.gather(
                dim=1, index=block_numbers.view(-1, 1)
            )
            block_ids = block_ids.view(-1)
            if self.uses_mrope:
                common_attn_metadata.slot_mapping = (
                    block_ids * self.block_size + clamped_positions[0] % self.block_size
                )
            else:
                common_attn_metadata.slot_mapping = (
                    block_ids * self.block_size + clamped_positions % self.block_size
                )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with the
            # padding tokens.
            common_attn_metadata.slot_mapping.masked_fill_(
                exceeds_max_model_len, PADDING_SLOT_ID
            )

            # Rebuild attention metadata
            extra_attn_metadata_args = {}
            extra_attn_metadata_args["num_tokens"] = (
                common_attn_metadata.seq_lens.cpu().numpy()
            )
            extra_attn_metadata_args["positions"] = positions.cpu()
            extra_attn_metadata_args["batch_pad"] = batch_bucket_size
            attn_metadata = attn_metadata_builder.build(
                common_prefix_len=0,
                common_attn_metadata=common_attn_metadata,
                fast_build=True,
                **extra_attn_metadata_args,
            )
            for layer_name in self.attn_layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata

            # copy inputs to buffer
            self.input_ids[:batch_size] = input_ids
            self._set_positions(batch_size, clamped_positions)
            self.hidden_states[: hidden_states.shape[0]] = hidden_states
            if self.supports_mm_inputs:
                self.inputs_embeds[:batch_size] = self.model.embed_input_ids(input_ids)

                input_ids = None
                inputs_embeds = self.inputs_embeds[:batch_size]
            else:
                # NOTE(RBLN): reshape tensors in the same way as the RBLN model runner.
                input_ids = self.input_ids[:batch_bucket_size].view(
                    batch_bucket_size, 1
                )
                positions = self.positions[:batch_bucket_size].view(
                    batch_bucket_size, 1
                )
                hidden_states = self.hidden_states[:batch_bucket_size].view(
                    batch_bucket_size, 1, -1
                )
                inputs_embeds = None

            # Run the model.
            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=batch_size,
                num_tokens_across_dp=None,
                num_padded_tokens=None,
            ):
                if per_layer_attn_metadata is not None:
                    for attn_metadata in per_layer_attn_metadata.values():
                        attn_metadata.kv_caches = self.runner.kv_caches

                hidden_states, logits = self.model_executable(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    inputs_embeds=inputs_embeds,
                    last_token_indices=None,
                )
            draft_token_ids = logits[:batch_size].argmax(dim=-1)
            draft_token_ids_list.append(draft_token_ids)

        # [batch_size, num_speculative_tokens]
        draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
        return draft_token_ids

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
    ) -> tuple[CommonAttentionMetadata, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        """
        token_indices_to_sample = eagle_prepare_inputs_padded(
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
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
            max_seq_len=common_attn_metadata.seq_lens.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return spec_common_attn_metadata, token_indices_to_sample

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
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = last_hidden_states
            else:
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
            self.model_executable = self._compile_model(model_wrapper)

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
            "mode": "strict",
        }
        if not envs.VLLM_DISABLE_COMPILE_CACHE:
            logger.info(
                "Once the model is compiled for the first time, "
                "the cached compiled binary will be reused."
            )
            options["cache_dir"] = os.path.join(envs.VLLM_CACHE_ROOT, "rbln")

        return torch.compile(
            model,
            backend="rbln",
            options=copy(options),
            dynamic=False,
        )
