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
import ast

import numpy as np
import torch
from vllm.config import VllmConfig
from vllm.forward_context import set_forward_context
from vllm.model_executor.models.llama_eagle3 import Eagle3LlamaForCausalLM
from vllm.multimodal import MULTIMODAL_REGISTRY
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.eagle import PADDING_SLOT_ID, EagleProposer
from vllm.v1.utils import CpuGpuBuffer

import vllm_rbln.utils as rbln_utils
from vllm_rbln.logger import init_logger
from vllm_rbln.v1.attention.backends.flash_attention import (
    RBLNFlashAttentionMetadata, RBLNFlashAttentionMetadataBuilder)

logger = init_logger(__name__)


def __custom_init__(self: EagleProposer,
                    vllm_config: VllmConfig,
                    device: torch.device,
                    runner=None):
    self.vllm_config = vllm_config
    self.speculative_config = vllm_config.speculative_config
    assert self.speculative_config is not None
    self.draft_model_config = self.speculative_config.draft_model_config
    self.method = self.speculative_config.method

    self.runner = runner
    self.device = device
    self.dtype = vllm_config.model_config.dtype
    self.max_model_len = vllm_config.model_config.max_model_len
    self.block_size = vllm_config.cache_config.block_size
    self.dp_rank = vllm_config.parallel_config.data_parallel_rank
    self.num_speculative_tokens = self.speculative_config.num_speculative_tokens
    self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
    self.token_arange_np = np.arange(self.max_num_tokens)
    # We need to get the hidden size from the draft model config because
    # the draft model's hidden size can be different from the target model's
    # hidden size (e.g., Llama 3.3 70B).
    self.hidden_size = self.draft_model_config.get_hidden_size()
    self.inputs_embeds_size = self.draft_model_config.get_inputs_embeds_size()

    # Multi-modal data support
    self.mm_registry = MULTIMODAL_REGISTRY
    self.supports_mm_inputs = self.mm_registry.supports_multimodal_inputs(
        vllm_config.model_config)

    self.attn_metadata_builder: RBLNFlashAttentionMetadataBuilder | None = None
    self.draft_indexer_metadata_builder: \
        RBLNFlashAttentionMetadataBuilder | None = None
    self.attn_layer_names: list[str] = []
    self.indexer_layer_names: list[str] = []
    self.eagle3_use_aux_hidden_state: bool = (
        self._get_eagle3_use_aux_hidden_state_from_config())

    # NOTE(RBLN): vllm-rbln does not use cudagraphs.
    self.use_cuda_graph = False

    # persistent buffers for cuda graph
    self.input_ids = torch.zeros(self.max_num_tokens,
                                 dtype=torch.int32,
                                 device=device)
    self.uses_mrope = self.vllm_config.model_config.uses_mrope
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
        self.mrope_positions = torch.zeros((3, self.max_num_tokens + 1),
                                           dtype=torch.int64,
                                           device=device)
    else:
        # RoPE need (max_num_tokens,)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=device)
    self.hidden_states = torch.zeros((self.max_num_tokens, self.hidden_size),
                                     dtype=self.dtype,
                                     device=device)

    # We need +1 here because the arange is used to set query_start_loc,
    # which has one more element than batch_size.
    max_batch_size = vllm_config.scheduler_config.max_num_seqs
    max_num_slots_for_arange = max(max_batch_size + 1, self.max_num_tokens)
    self.arange = torch.arange(max_num_slots_for_arange,
                               device=device,
                               dtype=torch.int32)

    self.inputs_embeds = torch.zeros(
        (self.max_num_tokens, self.inputs_embeds_size),
        dtype=self.dtype,
        device=device)

    self.backup_next_token_ids = CpuGpuBuffer(
        max_batch_size,
        dtype=torch.int32,
        pin_memory=is_pin_memory_available(),
        device=device,
        with_numpy=True,
    )

    # Determine allowed attention backends once during initialization.
    # NOTE(RBLN): vllm-rbln uses only RBLNFlashAttentionMetadata
    self.allowed_attn_types = (RBLNFlashAttentionMetadata, )

    # Parse the speculative token tree.
    spec_token_tree = self.speculative_config.speculative_token_tree
    self.tree_choices: list[tuple[int,
                                  ...]] = ast.literal_eval(spec_token_tree)
    tree_depth = len(self.tree_choices[-1])
    # Precompute per-level properties of the tree.
    num_drafts_per_level = [0] * tree_depth
    for node in self.tree_choices:
        num_drafts_per_level[len(node) - 1] += 1
    self.cu_drafts_per_level = [num_drafts_per_level[0]]
    self.child_drafts_per_level = [num_drafts_per_level[0]]
    for level in range(1, tree_depth):
        self.cu_drafts_per_level.append(self.cu_drafts_per_level[-1] +
                                        num_drafts_per_level[level])
        self.child_drafts_per_level.append(num_drafts_per_level[level] //
                                           num_drafts_per_level[level - 1])
    # Precompute draft position offsets in flattened tree.
    self.tree_draft_pos_offsets = torch.arange(1,
                                               len(self.tree_choices) + 1,
                                               device=device,
                                               dtype=torch.int32).repeat(
                                                   max_batch_size, 1)


def custom_propose(
    self: EagleProposer,
    target_token_ids: torch.Tensor,
    target_positions: torch.Tensor,
    target_hidden_states: torch.Tensor,
    next_token_ids: torch.Tensor,
    last_token_indices: torch.Tensor | None,
    common_attn_metadata: CommonAttentionMetadata,
    sampling_metadata: SamplingMetadata,
    mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
    num_rejected_tokens_gpu: torch.Tensor | None = None,
    *,
    kv_caches: list[torch.Tensor],
) -> torch.Tensor:
    max_len_per_req = (self.num_speculative_tokens + 1)

    # NOTE(RBLN): Logits tensor is a 2D tensor.
    if last_token_indices is None:
        last_token_indices = common_attn_metadata.query_start_loc[1:] - 1
        last_token_indices_2d = last_token_indices
    else:
        last_token_indices_2d = (last_token_indices % max_len_per_req).clamp(
            0, max_len_per_req - 1)

    num_tokens = target_token_ids.shape[0]
    batch_size = next_token_ids.shape[0]

    if self.method == "eagle3":
        assert isinstance(self.model, Eagle3LlamaForCausalLM)
        target_hidden_states = self.model.combine_hidden_states(
            target_hidden_states)
        assert target_hidden_states.shape[-1] == self.hidden_size
    # Shift the input ids by one token.
    # E.g., [a1, b1, b2, c1, c2, c3] -> [b1, b2, c1, c2, c3, c3]
    self.input_ids[:num_tokens - 1] = target_token_ids[1:]
    # Replace the last token with the next token.
    # E.g., [b1, b2, c1, c2, c3, c3] -> [a2, b2, b3, c2, c3, c4]
    self.input_ids[last_token_indices] = next_token_ids

    assert self.runner is not None

    if self.attn_metadata_builder is None:
        attn_metadata_builder = self.runner.attn_groups[0][
            0].get_metadata_builder()
    else:
        attn_metadata_builder = self.attn_metadata_builder

    extra_attn_metadata_args = {}
    extra_attn_metadata_args[
        "num_tokens"] = self.runner.input_batch.num_tokens_no_spec
    extra_attn_metadata_args["positions"] = target_positions.cpu()
    extra_attn_metadata_args[
        "batch_pad"] = self.runner.bucketing_manager.find_decode_batch_bucket(
            batch_size)
    attn_metadata = attn_metadata_builder.build(
        common_prefix_len=0,
        common_attn_metadata=common_attn_metadata,
        fast_build=True,
        **extra_attn_metadata_args)

    # FIXME: support hybrid kv for draft model (remove separate indexer)
    if self.draft_indexer_metadata_builder:
        draft_indexer_metadata = (
            self.draft_indexer_metadata_builder.build_for_drafting(
                common_attn_metadata=common_attn_metadata,
                draft_index=0,
            ))
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

    num_tokens_dp_padded, num_tokens_across_dp = self._pad_batch_across_dp(
        num_tokens_unpadded=num_tokens, num_tokens_padded=num_tokens)

    num_input_tokens = num_tokens_dp_padded
    if num_tokens_across_dp is not None:
        num_tokens_across_dp[self.dp_rank] = num_input_tokens

    # copy inputs to buffer for cudagraph
    self._set_positions(num_tokens, target_positions)

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
        input_ids = self.input_ids[:num_tokens]
        inputs_embeds = None

    # NOTE(RBLN): A temporary workaround;
    # reshapes input tensors in the same way as the RBLN model runner.
    is_prefills = self.runner.is_prefills()
    num_reqs = self.runner.input_batch.num_reqs
    input_ids = input_ids.view(num_reqs, -1)
    target_positions = target_positions.view(num_reqs, -1)
    if is_prefills[0]:
        input_ids = rbln_utils.pad(input_ids, -1, self.max_num_tokens)
        target_positions = rbln_utils.pad(target_positions, -1,
                                          self.max_num_tokens)
    else:
        input_ids = rbln_utils.pad(input_ids, 0, self.runner.max_num_seqs)
        target_positions = rbln_utils.pad(target_positions, -2,
                                          self.runner.max_num_seqs)
        last_token_indices_2d = rbln_utils.pad(last_token_indices_2d, 0,
                                               self.runner.max_num_seqs)
    target_hidden_states = target_hidden_states.unflatten(0, input_ids.shape)

    with set_forward_context(per_layer_attn_metadata,
                             self.vllm_config,
                             num_tokens=num_input_tokens,
                             num_tokens_across_dp=num_tokens_across_dp):
        if per_layer_attn_metadata is not None:
            for attn_metadata in per_layer_attn_metadata.values():
                attn_metadata.kv_caches = kv_caches

        ret_hidden_states = self.model(
            input_ids=input_ids,
            positions=target_positions,
            hidden_states=target_hidden_states,
            inputs_embeds=inputs_embeds,
        )
        if self.method == "mtp":
            last_hidden_states = ret_hidden_states
            hidden_states = last_hidden_states
        else:
            last_hidden_states, hidden_states = ret_hidden_states

    # NOTE(RBLN): A temporary workaround;
    # reshapes input tensors in the same way as the RBLN model runner.
    hidden_states = hidden_states.flatten(0, -2)
    last_hidden_states = last_hidden_states.flatten(0, -2)
    sample_hidden_states = last_hidden_states[last_token_indices]
    logits = self.model.compute_logits(sample_hidden_states)

    # Early exit if there is only one draft token to be generated.
    if self.num_speculative_tokens == 1:
        draft_token_ids = logits.argmax(dim=-1)
        return draft_token_ids.view(-1, 1)

    if is_prefills[0]:
        positions = target_positions[:, last_token_indices].view(-1)
    else:
        rows = torch.arange(target_positions.size(0),
                            device=target_positions.device)
        positions = target_positions[rows, last_token_indices_2d]

    if self.method in (
            "deepseek_mtp",
            "ernie_mtp",
            "longcat_flash_mtp",
            "pangu_ultra_moe_mtp",
    ):
        raise NotImplementedError(f"{self.method} is not supported")
    else:
        hidden_states = hidden_states[last_token_indices_2d]

    draft_token_ids = logits.argmax(dim=-1)

    if self.allowed_attn_types is not None and not isinstance(
            attn_metadata, self.allowed_attn_types):
        raise ValueError(
            f"Unsupported attention metadata type for speculative "
            "decoding with num_speculative_tokens > 1: "
            f"{type(attn_metadata)}. Supported types are: "
            f"{self.allowed_attn_types}")

    # Generate the remaining draft tokens.
    draft_token_ids_list = [draft_token_ids]

    batch_size_dp_padded, batch_size_across_dp = self._pad_batch_across_dp(
        num_tokens_unpadded=batch_size,
        num_tokens_padded=batch_size,
    )

    input_batch_size = batch_size_dp_padded
    if batch_size_across_dp is not None:
        batch_size_across_dp[self.dp_rank] = input_batch_size

    common_attn_metadata.num_actual_tokens = batch_size
    common_attn_metadata.max_query_len = 1
    common_attn_metadata.query_start_loc = self.arange[:batch_size + 1]
    common_attn_metadata.query_start_loc_cpu = torch.from_numpy(
        self.token_arange_np[:batch_size + 1]).clone()
    for _ in range(self.num_speculative_tokens - 1):
        # Update the inputs.
        # cast to int32 is crucial when eagle model is compiled.
        # tensor.argmax() returns int64 by default.
        input_ids = draft_token_ids_list[-1].int()
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
            positions = positions[:input_ids.size(0)]
            positions += 1
            exceeds_max_model_len = positions >= self.max_model_len
            clamped_positions = torch.where(exceeds_max_model_len, 0,
                                            positions)
        # For data integrity when async scheduling, we shouldn't use in place
        # operations in case they are modified in next step's `prepare_input`
        # of main model.
        # Increment the sequence lengths.
        common_attn_metadata.seq_lens += 1
        # For the requests that exceed the max model length, we set the
        # sequence length to 1 to minimize their overheads in attention.
        common_attn_metadata.seq_lens.masked_fill_(
            exceeds_max_model_len.view(-1), 1)

        # Compute the slot mapping.
        if self.uses_mrope:
            # all dimensions of positions are the same
            block_numbers = clamped_positions[0] // self.block_size
        else:
            block_numbers = clamped_positions // self.block_size
        block_ids = common_attn_metadata.block_table_tensor.gather(
            dim=1, index=block_numbers.view(-1, 1))
        block_ids = block_ids.view(-1)
        if self.uses_mrope:
            common_attn_metadata.slot_mapping = (
                block_ids * self.block_size +
                clamped_positions[0] % self.block_size)
        else:
            common_attn_metadata.slot_mapping = (
                block_ids * self.block_size +
                clamped_positions % self.block_size)
        # Mask out the slot mappings that exceed the max model length.
        # Otherwise, the KV cache will be inadvertently updated with the
        # padding tokens.
        common_attn_metadata.slot_mapping.masked_fill_(exceeds_max_model_len,
                                                       PADDING_SLOT_ID)

        # Rebuild attention metadata
        extra_attn_metadata_args = {}
        extra_attn_metadata_args[
            "num_tokens"] = common_attn_metadata.seq_lens.cpu().numpy()
        extra_attn_metadata_args["positions"] = positions.cpu()
        extra_attn_metadata_args["batch_pad"] = \
            self.runner.bucketing_manager.find_decode_batch_bucket(batch_size)
        attn_metadata = attn_metadata_builder.build(
            common_prefix_len=0,
            common_attn_metadata=common_attn_metadata,
            fast_build=True,
            **extra_attn_metadata_args)

        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata

        if self.supports_mm_inputs:
            self.inputs_embeds[:batch_size] = self.model.embed_input_ids(
                input_ids)
            input_ids = None
        else:
            inputs_embeds = None

        # NOTE(RBLN): A temporary workaround;
        # reshapes input tensors in the same way as the RBLN model runner.
        num_reqs = self.runner.input_batch.num_reqs
        input_ids = input_ids.view(num_reqs, -1)
        positions = positions.view(num_reqs, -1)
        if not is_prefills[0]:
            input_ids = rbln_utils.pad(input_ids, 0, self.runner.max_num_seqs)
            positions = rbln_utils.pad(positions, -2, self.runner.max_num_seqs)
        hidden_states = hidden_states.unflatten(0, input_ids.shape)

        # Run the model.
        with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=input_batch_size,
                num_tokens_across_dp=batch_size_across_dp,
        ):
            if per_layer_attn_metadata is not None:
                for attn_metadata in per_layer_attn_metadata.values():
                    attn_metadata.kv_caches = kv_caches

            ret_hidden_states = self.model(
                input_ids=input_ids,
                positions=positions,
                hidden_states=hidden_states,
                inputs_embeds=inputs_embeds,
            )
            if self.method == "mtp":
                last_hidden_states = ret_hidden_states
                hidden_states = ret_hidden_states
            else:
                last_hidden_states, hidden_states = ret_hidden_states

        # NOTE(RBLN): A temporary workaround;
        # reshapes input tensors in the same way as the RBLN model runner.
        hidden_states = hidden_states.flatten(0, -2)
        last_hidden_states = last_hidden_states.flatten(0, -2)
        logits = self.model.compute_logits(last_hidden_states[:batch_size])
        draft_token_ids = logits.argmax(dim=-1)
        draft_token_ids_list.append(draft_token_ids)

    # [batch_size, num_speculative_tokens]
    draft_token_ids = torch.stack(draft_token_ids_list, dim=1)
    return draft_token_ids


EagleProposer.__init__ = __custom_init__
EagleProposer.propose = custom_propose
