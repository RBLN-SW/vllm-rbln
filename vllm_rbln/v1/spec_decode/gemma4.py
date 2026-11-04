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
import torch
import torch.nn as nn
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.kv_cache_interface import SlidingWindowSpec
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.v1.attention.kv_cache_bindings import (
    attach_kv_cache_bindings,
    build_kv_cache_forward_context_kwargs,
)
from vllm_rbln.v1.spec_decode.eagle import RBLNEagleProposer


class RBLNGemma4Proposer(RBLNEagleProposer):
    """Gemma4 MTP: an assistant whose attention only reads the target's cache.

    Every draft step queries from the last target position, so each one runs
    one query per request with the same metadata. Only that query matters: the
    drafter writes no K/V, so the target positions before it add nothing.
    """

    model_returns_tuple = Gemma4Proposer.model_returns_tuple
    _maybe_share_embeddings = Gemma4Proposer._maybe_share_embeddings
    _maybe_share_lm_head = Gemma4Proposer._maybe_share_lm_head
    _setup_gemma4_kv_sharing = Gemma4Proposer._setup_gemma4_kv_sharing
    validate_same_kv_cache_group = Gemma4Proposer.validate_same_kv_cache_group
    initialize_attn_backend = Gemma4Proposer.initialize_attn_backend

    def load_model(self, target_model: nn.Module) -> None:
        target_attn_layer_names = set(
            get_layers_from_vllm_config(
                self.vllm_config,
                AttentionLayerBase,  # type: ignore[type-abstract]
            ).keys()
        )
        super().load_model(target_model)
        self._setup_gemma4_kv_sharing(target_attn_layer_names)

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
        num_reqs = self.runner.input_batch.num_reqs
        if self.runner.is_intermediate_chunked_prefill:
            return next_token_ids.new_zeros(
                (num_reqs, self.num_speculative_tokens), dtype=torch.long
            )
        if token_indices_to_sample is None:
            token_indices_to_sample = common_attn_metadata.query_start_loc[1:] - 1
        token_indices_to_sample = token_indices_to_sample[:num_reqs].long()
        positions = target_positions[token_indices_to_sample]
        hidden_states = target_hidden_states.view(-1, self.hidden_size)[
            token_indices_to_sample
        ]
        draft_token_ids = next_token_ids[:num_reqs]

        per_layer_attn_metadata, num_reqs_padded, forward_kwargs = (
            self._build_draft_step(num_reqs, positions)
        )
        draft_token_ids_list = []
        for _ in range(self.num_speculative_tokens):
            self.input_ids[:num_reqs] = draft_token_ids.int()
            input_ids, staged_positions, staged_hidden_states, _ = self._preprocess(
                num_reqs,
                num_reqs_padded,
                num_reqs,
                positions,
                hidden_states,
                is_prefill=False,
            )
            with set_forward_context(per_layer_attn_metadata, **forward_kwargs):
                hidden_states, draft_ids = self.model_executable(
                    input_ids=input_ids,
                    positions=staged_positions,
                    hidden_states=staged_hidden_states,
                    inputs_embeds=None,
                    token_indices_to_sample=None,
                )
            hidden_states = hidden_states[:num_reqs]
            draft_token_ids = self._to_target_token_ids(draft_ids[:num_reqs])
            draft_token_ids_list.append(draft_token_ids)
        return torch.stack(draft_token_ids_list, dim=1)

    @torch.inference_mode()
    def dummy_run(
        self,
        num_reqs: int,
        num_tokens_per_req: int,
        is_prefill: bool,
        *,
        num_padded_tokens: int | None = None,
    ) -> None:
        positions = self.positions[:num_reqs]
        per_layer_attn_metadata, num_reqs_padded, forward_kwargs = (
            self._build_draft_step(num_reqs, positions)
        )
        input_ids, staged_positions, hidden_states, _ = self._preprocess(
            num_reqs,
            num_reqs_padded,
            num_reqs,
            positions,
            self.hidden_states[:num_reqs],
            is_prefill=False,
        )
        with set_forward_context(per_layer_attn_metadata, **forward_kwargs):
            self.model_executable(
                input_ids=input_ids,
                positions=staged_positions,
                hidden_states=hidden_states,
                inputs_embeds=None,
                token_indices_to_sample=None,
            )

    def _build_draft_step(
        self, num_reqs: int, positions: torch.Tensor
    ) -> tuple[dict[str, object], int, dict]:
        """The attention metadata every draft step of this batch shares.

        Each query sits at its request's last target position and sees the
        keys up to it, within the window on a sliding layer. The op's write
        position is one past it, where the zero K/V do no harm.
        """
        batch_desc, num_tokens_across_dp = self._determine_batch_execution_and_padding(
            num_reqs, num_reqs, False, first_pass=False
        )
        num_reqs_padded = batch_desc.num_reqs_padded
        query_start_loc = self.arange[: num_reqs + 1]
        write_positions = torch.clamp(positions + 1, max=self.max_model_len - 1)
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            builder = attn_group.get_metadata_builder()
            gid = attn_group.kv_cache_group_id
            block_table = self.runner.input_batch.block_table[gid].get_cpu_tensor()
            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc,
                seq_lens=write_positions + 1,
                num_reqs=num_reqs,
                num_actual_tokens=num_reqs,
                max_query_len=1,
                max_seq_len=self.max_model_len,
                block_table_tensor=block_table[:num_reqs],
                slot_mapping=torch.tensor(0),  # dummy
            )
            attn_metadata = builder.build(
                common_attn_metadata=common_attn_metadata,
                positions=write_positions,
                is_prefill=False,
                batch_pad=num_reqs_padded,
                skip_attn_masks=True,
                is_prompt_sorted=self.runner.sort_batch_by_length,
            )
            if isinstance(builder.kv_cache_spec, SlidingWindowSpec):
                # sliding_window_attention_v1's key frame: the strip of
                # blocks from the one holding the window's first key.
                window = builder.kv_cache_spec.sliding_window
                block_size = builder.block_size
                num_blocks = (window + block_size - 1) // block_size + 1
                first_key = (
                    torch.clamp(write_positions - window + 1, min=0)
                    // block_size
                    * block_size
                )
                keys = first_key[:, None] + torch.arange(num_blocks * block_size)
                visible = (keys <= positions[:, None]) & (
                    keys > positions[:, None] - window
                )
                masks = torch.zeros(num_reqs_padded, 1, 1, num_blocks * block_size)
                masks[:num_reqs, 0, 0] = visible.to(masks.dtype)
            else:
                # gqa masks keys over the whole block table.
                keys = torch.arange(block_table.shape[1] * builder.block_size)
                visible = keys[None, :] <= positions[:, None]
                masks = torch.zeros(num_reqs_padded, 1, 1, 1, keys.numel())
                masks[:num_reqs, 0, 0, 0] = visible.to(masks.dtype)
            attn_metadata.attn_masks = builder._stage(
                masks.to(torch.float16 if builder.enforce_eager else torch.float32),
                "attn_masks",
            )
            attach_kv_cache_bindings(
                attn_metadata,
                self.runner.kv_caches,
                self.runner.kv_cache_bases,
                self.runner.kv_cache_view_infos,
            )
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        forward_kwargs = dict(
            vllm_config=self.vllm_config,
            num_tokens=num_reqs,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=batch_desc.num_tokens_padded,
            **build_kv_cache_forward_context_kwargs(self.runner.kv_cache_bases),
        )
        return per_layer_attn_metadata, num_reqs_padded, forward_kwargs
