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
import torch
import torch.nn as nn
from vllm.config import get_layers_from_vllm_config
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.v1.attention.backends.utils import CommonAttentionMetadata
from vllm.v1.spec_decode.gemma4 import Gemma4Proposer

from vllm_rbln.forward_context import set_forward_context
from vllm_rbln.model_executor.models.gemma4 import split_gate_up_weights
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
        split_gate_up_weights(self.model)
        self._setup_gemma4_kv_sharing(target_attn_layer_names)

        num_steps = self.num_speculative_tokens

        def draft_steps(
            input_ids: torch.Tensor,
            positions: torch.Tensor,
            hidden_states: torch.Tensor,
        ) -> torch.Tensor:
            # Every step queries the same position through the same metadata,
            # so the whole draft runs as one graph and only the ids leave it.
            draft_ids = []
            for _ in range(num_steps):
                sample_hidden_states, next_hidden_states = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    hidden_states=hidden_states,
                    inputs_embeds=None,
                )
                logits = self.model.compute_logits(
                    sample_hidden_states.view(-1, sample_hidden_states.shape[-1])
                )
                ids = torch.ops.rbln.argmax(logits)
                draft_ids.append(ids)
                input_ids = ids.view(input_ids.shape)
                hidden_states = next_hidden_states.view(hidden_states.shape)
            return torch.stack(draft_ids, dim=1)

        self.model_executable = self._compile_drafter(draft_steps)

    def propose(
        self,
        target_token_ids: torch.Tensor,
        target_positions: torch.Tensor,
        target_hidden_states: torch.Tensor,
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        mm_embed_inputs: tuple[list[tuple[int, int, torch.Tensor]], torch.Tensor]
        | None = None,
        num_rejected_tokens: torch.Tensor | None = None,
        target_mrope_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert target_mrope_positions is None, "Gemma4 does not use M-RoPE"
        num_reqs = self.runner.input_batch.num_reqs
        if (
            self.runner.is_intermediate_chunked_prefill
            or self.runner.is_strict_kv_producer
        ):
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
        per_layer_attn_metadata, num_reqs_padded, forward_kwargs = (
            self._build_draft_step(num_reqs, positions)
        )
        self.input_ids[:num_reqs] = next_token_ids[:num_reqs].int()
        input_ids, staged_positions, staged_hidden_states, _ = self._preprocess(
            num_reqs,
            num_reqs_padded,
            num_reqs,
            positions,
            hidden_states,
            is_prefill=False,
        )
        with set_forward_context(per_layer_attn_metadata, **forward_kwargs):
            draft_ids = self.model_executable(
                input_ids=input_ids,
                positions=staged_positions,
                hidden_states=staged_hidden_states,
            )
        return self._to_target_token_ids(draft_ids[:num_reqs])

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
            )

    def _build_draft_step(
        self, num_reqs: int, positions: torch.Tensor
    ) -> tuple[dict[str, object], int, dict]:
        """The attention metadata every draft step of this batch shares.

        Each query sits at its request's last target position and sees the
        keys up to it, within the window on a sliding layer. The KV-shared
        layers read the target's cache there and write nothing.
        """
        batch_desc, num_tokens_across_dp = self._determine_batch_execution_and_padding(
            num_reqs, num_reqs, False, first_pass=False
        )
        num_reqs_padded = batch_desc.num_reqs_padded
        query_start_loc = self.arange[: num_reqs + 1]
        assert torch.all(positions < self.max_model_len)
        per_layer_attn_metadata: dict[str, object] = {}
        for attn_group in self.draft_attn_groups:
            builder = attn_group.get_metadata_builder()
            gid = attn_group.kv_cache_group_id
            block_table = self.runner.input_batch.block_table[gid].get_cpu_tensor()
            common_attn_metadata = CommonAttentionMetadata(
                query_start_loc=query_start_loc,
                query_start_loc_cpu=query_start_loc,
                seq_lens=positions + 1,
                num_reqs=num_reqs,
                num_actual_tokens=num_reqs,
                max_query_len=1,
                max_seq_len=self.max_model_len,
                block_table_tensor=block_table[:num_reqs],
                slot_mapping=torch.tensor(0),  # dummy
            )
            attn_metadata = builder.build(
                common_attn_metadata=common_attn_metadata,
                positions=positions,
                is_prefill=False,
                batch_pad=num_reqs_padded,
                skip_attn_masks=True,
                is_prompt_sorted=self.runner.sort_batch_by_length,
            )
            self.runner.bind_kv_cache_to_metadata(attn_metadata)
            for layer_name in attn_group.layer_names:
                per_layer_attn_metadata[layer_name] = attn_metadata
        forward_kwargs = dict(
            vllm_config=self.vllm_config,
            num_tokens=num_reqs,
            num_tokens_across_dp=num_tokens_across_dp,
            num_padded_tokens=batch_desc.num_tokens_padded,
            **self.runner.kv_cache_forward_context_kwargs(),
        )
        return per_layer_attn_metadata, num_reqs_padded, forward_kwargs
