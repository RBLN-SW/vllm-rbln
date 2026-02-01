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
from vllm.config import VllmConfig
from vllm.logger import init_logger

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase
from .optimum_attention import (AttentionManager, InnerAttentionEntry,
                                InnerAttentionStrategy, InnerR1, InnerR2)

logger = init_logger(__name__)


class RBLNOptimumSlidingWindowAttentionForCausalLM(
        RBLNOptimumModelBase,
        RBLNOptimumDecoderMixin,
):
    """
    Supports text-only generation models with:
    - Sliding window attention
      - `block_tables` is not used because there is no full attention layer.
      - `local_block_tables` is only used for
        the sliding window attention layer.
    - Hybrid attention (full + sliding window layers)
      - `block_tables` and `local_block_tables` are both used.

    Note: Gemma3 uses hybrid attention but is multi-modal,
            so it uses another exclusive class.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(self.model.rbln_config,
                                         "use_multiple_decoder", False),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

        self.strategy = InnerAttentionStrategy()
        self.attention_manager: AttentionManager[InnerAttentionStrategy,
                                                 InnerAttentionEntry, InnerR1,
                                                 InnerR2] = AttentionManager(
                                                     self.strategy)
        self.is_hybrid = getattr(self.model.rbln_config, "cache_impl",
                                 None) == "hybrid"

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables

        finished_requests_ids = model_input.finished_requests_ids
        running_requests_ids = model_input.running_requests_ids
        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt

        # In prefill phase, the length of list must be 1
        sliding_window_table_ids = self.attention_manager.get(
            is_prompt,
            self.decoder_batch_size,
            running_requests_ids,
            finished_requests_ids,
        )

        kwargs = self.preprocess_for_decoder(is_prompt, block_tables,
                                             input_ids, cache_position)

        padded_batch_size = kwargs.pop("padded_batch_size",
                                       self.decoder_batch_size)

        # [prefill] the length of the padded cache is calculated
        # during the forward pass and stored in self.sliding_window_table.
        # [decode] `cache_position` and `position_ids` are distinguished
        # due to the padding space reserved for the sliding window.
        cache_position = kwargs.pop("cache_position")
        input_ids = kwargs.pop("input_ids")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            if self.model.prefill_decoder is None:
                raise version_error
            prefill_batch_idx = sliding_window_table_ids[0]
            local_block_table_id = torch.tensor([prefill_batch_idx],
                                                dtype=torch.int16)
            output = self.model.prefill_decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                local_block_tables=local_block_table_id,
                block_tables=block_tables if self.is_hybrid else None,
            )
            logits = output.logits
            assert len(running_requests_ids) == 1
            self.attention_manager.add(
                running_requests_id=running_requests_ids[0],
                local_table_id=prefill_batch_idx,
            )
        else:
            self.model.decoder = self.model.decoders[padded_batch_size]
            local_block_table_id, cache_position = \
                self.attention_manager.preprocess(
                sliding_window_table_ids,
                cache_position,
                request_nums,
                padded_batch_size,
            )
            logits = self.model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                local_block_tables=local_block_table_id,
                block_tables=block_tables if self.is_hybrid else None,
            ).logits

        if not is_prompt:
            logits = logits[:request_nums]
        return logits
