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
        assert self.kv_block_adapter is not None
        self.setup_decoder_mixin(
            attn_impl=self.attn_impl,
            vocab_size=self.model_config.get_vocab_size,
            use_multiple_decoder=getattr(
                self.model.rbln_config, "use_multiple_decoder", False
            ),
            default_batch_size=self.scheduler_config.max_num_seqs,
            decoder_batch_sizes=self.model.rbln_config.decoder_batch_sizes,
            num_blocks=self.kv_block_adapter._estimated_num_blocks(),
        )

        self.is_hybrid = getattr(self.model.rbln_config, "cache_impl", None) == "hybrid"

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables
        local_block_tables = model_input.local_block_tables
        assert local_block_tables is not None

        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt

        kwargs = self.preprocess_for_decoder(
            is_prompt, block_tables, input_ids, cache_position
        )

        padded_batch_size = kwargs.pop("padded_batch_size", self.decoder_batch_size)
        cache_position = kwargs.pop("cache_position")
        input_ids = kwargs.pop("input_ids")
        block_tables = kwargs.pop("block_tables")

        if is_prompt:
            if self.model.prefill_decoder is None:
                raise version_error
            output = self.model.prefill_decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                local_block_tables=local_block_tables,
                block_tables=block_tables if self.is_hybrid else None,
            )
            logits = output.logits
        else:
            self.model.decoder = self.model.decoders[padded_batch_size]
            logits = self.model.decoder(
                input_ids=input_ids,
                cache_position=cache_position,
                local_block_tables=self.pad_local_block_tables(
                    local_block_tables, padded_batch_size
                ),
                block_tables=block_tables if self.is_hybrid else None,
            ).logits
            logits = logits[:request_nums]
        return logits
