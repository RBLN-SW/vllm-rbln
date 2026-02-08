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
from vllm.model_executor.models.interfaces import SupportsLoRA

from vllm_rbln.logger import init_logger

from .base import ModelInputForRBLN, version_error
from .model_base import RBLNOptimumDecoderMixin, RBLNOptimumModelBase

logger = init_logger(__name__)


class RBLNOptimumForCausalLM(RBLNOptimumModelBase, RBLNOptimumDecoderMixin,
                             SupportsLoRA):

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

    def forward(self, model_input: ModelInputForRBLN,
                **kwargs) -> torch.Tensor:
        input_ids = model_input.input_tokens
        cache_position = model_input.input_positions
        block_tables = model_input.block_tables
        dummy_block = model_input.dummy_block

        request_nums = input_ids.shape[0]
        is_prompt = model_input.is_prompt

        kwargs = self.preprocess_for_decoder(is_prompt,
                                             block_tables,
                                             input_ids,
                                             cache_position,
                                             dummy_block=dummy_block)
        padded_batch_size = kwargs.pop("padded_batch_size",
                                       self.decoder_batch_size)

        if is_prompt:
            if self.model.prefill_decoder is None:
                raise version_error
            if model_input.cached_block_tables:
                self._copy_cached_kv_blocks(self.model.prefill_decoder,
                                            model_input.cached_block_tables,
                                            model_input.cached_lengths,
                                            block_tables)
            return self.model.prefill_decoder(**kwargs).logits
        else:
            self.model.decoder = self.model.decoders[padded_batch_size]

            logits = self.model.decoder(**kwargs).logits
            if self.attn_impl != "flash_attn":
                return logits[:request_nums]

            return logits[:model_input.block_tables.shape[0]]
