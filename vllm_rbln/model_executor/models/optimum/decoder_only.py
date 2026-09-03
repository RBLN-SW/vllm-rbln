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


class RBLNOptimumForCausalLM(
    RBLNOptimumModelBase, RBLNOptimumDecoderMixin, SupportsLoRA
):
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

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        request_nums = model_input.input_tokens.shape[0]

        if model_input.is_prompt:
            if self.model.prefill_decoder is None:
                raise version_error
            prefill_inputs = self.prepare_prefill_inputs(model_input)
            return self.model.prefill_decoder(
                input_ids=prefill_inputs.input_ids,
                cache_position=prefill_inputs.cache_position,
                block_tables=prefill_inputs.block_tables,
            ).logits
        else:
            decode_inputs = self.prepare_decode_inputs(
                model_input, dummy_block=model_input.dummy_block
            )
            self.model.decoder = self.model.decoders[decode_inputs.padded_batch_size]

            logits = self.model.decoder(
                input_ids=decode_inputs.input_ids,
                cache_position=decode_inputs.cache_position,
                block_tables=decode_inputs.block_tables,
            ).logits
            if self.attn_impl != "flash_attn":
                return logits[:request_nums]

            return logits[: model_input.block_tables.shape[0]]
