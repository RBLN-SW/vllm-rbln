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

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.model_executor.models.qwen2 import Qwen2ForCausalLM
from vllm.model_executor.models.utils import PPMissingLayer, maybe_prefix

qwen2_for_causal_lm_init = Qwen2ForCausalLM.__init__


def __qwen2_for_causal_lm__init__(
    self,
    vllm_config: VllmConfig,
    prefix: str = "",
):
    qwen2_for_causal_lm_init(self, vllm_config=vllm_config, prefix=prefix)
    config = self.config
    quant_config = self.quant_config

    if get_pp_group().is_last_rank:
        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )

        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)
    else:
        self.lm_head = PPMissingLayer()


Qwen2ForCausalLM.__init__ = __qwen2_for_causal_lm__init__
