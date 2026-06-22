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
from vllm.distributed import (
    tensor_model_parallel_all_gather,
    tensor_model_parallel_gather,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.vocab_parallel_embedding import VocabParallelEmbedding

import vllm_rbln.rbln_envs as envs


def logits_processor_get_logits(
    self,
    hidden_states: torch.Tensor,
    lm_head: VocabParallelEmbedding,
    embedding_bias: torch.Tensor | None,
) -> torch.Tensor | None:
    # Get the logits for the next tokens.
    logits = lm_head.quant_method.apply(lm_head, hidden_states, bias=embedding_bias)
    return logits


def logits_processor_gather_logits(self, logits: torch.Tensor) -> torch.Tensor:
    """gather/all-gather the logits tensor across model parallel group."""
    if self.use_all_gather:
        logits = tensor_model_parallel_all_gather(logits)
    else:
        # None may be returned for rank > 0
        logits = tensor_model_parallel_gather(logits)

    # Remove paddings in vocab (if any).
    if logits is not None:
        logits = logits[..., : self.org_vocab_size]
    return logits


if not envs.VLLM_RBLN_LOGITS_ALL_GATHER:
    LogitsProcessor._get_logits = logits_processor_get_logits
    LogitsProcessor._gather_logits = logits_processor_gather_logits
