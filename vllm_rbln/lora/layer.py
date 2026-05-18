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
import torch.nn.functional as F
from vllm.lora.layers import VocabParallelEmbeddingWithLoRA
from vllm.lora.layers.base_linear import BaseLinearLayerWithLoRA


def base_linear_patched_apply(
    self: BaseLinearLayerWithLoRA, x: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    output = self.base_layer.quant_method.apply(self.base_layer, x, bias)
    output_org_shape = output.shape

    x = x.reshape(
        -1, x.shape[-1]
    )  # [bs, seq_len, hidden_size] -> [bs * seq_len, hidden_size]
    output = output.reshape(
        -1, output.shape[-1]
    )  # [bs, seq_len, hidden_size] -> [bs * seq_len, hidden_size]

    lora_output: torch.Tensor = self.punica_wrapper.add_lora_linear(
        output, x, self.lora_a_stacked, self.lora_b_stacked, 1.0, self.output_slices
    )

    return lora_output.view(output_org_shape)


def vocab_parallel_embedding_patched_forward(
    self: VocabParallelEmbeddingWithLoRA, x: torch.Tensor
) -> torch.Tensor:
    # NOTE: It assumes that the batch size of prefill phase is always 1.
    is_prefill = x.shape[0] == 1
    added_tokens_mask = torch.where(x > self.base_layer.org_vocab_size - 1, 1, 0)
    num_tokens = x.size(1) if is_prefill else x.size(0)
    indices_1 = self.punica_wrapper._embeddings_indices[1][:num_tokens]

    if not is_prefill:
        indices_1 = indices_1.unsqueeze(1)
    full_lora_a_embeddings = F.embedding(
        x + indices_1,
        self.lora_a_stacked_2d,
    )
    full_output = self.base_layer.forward(x + (indices_1 * added_tokens_mask))

    full_output_org = full_output
    if full_output.ndim == 3:
        full_output = full_output.view(full_output.shape[0] * full_output.shape[1], -1)
    if full_lora_a_embeddings.ndim == 3:
        full_lora_a_embeddings = full_lora_a_embeddings.view(
            full_lora_a_embeddings.shape[0] * full_lora_a_embeddings.shape[1],
            -1,
        )

    lora_output: torch.Tensor = self.punica_wrapper.add_lora_embedding(
        full_output, full_lora_a_embeddings, self.lora_b_stacked, add_inputs=True
    )

    return lora_output.view_as(full_output_org)


BaseLinearLayerWithLoRA.apply = base_linear_patched_apply
VocabParallelEmbeddingWithLoRA.forward = vocab_parallel_embedding_patched_forward
