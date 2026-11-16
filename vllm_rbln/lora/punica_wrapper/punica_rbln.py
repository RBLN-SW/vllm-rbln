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

from typing import Union, final

import torch
from vllm.lora.punica_wrapper.punica_base import PunicaWrapperBase

from vllm_rbln.lora.inputs import LoRAInputs
from vllm_rbln.lora.mask import LoRAMask


@final
class PunicaWrapperRBLN(PunicaWrapperBase):
    def __init__(
        self,
        max_num_batched_tokens: int,
        max_batches: int,
        device: Union[torch.device, str],
        **kwargs,
    ):
        PunicaWrapperBase.__init__(self, max_num_batched_tokens, max_batches, device)
        self._embeddings_indices.fill_(0)

    def add_shrink(
        self,
        y: Union[tuple[torch.Tensor, ...], torch.Tensor],
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        scale: float,
        **kwargs,
    ) -> torch.Tensor | None:
        """
        Performs GEMM for multiple slices of lora_a.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (x @ lora_a_stacked[i]) * scale
        """
        raise NotImplementedError

    def add_expand(
        self,
        y: torch.Tensor,
        x: Union[tuple[torch.Tensor, ...], torch.Tensor],
        lora_b_stacked: tuple[torch.Tensor, ...],
        output_slices: tuple[int, ...],
        offset_start: int = 0,
        add_inputs: bool = True,
        **kwargs,
    ) -> torch.Tensor | None:
        """
        Performs GEMM and bias addition for multiple slices of lora_b.

        Semantics:
        offset = offset_start
        for i in range(len(lora_b_stacked)):
            slice = output_slices[i]
            y[:, offset:offset+slice] += x[i] @ lora_b_stacked[i]
            offset += slice
        """
        raise NotImplementedError

    def add_lora_embedding(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        add_inputs: bool = True,
        **kwargs,
    ) -> torch.Tensor:
        """
        Applies lora specifically for VocabParallelEmbeddingWithLoRA
        and this layer only requires the expand operation.

        Semantics:
        y += x @ lora_b_stacked
        """
        max_loras = lora_b_stacked.size(0)
        x = x.repeat(1, max_loras)  # [num_tokens, rank * max_loras]
        x = x * LoRAMask.get_lora_mask()
        lora_b_w = lora_b_stacked[:, 0, :, :].transpose(
            1, 2
        )  # [max_loras, rank, hidden_size]
        lora_b_w = lora_b_w.reshape(
            -1, lora_b_w.shape[2]
        )  # [max_loras * rank, hidden_size]
        out = x @ lora_b_w
        y += out

        return y

    def add_lora_linear(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: tuple[torch.Tensor, ...],
        lora_b_stacked: tuple[torch.Tensor, ...],
        scale: float,
        output_slices: tuple[int, ...],
        *,
        buffer: tuple[torch.Tensor, ...] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Applicable to linear-related lora.

        Semantics:
        for i in range(len(lora_a_stacked)):
            y[i] += (
                x[i].unsqueeze(0)
                @ lora_a_stacked[indicies[i], layer_idx, :, :]
                @ lora_b_stacked[indicies[i], layer_idx, :, :]
                * scale
            ).squeeze(0)
        """
        slice_offset = 0

        for slice_idx in range(len(output_slices)):
            lora_a_w = lora_a_stacked[slice_idx][:, 0, :, :]
            lora_a_w = lora_a_w.reshape(-1, lora_a_w.shape[2]).transpose(
                0, 1
            )  # [h1, max_loras * rank]
            lora_b_w = lora_b_stacked[slice_idx][:, 0, :, :].transpose(1, 2)
            lora_b_w = lora_b_w.reshape(-1, lora_b_w.shape[2])  # [max_loras * rank, h1]
            out = x @ lora_a_w  # [bs * seq_len, max_loras * rank]
            out = out * LoRAMask.get_lora_mask()
            out = out @ lora_b_w  # [bs * seq_len, h2]
            y[:, slice_offset : slice_offset + output_slices[slice_idx]] += out * scale
            slice_offset += output_slices[slice_idx]

        return y

    def add_lora_logits(
        self,
        y: torch.Tensor,
        x: torch.Tensor,
        lora_a_stacked: torch.Tensor,
        lora_b_stacked: torch.Tensor,
        scale,
        *,
        buffer: torch.Tensor | None = None,
        **kwargs,
    ) -> torch.Tensor | None:
        """
        Applies lora specifically for LogitsProcessorWithLoRA.

        Semantics:
        buffer = (x @ lora_a_stacked) * scale
        y += buffer @ lora_b_stacked
        """
        lora_a_w = lora_a_stacked[:, 0, :, :]
        lora_a_w = lora_a_w.reshape(-1, lora_a_w.shape[2]).transpose(0, 1)
        lora_b_w = lora_b_stacked[:, 0, :, :].transpose(1, 2)
        lora_b_w = lora_b_w.reshape(-1, lora_b_w.shape[2])
        out = x @ lora_a_w
        out = out * LoRAMask.get_lora_mask()[: x.shape[0]]
        out = out @ lora_b_w
        y[:, : out.shape[1]] += out * scale

        return y

    @property
    def sampler_indices_padded(self) -> torch.Tensor:
        return LoRAInputs.get_sampler_indices_padded()
