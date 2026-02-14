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

from typing import Union

import torch
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.model_executor.layers.pooler import DispatchPooler, Pooler
from vllm.model_executor.models import VllmModelForPooling
from vllm.tasks import PoolingTask
from vllm.v1.outputs import PoolerOutput
from vllm.v1.pool.metadata import PoolingMetadata

from .base import ModelInputForRBLN
from .model_base import RBLNOptimumModelBase

logger = init_logger(__name__)


class RBLNClassifierPooler(Pooler):
    """
    A pooler for RBLN models that simply wraps pre-processed
    hidden states into vLLM's PoolerOutput format.
    """

    def __init__(self) -> None:
        super().__init__()

    def get_supported_tasks(self) -> set[PoolingTask]:
        return {"classify", "score"}

    def forward(
        self,
        hidden_states: Union[torch.Tensor, list[torch.Tensor]],
        pooling_metadata: PoolingMetadata,
    ) -> PoolerOutput:
        # RBLN models return already pooled/processed states for classification
        # No additional pooling needed - just format for vllm compatibility
        return hidden_states


class RBLNOptimumForEncoderModel(RBLNOptimumModelBase, VllmModelForPooling):
    PAD_TOKEN_ID = 0
    is_pooling_model = True
    pooler: Pooler

    def __init__(
        self,
        vllm_config: VllmConfig,
    ) -> None:
        super().__init__(vllm_config=vllm_config)
        pooler_config = vllm_config.model_config.pooler_config
        # NOTE:
        # In vLLM, the default pooling type is set to LAST token pooling,
        # which is suitable for decoder or encoder-decoder models.
        # However, for encoder-only models (e.g., BERT),
        # CLS token pooling is more appropriate.
        # Therefore, we override the pooling type to CLS here.
        pooler_config.pooling_type = "CLS"
        assert pooler_config is not None
        # https://github.com/vllm-project/vllm/blob/72506c98349d6bcd32b4e33eec7b5513453c1502/docs/models/pooling_models.md?plain=1#L312
        # encode task is split into `token_embed` and `token_classify` tasks
        self.pooler = DispatchPooler(
            {
                "token_embed": Pooler.for_token_embed(pooler_config),
                "token_classify": Pooler.for_token_classify(pooler_config),
                "embed": Pooler.for_embed(pooler_config),
                "classify": RBLNClassifierPooler(),
                "score": RBLNClassifierPooler(),
            },
        )

    def is_classification_arch(self):
        architectures = getattr(
            self.model_config.hf_config,
            "architectures",
            [],
        )
        return len(architectures) > 0 and "Classification" in architectures[0]

    def preprocess(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ):
        batch_size, seq_len = input_ids.shape
        target_batch_size = self.batch_size

        def pad_if_needed(
            tensor: torch.Tensor | None,
        ) -> torch.Tensor | None:
            if tensor is None:
                return None

            if tensor.size(1) > self.rbln_model_config.max_seq_len:
                tensor = tensor[:, : self.rbln_model_config.max_seq_len]
            elif tensor.size(1) < self.rbln_model_config.max_seq_len:
                padded_tensor = torch.zeros(
                    batch_size,
                    self.rbln_model_config.max_seq_len,
                    dtype=tensor.dtype,
                )
                padded_tensor[:, : tensor.size(1)] = tensor
                tensor = padded_tensor

            if tensor.size(0) >= target_batch_size:
                return tensor
            padded = tensor.new_zeros((target_batch_size, tensor.size(1)))
            padded[:batch_size] = tensor
            return padded

        return (
            pad_if_needed(input_ids),
            pad_if_needed(positions),
        )

    def forward(self, model_input: ModelInputForRBLN, **kwargs) -> torch.Tensor:
        input_ids, positions = self.preprocess(
            model_input.input_tokens,
            model_input.input_positions,
        )

        max_position = torch.max(positions, dim=1).indices
        position_indices = torch.arange(
            positions.shape[1], device=positions.device
        ).unsqueeze(0)
        attention_mask = (position_indices <= max_position.unsqueeze(1)).long()
        request_nums = input_ids.shape[0]
        kwargs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }

        model_input_names = getattr(self.rbln_model_config, "model_input_names", None)
        if model_input_names is not None:
            rbln_model_input_names = self.rbln_model_config.model_input_names
            if "token_type_ids" in rbln_model_input_names:
                kwargs["token_type_ids"] = torch.zeros_like(input_ids)

        embeds = self.model.forward(**kwargs)

        hidden_states = embeds[0]
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]

        if not self.is_classification_arch():
            # Depad hidden_states for the number of requests and length.
            hidden_states = hidden_states[:request_nums]
            prompt_lens = max_position[:request_nums] + 1

            new_hidden_states = []
            for idx, prompt_len in enumerate(prompt_lens):
                new_hidden_states.append(hidden_states[idx, :prompt_len])
            hidden_states = torch.cat(new_hidden_states, dim=0)
        else:
            assert hidden_states.dim() == 2, (
                f"We expected the shape to be dim 2 ([batch, num_labels]), "
                f"but the current output is dim {hidden_states.dim()}."
            )
            hidden_states = hidden_states[:request_nums].squeeze(-1)

        return hidden_states
