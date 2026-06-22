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
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from vllm.config import VllmConfig
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.spec_decode.medusa import MedusaProposer

import vllm_rbln.envs as envs
from vllm_rbln.compilation import (
    build_process_group_dict,
    compile,
    create_compile_context,
)

if TYPE_CHECKING:
    from rebel.compile_context import CompileContext


class RBLNMedusaProposer(MedusaProposer):
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        compile_context: "CompileContext | None" = None,
    ) -> None:
        super().__init__(vllm_config, device)

        self.max_num_seqs = self.vllm_config.scheduler_config.max_num_seqs
        self.hidden_states = torch.zeros(
            self.max_num_seqs, self.hidden_size, device=self.device, dtype=self.dtype
        )
        self.compile_context = compile_context or create_compile_context(
            use_weight_sharing=True
        )

    def load_model(self, target_model: nn.Module) -> None:
        super().load_model(target_model)

        def model_wrapper(target_hidden_states: torch.Tensor) -> torch.Tensor:
            hidden_states = self.model(target_hidden_states)
            logits = self.model.compute_logits(hidden_states)
            return logits

        if (
            self.vllm_config.speculative_config.enforce_eager
            or not envs.VLLM_RBLN_COMPILE_MODEL
        ):
            self.model_executable = model_wrapper
        else:
            self.model_executable = compile(
                model_wrapper,
                dynamic=False,
                fullgraph=True,
                compile_context=self.compile_context,
                tensor_parallel_size=envs.VLLM_RBLN_TP_SIZE,
                process_group_dict=build_process_group_dict(),
                guard_filter_fn=torch.compiler.keep_tensor_guards_unsafe,
                mode="strict" if envs.VLLM_RBLN_COMPILE_STRICT_MODE else "",
            )

    def propose(
        self,
        target_hidden_states: torch.Tensor,  # [B, H]
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        batch_size = target_hidden_states.shape[0]
        self.hidden_states[:batch_size] = target_hidden_states

        # Generate blocks and compute logits
        logits = self.model_executable(self.hidden_states)

        # Compute argmax for each Medusa head and stack into a single tensor
        # Shape: [batch_size, num_heads]
        draft_tokens = torch.stack(
            [logit[:batch_size].argmax(dim=-1) for logit in logits], dim=1
        )

        return draft_tokens

    @torch.inference_mode()
    def dummy_run(self) -> None:
        self.model_executable(self.hidden_states)
