# Copyright 2026 Rebellions Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from copy import copy

import torch
from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dflash import _dflash_layer_causal

from vllm_rbln.v1.spec_decode.dflash import RBLNDFlashProposer


class RBLNDSparkProposer(RBLNDFlashProposer):
    """DSpark drafter on RBLN: the DFlash draft pass, then a Markov head."""

    def __init__(self, vllm_config: VllmConfig, device: torch.device, runner=None):
        speculative_config = vllm_config.speculative_config
        hf_config = speculative_config.draft_model_config.hf_config
        if getattr(hf_config, "dspark_draft_topk", None) is not None:
            raise NotImplementedError(
                "DSpark top-k drafting is not supported on RBLN: the draft pick "
                "is a greedy argmax over the whole draft vocabulary."
            )
        causality = {
            _dflash_layer_causal(hf_config, index)
            for index in range(hf_config.num_hidden_layers)
        }
        if len(causality) > 1:
            raise NotImplementedError(
                "A DSpark drafter mixing causal and non-causal layers is not "
                "supported on RBLN: the draft attention config has one "
                "causality setting."
            )

        as_dflash = copy(speculative_config)
        as_dflash.method = "dflash"
        view = copy(vllm_config)
        view.speculative_config = as_dflash
        super().__init__(view, device, runner)

        if getattr(hf_config, "sample_from_anchor", True):
            self.num_query_per_req = self.num_speculative_tokens

    def _check_rope_style(self, target_model) -> None:
        """Keep the drafter's own RoPE, as upstream runs DSpark."""

    def _draft_token_ids(
        self, hidden_states: torch.Tensor, input_ids: torch.Tensor
    ) -> torch.Tensor:
        """Greedy Markov drafting: each draft biases the next one's logits."""
        num_reqs = input_ids.shape[0]
        logits = self.model.compute_draft_logits(hidden_states).view(
            num_reqs, self.num_speculative_tokens, -1
        )
        previous = input_ids[:, 0]
        draft_ids = []
        for step in range(self.num_speculative_tokens):
            bias = self.model.markov_bias(self.model.markov_embed(previous))
            draft_id = torch.ops.rbln.argmax(logits[:, step] + bias)
            draft_ids.append(draft_id)
            previous = self.model.map_draft_to_target(draft_id)
        return torch.stack(draft_ids, dim=1).view(-1)
