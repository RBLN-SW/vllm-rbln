# Copyright 2026 Rebellions Inc. All rights reserved.
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
"""Name the DeepSeek EAGLE3 head's layers past the target's full depth.

The defect and the fix are `patches/llama_eagle3.py`'s, in the head that
`Eagle3DeepseekV2ForCausalLM` builds instead. `target_layer_count` does not move
with the name here: this head's decoder layer builds `DeepseekV2MLAAttention`
directly, so it never reaches the `llama.py` arithmetic that reads it.
"""

from vllm.config import VllmConfig
from vllm.model_executor.models.deepseek_eagle3 import DeepseekV2Eagle3Model

from vllm_rbln.patches import register_patch

# Captured at import time: the registry replaces targets outright, so wrapping
# upstream behaviour means holding the original here rather than copying its body.
_orig_model_init = DeepseekV2Eagle3Model.__init__


@register_patch(
    target="vllm.model_executor.models.deepseek_eagle3.DeepseekV2Eagle3Model.__init__",
    reason=(
        "Upstream names the EAGLE head's layers past the layer count on this "
        "pipeline rank, not past the target's full depth. The two agree at PP=1, so "
        "the name only goes wrong on a rank that does not start at zero -- where "
        "the drafter lives -- and the head then sorts in among the target's layers "
        "instead of landing after them. "
        "TODO(vllm-project/vllm#50514): delete once that lands and is released."
    ),
)
def patched_eagle3_deepseek_model_init(
    self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
) -> None:
    _orig_model_init(
        self,
        vllm_config=vllm_config,
        start_layer_id=vllm_config.model_config.get_total_num_hidden_layers(),
        prefix=prefix,
    )
