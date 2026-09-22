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
"""Name the DFlash head's layers past the target's full depth.

The defect and the fix are `patches/llama_eagle3.py`'s, in the head that
`DFlashQwen3ForCausalLM` builds instead. This head is several layers deep, so a
per-rank offset puts a whole band of names inside another stage's, not one.
"""

from __future__ import annotations

from vllm.config import VllmConfig
from vllm.model_executor.models.qwen3_dflash import DFlashQwen3Model

from vllm_rbln.patches.registry import register_patch

_orig_model_init = DFlashQwen3Model.__init__


@register_patch(
    target="vllm.model_executor.models.qwen3_dflash.DFlashQwen3Model.__init__",
    reason=(
        "Name the head's layers past the target's full depth, not past this "
        "pipeline rank's count, which is what upstream's "
        "`get_num_layers(parallel_config)` gives; the two agree only at PP=1. "
        "Elsewhere the head's names land in a band another stage owns, and "
        "upstream merges the workers' KV specs by name and asserts they match.\n"
        "TODO(vllm-project/vllm#50514): delete once that lands and is released."
    ),
)
def patched_dflash_qwen3_model_init(
    self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
) -> None:
    _orig_model_init(
        self,
        vllm_config=vllm_config,
        start_layer_id=vllm_config.model_config.get_total_num_hidden_layers(),
        prefix=prefix,
    )
