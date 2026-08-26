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
"""Name the EAGLE3 head's layers past the target's full depth.

Upstream names them past the count on *this* pipeline rank, which is only the
total when PP=1. See the patch's own reason string for what that breaks.

The draft->target vocab scatter is deliberately NOT patched here. Upstream builds
that index inside ``compute_logits`` from two input-independent operands, so
dynamo const-folds it into an anonymous constant that weight-free apply cannot
resolve by name -- an out-of-bounds host-op write that segfaults in KV warmup.
``RBLNEagleProposer`` already avoids it (``v1/spec_decode/eagle.py``): it stays in
draft-vocab space and maps after the argmax, so ``compute_logits`` is never the
path that runs.
"""

from vllm.config import VllmConfig
from vllm.model_executor.models.llama_eagle3 import (
    LlamaModel as Eagle3LlamaModel,
)

from vllm_rbln.patches import register_patch

# Captured at import time: the registry replaces targets outright, so wrapping
# upstream behaviour means holding the original here rather than copying its body.
_orig_model_init = Eagle3LlamaModel.__init__


@register_patch(
    target="vllm.model_executor.models.llama_eagle3.LlamaModel.__init__",
    reason=(
        "Name the EAGLE head's layers past the target's full depth, not past the "
        "count on this pipeline rank. Upstream passes "
        "`start_layer_id=model_config.get_num_layers(parallel_config)`, the per-rank "
        "count; with PP=1 that already equals the total, so the naming only goes "
        "wrong under pipeline parallelism, and then only on a rank that does not "
        "start at zero -- which is exactly where the drafter lives.\n"
        "\n"
        "The head's name then sorts in among the target's. On the 62-layer PP4 split "
        "stage 3 owns 47..61 and the head is named 15, so RBLN's KV ordering "
        "(`_get_ordered_layer_names`, which sorts on the raw index) puts the head "
        "first and every target layer's `name - start_layer` lookup is off by one. "
        "Nobody binds the last entry, and KV registration fails with "
        "`EnsureSyncedOnPhysicalView ... has no physical_view`.\n"
        "\n"
        "Naming past the full depth also makes the head's name identical on a "
        "prefill and a decode instance, which is what NIXL matches KV regions by, "
        "and keeps it clear of every target band for any split. The caller's "
        "`target_layer_count` is left alone: it pairs with the `layer_idx` the "
        "decoder layer is constructed with, which is the loop counter and not the "
        "index the name carries, and the only reader (`llama.py:193`) sits behind a "
        "`layer_types` walrus that this draft config leaves unset. "
        "TODO(vllm-project/vllm#50514): delete once that lands and is released."
    ),
)
def patched_eagle3_llama_model_init(
    self, *, vllm_config: VllmConfig, start_layer_id: int = 0, prefix: str = ""
) -> None:
    _orig_model_init(
        self,
        vllm_config=vllm_config,
        start_layer_id=vllm_config.model_config.hf_text_config.num_hidden_layers,
        prefix=prefix,
    )
