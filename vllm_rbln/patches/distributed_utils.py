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

from vllm.distributed.utils import get_pp_indices

from vllm_rbln import envs
from vllm_rbln.patches import register_patch

original_get_pp_indices = get_pp_indices


@register_patch(
    target="vllm.distributed.utils.get_pp_indices",
    reason=(
        "Honor VLLM_RBLN_NUM_HIDDEN_LAYERS: build only the first N decoder "
        "layers, so bring-up and tests compile a fraction of the model "
        "instead of all of it."
    ),
    condition=lambda: envs.VLLM_RBLN_NUM_HIDDEN_LAYERS > 0,
)
def patched_get_pp_indices(
    num_hidden_layers: int, pp_rank: int, pp_size: int
) -> tuple[int, int]:
    # `make_layers` is what truncating end_layer actually shrinks, but model
    # modules bind it at import time; it looks `get_pp_indices` up inside its
    # own body, so this target lands regardless of import order.
    layer_limit = envs.VLLM_RBLN_NUM_HIDDEN_LAYERS

    # `condition` gates on the value at apply time; this reads it per call, so
    # honor a later 0 rather than truncating to nothing.
    if layer_limit <= 0:
        return original_get_pp_indices(num_hidden_layers, pp_rank, pp_size)

    # A stage with no attention layer reports an attention-free spec worth
    # `num_blocks=1`, which `get_kv_cache_configs` imposes on every other stage.
    if layer_limit < pp_size:
        raise ValueError(
            f"VLLM_RBLN_NUM_HIDDEN_LAYERS={layer_limit} is smaller than "
            f"pipeline_parallel_size={pp_size}, which would leave a pipeline "
            "stage with no layer. Raise it to at least the pipeline parallel "
            "size."
        )

    return original_get_pp_indices(
        min(num_hidden_layers, layer_limit), pp_rank, pp_size
    )
