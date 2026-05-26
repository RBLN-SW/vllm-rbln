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

from collections.abc import Callable

import torch
from vllm.distributed import (
    get_tensor_model_parallel_world_size,
    tensor_model_parallel_all_reduce,
)
from vllm.model_executor.layers.fused_moe.layer import FusedMoE
from vllm.model_executor.layers.fused_moe.shared_fused_moe import SharedFusedMoE

from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

logger = init_logger(__name__)


@register_patch(
    target="vllm.model_executor.layers.fused_moe.shared_fused_moe.SharedFusedMoE.__init__",
    reason="To disable the upstream overlapped shared/fused MoE path on RBLN. (PR#293)",
)
def patched_shared_fused_moe_init(self: SharedFusedMoE, *args, **kwargs):
    FusedMoE.__init__(self, *args, **kwargs)

    # FIXME(RBLN) - disable use overlapped, not supported
    self.use_overlapped = False


@register_patch(
    target="vllm.model_executor.layers.fused_moe.shared_fused_moe.SharedFusedMoE.forward",
    reason=(
        "Replace SharedFusedMoE's forward while perserving shared expert output "
        "handling and required TP all-reduce behavior. (PR#367)"
    ),
)
def patched_shared_fused_moe_forward(
    self: SharedFusedMoE,
    hidden_states: torch.Tensor,
    router: Callable[[torch.Tensor], torch.Tensor],
) -> torch.Tensor:
    if self._shared_experts is not None:
        shared_out = self._shared_experts(hidden_states)

        if (
            self.reduce_results
            and get_tensor_model_parallel_world_size() > 1
            and self.must_reduce_shared_expert_outputs()
        ):
            shared_out = tensor_model_parallel_all_reduce(shared_out)
    else:
        shared_out = None

    fused_out = FusedMoE.forward(self, hidden_states, router)

    return shared_out, fused_out
