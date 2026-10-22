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
"""DFlash and DSpark drafting at config construction."""

from vllm.config import SpeculativeConfig, VllmConfig

from vllm_rbln.patches import register_patch

# `vars()` because mypy types a class-level property access as its return type;
# at runtime either form gives the descriptor.
upstream_max_num_new_slots_for_drafting = vars(SpeculativeConfig)[
    "max_num_new_slots_for_drafting"
]
upstream_v1_model_runner_unsupported_features = (
    VllmConfig._get_v1_model_runner_unsupported_features
)


def _reserve_no_slots_for_dflash(self: SpeculativeConfig) -> int:
    if self.method in ("dflash", "dspark"):
        return 0
    return upstream_max_num_new_slots_for_drafting.fget(self)


max_num_new_slots_for_drafting = register_patch(
    target="vllm.config.SpeculativeConfig.max_num_new_slots_for_drafting",
    key="vllm_rbln.patches.speculative_config.max_num_new_slots_for_drafting",
    owner_module="vllm_rbln.patches.speculative_config",
    reason=(
        "Upstream's reservation assumes a drafter that appends its draft "
        "tokens to the target's batch, which the RBLN DFlash and DSpark "
        "proposers do not: each runs its own graph over its own batch. Upstream "
        "has no per-platform way to say so, and `_set_max_num_scheduled_tokens` "
        "consumes the value inside `VllmConfig.__post_init__`, before any "
        "platform hook could correct it."
    ),
)(property(_reserve_no_slots_for_dflash))


@register_patch(
    target="vllm.config.VllmConfig._get_v1_model_runner_unsupported_features",
    reason=(
        "Upstream refuses DSpark outside Model Runner V2, the only runner with "
        "a GPU DSpark speculator. RBLN runs every model on its own runner, "
        "which serves DSpark through its DFlash proposer, and the refusal is "
        "raised inside `VllmConfig.__post_init__`, before any platform hook."
    ),
)
def v1_model_runner_unsupported_features(self: VllmConfig) -> list[str]:
    return [
        feature
        for feature in upstream_v1_model_runner_unsupported_features(self)
        if feature != "dspark speculative decoding"
    ]
