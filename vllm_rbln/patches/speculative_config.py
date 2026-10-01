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
"""Two answers `SpeculativeConfig` needs at construction, not afterwards.

Keep expert parallelism off the draft, and reserve no batch slots for DFlash
drafting.

`ParallelConfig` runs its validators at construction and
`_set_max_num_scheduled_tokens` reads the reservation inside
`VllmConfig.__post_init__`, so neither has a field to correct afterwards.
"""

from vllm.config import ParallelConfig, SpeculativeConfig

from vllm_rbln.patches import register_patch

# Captured at import time: the registry replaces targets outright, so wrapping
# upstream behaviour means holding the original here rather than copying its body.
_orig_create_draft_parallel_config = SpeculativeConfig.create_draft_parallel_config


@register_patch(
    target="vllm.config.SpeculativeConfig.create_draft_parallel_config",
    reason=(
        "Upstream hands the draft the target's enable_expert_parallel, and "
        "`verify_with_parallel_config` then refuses every EAGLE-family head on a "
        "MoE target: the head has no experts. Forwarding suits an MoE draft such "
        "as MTP and not a dense one, and the seam sees no draft model config to "
        "tell them apart, so this drops it for both."
    ),
)
def create_draft_parallel_config(
    target_parallel_config: ParallelConfig,
    speculative_draft_tensor_parallel_size: int,
) -> ParallelConfig:
    draft_parallel_config = _orig_create_draft_parallel_config(
        target_parallel_config, speculative_draft_tensor_parallel_size
    )
    draft_parallel_config.enable_expert_parallel = False
    return draft_parallel_config


# `vars()` because mypy types a class-level property access as its return type;
# at runtime either form gives the descriptor.
upstream_max_num_new_slots_for_drafting = vars(SpeculativeConfig)[
    "max_num_new_slots_for_drafting"
]


def _reserve_no_slots_for_dflash(self: SpeculativeConfig) -> int:
    if self.method == "dflash":
        return 0
    return upstream_max_num_new_slots_for_drafting.fget(self)


max_num_new_slots_for_drafting = register_patch(
    target="vllm.config.SpeculativeConfig.max_num_new_slots_for_drafting",
    key="vllm_rbln.patches.speculative_config.max_num_new_slots_for_drafting",
    owner_module="vllm_rbln.patches.speculative_config",
    reason=(
        "Upstream's reservation assumes a drafter that appends its draft "
        "tokens to the target's batch, which the RBLN DFlash proposer does not: "
        "it runs its own graph over its own batch. Upstream has no per-platform "
        "way to say so, and `_set_max_num_scheduled_tokens` consumes the value "
        "inside `VllmConfig.__post_init__`, before any platform hook could "
        "correct it."
    ),
)(property(_reserve_no_slots_for_dflash))
