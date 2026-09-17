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

Keep the draft model off the target's pipeline-parallel split, and reserve no
batch slots for DFlash drafting.

Upstream copies the target's `pipeline_parallel_size` onto the draft, which then
fails `ModelConfig.verify_with_parallel_config`: the EAGLE3 draft head does not
satisfy `SupportsPP`, because its `forward` takes no `intermediate_tensors`. The
draft only ever runs on the last stage -- `RBLNModelRunner` already constructs it
under `get_pp_group().is_last_rank` -- so inheriting the split describes a
topology that never exists.

`ParallelConfig` derives `world_size` in its validator, so the size has to be
right at construction; there is no field to correct afterwards.
"""

from vllm.config import ParallelConfig, SpeculativeConfig

from vllm_rbln.patches import register_patch


@register_patch(
    target="vllm.config.SpeculativeConfig.create_draft_parallel_config",
    reason=(
        "Upstream gives the draft the target's pipeline_parallel_size, so a "
        "pipeline-parallel target rejects every EAGLE-family draft head at config "
        "validation -- the heads implement no intermediate_tensors forward and so "
        "fail SupportsPP. Upstream cannot express 'this submodel runs on one stage' "
        "any other way; the field is the topology. "
        "TODO(vllm-project/vllm#50514): delete once that lands and is released."
    ),
)
def create_draft_parallel_config(
    target_parallel_config: ParallelConfig,
    speculative_draft_tensor_parallel_size: int,
) -> ParallelConfig:
    return ParallelConfig(
        pipeline_parallel_size=1,
        tensor_parallel_size=speculative_draft_tensor_parallel_size,
        distributed_executor_backend=target_parallel_config.distributed_executor_backend,
        max_parallel_loading_workers=target_parallel_config.max_parallel_loading_workers,
        disable_custom_all_reduce=target_parallel_config.disable_custom_all_reduce,
        ray_workers_use_nsight=target_parallel_config.ray_workers_use_nsight,
        placement_group=target_parallel_config.placement_group,
    )


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
