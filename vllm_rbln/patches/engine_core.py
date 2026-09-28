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
"""Two `EngineCore` patches: the shutdown that faults on RBLN, and the
post-step draft fetch the prefill path does not need.

``EngineCore.post_step`` fetches the drafts the next step verifies. Between
prefill chunks there is nothing to verify, and upstream's own consumer says so
-- ``Scheduler.update_draft_token_ids`` drops the value on arrival. Under PP
the fetch is a synchronous round-trip to the last stage, issued where
``step_with_batch_queue`` would otherwise refill ``batch_queue``, so paying it
for nothing costs the pipeline its depth.

The guard is that consumer's own condition moved ahead of the round-trip, so
the fetch resumes on exactly the step whose drafts the next one verifies: the
step that schedules a request's last chunk already reports
``is_prefill_chunk == False``.
"""

from importlib.metadata import version

import torch
from packaging.version import Version
from vllm.distributed import parallel_state
from vllm.v1.engine.core import EngineCore

from vllm_rbln.patches import register_patch

assert Version(version("torch_rbln")) < Version("0.12.0"), (
    "torch-rbln 0.12.0 fixes the empty_host_cache error. Delete "
    "patched_cleanup_dist_env_and_memory, _no_host_cache_to_empty and this "
    "assert."
)

original_cleanup_dist_env_and_memory = parallel_state.cleanup_dist_env_and_memory


def _no_host_cache_to_empty() -> None:
    """Stand in for torch.accelerator.empty_host_cache() during cleanup."""


@register_patch(
    # EngineCore is the only caller and it from-imports the name, so its own
    # binding is the one that has to change; replacing the definition in
    # parallel_state leaves that binding pointing at the original.
    target="vllm.v1.engine.core.cleanup_dist_env_and_memory",
    reason=(
        "torch.accelerator.empty_host_cache() faults in the RBLN accelerator "
        "once a device tensor has existed, so EngineCore takes the process "
        "down on its way out. Upstream guards the call with `except "
        "AttributeError`, which torch 2.9 made unreachable by shipping the "
        "API, and a fault is not an exception anyway. Neutralise that one "
        "call and delegate the rest."
    ),
)
def patched_cleanup_dist_env_and_memory(shutdown_ray: bool = False) -> None:
    accelerator = torch.accelerator
    original_empty_host_cache = accelerator.empty_host_cache
    accelerator.empty_host_cache = _no_host_cache_to_empty
    try:
        original_cleanup_dist_env_and_memory(shutdown_ray)
    finally:
        accelerator.empty_host_cache = original_empty_host_cache


@register_patch(
    target="vllm.v1.engine.core.EngineCore.post_step",
    reason=(
        "Skip the post-step draft fetch while every running request is still "
        "mid-prefill. The scheduler discards drafts for prefill chunks, but "
        "under PP the fetch is a synchronous round-trip to the last stage that "
        "stops the engine from refilling batch_queue, so the pipeline runs one "
        "microbatch deep instead of pipeline_parallel_size."
    ),
    key="vllm_rbln.patches.engine_core.post_step",
    owner_module="vllm_rbln.patches.engine_core",
)
def patched_post_step(self: EngineCore, model_executed: bool) -> None:
    if self.check_for_draft_tokens and not self.async_scheduling and model_executed:
        running = self.scheduler.running
        if running and all(request.is_prefill_chunk for request in running):
            return
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)
