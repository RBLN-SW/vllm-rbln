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
"""Guard RBLN shutdown and fetch drafts only when a request can verify them.

``EngineCore`` cleanup skips ``torch.accelerator.empty_host_cache()``, which
faults on RBLN after a device tensor has existed, until torch-rbln 0.12.0.

Under PP the pull is a synchronous round-trip to ``output_rank`` that stops the
engine refilling ``batch_queue``, so the pipeline runs one microbatch deep.
``is_prefill_chunk`` alone reads a just-scheduled last chunk as a decode;
holding a sampled token is the execution fact that it can verify.

The first decode step of each request goes unspeculated as a result, and
loosening this guard does not recover it.
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
        "Skip the post-step draft fetch while no running request can verify "
        "drafts yet. The scheduler discards drafts for prefill chunks, but "
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
        will_verify_drafts = any(
            not request.is_prefill_chunk and request.num_output_tokens > 0
            for request in running
        )
        if running and not will_verify_drafts:
            return
        draft_token_ids = self.model_executor.take_draft_token_ids()
        if draft_token_ids is not None:
            self.scheduler.update_draft_token_ids(draft_token_ids)
