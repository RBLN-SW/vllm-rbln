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
"""Fetch the post-step drafts only when a running request can verify them.

Under PP the pull is a synchronous round-trip to ``output_rank`` that stops the
engine refilling ``batch_queue``, so the pipeline runs one microbatch deep.
``is_prefill_chunk`` alone reads a just-scheduled last chunk as a decode;
holding a sampled token is the execution fact that it can verify.

The first decode step of each request goes unspeculated as a result, and
loosening this guard does not recover it.
"""

from vllm.v1.engine.core import EngineCore

from vllm_rbln.patches import register_patch


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
