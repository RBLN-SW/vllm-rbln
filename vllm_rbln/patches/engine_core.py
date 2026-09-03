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
"""Keep the post-step draft fetch out of the prefill path.

``EngineCore.post_step`` pulls the drafts out of the worker that produced them so
the scheduler can size and allocate the next verification step. It runs whenever
spec decode is on and async scheduling is off -- on RBLN that is every PP run,
since PP under async scheduling is not supported yet (see ``platform.py``).

Between decode steps the fetch is a true dependence: step N's drafts are what
step N+1 verifies. Between prefill chunks there is none, and upstream's own
consumer says so -- ``Scheduler.update_draft_token_ids`` drops the value on
arrival ("Ignore draft tokens for prefill chunks"). ``post_step`` only receives
``model_executed``, so it cannot tell the two apart.

At ``pipeline_parallel_size == 1`` that costs nothing: the engine has one batch
in flight and already blocks each step. Under PP it costs the pipeline. The
fetch is a synchronous round-trip to ``output_rank``, the last stage, issued
right after ``step_with_batch_queue`` returns early to refill ``batch_queue`` --
so the engine cannot get back to ``schedule()`` until the chunk has traversed
every stage, and the pipeline runs about one microbatch deep.

The guard is the consumer's own condition moved ahead of the round-trip. The
step that schedules a request's last chunk already reports ``is_prefill_chunk ==
False`` (``_update_after_schedule`` advances ``num_computed_tokens`` first), so
the fetch resumes on exactly the step whose drafts the next one will verify.

Self-disabling: with async scheduling the ``not async_scheduling`` term is false
and the guard is never reached.
"""

from vllm.v1.engine.core import EngineCore

from vllm_rbln.patches import register_patch


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
