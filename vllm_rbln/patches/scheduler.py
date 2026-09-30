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
"""Take the drafts off the output the worker already sent.

The pull in ``post_step`` only answers while the step that produced the drafts
is still the worker's batch, so a request's first decode step never sees the
drafts from its last prefill chunk. Reading them off ``ModelRunnerOutput``
removes the ordering entirely -- they arrive with the tokens they belong to.
"""

from vllm.v1.core.sched.scheduler import Scheduler

from vllm_rbln.patches import register_patch

_update_from_output = Scheduler.update_from_output


@register_patch(
    target="vllm.v1.core.sched.scheduler.Scheduler.update_from_output",
    reason=(
        "Apply the draft tokens the RBLN runner attaches to ModelRunnerOutput. "
        "Upstream pulls them in post_step with a separate RPC that only answers "
        "while the producing step is still the worker's batch, which strands "
        "the drafts a request's first decode step would verify."
    ),
    key="vllm_rbln.patches.scheduler.update_from_output",
    owner_module="vllm_rbln.patches.scheduler",
)
def patched_update_from_output(self, scheduler_output, model_output):
    engine_core_outputs = _update_from_output(self, scheduler_output, model_output)
    draft_token_ids = getattr(model_output, "rbln_draft_token_ids", None)
    if draft_token_ids is not None:
        self.update_draft_token_ids(draft_token_ids)
    return engine_core_outputs
