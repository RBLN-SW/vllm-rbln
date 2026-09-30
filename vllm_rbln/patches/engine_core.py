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
"""Drop the post-step draft pull -- the drafts come with the output now.

``patches/scheduler.py`` applies what the runner attaches to
``ModelRunnerOutput``, so the pull is redundant. Under PP it was also a
synchronous round-trip to ``output_rank`` that kept the engine from refilling
``batch_queue``, leaving the pipeline one microbatch deep at every request
boundary.

Self-disabling: with async scheduling the worker updates the drafts itself and
this never ran.
"""

from vllm.v1.engine.core import EngineCore

from vllm_rbln.patches import register_patch


@register_patch(
    target="vllm.v1.engine.core.EngineCore.post_step",
    reason=(
        "Drop the post-step draft pull. The RBLN runner attaches the drafts to "
        "ModelRunnerOutput and patches/scheduler.py applies them, so the pull "
        "is redundant; under PP it was a synchronous round-trip to the last "
        "stage that stopped the engine refilling batch_queue."
    ),
    key="vllm_rbln.patches.engine_core.post_step",
    owner_module="vllm_rbln.patches.engine_core",
)
def patched_post_step(self: EngineCore, model_executed: bool) -> None:
    return
