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

"""Fail a request whose logprobs hold a NaN with an internal error (HTTP 500).

TODO(rbln): delete once the intermittent NaN logprob is fixed.
"""

import numpy as np
from vllm.v1.engine import EngineCoreOutput, FinishReason
from vllm.v1.engine.output_processor import OutputProcessor, OutputProcessorOutput

from vllm_rbln.logger import init_logger
from vllm_rbln.patches import register_patch

logger = init_logger(__name__)

_upstream_process_outputs = OutputProcessor.process_outputs


@register_patch(
    target="vllm.v1.engine.output_processor.OutputProcessor.process_outputs",
    reason=(
        "A NaN logprob from the engine fails JSON serialization and is answered "
        "with HTTP 400, misleading the client into blaming its request. End the "
        "request with finish_reason 'error' instead, which every endpoint "
        "answers with HTTP 500."
    ),
    key="vllm_rbln.patches.nan_logprobs.process_outputs",
    owner_module="vllm_rbln.patches.nan_logprobs",
)
def patched_process_outputs(
    self: OutputProcessor,
    engine_core_outputs: list[EngineCoreOutput],
    *args,
    **kwargs,
) -> OutputProcessorOutput:
    nan_req_ids = []
    for output in engine_core_outputs:
        logprobs = output.new_logprobs
        if logprobs is not None and np.isnan(logprobs.logprobs).any():
            output.finish_reason = FinishReason.ERROR
            nan_req_ids.append(output.request_id)

    processed = _upstream_process_outputs(self, engine_core_outputs, *args, **kwargs)

    if nan_req_ids:
        logger.error(
            "Requests %s failed with an internal error during generation: "
            "NaN in their logprobs",
            nan_req_ids,
        )
        processed.reqs_to_abort.extend(
            req_id for req_id in nan_req_ids if req_id not in processed.reqs_to_abort
        )
    return processed
