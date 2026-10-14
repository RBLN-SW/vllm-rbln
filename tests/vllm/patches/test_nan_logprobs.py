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

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.engine import EngineCoreOutput, EngineCoreRequest
from vllm.v1.engine.output_processor import OutputProcessor
from vllm.v1.outputs import LogprobsLists


@pytest.mark.parametrize(
    ("logprob", "finish_reason", "abort_ids"),
    [
        pytest.param(float("nan"), "error", ["internal"], id="nan_during_generation"),
        pytest.param(-0.5, None, [], id="finite_logprobs"),
    ],
)
def test_process_logprobs(logprob, finish_reason, abort_ids):
    processor = OutputProcessor(tokenizer=None, log_stats=False)
    request = EngineCoreRequest(
        request_id="internal",
        external_req_id="external",
        prompt_token_ids=[1, 2],
        mm_features=None,
        sampling_params=SamplingParams(logprobs=0, detokenize=False),
        pooling_params=None,
        arrival_time=0.0,
        lora_request=None,
        cache_salt=None,
        data_parallel_rank=None,
    )
    processor.add_request(request, prompt=None)
    output = EngineCoreOutput(
        request_id="internal",
        new_token_ids=[3],
        new_logprobs=LogprobsLists(
            logprob_token_ids=np.array([[3]]),
            logprobs=np.array([[logprob]]),
            sampled_token_ranks=np.array([1]),
        ),
    )

    processed = processor.process_outputs([output])

    (result,) = processed.request_outputs
    assert result.outputs[0].finish_reason == finish_reason
    assert result.finished == (finish_reason is not None)
    assert processed.reqs_to_abort == abort_ids
    assert processor.has_request("internal") == (finish_reason is None)
    assert result.outputs[0].token_ids == [3]
    if finish_reason is None:
        assert result.outputs[0].logprobs[0][3].logprob == logprob
