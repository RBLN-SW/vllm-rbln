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
import pytest
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.sampler import Sampler

from tests.vllm.v1.sample.utils import build_input_batch


# top_k=0 and top_p=1.0 leave that filter unset.
@pytest.mark.parametrize(("top_k", "top_p"), [(1, 1.0), (0, 0.1), (1, 0.1)])
def test_reference_sampler_filters_without_triton(top_k, top_p):
    params = SamplingParams(temperature=1.0, top_k=top_k, top_p=top_p, logprobs=1)
    metadata = build_input_batch(
        requests=[("0", params)], vocab_size=4, build=build_logitsprocs
    ).sampling_metadata
    logits = torch.tensor([[0.0, 1.0, 2.0, 3.0]])
    expected_logprob = logits.log_softmax(dim=-1)[:, 3]

    output = Sampler()(logits, metadata)

    assert output.sampled_token_ids.tolist() == [[3]]
    assert output.logprobs_tensors is not None
    torch.testing.assert_close(output.logprobs_tensors.logprobs[:, 0], expected_logprob)
