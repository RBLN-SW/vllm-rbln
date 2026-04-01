# Copyright 2025 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
from vllm import LLM, SamplingParams

DENSE_MODELS = [
    "meta-llama/Llama-3.2-1B",
    # TODO(jiwoo.park) We will add more models.
    # "Qwen/Qwen3-0.6B",
]

MOE_MODELS = [
    "Qwen/Qwen1.5-MoE-A2.7B",
]

PROMPTS = [
    "The capital of France is",
]


@pytest.mark.parametrize("model", DENSE_MODELS)
@pytest.mark.parametrize("max_num_seqs", [1])
@pytest.mark.parametrize("max_tokens", [5])
def test_dense_models(
    monkeypatch: pytest.MonkeyPatch,
    model: str,
    max_num_seqs: int,
    max_tokens: int,
) -> None:
    with monkeypatch.context() as m:
        m.setenv("VLLM_RBLN_USE_VLLM_MODEL", "1")
        m.setenv("VLLM_DISABLE_COMPILE_CACHE", "1")

        prompts = PROMPTS

        sampling_params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        llm = LLM(
            model=model,
            max_model_len=4 * 1024,
            block_size=1024,
            enable_chunked_prefill=True,
            max_num_batched_tokens=128,
            max_num_seqs=max_num_seqs,
        )
        outputs = llm.generate(prompts, sampling_params)
        for output in outputs:
            generated_text = output.outputs[0].text
            assert len(generated_text) > 0
