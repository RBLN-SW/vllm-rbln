# Copyright 2025 Rebellions Inc. All rights reserved.
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

from vllm import LLM, SamplingParams

llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    block_size=1024,
    enable_chunked_prefill=True,
    max_num_batched_tokens=128,
    max_num_seqs=1,
    # The profile results can be visualized using https://ui.perfetto.dev/
    profiler_config={
        "profiler": "torch",
        "torch_profiler_dir": "./profile",
    },
)

# The first run initializes the compiled models.
# We don't want to capture that in profile results.
llm.generate(".", SamplingParams(temperature=0.0, max_tokens=2))

prompts = [
    "The president of the United States is",
]
sampling_params = SamplingParams(temperature=0.0)

llm.start_profile()
outputs = llm.generate(prompts, sampling_params)
llm.stop_profile()

# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
