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

prompts = [
    "The capital of France is",
]

qwen3_0_6_model_id = "Qwen/Qwen3-0.6B"
qwen3_1_7_model_id = "Qwen/Qwen3-1.7B"
qwen3_4_model_id = "Qwen/Qwen3-4B"
qwen3_8_model_id = "Qwen/Qwen3-8B"
qwen3_30_moe_model_id = "Qwen/Qwen3-30B-A3B"
qwen1_5_moe_model_id = "Qwen/Qwen1.5-MoE-A2.7B"

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)
llm = LLM(
    model=qwen1_5_moe_model_id,
    max_model_len=8 * 1024,
    block_size=1024,
    enable_chunked_prefill=True,
    max_num_batched_tokens=128,
    max_num_seqs=1,
    tensor_parallel_size=4,
    enable_expert_parallel=True,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
