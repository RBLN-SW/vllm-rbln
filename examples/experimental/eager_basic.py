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

import os

# enforce_eager bypasses torch.compile, so the model must run on real
# device='rbln' tensors rather than the compile-backend fake-CPU tensors used
# by the default path.
os.environ.setdefault("VLLM_RBLN_USE_VLLM_MODEL", "1")
os.environ.setdefault("VLLM_RBLN_USE_DEVICE_TENSOR", "1")

from vllm import LLM, SamplingParams  # noqa: E402

# Sample prompts.
prompts = [
    # "Hello, my name is",
    # "The president of the United States is",
    "The capital of France is",
    # "The future of AI is",
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=2)

# Create an LLM.
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    max_model_len=40 * 1024,
    block_size=1024,
    enable_chunked_prefill=True,
    max_num_batched_tokens=128,
    max_num_seqs=1,
    enforce_eager=True,
    gpu_memory_utilization=0.3,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
