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
    """Answer the question based on the context below.
    Keep the answer short and concise. Respond "Unsure about answer"
    if not sure about the answer.
    Context: Teplizumab traces its roots to a New Jersey drug company
    called Ortho Pharmaceutical. There, scientists generated an early
    version of the antibody, dubbed OKT3. Originally sourced from mice,
    the molecule was able to bind to the surface of T cells and limit
    their cell-killing potential. In 1986, it was approved to help prevent
    organ rejection after kidney transplants, making it the first therapeutic
    antibody allowed for human use.
    Question: What was OKT3 originally sourced from?
    Answer:"""
]

# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.0, max_tokens=32)

# 128 chunked prefill
llm = LLM(
    model="meta-llama/Llama-3.2-1B",
    max_model_len=128 * 1024,
    block_size=1024,
    enable_chunked_prefill=True,
    max_num_batched_tokens=128,
    max_num_seqs=1,
)

# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
"""
GPU Reference:
Prompt: 'Answer the question based on the context below.
Keep the answer short and concise. Respond "Unsure about answer"
if not sure about the answer.\n
Context: Teplizumab traces its roots
to a New Jersey drug company called Ortho Pharmaceutical.
There, scientists generated an early version of the antibody, dubbed OKT3.
Originally sourced from mice, the molecule was able to bind to the surface
of T cells and limit their cell-killing potential.
In 1986, it was approved to help prevent organ rejection
after kidney transplants,
making it the first therapeutic antibody allowed for human use.\n
Question: What was OKT3 originally sourced from?\n
Answer:',
Generated text: ' OKT3 was sourced from mice.\n
Explanation: OKT3 was sourced from mice.
It was originally sourced from mice.'
"""
