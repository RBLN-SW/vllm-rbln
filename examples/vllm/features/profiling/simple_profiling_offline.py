# Copyright 2026 Rebellions Inc. All rights reserved.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Capture an inference trace with the PyTorch profiler on RBLN.

Set RBLN_PROFILER=1 before launching to also enable RBLN device profiling.
Open the traces saved under ./profile in https://ui.perfetto.dev/.
"""

from vllm import LLM, SamplingParams
from vllm.config import ProfilerConfig


def main() -> None:
    llm = LLM(
        model="meta-llama/Llama-3.2-1B-Instruct",
        model_impl="vllm",
        block_size=1024,
        max_num_batched_tokens=512,
        enable_prefix_caching=False,
        profiler_config=ProfilerConfig(
            profiler="torch",
            torch_profiler_dir="./profile",
        ),
    )
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    sampling_params = SamplingParams(temperature=0.8, top_p=0.95)

    # Exclude first-use initialization from the trace.
    llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=2))

    llm.start_profile()
    outputs = llm.generate(prompts, sampling_params)
    llm.stop_profile()

    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt: {prompt!r}\nGenerated text: {generated_text!r}")
        print("-" * 50)


if __name__ == "__main__":
    main()
