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

"""Generate text using the vllm model path on RBLN.

Set model_impl="vllm" to explicitly select the vllm model path. If omitted,
model_impl defaults to "auto". Without another path selection, "auto" uses the
optimum model path for architectures supported by optimum-rbln and the vllm
model path otherwise.
"""

import argparse

from vllm import LLM, SamplingParams


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        model_impl="vllm",
        block_size=1024,
        max_num_batched_tokens=512,
    )
    prompts = [
        "Hello, my name is",
        "The capital of France is",
        "The future of AI is",
        "A good way to learn programming is",
    ]
    outputs = llm.generate(prompts, SamplingParams(temperature=0.0, max_tokens=32))
    # Print the outputs.
    print("\nGenerated Outputs:\n" + "-" * 60)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"Prompt:    {prompt!r}")
        print(f"Output:    {generated_text!r}")
        print("-" * 60)


if __name__ == "__main__":
    main()
