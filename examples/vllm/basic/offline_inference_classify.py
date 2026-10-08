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

"""Classify text using the vllm model path on RBLN."""

import argparse

from vllm import LLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="jason9693/Qwen2.5-1.5B-apeach")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        model_impl="vllm",
        runner="pooling",
        block_size=1024,
        max_num_batched_tokens=512,
    )
    prompts = [
        "Hello, my name is",
        "The president of the United States is",
        "The capital of France is",
        "The future of AI is",
    ]
    outputs = llm.classify(prompts)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for prompt, output in zip(prompts, outputs):
        probs = output.outputs.probs
        probs_trimmed = (str(probs[:16])[:-1] + ", ...]") if len(probs) > 16 else probs
        print(
            f"Prompt: {prompt!r} \n"
            f"Class Probabilities: {probs_trimmed} (size={len(probs)})"
        )
        print("-" * 60)


if __name__ == "__main__":
    main()
