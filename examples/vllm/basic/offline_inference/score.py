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

"""Score query-document pairs using the vllm model path on RBLN.

Qwen3-Embedding encodes each text separately and scores pairs by cosine similarity.
"""

import argparse

from vllm import LLM


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen3-Embedding-0.6B")
    args = parser.parse_args()

    llm = LLM(
        model=args.model,
        model_impl="vllm",
        runner="pooling",
        block_size=1024,
        max_num_batched_tokens=512,
    )
    query = "What is the capital of France?"
    documents = [
        "The capital of Brazil is Brasilia.",
        "The capital of France is Paris.",
    ]
    outputs = llm.score(query, documents)
    print("\nGenerated Outputs:\n" + "-" * 60)
    for document, output in zip(documents, outputs):
        print(f"Pair: {[query, document]!r}\nScore: {output.outputs.score}")
        print("-" * 60)


if __name__ == "__main__":
    main()
