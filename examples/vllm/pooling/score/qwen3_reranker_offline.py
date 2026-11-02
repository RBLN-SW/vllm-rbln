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

"""Score query-document pairs with Qwen3-Reranker-0.6B on RBLN.

hf_overrides converts the original yes/no token head into a sequence
classification head. The scoring template supplies the reranker instruction.
"""

from pathlib import Path

from vllm import LLM


def main() -> None:
    llm = LLM(
        model="Qwen/Qwen3-Reranker-0.6B",
        model_impl="vllm",
        runner="pooling",
        hf_overrides={
            "architectures": ["Qwen3ForSequenceClassification"],
            "classifier_from_token": ["no", "yes"],
            "is_original_qwen3_reranker": True,
        },
        block_size=1024,
        max_num_batched_tokens=512,
    )
    chat_template = (
        Path(__file__).parent / "template/qwen3_reranker.jinja"
    ).read_text()
    queries = ["What is the capital of China?", "Explain gravity"]
    documents = [
        "The capital of China is Beijing.",
        "Gravity is a force that attracts two bodies towards each other. "
        "It gives weight to physical objects and is responsible for the movement "
        "of planets around the sun.",
    ]
    outputs = llm.score(queries, documents, chat_template=chat_template)
    print("-" * 30)
    print("Relevance scores:", [output.outputs.score for output in outputs])
    print("-" * 30)


if __name__ == "__main__":
    main()
