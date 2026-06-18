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

import fire
import torch
from vllm import LLM


def get_input_prompts() -> list[str]:
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f"Instruct: {task_description}\nQuery:{query}"

    # Each query must come with a one-sentence instruction
    # that describes the task
    task = "Given a web search query, retrieve relevant passages that answer the query"

    queries = [
        get_detailed_instruct(task, "What is the capital of China?"),
        get_detailed_instruct(task, "Explain gravity"),
    ]
    documents = [
        "The capital of China is Beijing.",
        (
            "Gravity is a force that attracts two bodies towards each other. "
            "It gives weight to physical objects and "
            "is responsible for the movement of planets around the sun."
        ),
    ]

    inputs_texts = queries + documents
    return inputs_texts


def main(
    num_input_prompt: int = 2,
    model: str = "Qwen/Qwen3-Embedding-0.6B",
):
    llm = LLM(
        model=model,
        block_size=4096,
        runner="pooling",
    )
    prompt_list = get_input_prompts()
    if len(prompt_list) > 2 * num_input_prompt:
        raise RuntimeError(
            "The len(QUERIES) and len(DOCUMENTS) ",
            "should be equal with 2 * `num_input_prompt`.",
        )
    prompt_list = prompt_list[: num_input_prompt * 2]

    outputs = llm.encode(prompt_list, pooling_task="embed")

    embeddings = torch.stack([o.outputs.data for o in outputs])
    scores = embeddings[:num_input_prompt] @ embeddings[num_input_prompt:].T

    print(f"scores: {scores.tolist()}")


if __name__ == "__main__":
    fire.Fire(main)
